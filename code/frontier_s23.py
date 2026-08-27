"""s23 frontier: speed x fidelity of SID decoders on the TIGER/SID-MLP substrate.

One verify primitive (frozen-teacher exact rescore of candidate SIDs), three drafters:
  - none            -> teacher full beam (the reference; fidelity target = its top-10 items)
  - teacher-to-H    -> C4/horizon: teacher beam to depth H, trie-complete, rescore
  - student (SIDMLP)-> cascade: student full beam top-M proposals, teacher rescores

Modes:
  fid    : full-TEST fidelity + coverage + where-lost localization + cascade + C4 arms
  timing : per-arm latency (exclusive GPU), bs in {1,32}

Fidelity metric = jac@10 vs teacher beam-50 top-10 items (the s12/s22 ladder convention, bf16).
Sanity anchors: (A) rescore ranks teacher's own beam outputs consistently with beam order;
(B) plain student jac@10 reproduces the s22 ladder number.
"""
import sys, os, json, math, time, argparse
from collections import defaultdict
import torch, torch.nn.functional as F

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, os.path.join(R, "repos", "SID-MLP"))
os.chdir(os.path.join(R, "repos"))
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)      # teacher TIGER ckpt
ap.add_argument('--lens_path', required=True)       # student ckpt
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--mode', choices=['fid', 'timing'], default='fid')
ap.add_argument('--variant', default='base', choices=['base', 'pd', 'pdctx'])  # pd beam lands with the s23 students
ap.add_argument('--pd_depth', type=int, default=0, help='PD context depth override; 0 reads checkpoint args.')
ap.add_argument('--metrics_only', action='store_true', help='Skip cascade/C4 rescoring and only compute raw student safety metrics.')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--M', type=int, default=200)       # student proposal width
ap.add_argument('--teacher_beam', type=int, default=50)
ap.add_argument('--c4_H', type=int, default=3)
ap.add_argument('--c4_W', type=int, default=50)
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--timing_batches', type=int, default=16)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='')
ap.add_argument('--cov_out', default='')   # s49: per-request Coverage@C dump (npz)
                                           # for the doc 1.6A bootstrap CI
args = ap.parse_args()
args.encoder_ckpt = None; args.fp32 = False
dev = torch.device('cuda')
nb16 = configure_precision(args, dev)

pipe = build_tiger_pipeline(args)
tiger = pipe.model.to(dev).eval()
t5 = tiger.t5
if nb16: t5.bfloat16()
for p in t5.parameters(): p.requires_grad_(False)
tk = _torch_load(args.checkpoint, weights_only=False)
lens, lens_ckpt = build_sidmlp_from_checkpoint(args.lens_path, tk['t5.shared.weight'], dev)
CK_ARGS = (lens_ckpt.get('args', {}) or {}) if isinstance(lens_ckpt, dict) else {}
PD_DEPTH = int(args.pd_depth or CK_ARGS.get('pd_depth', 4))
PD_MODE = str(CK_ARGS.get('pd_mode', getattr(lens, 'pd_mode', 'full')) or 'full').lower()
if hasattr(lens, 'pd_depth'):
    lens.pd_depth = PD_DEPTH
if hasattr(lens, 'pd_mode'):
    lens.pd_mode = PD_MODE
if nb16: lens.bfloat16()
del tk
vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path, graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, keys4_sorted, vals4_sorted, _im,
 valid_items_4) = build_valid_tensors(vs, dev)
EOS = t5.config.eos_token_id
START = t5.config.decoder_start_token_id

# trie completion maps (vocab-space tuples)
pref2_map, pref3_map = defaultdict(list), defaultdict(list)
for quad in valid_items_4.keys():
    pref2_map[quad[:2]].append(quad)
    pref3_map[quad[:3]].append(quad)

def sid_to_item(tup):
    return valid_items_4.get(tuple(int(x) for x in tup), -1)

def dedup_items(sid_list):
    seen = set(); out = []
    for tup in sid_list:
        it = sid_to_item(tup)
        if it == -1 or it in seen: continue
        seen.add(it); out.append(it)
    return out

# ---------------- student beam (returns per-stage survivors) ----------------
def student_beam(enc, pad, W):
    """enc: (B,S,D) encoder states; pad: (B,S) bool padding mask. Returns:
    quads (B,W,4) final SID tokens ranked, plus per-stage survivor token tensors."""
    B = enc.shape[0]
    ctx = lens.encode(enc, pad)
    d1_lp = F.log_softmax(lens.head1(ctx).float(), dim=-1)
    W1 = min(W, CODEBOOK_SIZE)
    d1_scores, d1_idx = torch.topk(d1_lp, W1, dim=1)
    d1_tokens = d1_idx + DIGIT_OFFSETS[0]
    stage1 = d1_tokens.clone()
    ctx_flat = ctx.unsqueeze(1).expand(-1, W1, -1).reshape(B*W1, -1)
    d1_flat = d1_tokens.reshape(B*W1)
    logits2 = lens.head2(torch.cat([ctx_flat, lens.embedding(d1_flat)], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE-1)
    logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float('-inf'))
    d2_lp = F.log_softmax(logits2, 1).view(B, W1, CODEBOOK_SIZE)
    score2 = d1_scores.unsqueeze(-1) + d2_lp
    best2_score, best2_idx = torch.topk(score2.masked_fill(torch.isnan(score2), float('-inf')).view(B, -1), W, 1)
    sel_beam2 = best2_idx // CODEBOOK_SIZE; sel_d2 = best2_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam2); d2_tokens = sel_d2 + DIGIT_OFFSETS[1]
    stage2 = (d1_tokens.clone(), d2_tokens.clone())
    prefix_score = best2_score
    ctx_flat = ctx.unsqueeze(1).expand(-1, W, -1).reshape(B*W, -1)
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
    logits3 = lens.head3(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat)], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE-1)
    d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE-1)
    logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float('-inf'))
    d3_lp = F.log_softmax(logits3, 1).view(B, W, CODEBOOK_SIZE)
    score3 = prefix_score.unsqueeze(-1) + d3_lp
    best3_score, best3_idx = torch.topk(score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), W, 1)
    sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam3); d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
    d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
    stage3 = (d1_tokens.clone(), d2_tokens.clone(), d3_tokens.clone())
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1); d3_flat = d3_tokens.reshape(-1)
    logits4 = lens.head4(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat),
                                    lens.embedding(d3_flat)], 1)).float()
    key3 = triplet_key_int(d1_flat - DIGIT_OFFSETS[0], d2_flat - DIGIT_OFFSETS[1], d3_flat - DIGIT_OFFSETS[2])
    logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    d4_lp = F.log_softmax(logits4, 1)
    top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, 1)
    d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]
    quad = best3_score.unsqueeze(-1) + top_d4_val.view(B, W, CODEBOOK_SIZE)
    topq_score, topq_idx = torch.topk(quad.masked_fill(torch.isnan(quad), float('-inf')).view(B, -1), W, 1)
    beam_idx = topq_idx // CODEBOOK_SIZE
    d1f = torch.gather(d1_tokens, 1, beam_idx); d2f = torch.gather(d2_tokens, 1, beam_idx)
    d3f = torch.gather(d3_tokens, 1, beam_idx); d4f = torch.gather(d4_tokens.view(B, -1), 1, topq_idx)
    quads = torch.stack([d1f, d2f, d3f, d4f], -1)
    return quads, topq_score, stage1, stage2, stage3

# ---------------- PD student beam (s23 prefix-dependent-context variant) ----------------
def student_beam_pd(enc, pad, W):
    """Beam for SIDMLPPD: per-digit ctx recomputed per hypothesis (query = q0 + pproj(prefix))."""
    B, S, _ = enc.shape
    q0, k, v = lens.q0_kv(enc, pad)
    ctx1 = lens.ctx_from_q(q0, k, v, pad)
    d1_lp = F.log_softmax(lens.head1(ctx1).float(), dim=-1)
    W1 = min(W, CODEBOOK_SIZE)
    d1_scores, d1_idx = torch.topk(d1_lp, W1, dim=1)
    d1_tokens = d1_idx + DIGIT_OFFSETS[0]
    stage1 = d1_tokens.clone()

    def expand(t, w):
        return t.unsqueeze(1).expand(B, w, *t.shape[1:]).reshape(B * w, *t.shape[1:])
    k_e, v_e = expand(k, W1), expand(v, W1)
    pad_e = expand(pad, W1)
    q0_e = expand(q0, W1)
    d1_flat = d1_tokens.reshape(-1)
    e1 = lens.embedding(d1_flat)
    ctx1_w1 = expand(ctx1, W1)
    ctx2 = lens.prefix_ctx(q0_e, ctx1_w1, k_e, v_e, pad_e, lens.pproj2(e1)) if PD_DEPTH >= 2 else ctx1_w1
    logits2 = lens.head2(torch.cat([ctx2, e1], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE-1)
    logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float('-inf'))
    d2_lp = F.log_softmax(logits2, 1).view(B, W1, CODEBOOK_SIZE)
    score2 = d1_scores.unsqueeze(-1) + d2_lp
    best2_score, best2_idx = torch.topk(score2.masked_fill(torch.isnan(score2), float('-inf')).view(B, -1), W, 1)
    sel_beam2 = best2_idx // CODEBOOK_SIZE; sel_d2 = best2_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam2); d2_tokens = sel_d2 + DIGIT_OFFSETS[1]
    stage2 = (d1_tokens.clone(), d2_tokens.clone())
    prefix_score = best2_score
    if W != W1:
        k_e, v_e, pad_e, q0_e = expand(k, W), expand(v, W), expand(pad, W), expand(q0, W)
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
    e1 = lens.embedding(d1_flat); e2 = lens.embedding(d2_flat)
    ctx1_w = expand(ctx1, W)
    ctx3 = lens.prefix_ctx(q0_e, ctx1_w, k_e, v_e, pad_e, lens.pproj3(torch.cat([e1, e2], 1))) if PD_DEPTH >= 3 else ctx1_w
    logits3 = lens.head3(torch.cat([ctx3, e1, e2], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE-1)
    d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE-1)
    logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float('-inf'))
    d3_lp = F.log_softmax(logits3, 1).view(B, W, CODEBOOK_SIZE)
    score3 = prefix_score.unsqueeze(-1) + d3_lp
    best3_score, best3_idx = torch.topk(score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), W, 1)
    sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam3); d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
    d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
    stage3 = (d1_tokens.clone(), d2_tokens.clone(), d3_tokens.clone())
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1); d3_flat = d3_tokens.reshape(-1)
    e1 = lens.embedding(d1_flat); e2 = lens.embedding(d2_flat); e3 = lens.embedding(d3_flat)
    ctx1_w = expand(ctx1, W)
    ctx4 = lens.prefix_ctx(q0_e, ctx1_w, k_e, v_e, pad_e, lens.pproj4(torch.cat([e1, e2, e3], 1))) if PD_DEPTH >= 4 else ctx1_w
    logits4 = lens.head4(torch.cat([ctx4, e1, e2, e3], 1)).float()
    key3 = triplet_key_int(d1_flat - DIGIT_OFFSETS[0], d2_flat - DIGIT_OFFSETS[1], d3_flat - DIGIT_OFFSETS[2])
    logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    d4_lp = F.log_softmax(logits4, 1)
    top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, 1)
    d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]
    quad = best3_score.unsqueeze(-1) + top_d4_val.view(B, W, CODEBOOK_SIZE)
    topq_score, topq_idx = torch.topk(quad.masked_fill(torch.isnan(quad), float('-inf')).view(B, -1), W, 1)
    beam_idx = topq_idx // CODEBOOK_SIZE
    d1f = torch.gather(d1_tokens, 1, beam_idx); d2f = torch.gather(d2_tokens, 1, beam_idx)
    d3f = torch.gather(d3_tokens, 1, beam_idx); d4f = torch.gather(d4_tokens.view(B, -1), 1, topq_idx)
    quads = torch.stack([d1f, d2f, d3f, d4f], -1)
    return quads, topq_score, stage1, stage2, stage3

# ---------------- JP student beam (s24 arm C: joint occupied-prefix head) ----------------
def student_beam_jp(enc, pad, W):
    """One topk over occupied (d1,d2) classes replaces beam stages 1-2; suffix identical."""
    B = enc.shape[0]
    ctx = lens.encode(enc, pad)
    jp_lp = F.log_softmax(lens.headJP(ctx).float(), dim=-1)
    W = min(W, jp_lp.shape[1])
    prefix_score, pre_idx = torch.topk(jp_lp, W, 1)
    pairs = lens.jp_pairs[pre_idx]
    d1_tokens = pairs[:, :, 0] + DIGIT_OFFSETS[0]
    d2_tokens = pairs[:, :, 1] + DIGIT_OFFSETS[1]
    stage1 = d1_tokens.clone(); stage2 = (d1_tokens.clone(), d2_tokens.clone())
    ctx_flat = ctx.unsqueeze(1).expand(-1, W, -1).reshape(B*W, -1)
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
    logits3 = lens.head3(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat)], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE-1)
    d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE-1)
    logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float('-inf'))
    d3_lp = F.log_softmax(logits3, 1).view(B, W, CODEBOOK_SIZE)
    score3 = prefix_score.unsqueeze(-1) + d3_lp
    best3_score, best3_idx = torch.topk(score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), W, 1)
    sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam3); d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
    d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
    stage3 = (d1_tokens.clone(), d2_tokens.clone(), d3_tokens.clone())
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1); d3_flat = d3_tokens.reshape(-1)
    logits4 = lens.head4(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat),
                                    lens.embedding(d3_flat)], 1)).float()
    key3 = triplet_key_int(d1_flat - DIGIT_OFFSETS[0], d2_flat - DIGIT_OFFSETS[1], d3_flat - DIGIT_OFFSETS[2])
    logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    d4_lp = F.log_softmax(logits4, 1)
    top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, 1)
    d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]
    quad = best3_score.unsqueeze(-1) + top_d4_val.view(B, W, CODEBOOK_SIZE)
    topq_score, topq_idx = torch.topk(quad.masked_fill(torch.isnan(quad), float('-inf')).view(B, -1), W, 1)
    beam_idx = topq_idx // CODEBOOK_SIZE
    d1f = torch.gather(d1_tokens, 1, beam_idx); d2f = torch.gather(d2_tokens, 1, beam_idx)
    d3f = torch.gather(d3_tokens, 1, beam_idx); d4f = torch.gather(d4_tokens.view(B, -1), 1, topq_idx)
    quads = torch.stack([d1f, d2f, d3f, d4f], -1)
    return quads, topq_score, stage1, stage2, stage3

CK_VARIANT = CK_ARGS.get('variant', 'base')
if CK_VARIANT == 'jp':
    student_beam = student_beam_jp
    print('[s24] JP checkpoint detected -> using student_beam_jp', flush=True)
elif CK_VARIANT != 'base':
    if PD_MODE == 'train_only':
        print(f'[s33] train_only PD checkpoint detected (variant={CK_VARIANT}, pd_depth={PD_DEPTH}) -> using base one-shot student_beam', flush=True)
    else:
        student_beam = student_beam_pd
        print(f'[s23/s33] PD checkpoint detected (variant={CK_VARIANT}, pd_depth={PD_DEPTH}, pd_mode={PD_MODE}) -> using student_beam_pd', flush=True)

# ---------------- exact teacher rescore (the shared verify primitive) ----------------
@torch.inference_mode()
def teacher_rescore(enc, attn, cand_user, cand_tokens, chunk=8192):
    """enc: (B,S,D); attn: (B,S); cand_user: (N,) long user row per candidate;
    cand_tokens: (N,4) vocab-space SID tokens. Returns (N,) float scores =
    sum logprob(d1..d4) + logprob(eos | d1..d4)  (the HF beam accumulation objective)."""
    N = cand_tokens.shape[0]
    scores = torch.empty(N, dtype=torch.float32, device=enc.device)
    dec_in = torch.cat([torch.full((N, 1), START, dtype=torch.long, device=enc.device), cand_tokens], 1)
    for lo in range(0, N, chunk):
        hi = min(N, lo + chunk)
        rows = cand_user[lo:hi]
        eo = BaseModelOutput(last_hidden_state=enc.index_select(0, rows))
        am = attn.index_select(0, rows)
        out = t5(encoder_outputs=eo, attention_mask=am, decoder_input_ids=dec_in[lo:hi], return_dict=True)
        lp = F.log_softmax(out.logits.float(), dim=-1)  # (n,5,V)
        tok_lp = lp[:, :4, :].gather(2, cand_tokens[lo:hi].unsqueeze(-1)).squeeze(-1).sum(1)
        eos_lp = lp[:, 4, EOS]
        scores[lo:hi] = tok_lp + eos_lp
    return scores

def teacher_beam_topk(d, n_return):
    return tiger.generate(d, n_return_sequences=n_return)  # (B, n_return, 4)

@torch.inference_mode()
def teacher_prefix_beam(input_ids, attn, H, W):
    """teacher beam to depth H; returns (B, W, H) prefix tokens."""
    out = t5.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=H,
                      num_beams=W, num_return_sequences=W, use_cache=True,
                      early_stopping=False)
    pref = out[:, 1:1+H]
    return pref.view(input_ids.shape[0], W, H)

def jac(a, b):
    A, B_ = set(a), set(b)
    return len(A & B_) / max(len(A | B_), 1)

def hit_at(items, gt, k):
    return int(gt in items[:k])

def ndcg_at(items, gt, k):
    for rank, item in enumerate(items[:k], start=1):
        if item == gt:
            return 1.0 / math.log2(rank + 1)
    return 0.0

# ---------------- fid mode ----------------
def run_fid():
    ds = pipe.tokenized_datasets['test']
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=pipe.tokenizer.collate_fn['test'])
    M = args.M
    MS = [m for m in [10, 20, 50, 100, 200] if m <= M]
    COV_ROWS = {m: [] for m in MS} if args.cov_out else None   # s49
    COV_DEN = [] if args.cov_out else None                     # s49
    CMS = [10, 20, 50, M]  # s24: cascade-M sweep for the efficiency frontier
    agg = {
        'n': 0,
        'teacher_r5': 0, 'teacher_r10': 0, 'teacher_ndcg5': 0.0, 'teacher_ndcg10': 0.0,
        'student_r5': 0, 'student_r10': 0, 'student_ndcg5': 0.0, 'student_ndcg10': 0.0,
        'student_jac10': 0.0,
        'cov': {m: 0.0 for m in MS}, 'cov_all': {m: 0 for m in MS},
        'lost': {'d1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'found': 0, 'tot': 0},
        'cascade': {m: {'jac10': 0.0, 'setid': 0, 'top1': 0, 'r5': 0, 'r10': 0,
                        'ndcg5': 0.0, 'ndcg10': 0.0} for m in CMS},
        'c4': {'jac10': 0.0, 'setid': 0, 'top1': 0, 'r5': 0, 'r10': 0,
               'ndcg5': 0.0, 'ndcg10': 0.0, 'ncand': 0.0},
        'c4h2': {'jac10': 0.0, 'setid': 0, 'top1': 0, 'r5': 0, 'r10': 0,
                 'ndcg5': 0.0, 'ndcg10': 0.0, 'ncand': 0.0},
        'anchor_rescore_top1_match': 0, 'anchor_n': 0,
    }
    ncand_hist = []
    t0 = time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches: break
            d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            input_ids = d['input_ids']; attn = d['attention_mask']; B = input_ids.shape[0]
            labs = batch['labels'].tolist()
            # teacher reference
            preds_t = teacher_beam_topk(d, args.teacher_beam)
            pt = preds_t.tolist()
            # shared encoder
            enc = t5.encoder(input_ids=input_ids, attention_mask=attn, return_dict=True).last_hidden_state
            pad = ~attn.bool()
            # student proposals
            quads, sc, st1, st2, st3 = student_beam(enc, pad, M)
            ps = quads.tolist()
            s1 = st1.tolist(); s2a, s2b = [x.tolist() for x in st2]
            s3a, s3b, s3c = [x.tolist() for x in st3]
            c4tmp = {}
            if not args.metrics_only:
                # cascade rescore of student proposals
                cu = torch.arange(B, device=dev).repeat_interleave(M)
                ct = quads.view(B*M, 4)
                cs = teacher_rescore(enc, attn, cu, ct).view(B, M)
                # C4 arms
                for Hkey, H in (('c4', args.c4_H), ('c4h2', 2)):
                    prefs = teacher_prefix_beam(input_ids, attn, H, args.c4_W)
                    pl = prefs.tolist()
                    cand_rows, cand_users, ncands = [], [], []
                    pmap = pref3_map if H == 3 else pref2_map
                    for r in range(B):
                        seen = set()
                        for p in pl[r]:
                            for quad in pmap.get(tuple(p), ()):
                                seen.add(quad)
                        ncands.append(len(seen))
                        for quad in seen:
                            cand_rows.append(list(quad)); cand_users.append(r)
                    if not cand_rows:
                        c4tmp[Hkey] = (defaultdict(list), ncands); continue
                    ctok = torch.tensor(cand_rows, dtype=torch.long, device=dev)
                    cusr = torch.tensor(cand_users, dtype=torch.long, device=dev)
                    csc = teacher_rescore(enc, attn, cusr, ctok)
                    per_user = defaultdict(list)
                    csc_l = csc.tolist()
                    for i in range(len(cand_users)):
                        per_user[cand_users[i]].append((csc_l[i], tuple(cand_rows[i])))
                    c4tmp[Hkey] = (per_user, ncands)
                    if Hkey == 'c4': ncand_hist.extend(ncands)
            # teacher rescore of teacher's own beams (sanity anchor A) - first 6 batches
            anchor_scores = None
            if not args.metrics_only and bi < 6:
                au, arows = [], []
                for r in range(B):
                    for q in pt[r]:
                        if sid_to_item(q) != -1:
                            au.append(r); arows.append(list(q))
                if arows:
                    a_sc = teacher_rescore(enc, attn, torch.tensor(au, dtype=torch.long, device=dev),
                                           torch.tensor(arows, dtype=torch.long, device=dev))
                    anchor_scores = (au, arows, a_sc.tolist())
            # ---- per-user metrics ----
            for r in range(B):
                gt = tuple(int(x) for x in labs[r][:4]); gt_it = sid_to_item(gt)
                if gt_it == -1: continue
                t_items = dedup_items(pt[r]); t10 = t_items[:10]
                if not t10: continue
                t10_quads = {}
                seen = set()
                for q in pt[r]:
                    it = sid_to_item(q)
                    if it == -1 or it in seen: continue
                    seen.add(it)
                    if it in t10 and it not in t10_quads: t10_quads[it] = tuple(q)
                s_items_full = dedup_items(ps[r])
                agg['n'] += 1
                agg['teacher_r5'] += hit_at(t_items, gt_it, 5)
                agg['teacher_r10'] += hit_at(t_items, gt_it, 10)
                agg['teacher_ndcg5'] += ndcg_at(t_items, gt_it, 5)
                agg['teacher_ndcg10'] += ndcg_at(t_items, gt_it, 10)
                s10 = s_items_full[:10]
                agg['student_r5'] += hit_at(s_items_full, gt_it, 5)
                agg['student_r10'] += hit_at(s_items_full, gt_it, 10)
                agg['student_ndcg5'] += ndcg_at(s_items_full, gt_it, 5)
                agg['student_ndcg10'] += ndcg_at(s_items_full, gt_it, 10)
                agg['student_jac10'] += jac(t10, s10)
                _t10set = set(t10)
                for m in MS:
                    sm = set(s_items_full[:m])
                    inter = len(_t10set & sm)
                    agg['cov'][m] += inter / len(t10)
                    agg['cov_all'][m] += int(inter == len(t10))
                    if COV_ROWS is not None:
                        COV_ROWS[m].append(inter)
                if COV_ROWS is not None:
                    COV_DEN.append(len(t10))
                sm_full = set(s_items_full)
                set1 = set(s1[r]); set2 = set(zip(s2a[r], s2b[r]))
                set3 = set(zip(s3a[r], s3b[r], s3c[r]))
                for it in t10:
                    agg['lost']['tot'] += 1
                    q = t10_quads[it]
                    if it in sm_full: agg['lost']['found'] += 1
                    elif q[0] not in set1: agg['lost']['d1'] += 1
                    elif (q[0], q[1]) not in set2: agg['lost']['d2'] += 1
                    elif (q[0], q[1], q[2]) not in set3: agg['lost']['d3'] += 1
                    else: agg['lost']['d4'] += 1
                if not args.metrics_only:
                    order = torch.argsort(cs[r], descending=True).tolist()
                    for m in CMS:
                        keep = [o for o in order if o < m]
                        ranked = [tuple(ps[r][o]) for o in keep]
                        c_items = dedup_items(ranked)[:10]
                        a = agg['cascade'][m]
                        a['jac10'] += jac(t10, c_items)
                        a['setid'] += int(set(c_items) == set(t10) and len(c_items) > 0)
                        a['top1'] += int(len(c_items) > 0 and len(t10) > 0 and c_items[0] == t10[0])
                        a['r5'] += hit_at(c_items, gt_it, 5)
                        a['r10'] += hit_at(c_items, gt_it, 10)
                        a['ndcg5'] += ndcg_at(c_items, gt_it, 5)
                        a['ndcg10'] += ndcg_at(c_items, gt_it, 10)
                    for Hkey in ('c4', 'c4h2'):
                        per_user, ncands = c4tmp[Hkey]
                        lst = sorted(per_user.get(r, []), key=lambda x: -x[0])
                        c4_items = dedup_items([q for _, q in lst])[:10]
                        a = agg[Hkey]
                        a['jac10'] += jac(t10, c4_items)
                        a['setid'] += int(set(c4_items) == set(t10) and len(c4_items) > 0)
                        a['top1'] += int(len(c4_items) > 0 and len(t10) > 0 and c4_items[0] == t10[0])
                        a['r5'] += hit_at(c4_items, gt_it, 5)
                        a['r10'] += hit_at(c4_items, gt_it, 10)
                        a['ndcg5'] += ndcg_at(c4_items, gt_it, 5)
                        a['ndcg10'] += ndcg_at(c4_items, gt_it, 10)
                        a['ncand'] += ncands[r]
            if anchor_scores is not None:
                au, arows, asc = anchor_scores
                by_user = defaultdict(list)
                for u, row, s in zip(au, arows, asc):
                    by_user[u].append((s, tuple(row)))
                for r in range(B):
                    if r not in by_user: continue
                    t_items = dedup_items(pt[r])
                    if not t_items: continue
                    ranked_items = dedup_items([q for _, q in sorted(by_user[r], key=lambda x: -x[0])])
                    agg['anchor_n'] += 1
                    agg['anchor_rescore_top1_match'] += int(len(ranked_items) > 0 and ranked_items[0] == t_items[0])
            if bi % 20 == 0:
                print(f'batch {bi} n={agg["n"]} elapsed={time.time()-t0:.0f}s', flush=True)
    n = max(agg['n'], 1)
    out = {'tag': args.tag, 'category': args.category, 'variant': args.variant,
           'checkpoint_variant': CK_VARIANT, 'pd_depth': PD_DEPTH if CK_VARIANT != 'base' else None,
           'pd_mode': PD_MODE if CK_VARIANT != 'base' else None,
           'metrics_only': bool(args.metrics_only), 'M': M,
           'teacher_beam': args.teacher_beam, 'n': agg['n'],
           'teacher_r5': agg['teacher_r5']/n, 'teacher_ndcg5': agg['teacher_ndcg5']/n,
           'teacher_r10': agg['teacher_r10']/n, 'teacher_ndcg10': agg['teacher_ndcg10']/n,
           'student_r5': agg['student_r5']/n, 'student_ndcg5': agg['student_ndcg5']/n,
           'student_r10': agg['student_r10']/n, 'student_ndcg10': agg['student_ndcg10']/n,
           'student_jac10': agg['student_jac10']/n,
           'coverage': {m: agg['cov'][m]/n for m in MS},
           'cov_out': args.cov_out or None,
           'coverage_all10': {m: agg['cov_all'][m]/n for m in MS},
           'lost': agg['lost']}
    if COV_ROWS is not None:
        import numpy as _np
        _cols = {('inter_at_%d' % m): _np.asarray(COV_ROWS[m], dtype=_np.int16) for m in MS}
        _np.savez_compressed(args.cov_out, den=_np.asarray(COV_DEN, dtype=_np.int16),
                             Cs=_np.asarray(MS, dtype=_np.int32), **_cols)
        print('[cov_out] wrote %s rows=%d Cs=%s' % (args.cov_out, len(COV_DEN), MS), flush=True)
    if not args.metrics_only:
        out.update({
            'cascade': {m: {k: v/n for k, v in a.items()} for m, a in agg['cascade'].items()},
            'c4_H3': {k: v/n for k, v in agg['c4'].items()},
            'c4_H2': {k: v/n for k, v in agg['c4h2'].items()},
            'anchor_rescore_top1_match': agg['anchor_rescore_top1_match']/max(agg['anchor_n'], 1),
            'anchor_n': agg['anchor_n']})
    if ncand_hist:
        srt = sorted(ncand_hist)
        out['c4_H3']['ncand_p95'] = srt[int(0.95*len(srt))]
        out['c4_H3']['ncand_max'] = srt[-1]
    with open(args.out, 'w') as f: json.dump(out, f, indent=2)
    print('=== FID SUMMARY', args.tag, args.category, '===')
    print(json.dumps(out, indent=2)[:3000])
    print('FRONTIER_FID_DONE', flush=True)

# ---------------- timing mode ----------------
def run_timing():
    ds = pipe.tokenized_datasets['test']
    res = {}
    for bs in (32, 1):
        loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=pipe.tokenizer.collate_fn['test'])
        nb = args.timing_batches + 2 if bs == 32 else args.timing_batches * 8 + 2
        batches = []
        for bi, batch in enumerate(loader):
            if bi >= nb: break
            batches.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()})
        def timed(fn):
            with torch.inference_mode():
                for b in batches[:2]: fn(b)          # warmup
                torch.cuda.synchronize(); t0 = time.time()
                nq = 0
                for b in batches[2:]:
                    fn(b); nq += b['input_ids'].shape[0]
                torch.cuda.synchronize()
                return (time.time() - t0) / nq * 1000.0  # ms/query
        arms = {}
        arms['encoder_only'] = timed(lambda b: t5.encoder(input_ids=b['input_ids'],
                                                          attention_mask=b['attention_mask'], return_dict=True))
        arms[f'teacher_b{args.teacher_beam}'] = timed(lambda b: teacher_beam_topk(b, args.teacher_beam))
        arms['teacher_b10'] = timed(lambda b: tiger.generate(b, n_return_sequences=10))
        def _student(b, W):
            enc = t5.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask'],
                             return_dict=True).last_hidden_state
            return student_beam(enc, ~b['attention_mask'].bool(), W)
        arms['student_M50'] = timed(lambda b: _student(b, 50))
        arms[f'student_M{args.M}'] = timed(lambda b: _student(b, args.M))
        def _cascade(b, W):
            enc = t5.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask'],
                             return_dict=True).last_hidden_state
            quads, sc, *_ = student_beam(enc, ~b['attention_mask'].bool(), W)
            B = quads.shape[0]
            cu = torch.arange(B, device=dev).repeat_interleave(W)
            return teacher_rescore(enc, b['attention_mask'], cu, quads.view(B*W, 4))
        arms['cascade_M50'] = timed(lambda b: _cascade(b, 50))
        arms[f'cascade_M{args.M}'] = timed(lambda b: _cascade(b, args.M))
        def _c4(b, H, W):
            prefs = teacher_prefix_beam(b['input_ids'], b['attention_mask'], H, W)
            enc = t5.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask'],
                             return_dict=True).last_hidden_state
            pl = prefs.tolist(); B = prefs.shape[0]
            pmap = pref3_map if H == 3 else pref2_map
            cand_rows, cand_users = [], []
            for r in range(B):
                seen = set()
                for p in pl[r]:
                    for quad in pmap.get(tuple(p), ()):
                        seen.add(quad)
                for quad in seen:
                    cand_rows.append(list(quad)); cand_users.append(r)
            if not cand_rows: return None
            return teacher_rescore(enc, b['attention_mask'],
                                   torch.tensor(cand_users, dtype=torch.long, device=dev),
                                   torch.tensor(cand_rows, dtype=torch.long, device=dev))
        arms['c4_H3_W50'] = timed(lambda b: _c4(b, 3, args.c4_W))
        arms['c4_H2_W50'] = timed(lambda b: _c4(b, 2, args.c4_W))
        res[f'bs{bs}'] = arms
        print(f'--- bs={bs} ---')
        for k, v in arms.items(): print(f'{k:>18s}: {v:8.2f} ms/q', flush=True)
    out = {'tag': args.tag, 'category': args.category, 'mode': 'timing', 'timing': res}
    with open(args.out, 'w') as f: json.dump(out, f, indent=2)
    print('FRONTIER_TIMING_DONE', flush=True)

if args.mode == 'fid':
    run_fid()
else:
    run_timing()
