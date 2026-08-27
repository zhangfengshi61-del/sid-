"""s34: popularity-bucket + prefix-diagnostic eval of a plain (non-PD) student vs teacher.

Per test user (full TEST): teacher beam-50 (cached per category), student masked-valid
factorized search W=50 (deployed procedure), then:
  - R@5/10, NDCG@5/10 aggregate + per popularity bucket (cold/tail/mid/head ==
    cascade_bucket_s23 tertile convention), teacher and student.
  - PS@d (prefix survival): student = exact rank of the GT's depth-d prefix in the
    student search frontier at depth d (width 50); teacher = lower-bound membership of
    the GT depth-d prefix among the FINAL beam-50 codes truncated to depth d (exact at d=4).
  - coverage/misrank split: GT in student top-50 but not top-10 = ranking failure;
    GT absent from top-50 = support failure.
  - PopGap@10: mean train-pop of top-10 minus GT pop (teacher and student rows).
  - displacement bucket counts: buckets of teacher-minus-student and student-minus-teacher.
  - prefix-deficit raw material: per-candidate per-depth teacher and student logprobs on
    the shared candidate set C = {GT} + t10 + b10 (padded to 21 with GT repeats).
Aggregate JSON + per-user npz dump (gt_pop raw counts included so buckets can be re-cut
offline under any convention). Non-PD students only (pd arms need frontier_s23 pd_depth path).
"""
import sys, os, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from collections import defaultdict

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
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--lens_path', required=True)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--W', type=int, default=50)
ap.add_argument('--teacher_beam', type=int, default=50)
ap.add_argument('--teacher_cache', default='')   # .pt path; created if missing, reused if present
ap.add_argument('--pop_dump', default='')        # optional popcounts npz path, written once
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--per_user_out', required=True)
ap.add_argument('--tag', default='')
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
lens, _ck = build_sidmlp_from_checkpoint(args.lens_path, tk['t5.shared.weight'], dev)
if nb16: lens.bfloat16()
del tk

vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path, graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, keys4_sorted, vals4_sorted, _im,
 valid_items_4) = build_valid_tensors(vs, dev)
EOS = t5.config.eos_token_id
START = t5.config.decoder_start_token_id

def sid_to_item(tup):
    return valid_items_4.get(tuple(int(x) for x in tup), -1)

def dedup_items(sid_list):
    seen = set(); out = []
    for tup in sid_list:
        it = sid_to_item(tup)
        if it == -1 or it in seen: continue
        seen.add(it); out.append(it)
    return out

def jac(a, b):
    A, B_ = set(a), set(b)
    return len(A & B_) / max(len(A | B_), 1)

def rank_of(items, gt_it, k):
    try:
        r = items[:k].index(gt_it)
        return r
    except ValueError:
        return -1

# popularity from TRAIN interactions (== cascade_bucket_s23 / taskrr_eval_s28)
pop = defaultdict(int)
tl = DataLoader(pipe.tokenized_datasets['train'], batch_size=256, shuffle=False,
                collate_fn=pipe.tokenizer.collate_fn['train'])
with torch.inference_mode():
    for b in tl:
        for row in b['labels'].tolist():
            it = sid_to_item(row[:4])
            if it != -1: pop[it] += 1
counts_sorted = sorted(pop.values())
LO = counts_sorted[len(counts_sorted)//3]; HI = counts_sorted[2*len(counts_sorted)//3]
def pop_bucket(it):
    c = pop.get(it, 0)
    if c == 0: return 0      # cold
    if c <= LO: return 1     # tail
    if c <= HI: return 2     # mid
    return 3                 # head
BNAME = ['cold', 'tail', 'mid', 'head']
print(f'[pop] items={len(pop)} tertiles lo={LO} hi={HI}', flush=True)
if args.pop_dump and not os.path.exists(args.pop_dump):
    its = np.array(sorted(pop.keys()), dtype=np.int64)
    cts = np.array([pop[int(i)] for i in its], dtype=np.int64)
    np.savez_compressed(args.pop_dump, items=its, counts=cts, lo=LO, hi=HI)
    print(f'[pop] dumped {args.pop_dump}', flush=True)

def student_beam_frontiers(enc, pad, W):
    """Deployed masked-valid factorized search, width W, returning per-depth frontiers.
    Frontiers are score-ordered code-prefix tensors: f1 [B,W], f2 [B,W,2], f3 [B,W,3],
    quads [B,W,4] (token-id space)."""
    B = enc.shape[0]
    ctx = lens.encode(enc, pad)
    d1_lp = F.log_softmax(lens.head1(ctx).float(), dim=-1)
    W1 = min(W, CODEBOOK_SIZE)
    d1_scores, d1_idx = torch.topk(d1_lp, W1, dim=1)
    d1_tokens = d1_idx + DIGIT_OFFSETS[0]
    f1 = d1_tokens.clone()
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
    f2 = torch.stack([d1_tokens, d2_tokens], -1)
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
    f3 = torch.stack([d1_tokens, d2_tokens, d3_tokens], -1)
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
    return quads, topq_score, ctx, f1, f2, f3

def list_scores_comp(model, ctx, cand_cb, masks):
    mask_d1_d2_, mask_d1d2_d3_, d4k, d4m = masks
    B, M, _ = cand_cb.shape
    c1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    c2 = cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    c3 = cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)
    c4 = cand_cb[:, :, 3].clamp(0, CODEBOOK_SIZE - 1)
    v1 = (c1 + DIGIT_OFFSETS[0]).reshape(-1)
    v2 = (c2 + DIGIT_OFFSETS[1]).reshape(-1)
    v3 = (c3 + DIGIT_OFFSETS[2]).reshape(-1)
    lp1 = F.log_softmax(model.head1(ctx).float(), dim=-1).gather(1, c1)
    D = ctx.shape[1]
    ctx_f = ctx.unsqueeze(1).expand(-1, M, -1).reshape(B * M, D)
    e1 = model.embedding(v1); e2 = model.embedding(v2); e3 = model.embedding(v3)
    l2 = model.head2(torch.cat([ctx_f, e1], 1)).float()
    l2 = l2.masked_fill(~mask_d1_d2_[c1.reshape(-1)], float('-inf'))
    lp2 = F.log_softmax(l2, 1).gather(1, c2.reshape(-1, 1)).view(B, M)
    l3 = model.head3(torch.cat([ctx_f, e1, e2], 1)).float()
    l3 = l3.masked_fill(~mask_d1d2_d3_[c1.reshape(-1), c2.reshape(-1)], float('-inf'))
    lp3 = F.log_softmax(l3, 1).gather(1, c3.reshape(-1, 1)).view(B, M)
    l4 = model.head4(torch.cat([ctx_f, e1, e2, e3], 1)).float()
    key3 = triplet_key_int(c1.reshape(-1), c2.reshape(-1), c3.reshape(-1))
    l4 = l4.masked_fill(~gather_d4_mask(d4k, d4m, key3), float('-inf'))
    lp4 = F.log_softmax(l4, 1).gather(1, c4.reshape(-1, 1)).view(B, M)
    return torch.stack([lp1, lp2, lp3, lp4], dim=-1)

@torch.inference_mode()
def teacher_steps(enc, attn, cand_user, cand_tokens, chunk=8192):
    """Teacher-forced per-step logprobs [N,4] + eos lp [N]."""
    N = cand_tokens.shape[0]
    steps = torch.empty(N, 4, dtype=torch.float32, device=enc.device)
    eosl = torch.empty(N, dtype=torch.float32, device=enc.device)
    dec_in = torch.cat([torch.full((N, 1), START, dtype=torch.long, device=enc.device), cand_tokens], 1)
    for lo in range(0, N, chunk):
        hi = min(N, lo + chunk)
        rows = cand_user[lo:hi]
        eo = BaseModelOutput(last_hidden_state=enc.index_select(0, rows))
        am = attn.index_select(0, rows)
        out = t5(encoder_outputs=eo, attention_mask=am, decoder_input_ids=dec_in[lo:hi], return_dict=True)
        lp = F.log_softmax(out.logits.float(), dim=-1)
        steps[lo:hi] = lp[:, :4, :].gather(2, cand_tokens[lo:hi].unsqueeze(-1)).squeeze(-1)
        eosl[lo:hi] = lp[:, 4, EOS]
    return steps, eosl

OFF = torch.tensor(DIGIT_OFFSETS, device=dev)
LM = (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted)
MC = 21  # candidate-set pad size: {GT} + t10 + b10

ds = pipe.tokenized_datasets['test']
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn['test'])

# ---- teacher beam cache ----
t_cache = None; t_cache_new = []
if args.teacher_cache and os.path.exists(args.teacher_cache):
    t_cache = torch.load(args.teacher_cache, map_location='cpu')['codes']  # [Nt, beam, 4] int32
    print(f'[tcache] loaded {args.teacher_cache} rows={t_cache.shape[0]}', flush=True)

agg = defaultdict(float)
bkt = {b: defaultdict(float) for b in range(4)}
exp10 = {s: np.zeros(4, dtype=np.int64) for s in ('teacher', 'student')}
PU = defaultdict(list)
C_buck, C_flags, C_pop, C_tlp, C_slp, C_teos, C_off = [], [], [], [], [], [], [0]
row_ptr = 0
t0 = time.time()
with torch.inference_mode():
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches: break
        d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        input_ids = d['input_ids']; attn = d['attention_mask']; B = input_ids.shape[0]
        labs = batch['labels'].tolist()
        if t_cache is not None:
            TQ = t_cache[row_ptr:row_ptr+B].to(dev).long()
        else:
            preds_t = tiger.generate(d, n_return_sequences=args.teacher_beam)
            TQ = preds_t.view(B, args.teacher_beam, -1)[:, :, :4].long() \
                 if preds_t.dim() != 3 else preds_t[:, :, :4].long()
            t_cache_new.append(TQ.to(torch.int32).cpu())
        row_ptr += B
        pt = TQ.tolist()
        enc = t5.encoder(input_ids=input_ids, attention_mask=attn, return_dict=True).last_hidden_state
        pad = ~attn.bool()
        quads, beam_sc, ctx, f1, f2, f3 = student_beam_frontiers(enc, pad, args.W)
        ps_codes = quads.tolist()
        F1 = f1.tolist(); F2 = f2.tolist(); F3 = f3.tolist()
        # ---- per-user candidate sets (built on CPU, scored batched) ----
        cand_rows = []
        for r in range(B):
            gt = tuple(int(x) for x in labs[r][:4]); gt_it = sid_to_item(gt)
            t_items = dedup_items(pt[r]); s_items = dedup_items(ps_codes[r])
            cand_rows.append((gt, gt_it, t_items, s_items))
        cand_tok = torch.zeros(B, MC, 4, dtype=torch.long, device=dev)
        cand_n = []
        for r in range(B):
            gt, gt_it, t_items, s_items = cand_rows[r]
            t10c = [tuple(c) for c in pt[r]][:0]  # placeholder; rebuilt below at item level
            # candidate CODES: gt + first codes of t10 items + b10 items (item-level dedup order)
            it2code_t = {}
            for c in pt[r]:
                it = sid_to_item(c[:4])
                if it != -1 and it not in it2code_t: it2code_t[it] = tuple(c[:4])
            it2code_s = {}
            for c in ps_codes[r]:
                it = sid_to_item(c[:4])
                if it != -1 and it not in it2code_s: it2code_s[it] = tuple(c[:4])
            cands = [gt]
            for it in t_items[:10]: cands.append(it2code_t[it])
            for it in s_items[:10]:
                cc = it2code_s[it]
                cands.append(cc)
            cands = cands[:MC]
            n = len(cands)
            while len(cands) < MC: cands.append(gt)
            cand_n.append(n)
            cand_tok[r] = torch.tensor(cands, dtype=torch.long, device=dev)
        cb = (cand_tok - OFF.view(1, 1, 4)).clamp(0, CODEBOOK_SIZE - 1)
        slp = list_scores_comp(lens, ctx, cb, LM)                      # [B,MC,4]
        cu = torch.arange(B, device=dev).repeat_interleave(MC)
        tst, teos = teacher_steps(enc, attn, cu, cand_tok.view(-1, 4))
        tst = tst.view(B, MC, 4); teos = teos.view(B, MC)
        slp_l = slp.cpu().numpy(); tst_l = tst.cpu().numpy(); teos_l = teos.cpu().numpy()
        for r in range(B):
            gt, gt_it, t_items, s_items = cand_rows[r]
            if gt_it == -1: continue
            t10 = t_items[:10]
            if not t10: continue
            b10 = s_items[:10]
            gt_pop = pop.get(gt_it, 0); gb = pop_bucket(gt_it)
            agg['n'] += 1; bkt[gb]['n'] += 1
            # ranks and metrics
            tr = rank_of(t_items, gt_it, 10); sr = rank_of(s_items, gt_it, 10)
            tr50 = rank_of(t_items, gt_it, 50); sr50 = rank_of(s_items, gt_it, 50)
            for pre, rk in (('t', tr), ('s', sr)):
                hit10 = int(rk >= 0); nd10 = 1.0/np.log2(rk+2.0) if rk >= 0 else 0.0
                hit5 = int(0 <= rk < 5); nd5 = 1.0/np.log2(rk+2.0) if 0 <= rk < 5 else 0.0
                agg[f'{pre}_r10'] += hit10; agg[f'{pre}_n10'] += nd10
                agg[f'{pre}_r5'] += hit5; agg[f'{pre}_n5'] += nd5
                bkt[gb][f'{pre}_r10'] += hit10; bkt[gb][f'{pre}_n10'] += nd10
                bkt[gb][f'{pre}_r5'] += hit5; bkt[gb][f'{pre}_n5'] += nd5
            agg['jac10'] += jac(t10, b10)
            bkt[gb]['cov50'] += int(sr50 >= 0)
            bkt[gb]['tcov50'] += int(tr50 >= 0)
            bkt[gb]['misrank'] += int(sr50 >= 0 and sr < 0)
            # PS: student exact frontier ranks
            gtl = list(gt)
            ps_s = [-1, -1, -1, -1]
            try: ps_s[0] = F1[r].index(gtl[0])
            except ValueError: pass
            try: ps_s[1] = F2[r].index(gtl[:2])
            except ValueError: pass
            try: ps_s[2] = F3[r].index(gtl[:3])
            except ValueError: pass
            try: ps_s[3] = ps_codes[r].index(gtl)
            except ValueError: pass
            # PS teacher lower bound: GT prefix among final beam codes truncated to d
            ps_t = [0, 0, 0, 0]
            for dd in range(4):
                pref = tuple(gtl[:dd+1])
                ps_t[dd] = int(any(tuple(c[:dd+1]) == pref for c in pt[r]))
            for dd in range(4):
                agg[f'ps_s_d{dd+1}'] += int(ps_s[dd] >= 0)
                agg[f'ps_t_d{dd+1}'] += ps_t[dd]
                bkt[gb][f'ps_s_d{dd+1}'] += int(ps_s[dd] >= 0)
                bkt[gb][f'ps_t_d{dd+1}'] += ps_t[dd]
            # popularity of served lists
            mp_b10 = float(np.mean([pop.get(it, 0) for it in b10])) if b10 else 0.0
            mp_t10 = float(np.mean([pop.get(it, 0) for it in t10]))
            for it in t10: exp10['teacher'][pop_bucket(it)] += 1
            for it in b10: exp10['student'][pop_bucket(it)] += 1
            # displacement bucket counts
            dts = np.zeros(4, dtype=np.int64); dst = np.zeros(4, dtype=np.int64)
            sb10 = set(b10); st10 = set(t10)
            for it in t10:
                if it not in sb10: dts[pop_bucket(it)] += 1
            for it in b10:
                if it not in st10: dst[pop_bucket(it)] += 1
            # per-user dump
            PU['gt_pop'].append(gt_pop); PU['gt_bucket'].append(gb)
            PU['t_rank'].append(tr); PU['s_rank'].append(sr)
            PU['t_rank50'].append(tr50); PU['s_rank50'].append(sr50)
            PU['ps_s'].append(ps_s); PU['ps_t'].append(ps_t)
            PU['meanpop_t10'].append(mp_t10); PU['meanpop_b10'].append(mp_b10)
            PU['disp_t_not_s'].append(dts.tolist()); PU['disp_s_not_t'].append(dst.tolist())
            PU['jac10'].append(jac(t10, b10))
            # candidate dump (ragged)
            n = cand_n[r]
            gtset = {gt_it}
            for ci in range(n):
                code = tuple(int(x) for x in cand_tok[r, ci].tolist())
                it = sid_to_item(code)
                fl = 0
                if ci == 0: fl |= 1
                if it in st10: fl |= 2
                if it in sb10: fl |= 4
                C_buck.append(pop_bucket(it)); C_flags.append(fl)
                C_pop.append(pop.get(it, 0))
                C_tlp.append(tst_l[r, ci]); C_slp.append(slp_l[r, ci]); C_teos.append(teos_l[r, ci])
            C_off.append(C_off[-1] + n)
        if bi % 20 == 0:
            print(f'batch {bi} n={int(agg["n"])} elapsed={time.time()-t0:.0f}s', flush=True)

if t_cache is None and args.teacher_cache and t_cache_new:
    torch.save({'codes': torch.cat(t_cache_new, 0)}, args.teacher_cache)
    print(f'[tcache] wrote {args.teacher_cache}', flush=True)

n = max(int(agg['n']), 1)
out = {'tag': args.tag, 'category': args.category, 'W': args.W, 'n': int(agg['n']),
       'lens_path': args.lens_path, 'bucket_def': f'train-count tertiles lo={LO} hi={HI} (cold=0)',
       'precision': 'bf16-compute-fp32-logits (matches prior frontier/taskrr evals)',
       'ps_teacher_note': 'teacher PS is a lower bound from final beam-50 truncations; exact at d4',
       'jac10': agg['jac10']/n}
for pre in ('t', 's'):
    for m in ('r5', 'n5', 'r10', 'n10'):
        out[f'{pre}_{m}'] = agg[f'{pre}_{m}']/n
for dd in range(4):
    out[f'ps_s_d{dd+1}'] = agg[f'ps_s_d{dd+1}']/n
    out[f'ps_t_d{dd+1}'] = agg[f'ps_t_d{dd+1}']/n
out['buckets'] = {}
for gb in range(4):
    bn = max(int(bkt[gb]['n']), 1)
    out['buckets'][BNAME[gb]] = {
        'n': int(bkt[gb]['n']),
        **{f'{pre}_{m}': bkt[gb][f'{pre}_{m}']/bn for pre in ('t', 's') for m in ('r5', 'n5', 'r10', 'n10')},
        'cov50': bkt[gb]['cov50']/bn, 'tcov50': bkt[gb]['tcov50']/bn,
        'misrank': bkt[gb]['misrank']/bn,
        **{f'ps_s_d{dd+1}': bkt[gb][f'ps_s_d{dd+1}']/bn for dd in range(4)},
        **{f'ps_t_d{dd+1}': bkt[gb][f'ps_t_d{dd+1}']/bn for dd in range(4)}}
out['exposure10'] = {s: {BNAME[i]: int(exp10[s][i]) for i in range(4)} for s in exp10}
with open(args.out, 'w') as f: json.dump(out, f, indent=2)
np.savez_compressed(args.per_user_out,
    gt_pop=np.array(PU['gt_pop'], dtype=np.int64),
    gt_bucket=np.array(PU['gt_bucket'], dtype=np.int8),
    t_rank=np.array(PU['t_rank'], dtype=np.int8), s_rank=np.array(PU['s_rank'], dtype=np.int8),
    t_rank50=np.array(PU['t_rank50'], dtype=np.int8), s_rank50=np.array(PU['s_rank50'], dtype=np.int8),
    ps_s=np.array(PU['ps_s'], dtype=np.int16), ps_t=np.array(PU['ps_t'], dtype=np.int8),
    meanpop_t10=np.array(PU['meanpop_t10'], dtype=np.float32),
    meanpop_b10=np.array(PU['meanpop_b10'], dtype=np.float32),
    disp_t_not_s=np.array(PU['disp_t_not_s'], dtype=np.int8),
    disp_s_not_t=np.array(PU['disp_s_not_t'], dtype=np.int8),
    jac10=np.array(PU['jac10'], dtype=np.float32),
    cand_offsets=np.array(C_off, dtype=np.int64),
    cand_bucket=np.array(C_buck, dtype=np.int8), cand_flags=np.array(C_flags, dtype=np.int8),
    cand_pop=np.array(C_pop, dtype=np.int64),
    cand_tlp=np.array(C_tlp, dtype=np.float32), cand_slp=np.array(C_slp, dtype=np.float32),
    cand_teos=np.array(C_teos, dtype=np.float32),
    bucket_lo=LO, bucket_hi=HI)
print('=== BUCKET DIAG SUMMARY', args.tag, args.category, '===')
print(json.dumps({k: v for k, v in out.items() if k != 'buckets'}, indent=2))
print(json.dumps(out['buckets'], indent=2))
print('BUCKET_DIAG_DONE', flush=True)
