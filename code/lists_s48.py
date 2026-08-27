"""s48: per-request SERVED-LIST dumper — the single artifact every s48 offline metric derives from.

Motivation (待完成实验 doc §0): "原始逐请求输出，不能只保存最终均值". Every metric the doc asks
for -- Jac@5/10/20, Overlap@K, R/N@K, Head/Mid/Tail buckets, Hit-Swap decomposition, conditional
agreement, NDCG win/loss mass, bootstrap CIs -- is a pure function of, per test request:
    (ground-truth item, teacher served list, method served list).
So we dump exactly that, ONCE per (method, dataset, seed), and compute everything offline in
code/metrics_s48.py. This also enforces the doc's protocol requirement that every method is scored
against "同一 TIGER Top-K cache、测试用户和 item universe": the teacher list comes from the shared
runs/tbeam50_<CAT>.pt cache and the test loader order is identical for every method.

Protocol fixed here (recorded into the npz + manifest):
  - decode width W = 50 for BOTH teacher and method (the doc's deployment `beam=50`). NOTE the
    project's previously-registered ServeSID jac@10 used the frontier_s23 student width M=200;
    measured W-sensitivity is <= 0.003 jac@10 and unsigned (see runs/s48_wsens.json).
  - lists are item-level: invalid SIDs dropped, then deduplicated, then truncated to --topk (20).
  - popularity buckets = train-interaction-count tertiles with cold(count==0) split out, i.e. the
    exact convention of code/bucket_diag_s34.py, so s48 rows are comparable to every earlier table.

Backends (--method): `sidmlp` = any ServeSID/SID-MLP-family checkpoint (r0base, lkpf, serveU4,
serveU4pair, full, dauni, daonly, ...) decoded with the deployed masked-valid factorized search;
`tiger` = the teacher itself (dumped as its own method so TIGER gets a row in every table and the
identity jac==1 acts as a pipeline self-check).

Emits: <out>.npz (schema below) + <out>.json (aggregate sanity block, for cross-checking against
the established frontier_s23 / bucket_diag_s34 numbers before any of it is trusted).
"""
import sys, os, json, time, argparse, hashlib
import numpy as np
import torch, torch.nn.functional as F
from collections import defaultdict

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, f'{R}/repos/SID-MLP')
os.chdir(f'{R}/repos')
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from torch.utils.data import DataLoader
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)          # TIGER teacher .pth (also supplies embeddings)
ap.add_argument('--lens_path', default='')              # student ckpt; required for --method sidmlp
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--method', default='sidmlp', choices=['sidmlp', 'tiger', 'precomputed'])
ap.add_argument('--tiger_beams', type=int, default=0)  # s49: override the TIGER model
                                                       # config num_beams (0 = leave as-is);
                                                       # needed for W > 50
ap.add_argument('--codes_in', default='')              # s49: .pt {'codes': (N, >=W, 4)} TOKEN-ID
                                                       # served codes from another decoder
                                                       # (PAD-TIGER --dump_codes); row-aligned
                                                       # to this same test loader
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--W', type=int, default=50)            # deployment decode width, teacher AND student
ap.add_argument('--topk', type=int, default=20)         # dumped list depth (>= max K the metrics need)
ap.add_argument('--teacher_cache', default='')
ap.add_argument('--pop_cache', default='')              # npz of train pop counts; built if missing
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--out', required=True)                 # path WITHOUT extension
ap.add_argument('--tag', default='')
ap.add_argument('--seed_tag', default='')
args = ap.parse_args()
args.encoder_ckpt = None; args.fp32 = False
dev = torch.device('cuda')
nb16 = configure_precision(args, dev)
TOPK = args.topk

pipe = build_tiger_pipeline(args)
tiger = pipe.model.to(dev).eval()
if args.tiger_beams:
    _old_nb = tiger.config['num_beams']
    tiger.config['num_beams'] = int(args.tiger_beams)
    print(f'[beams] TIGER num_beams {_old_nb} -> {args.tiger_beams}', flush=True)
t5 = tiger.t5
if nb16: t5.bfloat16()
for p in t5.parameters(): p.requires_grad_(False)

lens = None
if args.method == 'sidmlp':
    if not args.lens_path: raise SystemExit('--method sidmlp requires --lens_path')
    tk = _torch_load(args.checkpoint, weights_only=False)
    lens, _ck = build_sidmlp_from_checkpoint(args.lens_path, tk['t5.shared.weight'], dev)
    if nb16: lens.bfloat16()
    del tk

vs = find_valid_set(args.dataset_tag, explicit_path=None,
                    lens_path=(args.lens_path or None), graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, keys4_sorted, vals4_sorted, _im,
 valid_items_4) = build_valid_tensors(vs, dev)


def sid_to_item(tup):
    return valid_items_4.get(tuple(int(x) for x in tup), -1)


def dedup_items(sid_list, cap):
    """item-level: drop invalid SIDs, dedup preserving rank order, truncate. Returns
    (items, n_invalid, n_raw)."""
    seen = set(); out = []; ninv = 0
    for tup in sid_list:
        it = sid_to_item(tup)
        if it == -1:
            ninv += 1; continue
        if it in seen: continue
        seen.add(it); out.append(it)
        if len(out) >= cap: break
    return out, ninv, len(sid_list)


# ---------- popularity buckets: train-count tertiles, cold split out (== bucket_diag_s34) ----------
pop = None
if args.pop_cache and os.path.exists(args.pop_cache):
    z = np.load(args.pop_cache)
    pop = {int(i): int(c) for i, c in zip(z['items'], z['counts'])}
    LO, HI = int(z['lo']), int(z['hi'])
    print(f'[pop] loaded {args.pop_cache} items={len(pop)} lo={LO} hi={HI}', flush=True)
else:
    pop = defaultdict(int)
    tl = DataLoader(pipe.tokenized_datasets['train'], batch_size=256, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn['train'])
    with torch.inference_mode():
        for b in tl:
            for row in b['labels'].tolist():
                it = sid_to_item(row[:4])
                if it != -1: pop[it] += 1
    cs = sorted(pop.values())
    LO = cs[len(cs) // 3]; HI = cs[2 * len(cs) // 3]
    print(f'[pop] items={len(pop)} tertiles lo={LO} hi={HI}', flush=True)
    if args.pop_cache:
        its = np.array(sorted(pop.keys()), dtype=np.int64)
        np.savez_compressed(args.pop_cache, items=its,
                            counts=np.array([pop[int(i)] for i in its], dtype=np.int64), lo=LO, hi=HI)


def pop_bucket(it):
    c = pop.get(it, 0)
    if c == 0: return 0      # cold
    if c <= LO: return 1     # tail
    if c <= HI: return 2     # mid
    return 3                 # head


def student_beam(enc, pad, W):
    """Deployed masked-valid factorized search, width W -> (B, W, 4) token-id quads, score-ordered.
    Verbatim geometry of bucket_diag_s34.student_beam_frontiers / frontier_s23.student_beam."""
    B = enc.shape[0]
    ctx = lens.encode(enc, pad)
    d1_lp = F.log_softmax(lens.head1(ctx).float(), dim=-1)
    W1 = min(W, CODEBOOK_SIZE)
    d1_scores, d1_idx = torch.topk(d1_lp, W1, dim=1)
    d1_tokens = d1_idx + DIGIT_OFFSETS[0]
    ctx_flat = ctx.unsqueeze(1).expand(-1, W1, -1).reshape(B * W1, -1)
    d1_flat = d1_tokens.reshape(B * W1)
    logits2 = lens.head2(torch.cat([ctx_flat, lens.embedding(d1_flat)], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
    logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float('-inf'))
    d2_lp = F.log_softmax(logits2, 1).view(B, W1, CODEBOOK_SIZE)
    score2 = d1_scores.unsqueeze(-1) + d2_lp
    best2_score, best2_idx = torch.topk(
        score2.masked_fill(torch.isnan(score2), float('-inf')).view(B, -1), W, 1)
    sel_beam2 = best2_idx // CODEBOOK_SIZE; sel_d2 = best2_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam2); d2_tokens = sel_d2 + DIGIT_OFFSETS[1]
    prefix_score = best2_score
    ctx_flat = ctx.unsqueeze(1).expand(-1, W, -1).reshape(B * W, -1)
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1)
    logits3 = lens.head3(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat)], 1)).float()
    d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
    d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE - 1)
    logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float('-inf'))
    d3_lp = F.log_softmax(logits3, 1).view(B, W, CODEBOOK_SIZE)
    score3 = prefix_score.unsqueeze(-1) + d3_lp
    best3_score, best3_idx = torch.topk(
        score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), W, 1)
    sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam3); d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
    d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
    d1_flat = d1_tokens.reshape(-1); d2_flat = d2_tokens.reshape(-1); d3_flat = d3_tokens.reshape(-1)
    logits4 = lens.head4(torch.cat([ctx_flat, lens.embedding(d1_flat), lens.embedding(d2_flat),
                                    lens.embedding(d3_flat)], 1)).float()
    key3 = triplet_key_int(d1_flat - DIGIT_OFFSETS[0], d2_flat - DIGIT_OFFSETS[1],
                           d3_flat - DIGIT_OFFSETS[2])
    logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    d4_lp = F.log_softmax(logits4, 1)
    top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, 1)
    d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]
    quad = best3_score.unsqueeze(-1) + top_d4_val.view(B, W, CODEBOOK_SIZE)
    topq_score, topq_idx = torch.topk(
        quad.masked_fill(torch.isnan(quad), float('-inf')).view(B, -1), W, 1)
    beam_idx = topq_idx // CODEBOOK_SIZE
    d1f = torch.gather(d1_tokens, 1, beam_idx); d2f = torch.gather(d2_tokens, 1, beam_idx)
    d3f = torch.gather(d3_tokens, 1, beam_idx); d4f = torch.gather(d4_tokens.view(B, -1), 1, topq_idx)
    return torch.stack([d1f, d2f, d3f, d4f], -1)


ds = pipe.tokenized_datasets['test']
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn['test'])

m_codes = None
if args.method == 'precomputed':
    if not args.codes_in:
        raise SystemExit('--method precomputed requires --codes_in')
    m_codes = torch.load(args.codes_in, map_location='cpu')['codes'].long()
    print(f'[codes_in] loaded {args.codes_in} rows={m_codes.shape[0]} '
          f'width={m_codes.shape[1]}', flush=True)
    if m_codes.shape[1] < args.W:
        raise SystemExit(f'--codes_in width {m_codes.shape[1]} < requested W {args.W}')

t_cache = None; t_cache_new = []
if args.teacher_cache and os.path.exists(args.teacher_cache):
    t_cache = torch.load(args.teacher_cache, map_location='cpu')['codes']
    print(f'[tcache] loaded {args.teacher_cache} rows={t_cache.shape[0]}', flush=True)
    if t_cache.shape[1] < args.W:
        raise SystemExit(f'teacher cache beam {t_cache.shape[1]} < requested W {args.W}')

GT, GP, GB = [], [], []
TT, ST, TN, SN, SRAW, SINV, VALID = [], [], [], [], [], [], []
row_ptr = 0
t0 = time.time()
with torch.inference_mode():
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches: break
        d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        input_ids = d['input_ids']; attn = d['attention_mask']; B = input_ids.shape[0]
        labs = batch['labels'].tolist()
        if t_cache is not None:
            TQ = t_cache[row_ptr:row_ptr + B, :args.W].to(dev).long()
        else:
            preds_t = tiger.generate(d, n_return_sequences=args.W)
            TQ = (preds_t.view(B, args.W, -1)[:, :, :4].long() if preds_t.dim() != 3
                  else preds_t[:, :, :4].long())
            t_cache_new.append(TQ.to(torch.int32).cpu())
        row0 = row_ptr
        row_ptr += B
        pt = TQ.tolist()
        if args.method == 'sidmlp':
            enc = t5.encoder(input_ids=input_ids, attention_mask=attn,
                             return_dict=True).last_hidden_state
            ps = student_beam(enc, ~attn.bool(), args.W).tolist()
        elif args.method == 'precomputed':        # s49: served codes decoded elsewhere
            MQ = m_codes[row0:row0 + B, :args.W]
            if MQ.shape[0] != B:
                raise SystemExit(f'--codes_in exhausted at row {row0} '
                                 f'(rows={m_codes.shape[0]}, need {row0 + B})')
            ps = MQ.tolist()
        else:                                    # tiger: the method IS the teacher
            ps = pt
        for r in range(B):
            gt = tuple(int(x) for x in labs[r][:4]); gt_it = sid_to_item(gt)
            t_items, _tinv, _traw = dedup_items(pt[r], TOPK)
            s_items, sinv, sraw = dedup_items(ps[r], TOPK)
            GT.append(gt_it); GP.append(pop.get(gt_it, 0) if gt_it != -1 else -1)
            GB.append(pop_bucket(gt_it) if gt_it != -1 else -1)
            TT.append(t_items + [-1] * (TOPK - len(t_items)))
            ST.append(s_items + [-1] * (TOPK - len(s_items)))
            TN.append(len(t_items)); SN.append(len(s_items))
            SRAW.append(sraw); SINV.append(sinv)
            # doc protocol: a request counts only if the GT is a resolvable item AND the teacher
            # served a non-empty list (identical filter to frontier_s23 / bucket_diag_s34).
            VALID.append(int(gt_it != -1 and len(t_items) > 0))
        if bi % 50 == 0:
            print(f'batch {bi} rows={len(GT)} elapsed={time.time()-t0:.0f}s', flush=True)

if t_cache is None and args.teacher_cache and t_cache_new:
    torch.save({'codes': torch.cat(t_cache_new, 0)}, args.teacher_cache)
    print(f'[tcache] wrote {args.teacher_cache}', flush=True)

gt = np.array(GT, dtype=np.int32); gp = np.array(GP, dtype=np.int64)
gb = np.array(GB, dtype=np.int8); valid = np.array(VALID, dtype=np.int8)
tt = np.array(TT, dtype=np.int32); st = np.array(ST, dtype=np.int32)
tn = np.array(TN, dtype=np.int16); sn = np.array(SN, dtype=np.int16)
sraw = np.array(SRAW, dtype=np.int16); sinv = np.array(SINV, dtype=np.int16)

npz_path = args.out + '.npz'
np.savez_compressed(npz_path, gt_item=gt, gt_pop=gp, gt_bucket=gb, valid=valid,
                    t_top=tt, s_top=st, t_n=tn, s_n=sn, s_raw=sraw, s_invalid=sinv,
                    bucket_lo=LO, bucket_hi=HI, n_test=len(GT), W=args.W, topk=TOPK)

# ---- aggregate sanity block: must reproduce the established frontier/bucketdiag numbers ----
m = valid.astype(bool)
def _rank(lists, k):
    """1-based rank of gt in the first k entries, 0 if absent."""
    sub = lists[:, :k]
    hit = (sub == gt[:, None])
    has = hit.any(1)
    return np.where(has, hit.argmax(1) + 1, 0)

san = {'tag': args.tag, 'category': args.category, 'method': args.method,
       'seed_tag': args.seed_tag, 'lens_path': args.lens_path, 'W': args.W, 'topk': TOPK,
       'n_rows': int(len(GT)), 'n_valid': int(m.sum()),
       'bucket_lo': LO, 'bucket_hi': HI,
       'mean_s_invalid': float(sinv[m].mean()), 'mean_s_n': float(sn[m].mean()),
       'short_at_20': int((sn[m] < 20).sum()), 'short_at_10': int((sn[m] < 10).sum())}
for k in (5, 10, 20):
    tr = _rank(tt, k)[m]; sr = _rank(st, k)[m]
    san[f'teacher_r{k}'] = float((tr > 0).mean())
    san[f'teacher_n{k}'] = float(np.where(tr > 0, 1.0 / np.log2(np.maximum(tr, 1) + 1.0), 0.0).mean())
    san[f'student_r{k}'] = float((sr > 0).mean())
    san[f'student_n{k}'] = float(np.where(sr > 0, 1.0 / np.log2(np.maximum(sr, 1) + 1.0), 0.0).mean())
    inter = np.zeros(int(m.sum())); union = np.zeros(int(m.sum()))
    T = tt[m][:, :k]; S = st[m][:, :k]
    for i in range(T.shape[0]):
        a = set(int(x) for x in T[i] if x >= 0); b = set(int(x) for x in S[i] if x >= 0)
        inter[i] = len(a & b); union[i] = max(len(a | b), 1)
    san[f'jac{k}'] = float((inter / union).mean())
    san[f'overlap{k}'] = float((inter / k).mean())
with open(args.out + '.json', 'w') as f:
    json.dump(san, f, indent=2)
h = hashlib.sha256(open(npz_path, 'rb').read()).hexdigest()[:16]
print(f'=== S48 LISTS {args.tag} {args.category} {args.method} ===')
print(json.dumps(san, indent=2))
print(f'npz={npz_path} sha256_16={h} elapsed={time.time()-t0:.0f}s')
print('S48_LISTS_DONE', flush=True)
