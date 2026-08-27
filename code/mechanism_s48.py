"""s48 mechanism validation — doc §1.6B HDrift / §1.7C boundary FlipRate / §1.7D attribution
faithfulness, all three derived from ONE scoring pass over the shared teacher candidate support.

Protocol (identical for every arm, so arms are directly comparable):
  candidate support = the TEACHER's cached beam-50 list for that request
      (`runs/tbeam50_<CAT>.pt`, the same cache `lists_s48.py` / `bucket_diag_s34.py` use, so the
       doc's "same TIGER Top-K cache / same test users / same item universe" holds by construction),
      item-level deduplicated with invalid SIDs dropped (an invalid SID gets -inf under the
      valid-set mask => a NaN in every softmax below; dropping it is mandatory, not cosmetic),
      then RE-SORTED by the exact teacher total score.
  scored twice:
      teacher: exact per-digit logp + eos logp  (`teacher_steps`, verbatim from bucket_diag_s34)
               => total = sum(4 digits) + logp(eos) == the beam objective the cache was ranked by.
      student: deployed factorized per-digit logp (`list_scores_comp`, verbatim from
               bucket_diag_s34; mathematically identical to sid_mlp.losses.student_list_scores_steps)
               => total = sum(4 digits).  No student beam is run: the support is the teacher's.

Deliverable A (§1.6B) — per-depth HDrift_d = D_KL(q_T^(d) || q_S^(d)), d=1..4, where q^(d) is the
  candidate-list softmax MARGINALISED onto shared depth-d code prefixes.  `serve_kl_terms` below is
  the body of `sid_mlp.losses.serve_kl_loss` (s43e E1 — the loss the main method trains) with only
  the final `(w1*kl1+..+w4*kl4).mean()*(T**2)` reduction removed, so kl1..kl4 are exactly the four
  terms that loss optimises; same eps guards, same prefix grouping, same -1e9 padding convention.
  NOTE the training loss additionally multiplies by temperature**2; HDrift is reported as the raw
  KL at temperature tau (identical at the tau=1.0 default), flagged in the json.

Deliverable B (§1.7C) — boundary FlipRate by teacher-margin quintile.  Pairs match the TRAINING
  definition (`sid_mlp.losses.boundary_pair_loss` / `da_cdrs_loss`): positives = teacher ranks
  1..K (--pair_k), negatives = teacher ranks K+1..B (--pair_b), ranks from the EXACT teacher total
  ordering.  FlipRate = Pr[s_S(i+) <= s_S(i-)].  margin = t_score(i+) - t_score(i-).  Quintile edges
  are computed ONCE over this run's pooled pairs and written to the json/npz; --margin_edges forces
  a given set so several arms can be re-binned onto identical edges.  CIs resample REQUESTS (pairs
  inside a request are dependent), all columns sharing one draw so the CIs are paired.

Deliverable C (§1.7D) — attribution faithfulness on the same pairs.  The attribution block
  (dG_S / dG_T / G_S / G_tilde / A / w) is copied verbatim from `sid_mlp.losses.da_cdrs_loss`; only
  its Huber residual term is dropped (it needs the step_scale artifact and plays no part in
  faithfulness) and the corrected-depth count Dc is made a parameter.
  ** Read this before interpreting the numbers **: da_cdrs_loss defines A_d as the counterfactual
  loss reduction itself, A_d = [l(G_S) - l(G_S - dG_S^d + dG_T^d)]_+, and picks the top-attributed
  depth by argmax_d A_d.  Therefore "reduction at the top-attributed depth" == max_d A_d >= "reduction
  at a random depth" BY CONSTRUCTION, and likewise for the un-flip rate (l is monotone in G).  The
  top-vs-random increment measures how CONCENTRATED the depth attribution is, NOT whether it is
  correct.  The non-tautological comparison is top vs UNIFORM (all-depth) replacement: if a single
  top depth recovers most of what replacing every depth recovers, one depth explains the boundary
  error.  Both are reported.  Reported for Dc=3 (the method's own d<D corrected set — PRIMARY) and
  Dc=4 (the doc's requested d1..d4 view — replacing at all four depths makes G_tilde == G_T exactly).

Tensor spaces (getting this wrong silently yields zero valid candidates):
  * teacher cache `runs/tbeam50_<CAT>.pt['codes']`  -> TOKEN-ID space (vocab, digit-offset applied)
  * `labels` from the collate_fn                    -> TOKEN-ID space
  * `sid_to_item` / `valid_items_4`                 -> keyed in TOKEN-ID space
  * `teacher_steps` / `t5` decoder inputs           -> TOKEN-ID space
  * `list_scores_comp`, `serve_kl_terms` grouping   -> CODEBOOK space (0..255); cand_cb = tok - OFF
  * `model.embedding(...)` inside list_scores_comp  -> TOKEN-ID space (it re-adds DIGIT_OFFSETS)

Emits <out>.npz (per-request arrays: the full per-candidate teacher/student per-digit logp, the
codes, the valid mask -> every aggregate below is re-derivable offline) + <out>.json (aggregates,
CIs, margin edges, every protocol parameter) and prints MECHANISM_S48_DONE.
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
from transformers.modeling_outputs import BaseModelOutput
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)          # TIGER teacher .pth (also supplies embeddings)
ap.add_argument('--lens_path', required=True)           # student ckpt (the arm under diagnosis)
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--W', type=int, default=50)            # candidate support width (the doc's beam=50)
ap.add_argument('--teacher_cache', default='')          # runs/tbeam50_<CAT>.pt; built if missing
ap.add_argument('--pop_cache', default='')              # npz of train pop counts; built if missing
ap.add_argument('--pair_k', type=int, default=10)       # positives = teacher ranks 1..K
ap.add_argument('--pair_b', type=int, default=40)       # negatives = teacher ranks K+1..B
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--gamma', type=float, default=0.05)    # boundary hinge scale (training PAIR_GAMMA)
ap.add_argument('--margin_edges', default='')           # comma-separated; forces the §1.7C bins
ap.add_argument('--seed', type=int, default=0)          # random-depth control + subset + bootstrap
ap.add_argument('--max_users', type=int, default=0)     # 0 = full test; >0 = seeded random subset
ap.add_argument('--boot', type=int, default=2000)       # bootstrap replicates (0 = point only)
ap.add_argument('--out', required=True)                 # path WITHOUT extension
ap.add_argument('--tag', default='')
ap.add_argument('--seed_tag', default='')
args = ap.parse_args()
args.encoder_ckpt = None; args.fp32 = False
dev = torch.device('cuda')
nb16 = configure_precision(args, dev)
MC = args.W
K_POS = args.pair_k
B_HI = min(args.pair_b, MC)
if K_POS >= B_HI:
    raise SystemExit(f'--pair_k {K_POS} must be < min(--pair_b, W) = {B_HI}')
if args.temperature <= 0 or args.gamma <= 0:
    raise SystemExit('--temperature and --gamma must be > 0 (they are divisors)')

pipe = build_tiger_pipeline(args)
tiger = pipe.model.to(dev).eval()
t5 = tiger.t5
if nb16: t5.bfloat16()
for p in t5.parameters(): p.requires_grad_(False)
tk = _torch_load(args.checkpoint, weights_only=False)
lens, _ck = build_sidmlp_from_checkpoint(args.lens_path, tk['t5.shared.weight'], dev)
if nb16: lens.bfloat16()
lens.eval()
del tk

vs = find_valid_set(args.dataset_tag, explicit_path=None,
                    lens_path=args.lens_path, graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, keys4_sorted, vals4_sorted, _im,
 valid_items_4) = build_valid_tensors(vs, dev)
EOS = t5.config.eos_token_id
START = t5.config.decoder_start_token_id
OFF = torch.tensor(DIGIT_OFFSETS, device=dev)
LM = (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted)


def sid_to_item(tup):
    """TOKEN-ID 4-tuple -> item id, or -1 if the SID is not in the valid set."""
    return valid_items_4.get(tuple(int(x) for x in tup), -1)


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


# =============================== reused scorers (VERBATIM) ====================================
# Both copied character-for-character out of code/bucket_diag_s34.py (those scripts are top-level
# imperative, so they cannot be imported without rebuilding the whole pipeline).

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


# =============================== deliverable A: per-depth HDrift ==============================

def serve_kl_terms(s_student, s_teacher, cand_cb, valid_mask, temperature=1.0):
    """PER-REQUEST (kl1, kl2, kl3, kl4), each (B,).  Body copied verbatim from
    sid_mlp.losses.serve_kl_loss; ONLY the final `(w1*kl1+w2*kl2+w3*kl3+w4*kl4).mean()*(T**2)`
    reduction is removed.  cand_cb is CODEBOOK space."""
    B, M = s_student.shape
    sS = (s_student / temperature).masked_fill(~valid_mask, -1e9)
    sT = (s_teacher / temperature).masked_fill(~valid_mask, -1e9)
    log_qS = F.log_softmax(sS, dim=-1)
    log_pT = F.log_softmax(sT, dim=-1)
    pT = log_pT.exp(); pS = log_qS.exp()
    eps = 1e-9
    kl4 = (pT * (log_pT - log_qS)).sum(-1)                                    # d4 == list_kl
    g1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    PT1 = torch.zeros(B, CODEBOOK_SIZE, device=s_student.device, dtype=pT.dtype).scatter_add_(1, g1, pT)
    PS1 = torch.zeros(B, CODEBOOK_SIZE, device=s_student.device, dtype=pS.dtype).scatter_add_(1, g1, pS)
    s1 = PT1 > eps
    kl1 = (PT1 * (torch.log(PT1.clamp_min(eps)) - torch.log(PS1.clamp_min(eps)))).masked_fill(~s1, 0.0).sum(-1)
    g2 = g1 * CODEBOOK_SIZE + cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    K2 = CODEBOOK_SIZE * CODEBOOK_SIZE
    PT2 = torch.zeros(B, K2, device=s_student.device, dtype=pT.dtype).scatter_add_(1, g2, pT)
    PS2 = torch.zeros(B, K2, device=s_student.device, dtype=pS.dtype).scatter_add_(1, g2, pS)
    s2 = PT2 > eps
    kl2 = (PT2 * (torch.log(PT2.clamp_min(eps)) - torch.log(PS2.clamp_min(eps)))).masked_fill(~s2, 0.0).sum(-1)
    g3 = (g2 * CODEBOOK_SIZE + cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)).masked_fill(~valid_mask, -1)
    eq = (g3.unsqueeze(2) == g3.unsqueeze(1)) & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
    eqf = eq.to(pT.dtype)
    mT = torch.bmm(eqf, pT.unsqueeze(-1)).squeeze(-1)
    mS = torch.bmm(eqf, pS.unsqueeze(-1)).squeeze(-1)
    cnt = eq.sum(-1).clamp_min(1).to(pT.dtype)
    term3 = (mT / cnt) * (torch.log(mT.clamp_min(eps)) - torch.log(mS.clamp_min(eps)))
    kl3 = term3.masked_fill(~valid_mask | (mT <= eps), 0.0).sum(-1)
    return kl1, kl2, kl3, kl4


# ============================= deliverable C: attribution block ===============================

def attribution_chunk(t_steps, s_steps, valid, gamma, Dc, gen):
    """Per-pair counterfactual depth attribution + the three replacement reductions, for one
    request chunk.  dG_S / dG_T / G_S / G_tilde / A / w are verbatim `da_cdrs_loss`; its Huber
    residual term is dropped and Dc (= Dd-1 there) is a parameter.
    t_steps/s_steps: (B, M, 4) per-digit logp, teacher-exact-sorted.  valid: (B, M) bool.
    Returns per-request sums (pair-weighted) + global counters."""
    sp = s_steps[:, :K_POS]; sn = s_steps[:, K_POS:B_HI]
    tp = t_steps[:, :K_POS].float(); tn = t_steps[:, K_POS:B_HI].float()
    dG_S = sp.unsqueeze(2) - sn.unsqueeze(1)             # (B,np,neg,D) student per-depth incr gap
    dG_T = tp.unsqueeze(2) - tn.unsqueeze(1)             # teacher per-depth incr gap
    vp = valid[:, :K_POS]; vn = valid[:, K_POS:B_HI]
    pm = (vp.unsqueeze(2) & vn.unsqueeze(1)).float()     # (B,np,neg) active pair
    G_S = dG_S.sum(-1)                                   # (B,np,neg) total student gap
    G_tilde = G_S.unsqueeze(-1) - dG_S[..., :Dc] + dG_T[..., :Dc]   # replace depth-d increment
    l_base = F.softplus(-G_S / gamma)                    # == the boundary_pair_loss hinge
    l_tilde = F.softplus(-G_tilde / gamma)
    A = F.relu(l_base.unsqueeze(-1) - l_tilde)
    w = (A / A.sum(-1, keepdim=True).clamp_min(1e-9)).detach()       # normalized over the Dc depths
    top = A.argmax(-1)                                              # (B,np,neg)
    rnd = torch.randint(0, Dc, top.shape, generator=gen, device=top.device)
    G_top = G_tilde.gather(-1, top.unsqueeze(-1)).squeeze(-1)
    G_rnd = G_tilde.gather(-1, rnd.unsqueeze(-1)).squeeze(-1)
    G_uni = G_S - dG_S[..., :Dc].sum(-1) + dG_T[..., :Dc].sum(-1)   # replace at EVERY corrected depth
    red_top = l_base - F.softplus(-G_top / gamma)
    red_rnd = l_base - F.softplus(-G_rnd / gamma)
    red_uni = l_base - F.softplus(-G_uni / gamma)
    flipped = ((G_S <= 0).float() * pm)                             # the §1.7C flip, pair-level
    out = {
        'n_pairs': pm.sum((1, 2)),
        'attr': (w * pm.unsqueeze(-1)).sum((1, 2)),                 # (B, Dc)
        'l_base': (l_base * pm).sum((1, 2)),
        'red_top': (red_top * pm).sum((1, 2)),
        'red_rnd': (red_rnd * pm).sum((1, 2)),
        'red_uni': (red_uni * pm).sum((1, 2)),
        'n_flip': flipped.sum((1, 2)),
        'unflip_top': (flipped * (G_top > 0).float()).sum((1, 2)),
        'unflip_rnd': (flipped * (G_rnd > 0).float()).sum((1, 2)),
        'unflip_uni': (flipped * (G_uni > 0).float()).sum((1, 2)),
        'top_hist': torch.stack([((top == d).float() * pm).sum() for d in range(Dc)]),
        'n_zero_attr': ((A.sum(-1) <= 0).float() * pm).sum(),
    }
    return out


# =================================== bootstrap (request-level) =================================

def boot_ci(num_mat, den_mat, n_boot, seed):
    """Request-level bootstrap of column ratios sum(num)/sum(den).  num_mat/den_mat are
    (n_requests, m) PER-REQUEST sums (a plain mean is a ones() denominator column).  Requests are
    resampled with replacement; ALL columns share the same draws, so the CIs are paired across
    columns (needed for the top-vs-random / top-vs-uniform increments and the quintile rows).
    Returns (point[m], lo[m], hi[m]) with np.nan where undefined."""
    n, m = num_mat.shape
    point = np.full(m, np.nan); lo = np.full(m, np.nan); hi = np.full(m, np.nan)
    if n == 0:
        return point, lo, hi
    num_tot = num_mat.sum(0); den_tot = den_mat.sum(0)
    nz = den_tot != 0.0
    point[nz] = num_tot[nz] / den_tot[nz]
    if n_boot <= 0:
        return point, lo, hi
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, m), dtype=np.float64)
    # replicates are drawn in chunks purely to cap memory; the RNG stream is consumed
    # sequentially, so the chunk size does not affect the result.
    chunk = max(1, min(256, int(1_000_000 // max(n, 1))))
    done = 0
    while done < n_boot:
        c = min(chunk, n_boot - done)
        ridx = rng.integers(0, n, size=(c, n))
        flat = ridx + (np.arange(c, dtype=np.int64) * n)[:, None]
        w = np.bincount(flat.ravel(), minlength=c * n).reshape(c, n).astype(np.float64)
        a = w @ num_mat
        d = w @ den_mat
        out = np.full((c, m), np.nan)
        np.divide(a, d, out=out, where=(d != 0.0))
        reps[done:done + c, :] = out
        done += c
    for j in range(m):
        col = reps[:, j]; col = col[np.isfinite(col)]
        if col.size >= 2:
            lo[j], hi[j] = np.percentile(col, [2.5, 97.5])
    return point, lo, hi


COLS = []            # (key, num (n,), den (n,))


def add_col(key, num, den):
    COLS.append((key, np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)))


def jnum(v):
    """json-safe: NaN/inf -> None (never emit NaN)."""
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def stat(res, key):
    p, l, h = res[key]
    return {'mean': jnum(p), 'ci_lo': jnum(l), 'ci_hi': jnum(h)}


# ================================ pass 1: the single scoring pass =============================
ds = pipe.tokenized_datasets['test']
n_test = len(ds)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn['test'])

t_cache = None; t_cache_new = []
if args.teacher_cache and os.path.exists(args.teacher_cache):
    t_cache = torch.load(args.teacher_cache, map_location='cpu')['codes']   # [Nt, beam, 4] TOKEN-ID
    print(f'[tcache] loaded {args.teacher_cache} rows={t_cache.shape[0]}', flush=True)
    if t_cache.shape[1] < args.W:
        raise SystemExit(f'teacher cache beam {t_cache.shape[1]} < requested W {args.W}')

sel_set = None; last_needed = n_test - 1
if args.max_users and args.max_users < n_test:
    _rng = np.random.default_rng([args.seed, 7])
    sel = np.sort(_rng.choice(n_test, size=args.max_users, replace=False))
    sel_set = set(int(x) for x in sel)
    last_needed = int(sel[-1])
    print(f'[subset] max_users={args.max_users} of n_test={n_test} '
          f'(seed={args.seed}); last needed row {last_needed}', flush=True)

RIDX, GTI, GPOP, GBKT, NCAND = [], [], [], [], []
CB, VAL, TST, SLP, TEOS = [], [], [], [], []
n_skip_gt = 0; n_skip_empty = 0; n_seen = 0
reorder_set = 0; reorder_pos = 0; reorder_den = 0
row_ptr = 0
t0 = time.time()
with torch.inference_mode():
    for bi, batch in enumerate(loader):
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
        base = row_ptr
        row_ptr += B
        pt = TQ.tolist()                                    # TOKEN-ID space

        keep, cand_codes, cand_n = [], [], []
        for r in range(B):
            g = base + r
            if sel_set is not None and g not in sel_set:
                continue
            n_seen += 1
            gt = tuple(int(x) for x in labs[r][:4]); gt_it = sid_to_item(gt)
            if gt_it == -1:
                n_skip_gt += 1; continue
            seen = set(); codes = []
            for c in pt[r]:
                cc = tuple(int(x) for x in c[:4])
                it = sid_to_item(cc)
                if it == -1 or it in seen:                  # invalid SID -> -inf -> NaN; must drop
                    continue
                seen.add(it); codes.append(cc)
            if not codes:
                n_skip_empty += 1; continue
            nvalid = len(codes)
            codes = codes + [codes[0]] * (MC - nvalid)      # pad slots are masked out everywhere
            keep.append(r); cand_codes.append(codes); cand_n.append(nvalid)
            RIDX.append(g); GTI.append(gt_it)
            GPOP.append(pop.get(gt_it, 0)); GBKT.append(pop_bucket(gt_it))
        if keep:
            enc = t5.encoder(input_ids=input_ids, attention_mask=attn,
                             return_dict=True).last_hidden_state
            pad = ~attn.bool()
            ctx = lens.encode(enc, pad)
            krow = torch.tensor(keep, dtype=torch.long, device=dev)
            Bk = len(keep)
            cand_tok = torch.tensor(cand_codes, dtype=torch.long, device=dev)          # TOKEN-ID
            cand_cb = (cand_tok - OFF.view(1, 1, 4)).clamp(0, CODEBOOK_SIZE - 1)       # CODEBOOK
            ncand = torch.tensor(cand_n, dtype=torch.long, device=dev)
            vmask = torch.arange(MC, device=dev).unsqueeze(0) < ncand.unsqueeze(1)     # (Bk, MC)
            slp = list_scores_comp(lens, ctx.index_select(0, krow), cand_cb, LM)       # (Bk,MC,4)
            cu = krow.repeat_interleave(MC)
            tst, teos = teacher_steps(enc, attn, cu, cand_tok.view(-1, 4))
            tst = tst.view(Bk, MC, 4); teos = teos.view(Bk, MC)
            # ---- re-sort the support by the EXACT teacher total (= the beam objective) ----
            t_tot = tst.sum(-1) + teos
            key = t_tot.masked_fill(~vmask, float('-inf'))
            _sv, order = torch.sort(key, dim=1, descending=True, stable=True)
            full = order[:, :K_POS]
            ok = ncand >= K_POS                              # else the "top-K" includes pad slots
            reorder_set += int(((full >= K_POS).any(1) & ok).sum())
            reorder_pos += int(((full != torch.arange(K_POS, device=dev)).any(1) & ok).sum())
            reorder_den += int(ok.sum())
            gi = order.unsqueeze(-1).expand(-1, -1, 4)
            tst = tst.gather(1, gi); slp = slp.gather(1, gi); cand_cb = cand_cb.gather(1, gi)
            teos = teos.gather(1, order); vmask = vmask.gather(1, order)
            CB.append(cand_cb.to(torch.int16).cpu().numpy())
            VAL.append(vmask.cpu().numpy())
            TST.append(tst.cpu().numpy()); SLP.append(slp.cpu().numpy())
            TEOS.append(teos.cpu().numpy())
            NCAND.extend(cand_n)
        if bi % 50 == 0:
            el = time.time() - t0
            print(f'batch {bi} rows={len(RIDX)} elapsed={el:.0f}s', flush=True)
        if t_cache is not None and row_ptr > last_needed:
            break

if t_cache is None and args.teacher_cache and t_cache_new:
    torch.save({'codes': torch.cat(t_cache_new, 0)}, args.teacher_cache)
    print(f'[tcache] wrote {args.teacher_cache}', flush=True)

n = len(RIDX)
if n == 0:
    raise SystemExit('no usable requests (every request skipped) — check the teacher cache / valid set')
req_index = np.array(RIDX, dtype=np.int64)
gt_item = np.array(GTI, dtype=np.int64)
gt_pop = np.array(GPOP, dtype=np.int64)
gt_bucket = np.array(GBKT, dtype=np.int8)
n_cand = np.array(NCAND, dtype=np.int32)
cand_cb_np = np.concatenate(CB, 0)                      # (n, MC, 4) int16, CODEBOOK space
cand_val_np = np.concatenate(VAL, 0)                    # (n, MC) bool
t_steps_np = np.concatenate(TST, 0).astype(np.float32)  # (n, MC, 4) teacher per-digit logp
s_steps_np = np.concatenate(SLP, 0).astype(np.float32)  # (n, MC, 4) student per-digit logp
t_eos_np = np.concatenate(TEOS, 0).astype(np.float32)   # (n, MC) teacher eos logp
del CB, VAL, TST, SLP, TEOS
t_tot_np = t_steps_np.sum(-1) + t_eos_np                # exact teacher total == beam objective
s_tot_np = s_steps_np.sum(-1)                           # deployed student factorized total
print(f'[pass1] n_used={n} n_seen={n_seen} skip_gt={n_skip_gt} skip_empty={n_skip_empty} '
      f'elapsed={time.time()-t0:.0f}s', flush=True)

# ---------------------------- deliverable A: per-depth HDrift --------------------------------
HD = np.zeros((n, 4), dtype=np.float32)
KLC = 128                                               # (B, 65536) buffers inside serve_kl_terms
with torch.inference_mode():
    for lo in range(0, n, KLC):
        hi = min(n, lo + KLC)
        cb_t = torch.from_numpy(cand_cb_np[lo:hi]).to(dev).long()
        vm_t = torch.from_numpy(cand_val_np[lo:hi]).to(dev)
        sS = torch.from_numpy(s_tot_np[lo:hi]).to(dev)
        sT = torch.from_numpy(t_tot_np[lo:hi]).to(dev)
        k1, k2, k3, k4 = serve_kl_terms(sS, sT, cb_t, vm_t, args.temperature)
        HD[lo:hi] = torch.stack([k1, k2, k3, k4], -1).float().cpu().numpy()
ones = np.ones(n, dtype=np.float64)
for dd in range(4):
    add_col(f'hdrift_d{dd+1}', HD[:, dd], ones)
add_col('hdrift_sum', HD.sum(1), ones)

# ------------------- deliverable B: boundary pairs, margins, quintile bins --------------------
mg_l, fl_l, rq_l = [], [], []
PC = 2048
for lo in range(0, n, PC):
    hi = min(n, lo + PC)
    tp = t_tot_np[lo:hi, :K_POS]; tn = t_tot_np[lo:hi, K_POS:B_HI]
    sp = s_tot_np[lo:hi, :K_POS]; sn = s_tot_np[lo:hi, K_POS:B_HI]
    vp = cand_val_np[lo:hi, :K_POS]; vn = cand_val_np[lo:hi, K_POS:B_HI]
    pmm = vp[:, :, None] & vn[:, None, :]
    if not pmm.any():
        continue
    mg = (tp[:, :, None] - tn[:, None, :])[pmm].astype(np.float32)
    fl = (sp[:, :, None] <= sn[:, None, :])[pmm]
    rq = (np.nonzero(pmm)[0] + lo).astype(np.int32)
    mg_l.append(mg); fl_l.append(fl); rq_l.append(rq)
if mg_l:
    pair_margin = np.concatenate(mg_l); pair_flip = np.concatenate(fl_l)
    pair_req = np.concatenate(rq_l)
else:
    pair_margin = np.zeros(0, dtype=np.float32); pair_flip = np.zeros(0, dtype=bool)
    pair_req = np.zeros(0, dtype=np.int32)
del mg_l, fl_l, rq_l
n_pairs_total = int(pair_margin.size)
if n_pairs_total == 0:
    print('[WARN] zero boundary pairs — every request has fewer than pair_k+1 valid candidates; '
          'deliverables B and C will be empty', flush=True)

edges_source = 'quantile_quintiles'
if args.margin_edges.strip():
    edges = np.array(sorted(float(x) for x in args.margin_edges.split(',') if x.strip()),
                     dtype=np.float64)
    edges_source = 'forced'
elif n_pairs_total > 0:
    edges = np.quantile(pair_margin.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
else:
    edges = np.zeros(0, dtype=np.float64)
NBIN = int(len(edges)) + 1
bin_id = (np.digitize(pair_margin.astype(np.float64), edges) if n_pairs_total > 0
          else np.zeros(0, dtype=np.int64))
flat = pair_req.astype(np.int64) * NBIN + bin_id
bin_n = np.bincount(flat, minlength=n * NBIN).reshape(n, NBIN).astype(np.float64)
bin_f = np.bincount(flat, weights=pair_flip.astype(np.float64),
                    minlength=n * NBIN).reshape(n, NBIN)
bin_m = np.bincount(flat, weights=pair_margin.astype(np.float64),
                    minlength=n * NBIN).reshape(n, NBIN)
pair_n_req = bin_n.sum(1)
add_col('fliprate_all', bin_f.sum(1), pair_n_req)
add_col('margin_mean_all', bin_m.sum(1), pair_n_req)
for q in range(NBIN):
    add_col(f'fliprate_q{q}', bin_f[:, q], bin_n[:, q])
    add_col(f'margin_mean_q{q}', bin_m[:, q], bin_n[:, q])

# ---------------------- deliverable C: attribution faithfulness ------------------------------
CVARS = {}
AC = 1024
for Dc in (3, 4):
    gen = torch.Generator(device=dev); gen.manual_seed(int(args.seed) + 1000 * Dc)
    acc = {k: np.zeros(n, dtype=np.float64) for k in
           ('n_pairs', 'l_base', 'red_top', 'red_rnd', 'red_uni',
            'n_flip', 'unflip_top', 'unflip_rnd', 'unflip_uni')}
    attr = np.zeros((n, Dc), dtype=np.float64)
    top_hist = np.zeros(Dc, dtype=np.float64); n_zero = 0.0
    with torch.inference_mode():
        for lo in range(0, n, AC):
            hi = min(n, lo + AC)
            ts = torch.from_numpy(t_steps_np[lo:hi]).to(dev)
            ss = torch.from_numpy(s_steps_np[lo:hi]).to(dev)
            vv = torch.from_numpy(cand_val_np[lo:hi]).to(dev)
            o = attribution_chunk(ts, ss, vv, args.gamma, Dc, gen)
            for k in acc:
                acc[k][lo:hi] = o[k].double().cpu().numpy()
            attr[lo:hi] = o['attr'].double().cpu().numpy()
            top_hist += o['top_hist'].double().cpu().numpy()
            n_zero += float(o['n_zero_attr'].item())
    CVARS[Dc] = {'acc': acc, 'attr': attr, 'top_hist': top_hist, 'n_zero': n_zero}
    npair = acc['n_pairs']
    for dd in range(Dc):
        add_col(f'attr{Dc}_d{dd+1}', attr[:, dd], npair)
    add_col(f'lbase{Dc}', acc['l_base'], npair)
    add_col(f'red{Dc}_top', acc['red_top'], npair)
    add_col(f'red{Dc}_rand', acc['red_rnd'], npair)
    add_col(f'red{Dc}_unif', acc['red_uni'], npair)
    add_col(f'inc{Dc}_top_minus_rand', acc['red_top'] - acc['red_rnd'], npair)
    add_col(f'inc{Dc}_top_minus_unif', acc['red_top'] - acc['red_uni'], npair)
    add_col(f'unflip{Dc}_top', acc['unflip_top'], acc['n_flip'])
    add_col(f'unflip{Dc}_rand', acc['unflip_rnd'], acc['n_flip'])
    add_col(f'unflip{Dc}_unif', acc['unflip_uni'], acc['n_flip'])

# ------------------------------------- bootstrap ---------------------------------------------
keys = [c[0] for c in COLS]
num_mat = np.column_stack([c[1] for c in COLS])
den_mat = np.column_stack([c[2] for c in COLS])
tb = time.time()
pt_, lo_, hi_ = boot_ci(num_mat, den_mat, args.boot, [args.seed, 20480])
RES = {k: (pt_[j], lo_[j], hi_[j]) for j, k in enumerate(keys)}
print(f'[boot] n_requests={n} cols={len(keys)} reps={args.boot} {time.time()-tb:.1f}s', flush=True)

# --------------------------------------- output ----------------------------------------------
npz_path = args.out + '.npz'
save = dict(
    req_index=req_index, gt_item=gt_item.astype(np.int32), gt_pop=gt_pop, gt_bucket=gt_bucket,
    n_cand=n_cand, cand_cb=cand_cb_np, cand_valid=cand_val_np.astype(np.uint8),
    t_steps=t_steps_np, t_eos=t_eos_np, s_steps=s_steps_np,
    hdrift=HD, margin_edges=np.asarray(edges, dtype=np.float64),
    bin_pair_n=bin_n.astype(np.int64), bin_flip_n=bin_f.astype(np.int64),
    bin_margin_sum=bin_m.astype(np.float64),
    bucket_lo=LO, bucket_hi=HI, n_used=n, n_test=n_test, W=args.W, MC=MC,
    pair_k=K_POS, pair_b=B_HI, temperature=args.temperature, gamma=args.gamma, seed=args.seed)
for Dc in (3, 4):
    cv = CVARS[Dc]
    save[f'attr{Dc}_sum'] = cv['attr']
    for k, v in cv['acc'].items():
        save[f'c{Dc}_{k}'] = v
np.savez_compressed(npz_path, **save)

out = {
    'tag': args.tag, 'seed_tag': args.seed_tag, 'category': args.category,
    'lens_path': args.lens_path, 'checkpoint': args.checkpoint, 'dataset_tag': args.dataset_tag,
    'teacher_cache': args.teacher_cache, 'pop_cache': args.pop_cache,
    'protocol': {
        'candidate_support': 'teacher cached beam-W list, invalid SIDs dropped, item-level '
                             'dedup, re-sorted by the exact teacher total score',
        'teacher_score': 'sum(per-digit logp d1..d4) + logp(eos)  [exact beam objective]',
        'student_score': 'sum(deployed factorized per-digit logp d1..d4)  [no eos term]',
        'W': args.W, 'MC': MC, 'pair_k': K_POS, 'pair_b': B_HI,
        'temperature': args.temperature,
        'temperature_sq_factor_applied': False,
        'temperature_note': 'serve_kl_loss multiplies its sum by temperature**2; HDrift is the raw '
                            'KL at this temperature (identical at the tau=1.0 default)',
        'gamma': args.gamma, 'gamma_note': 'boundary hinge scale; training default PAIR_GAMMA=0.05',
        'seed': args.seed, 'boot_replicates': args.boot, 'batch_size': args.batch_size,
        'bucket_lo': LO, 'bucket_hi': HI,
        'bucket_def': f'train-count tertiles lo={LO} hi={HI} (cold=0), == bucket_diag_s34',
        'precision': 'bf16-compute-fp32-logits (matches bucket_diag_s34 / lists_s48)',
    },
    'counts': {
        'n_test': n_test, 'max_users': args.max_users,
        'n_requests_selected': n_seen, 'n_used': n,
        'n_skip_gt_unresolvable': n_skip_gt, 'n_skip_teacher_list_empty': n_skip_empty,
        'subsampled': bool(sel_set is not None),
        'mean_n_cand': float(n_cand.mean()),
        'n_requests_with_pairs': int((pair_n_req > 0).sum()),
        'n_pairs_total': n_pairs_total,
        'mean_pairs_per_request': jnum(n_pairs_total / n),
    },
    'sanity': {
        'mean_teacher_eos_logp': jnum(float(t_eos_np[cand_val_np].mean())
                                      if cand_val_np.any() else np.nan),
        'eos_note': 'the eos term enters the teacher TOTAL (ranking/margin) but not the per-digit '
                    'increments used by the attribution — same convention as da_cdrs_loss',
        'frac_topk_set_changed_by_exact_rescore': jnum(reorder_set / reorder_den if reorder_den else np.nan),
        'frac_topk_order_changed_by_exact_rescore': jnum(reorder_pos / reorder_den if reorder_den else np.nan),
        'rescore_denominator': reorder_den,
        # deliverable B counts pairs/flips from the per-candidate TOTALS; deliverable C recounts
        # them from the per-DEPTH increments (G_S = sum_d dG_S^d).  Same quantity by two paths and
        # two tensor layouts -> equal counts is the cheap guard that the codebook/token spaces and
        # the exact-teacher sort line up.  Tiny float-summation disagreements at G_S == 0 are possible.
        'pair_count_crosscheck': [n_pairs_total, int(CVARS[3]['acc']['n_pairs'].sum())],
        'flip_count_crosscheck': [int(bin_f.sum()), int(CVARS[3]['acc']['n_flip'].sum())],
    },
    'A_hdrift': {
        'definition': 'D_KL(q_T^(d) || q_S^(d)) over the candidate list marginalised onto shared '
                      'depth-d code prefixes; body verbatim from sid_mlp.losses.serve_kl_loss '
                      '(kl1/kl2/kl3/kl4) minus its weighted-sum-and-mean reduction',
        'per_depth': {f'd{dd+1}': stat(RES, f'hdrift_d{dd+1}') for dd in range(4)},
        'sum_d1_d4': stat(RES, 'hdrift_sum'),
    },
    'B_fliprate': {
        'definition': 'FlipRate = Pr[s_S(i+) <= s_S(i-)] over pairs {teacher rank 1..K} x '
                      '{teacher rank K+1..B}; margin = t_score(i+) - t_score(i-)',
        'margin_edges': [float(x) for x in edges],
        'edges_source': edges_source,
        'bin_convention': 'np.digitize(margin, edges) — bin q covers [edges[q-1], edges[q])',
        'ci_note': 'bootstrap resamples REQUESTS (pairs within a request are dependent)',
        'overall': stat(RES, 'fliprate_all'),
        'overall_margin_mean': stat(RES, 'margin_mean_all'),
        'quintiles': [{
            'bin': q,
            'edge_lo': (float(edges[q - 1]) if q > 0 else None),
            'edge_hi': (float(edges[q]) if q < len(edges) else None),
            'n_pairs': int(bin_n[:, q].sum()),
            'n_requests': int((bin_n[:, q] > 0).sum()),
            'margin_mean': jnum(RES[f'margin_mean_q{q}'][0]),
            'fliprate': jnum(RES[f'fliprate_q{q}'][0]),
            'fliprate_ci_lo': jnum(RES[f'fliprate_q{q}'][1]),
            'fliprate_ci_hi': jnum(RES[f'fliprate_q{q}'][2]),
        } for q in range(NBIN)],
    },
    'C_attribution': {
        'definition': 'A_d = [l(G_S) - l(G_S - dG_S^d + dG_T^d)]_+ normalised over the corrected '
                      'depths, l(G) = softplus(-G/gamma) — verbatim sid_mlp.losses.da_cdrs_loss',
        'tautology_warning': 'top-attributed = argmax_d A_d, and A_d IS the counterfactual loss '
                             'reduction, so red_top >= red_rand (and unflip_top >= unflip_rand) '
                             'hold BY CONSTRUCTION. top-vs-random measures how CONCENTRATED the '
                             'attribution is; top-vs-uniform (all-depth replacement) is the '
                             'non-trivial comparison — top ~= unif means one depth explains the '
                             'boundary error.',
        'primary_variant': 'Dc3',
    },
    'macros_note': 'paper numbers must be registered as N.* measures from this json, not copied',
}
for Dc in (3, 4):
    cv = CVARS[Dc]
    npair_tot = float(cv['acc']['n_pairs'].sum())
    nflip_tot = float(cv['acc']['n_flip'].sum())
    out['C_attribution'][f'Dc{Dc}'] = {
        'corrected_depths': list(range(1, Dc + 1)),
        'note': ('the method\'s own set: da_cdrs_loss corrects depths d < D (=3)' if Dc == 3 else
                 'the doc\'s d1..d4 view: replacing at all four depths makes G_tilde == G_T exactly'),
        'attribution_mean': {f'd{dd+1}': stat(RES, f'attr{Dc}_d{dd+1}') for dd in range(Dc)},
        'top_depth_fraction': {f'd{dd+1}': jnum(cv['top_hist'][dd] / npair_tot if npair_tot else np.nan)
                               for dd in range(Dc)},
        'frac_pairs_zero_attribution': jnum(cv['n_zero'] / npair_tot if npair_tot else np.nan),
        'n_pairs': int(npair_tot), 'n_flipped_pairs': int(nflip_tot),
        'base_pair_loss': stat(RES, f'lbase{Dc}'),
        'reduction': {'top_attributed': stat(RES, f'red{Dc}_top'),
                      'random_depth': stat(RES, f'red{Dc}_rand'),
                      'uniform_all_depth': stat(RES, f'red{Dc}_unif')},
        'increment': {'top_minus_random': stat(RES, f'inc{Dc}_top_minus_rand'),
                      'top_minus_uniform': stat(RES, f'inc{Dc}_top_minus_unif')},
        'unflip_rate': {'top_attributed': stat(RES, f'unflip{Dc}_top'),
                        'random_depth': stat(RES, f'unflip{Dc}_rand'),
                        'uniform_all_depth': stat(RES, f'unflip{Dc}_unif'),
                        'note': 'denominator = pairs the student actually flips (G_S <= 0)'},
    }
with open(args.out + '.json', 'w') as f:
    json.dump(out, f, indent=2)

h = hashlib.sha256(open(npz_path, 'rb').read()).hexdigest()[:16]
print(f'=== S48 MECHANISM {args.tag} {args.category} ===')
print(json.dumps({k: out[k] for k in ('counts', 'sanity', 'A_hdrift')}, indent=2))
print(json.dumps(out['B_fliprate'], indent=2))
print(json.dumps(out['C_attribution']['Dc3'], indent=2))
print(f'npz={npz_path} sha256_16={h} elapsed={time.time()-t0:.0f}s')
print('MECHANISM_S48_DONE', flush=True)
