# s46 TCS-Pair (Tree-Calibrated Served-Swap Pair Distillation) — OFFLINE swap-mining + tree-local
# scale builder + the X0 diagnostics (delivery doc §5 D1-D4). Runs ONCE from the frozen HServe
# (serveU4) warm-start checkpoint. Produces (a) a fixed pair store for training and (b) the D1-D4
# gate report. No training here. Mirrors augment_dump_cdrs.py's pipeline (teacher t5 + packed enc
# cache aligned with the beam dump rows = TRAIN users) + frontier_s23.py's student beam + trie.
#
# Swap mining (delivery §3.2 A): per train user u, teacher served top-K = dedup(dump codes) top-K;
# warm-student served top-K = dedup(HServe student beam) top-K; P_u = teacher\student (pushed-out
# teacher members), N_u = student\teacher (student intruders), pairs = P_u x N_u (cap MAX by most-
# violated warm-student margin). Tree-local scale (delivery §3.2 B): sigma_T(u,p) = 1.4826*MAD of
# teacher valid-child logp at the pair's first-divergence shared prefix p (MAD is shift-invariant,
# so full-vocab log_softmax gathered at valid children == the renormalized child logp for MAD).
import sys, os, json, time, argparse, math
from collections import defaultdict
import torch, torch.nn.functional as F

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, os.path.join(R, "repos", "SID-MLP"))
os.chdir(os.path.join(R, "repos"))
for k in ['HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ[k] = '1'
from transformers.modeling_outputs import BaseModelOutput
from sid_mlp.inference import (build_tiger_pipeline, build_sidmlp_from_checkpoint,
                               configure_precision, _torch_load, find_valid_set)
from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int
from sid_mlp.losses import student_list_scores

ap = argparse.ArgumentParser()
ap.add_argument('--category', required=True)
ap.add_argument('--checkpoint', required=True)      # teacher TIGER ckpt
ap.add_argument('--lens_path', required=True)       # HServe (serveU4) warm-start student ckpt
ap.add_argument('--sem_ids', required=True)
ap.add_argument('--dataset_tag', required=True)
ap.add_argument('--graph_dir', required=True)
ap.add_argument('--dataset', default='AmazonReviews2023')
ap.add_argument('--in_dump', required=True)         # teacher beam dump (teacher top-K + alignment)
ap.add_argument('--out_pairs', required=True)
ap.add_argument('--out_diag', required=True)
ap.add_argument('--K', type=int, default=10)        # served top-K
ap.add_argument('--beam_W', type=int, default=50)   # student beam width
ap.add_argument('--max_pairs', type=int, default=32)
ap.add_argument('--batch_size', type=int, default=256)
ap.add_argument('--chunk', type=int, default=8192)
ap.add_argument('--max_users', type=int, default=0, help='0=all train users; >0 = smoke subset')
ap.add_argument('--d4_users', type=int, default=3000, help='#users for the D4 boundary-flip AUC test')
ap.add_argument('--d4_neg', type=int, default=20, help='#negatives (ranks K..K+d4_neg) per pos in D4')
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
if nb16: lens.bfloat16()
lens.eval()
for p in lens.parameters(): p.requires_grad_(False)
del tk
EOS = t5.config.eos_token_id; START = t5.config.decoder_start_token_id

vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path, graph_dir=args.graph_dir)
(mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted, keys4_sorted, vals4_sorted, _im,
 valid_items_4) = build_valid_tensors(vs, dev)
MASKS = (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted)


_OFF = DIGIT_OFFSETS
# valid_items_4 is VOCAB-keyed (beam.py does d1-DIGIT_OFFSETS[0]); dump codes + student_beam here
# are CODEBOOK-space, so build a codebook-keyed item map for consistent lookup.
item_of_cb = {(k[0] - _OFF[0], k[1] - _OFF[1], k[2] - _OFF[2], k[3] - _OFF[3]): v
              for k, v in valid_items_4.items()}


def sid_to_item(tup):  # tup in CODEBOOK space 0..255
    return item_of_cb.get(tuple(int(x) for x in tup), -1)


# ---------------- student base beam (verbatim from frontier_s23; HServe=serveU4 is 'base') --------
def student_beam(enc, pad, W):
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
    best2_score, best2_idx = torch.topk(score2.masked_fill(torch.isnan(score2), float('-inf')).view(B, -1), W, 1)
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
    best3_score, best3_idx = torch.topk(score3.masked_fill(torch.isnan(score3), float('-inf')).view(B, -1), W, 1)
    sel_beam3 = best3_idx // CODEBOOK_SIZE; sel_d3 = best3_idx % CODEBOOK_SIZE
    d1_tokens = torch.gather(d1_tokens, 1, sel_beam3); d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
    d3_tokens = sel_d3 + DIGIT_OFFSETS[2]
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
    quads = torch.stack([d1f, d2f, d3f, d4f], -1)  # vocab-space tokens
    return quads - torch.tensor(DIGIT_OFFSETS, device=dev).view(1, 1, 4)  # -> codebook 0..255


@torch.inference_mode()
def teacher_total_score(enc_b, attn, cand_user, cand_cb, chunk):
    """exact teacher-forced total = sum logp(d1..d4)+logp(eos), the beam objective. cand_cb (N,4) cb."""
    N = cand_cb.shape[0]
    OFF = torch.tensor(DIGIT_OFFSETS, dtype=torch.long, device=dev).view(1, 4)
    tok = cand_cb + OFF
    dec_in = torch.cat([torch.full((N, 1), START, dtype=torch.long, device=dev), tok], 1)
    out = torch.empty(N, dtype=torch.float32, device=dev)
    for lo in range(0, N, chunk):
        hi = min(N, lo + chunk); rows = cand_user[lo:hi]
        eo = BaseModelOutput(last_hidden_state=enc_b.index_select(0, rows))
        am = attn.index_select(0, rows)
        o = t5(encoder_outputs=eo, attention_mask=am, decoder_input_ids=dec_in[lo:hi], return_dict=True)
        lp = F.log_softmax(o.logits.float(), dim=-1)
        out[lo:hi] = lp[:, :4, :].gather(2, tok[lo:hi].unsqueeze(-1)).squeeze(-1).sum(1) + lp[:, 4, EOS]
    return out


@torch.inference_mode()
def teacher_tree_scale(enc_b, attn, pf_user, pf_prefix, pf_len, chunk=4096):
    """sigma_T(u,p) = 1.4826 * MAD{ log p_T(child | u, p) : child in valid children of p }.
    pf_prefix: (N,3) cb (only first pf_len used); pf_len: (N,) in {0,1,2,3} = shared-prefix length
    (= first_div_depth-1). Reads teacher next-digit logp at depth pf_len (0-idx position pf_len) and
    MADs over the trie-valid children. Processed grouped by pf_len."""
    N = pf_prefix.shape[0]
    OFF = torch.tensor(DIGIT_OFFSETS, dtype=torch.long, device=dev)
    sigma = torch.full((N,), 1e-4, dtype=torch.float32, device=dev)
    for L in (0, 1, 2, 3):
        sel = (pf_len == L).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        for lo in range(0, sel.numel(), chunk):
            idx = sel[lo:lo + chunk]
            rows = pf_user[idx]
            eo = BaseModelOutput(last_hidden_state=enc_b.index_select(0, rows))
            am = attn.index_select(0, rows)
            if L == 0:
                dec = torch.full((idx.numel(), 1), START, dtype=torch.long, device=dev)
            else:
                pf = pf_prefix[idx, :L] + OFF[:L].view(1, L)
                dec = torch.cat([torch.full((idx.numel(), 1), START, dtype=torch.long, device=dev), pf], 1)
            o = t5(encoder_outputs=eo, attention_mask=am, decoder_input_ids=dec, return_dict=True)
            logp = F.log_softmax(o.logits[:, -1, :].float(), dim=-1)      # next-digit at depth L
            # valid children mask (codebook tokens at digit offset L)
            child_tok = (torch.arange(CODEBOOK_SIZE, device=dev) + DIGIT_OFFSETS[L]).view(1, -1)
            clp = logp.gather(1, child_tok.expand(idx.numel(), -1))       # (n,256) child logp
            if L == 0:
                vm = torch.ones(idx.numel(), CODEBOOK_SIZE, dtype=torch.bool, device=dev)  # d1 unmasked
            elif L == 1:
                c1 = pf_prefix[idx, 0].clamp(0, CODEBOOK_SIZE - 1)
                vm = mask_d1_d2[c1]
            elif L == 2:
                c1 = pf_prefix[idx, 0].clamp(0, CODEBOOK_SIZE - 1)
                c2 = pf_prefix[idx, 1].clamp(0, CODEBOOK_SIZE - 1)
                vm = mask_d1d2_d3[c1, c2]
            else:
                key3 = triplet_key_int(pf_prefix[idx, 0].clamp(0, CODEBOOK_SIZE - 1),
                                       pf_prefix[idx, 1].clamp(0, CODEBOOK_SIZE - 1),
                                       pf_prefix[idx, 2].clamp(0, CODEBOOK_SIZE - 1))
                vm = gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3)
            # MAD over valid children (VECTORIZED; >=2 valid needed, else keep 1e-4)
            masked = torch.where(vm, clp, torch.tensor(float('nan'), device=dev))
            med = torch.nanmedian(masked, dim=1).values
            mad = torch.nanmedian((masked - med.unsqueeze(1)).abs(), dim=1).values
            sig = 1.4826 * mad
            sig = torch.where(vm.sum(1) >= 2, sig, torch.full_like(sig, 1e-4))
            sigma[idx] = torch.nan_to_num(sig, nan=1e-4).clamp_min(1e-4)
    return sigma


def first_div(a, b):
    """a,b: 4-tuples cb. returns (depth 1..4, shared_prefix list len depth-1). depth=first position
    they differ (1-based); if identical returns 4 (shouldn't happen for distinct items)."""
    for d in range(4):
        if a[d] != b[d]:
            return d + 1, list(a[:d])
    return 4, list(a[:3])


# ---------------- load teacher dump + packed enc cache (aligned train users) --------------------
D = torch.load(args.in_dump, map_location='cpu', weights_only=False)
codes = D['codes']       # (Ntr,W,4) uint8 cb, teacher-score-sorted
scores = D['scores']     # (Ntr,W) fp16 exact total
nvalid = D['nvalid']     # (Ntr,)
Ntr, Wd, _ = codes.shape
cache_file = os.path.join(args.graph_dir, 'hidden_states', args.dataset_tag, 'packed_cache', 'k0_fp32.pt')
cached = torch.load(cache_file, map_location='cpu', weights_only=False, mmap=True)
enc_all = cached['enc_seq']; msk_all = cached['enc_mask']
assert enc_all.shape[0] == Ntr, (enc_all.shape[0], Ntr)
NU = Ntr if args.max_users <= 0 else min(args.max_users, Ntr)
print(f'[tcs] dump Ntr={Ntr} W={Wd}; mining over {NU} train users; K={args.K} max_pairs={args.max_pairs}', flush=True)

# ---------------- mine swaps + build pair records --------------------
recs = []  # each: dict(user, pos_code[4], neg_code[4], t_pos, t_neg, ws_pos, ws_neg, first_div, sigma)
n_users_with_swap = 0
pu_sizes, nu_sizes, pair_counts, displaced = [], [], [], []
bs = args.batch_size; t0 = time.time()
for blo in range(0, NU, bs):
    bhi = min(NU, blo + bs)
    enc = enc_all[blo:bhi].to(dev, non_blocking=True)
    pad = msk_all[blo:bhi].to(dev, non_blocking=True)
    attn = (~pad).long()
    enc_b = enc.bfloat16() if nb16 else enc
    B = enc.shape[0]
    sq = student_beam(enc_b, pad, args.beam_W)                  # (B,W,4) cb
    sq_l = sq.tolist()
    cb_b = codes[blo:bhi].long()                                # teacher codes (B,Wd,4) cb
    nv_b = nvalid[blo:bhi].tolist()
    # per-user mining; collect batch-local pair rows for one teacher pass
    b_pos_cb, b_neg_cb, b_user = [], [], []
    b_meta = []  # (global_user, first_div, shared_prefix(list len3 padded))
    for r in range(B):
        # teacher top-K items+quads
        t_items, t_quads, seen = [], [], set()
        for q in cb_b[r][:nv_b[r]].tolist():
            it = sid_to_item(q)
            if it == -1 or it in seen: continue
            seen.add(it); t_items.append(it); t_quads.append(tuple(q))
            if len(t_items) >= args.K: break
        # student top-K items+quads
        s_items, s_quads, sseen = [], [], set()
        for q in sq_l[r]:
            it = sid_to_item(q)
            if it == -1 or it in sseen: continue
            sseen.add(it); s_items.append(it); s_quads.append(tuple(q))
            if len(s_items) >= args.K: break
        t_set, s_set = set(t_items), set(s_items)
        pos = [(it, q) for it, q in zip(t_items, t_quads) if it not in s_set]  # pushed-out teacher
        neg = [(it, q) for it, q in zip(s_items, s_quads) if it not in t_set]  # student intruders
        pu_sizes.append(len(pos)); nu_sizes.append(len(neg))
        displaced.append(len(pos))
        if not pos or not neg:
            pair_counts.append(0); continue
        n_users_with_swap += 1
        pairs = [(pq, nq) for (_, pq) in pos for (_, nq) in neg]
        pair_counts.append(len(pairs))
        for pq, nq in pairs:
            fd, sp = first_div(pq, nq)
            sp3 = (sp + [0, 0, 0])[:3]
            b_pos_cb.append(list(pq)); b_neg_cb.append(list(nq)); b_user.append(r)
            b_meta.append((blo + r, fd, sp3))
    if not b_pos_cb:
        continue
    # teacher exact scores for all pos & neg of this batch (one pass each)
    pcb = torch.tensor(b_pos_cb, dtype=torch.long, device=dev)
    ncb = torch.tensor(b_neg_cb, dtype=torch.long, device=dev)
    urow = torch.tensor(b_user, dtype=torch.long, device=dev)
    t_pos = teacher_total_score(enc_b, attn, urow, pcb, args.chunk)
    t_neg = teacher_total_score(enc_b, attn, urow, ncb, args.chunk)
    # warm-student factorized scores for pos & neg (deployed scorer over the 2-cand list per pair)
    cand2 = torch.stack([pcb, ncb], dim=1)                      # (P,2,4)
    ws = student_list_scores(lens, enc_b.index_select(0, urow), pad.index_select(0, urow), cand2, MASKS)
    ws_pos, ws_neg = ws[:, 0], ws[:, 1]
    # tree scale at each pair's shared prefix
    fd_t = torch.tensor([m[1] for m in b_meta], dtype=torch.long, device=dev)
    sp_t = torch.tensor([m[2] for m in b_meta], dtype=torch.long, device=dev)
    sigma = teacher_tree_scale(enc_b, attn, urow, sp_t, (fd_t - 1))
    t_pos_l = t_pos.tolist(); t_neg_l = t_neg.tolist()
    wsp_l = ws_pos.tolist(); wsn_l = ws_neg.tolist(); sig_l = sigma.tolist()
    for i, m in enumerate(b_meta):
        recs.append({'user': m[0], 'pos_code': b_pos_cb[i], 'neg_code': b_neg_cb[i],
                     't_pos': t_pos_l[i], 't_neg': t_neg_l[i], 'ws_pos': wsp_l[i], 'ws_neg': wsn_l[i],
                     'first_div': m[1], 'sigma': sig_l[i]})
    if (blo // bs) % 20 == 0:
        el = time.time() - t0
        print(f'[tcs] {bhi}/{NU} users, {len(recs)} pairs, swap-users {n_users_with_swap}, '
              f'el {el:.0f}s eta {el/max(bhi,1)*(NU-bhi):.0f}s', flush=True)

# ---------------- cap per-user to max_pairs by most-violated warm-student margin --------------------
by_user = defaultdict(list)
for rc in recs:
    by_user[rc['user']].append(rc)
kept = []
for u, lst in by_user.items():
    lst.sort(key=lambda x: (x['ws_pos'] - x['ws_neg']))   # most-violated (smallest student margin) first
    kept.extend(lst[:args.max_pairs])
print(f'[tcs] pairs before cap {len(recs)} -> after cap {len(kept)} (max_pairs={args.max_pairs})', flush=True)

# ================= D1-D4 DIAGNOSTICS (delivery §5) =================
diag = {'category': args.category, 'lens_path': args.lens_path, 'n_users': NU, 'K': args.K,
        'max_pairs': args.max_pairs}


def pct(x, p):
    if not x: return 0.0
    xs = sorted(x); return xs[min(len(xs) - 1, int(p * len(xs)))]


# D1: residual served-swap counts
frac_swap = n_users_with_swap / max(NU, 1)
diag['D1'] = {
    'frac_users_with_swap': frac_swap,
    'pos_mean': sum(pu_sizes) / max(len(pu_sizes), 1), 'pos_p90': pct(pu_sizes, 0.9),
    'neg_mean': sum(nu_sizes) / max(len(nu_sizes), 1), 'neg_p90': pct(nu_sizes, 0.9),
    'pairs_per_user_mean_raw': sum(pair_counts) / max(len(pair_counts), 1),
    'pairs_per_user_p90_raw': pct(pair_counts, 0.9),
    'displaced_mean': sum(displaced) / max(len(displaced), 1),
    'total_pairs_raw': len(recs), 'total_pairs_capped': len(kept),
    'GATE_pass': frac_swap >= 0.10, 'GATE_rule': 'frac_users_with_swap >= 0.10',
}
# D2: first-divergence distribution (over capped pairs)
fdc = defaultdict(int)
for r in kept: fdc[r['first_div']] += 1
tot = max(len(kept), 1)
diag['D2'] = {'first_div_frac': {str(d): fdc.get(d, 0) / tot for d in (1, 2, 3, 4)}, 'n': len(kept)}
# D3: subtree scale heterogeneity
sig_all = [r['sigma'] for r in kept]
sub = defaultdict(list)
for r in kept:
    key = (r['first_div'], tuple(r['pos_code'][:r['first_div'] - 1]))
    sub[key].append(r['sigma'])
sub_means = [sum(v) / len(v) for v in sub.values() if len(v) >= 1]
glob_mean = sum(sig_all) / max(len(sig_all), 1)
glob_sd = (sum((s - glob_mean) ** 2 for s in sig_all) / max(len(sig_all), 1)) ** 0.5
cv_overall = glob_sd / max(glob_mean, 1e-9)
sm_mean = sum(sub_means) / max(len(sub_means), 1)
sm_sd = (sum((s - sm_mean) ** 2 for s in sub_means) / max(len(sub_means), 1)) ** 0.5
cv_across = sm_sd / max(sm_mean, 1e-9)
by_depth = {}
for d in (1, 2, 3, 4):
    ds = [r['sigma'] for r in kept if r['first_div'] == d]
    if ds:
        by_depth[str(d)] = {'n': len(ds), 'mean': sum(ds) / len(ds), 'median': pct(ds, 0.5),
                            'p10': pct(ds, 0.1), 'p90': pct(ds, 0.9)}
diag['D3'] = {
    'sigma_count': len(sig_all), 'sigma_mean': glob_mean, 'sigma_median': pct(sig_all, 0.5),
    'sigma_p10': pct(sig_all, 0.1), 'sigma_p90': pct(sig_all, 0.9),
    'cv_overall': cv_overall, 'n_subtrees': len(sub_means), 'cv_across_subtrees': cv_across,
    'by_first_div_depth': by_depth,
    'GATE_pass': cv_across >= 0.10, 'GATE_rule': 'cv_across_subtrees >= 0.10 (else scale ~constant, drop tree branch)',
}

# ---------------- D4: raw vs tree-normalized teacher margin -> student boundary flip (AUC) ----------
def auc(scores_, labels_):
    order = sorted(range(len(scores_)), key=lambda i: scores_[i])
    npos = sum(labels_); nneg = len(labels_) - npos
    if npos == 0 or nneg == 0: return float('nan')
    rank_sum = 0.0; i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores_[order[j]] == scores_[order[i]]: j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for kk in range(i, j):
            if labels_[order[kk]] == 1: rank_sum += avg_rank
        i = j
    return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


NPOS_B, NNEG_HI = args.K, 40
d4_raw_m, d4_norm_m, d4_flip = [], [], []
du = min(args.d4_users, NU)
for blo in range(0, du, bs):
    bhi = min(du, blo + bs)
    enc = enc_all[blo:bhi].to(dev); pad = msk_all[blo:bhi].to(dev); attn = (~pad).long()
    enc_b = enc.bfloat16() if nb16 else enc
    B = enc.shape[0]
    cb_b = codes[blo:bhi].long().to(dev)                       # (B,Wd,4)
    nv_b = nvalid[blo:bhi]
    ws = student_list_scores(lens, enc_b, pad, cb_b[:, :NNEG_HI], MASKS)   # (B,40) warm-student
    sc_b = scores[blo:bhi, :NNEG_HI].float()                   # teacher totals (dump)
    pf_user, pf_prefix, pf_len, raw_m, flip = [], [], [], [], []
    for r in range(B):
        nvr = int(nv_b[r])
        for pi in range(min(NPOS_B, nvr)):
            for ni in range(NPOS_B, min(NPOS_B + args.d4_neg, nvr)):
                pq = cb_b[r, pi].tolist(); nq = cb_b[r, ni].tolist()
                if pq == nq: continue
                fd, sp = first_div(tuple(pq), tuple(nq))
                mt = float(sc_b[r, pi] - sc_b[r, ni])           # teacher margin
                ms = float(ws[r, pi] - ws[r, ni])               # student margin
                pf_user.append(r); pf_prefix.append((sp + [0, 0, 0])[:3]); pf_len.append(fd - 1)
                raw_m.append(mt); flip.append(1 if ms < 0 else 0)
    if not raw_m:
        continue
    urow = torch.tensor(pf_user, dtype=torch.long, device=dev)
    sp_t = torch.tensor(pf_prefix, dtype=torch.long, device=dev)
    fl_t = torch.tensor(pf_len, dtype=torch.long, device=dev)
    sig = teacher_tree_scale(enc_b, attn, urow, sp_t, fl_t).tolist()
    for i in range(len(raw_m)):
        d4_raw_m.append(raw_m[i]); d4_norm_m.append(raw_m[i] / max(sig[i], 1e-4)); d4_flip.append(flip[i])
auc_raw = auc([-m for m in d4_raw_m], d4_flip)
auc_norm = auc([-m for m in d4_norm_m], d4_flip)
diag['D4'] = {
    'n_pairs': len(d4_flip), 'flip_rate': sum(d4_flip) / max(len(d4_flip), 1),
    'auc_raw_margin': auc_raw, 'auc_tree_normalized_margin': auc_norm,
    'auc_gain': (auc_norm - auc_raw) if not (math.isnan(auc_raw) or math.isnan(auc_norm)) else None,
    'GATE_pass': (not math.isnan(auc_norm) and not math.isnan(auc_raw) and auc_norm > auc_raw + 0.005),
    'GATE_rule': 'tree-normalized AUC > raw AUC + 0.005 (else normalization does not help flip prediction)',
}

# ---------------- save pairs + diag --------------------
torch.save({'category': args.category, 'lens_path': args.lens_path, 'K': args.K,
            'max_pairs': args.max_pairs, 'pairs': kept,
            'meta': {'n_users': NU, 'in_dump': args.in_dump}}, args.out_pairs)
with open(args.out_diag, 'w') as f:
    json.dump(diag, f, indent=2)
print('==== D1-D4 GATE SUMMARY ====', flush=True)
print(json.dumps(diag, indent=2)[:4000], flush=True)
print(f'[tcs] SAVED pairs -> {args.out_pairs} ({len(kept)} pairs) ; diag -> {args.out_diag}', flush=True)
print('TCS_MINE_DONE', flush=True)
