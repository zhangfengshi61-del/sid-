"""Loss functions for SID-MLP distillation training."""

import torch
import torch.nn.functional as F

from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int


def kl_distillation_loss(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)


def combined_loss(student_logits, teacher_logits, gt_labels, temperature=1.0, alpha=0.7):
    kl_loss = kl_distillation_loss(student_logits, teacher_logits, temperature)
    ce_loss = F.cross_entropy(student_logits, gt_labels)
    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def loss_fn(args, student_logits, teacher_logits, gt_labels):
    if args.loss == "kl":
        return kl_distillation_loss(student_logits, teacher_logits, args.temperature)
    if args.loss == "ce":
        return F.cross_entropy(student_logits, gt_labels)
    return combined_loss(student_logits, teacher_logits, gt_labels, args.temperature, args.alpha)


def d4_loss_fn(args, logits4, tl4, gd4):
    if args.d4_teacher == "kl":
        return kl_distillation_loss(logits4, tl4, args.temperature)
    if args.d4_teacher == "ce":
        return F.cross_entropy(logits4, gd4)
    return loss_fn(args, logits4, tl4, gd4)


@torch.no_grad()
def topk_correct(logits: torch.Tensor, labels_cb: torch.Tensor, k: int = 5):
    topk_idx = logits.topk(k, dim=-1).indices
    c1 = (topk_idx[:, 0] == labels_cb).sum().item()
    ck = (topk_idx == labels_cb.unsqueeze(1)).any(dim=1).sum().item()
    return c1, ck


# ---------------------------------------------------------------------------
# s24: search-distribution distillation.
# All losses above supervise teacher-forced GT-prefix digit conditionals; the deployed
# beam instead consumes (a) conditionals on non-GT prefixes and (b) the relative joint
# calibration across candidate codes. These losses supervise exactly that, from a
# precomputed dump of the teacher's beam top-M complete codes + exact sequence scores.
# ---------------------------------------------------------------------------

def _gather_d4_mask_grad(keys3_sorted, masks3_sorted, key3_query):
    """gather_d4_mask minus the @torch.inference_mode() decorator (whose output tensors
    cannot be saved for backward)."""
    if keys3_sorted.numel() == 0:
        return torch.zeros((key3_query.numel(), CODEBOOK_SIZE), dtype=torch.bool, device=key3_query.device)
    idx = torch.searchsorted(keys3_sorted, key3_query).clamp(0, keys3_sorted.numel() - 1)
    found = keys3_sorted[idx] == key3_query
    out = torch.zeros((key3_query.numel(), CODEBOOK_SIZE), dtype=torch.bool, device=key3_query.device)
    if found.any():
        out[found] = masks3_sorted[idx[found]]
    return out


def student_list_scores(model, enc_seq, enc_mask, cand_cb, masks):
    """Deployed-equivalent factorized score of complete candidate codes.

    cand_cb: (B, M, 4) long, CODEBOOK-space digits (0..255).
    masks: (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted) — CLONED out of
    inference mode by the caller.
    Scoring reproduces the deployed student beam exactly: level-1 log-softmax UNmasked;
    levels 2-4 valid-set-masked then log-softmaxed (renormalized). Returns (B, M) scores
    with gradients into the context module + all four heads at the candidates' prefixes
    (including non-GT prefixes — the supervision GT-prefix KD never provides).
    """
    mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted = masks
    B, M, _ = cand_cb.shape
    c1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    c2 = cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    c3 = cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)
    c4 = cand_cb[:, :, 3].clamp(0, CODEBOOK_SIZE - 1)
    v1 = (c1 + DIGIT_OFFSETS[0]).reshape(-1)
    v2 = (c2 + DIGIT_OFFSETS[1]).reshape(-1)
    v3 = (c3 + DIGIT_OFFSETS[2]).reshape(-1)

    ctx = model.encode(enc_seq, enc_mask)                                   # (B, D)
    lp1_all = F.log_softmax(model.head1(ctx).float(), dim=-1)               # (B, 256) unmasked
    lp1 = lp1_all.gather(1, c1)                                             # (B, M)

    D = ctx.shape[1]
    ctx_f = ctx.unsqueeze(1).expand(-1, M, -1).reshape(B * M, D)
    e1 = model.embedding(v1)
    e2 = model.embedding(v2)
    e3 = model.embedding(v3)

    l2 = model.head2(torch.cat([ctx_f, e1], 1)).float()
    l2 = l2.masked_fill(~mask_d1_d2[c1.reshape(-1)], float('-inf'))
    lp2 = F.log_softmax(l2, 1).gather(1, c2.reshape(-1, 1)).view(B, M)

    l3 = model.head3(torch.cat([ctx_f, e1, e2], 1)).float()
    l3 = l3.masked_fill(~mask_d1d2_d3[c1.reshape(-1), c2.reshape(-1)], float('-inf'))
    lp3 = F.log_softmax(l3, 1).gather(1, c3.reshape(-1, 1)).view(B, M)

    l4 = model.head4(torch.cat([ctx_f, e1, e2, e3], 1)).float()
    key3 = triplet_key_int(c1.reshape(-1), c2.reshape(-1), c3.reshape(-1))
    l4 = l4.masked_fill(~_gather_d4_mask_grad(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    lp4 = F.log_softmax(l4, 1).gather(1, c4.reshape(-1, 1)).view(B, M)

    return lp1 + lp2 + lp3 + lp4


def list_kl_loss(s_student, s_teacher, valid_mask, temperature=1.0):
    """KL( softmax(s_T/tau) || softmax(s_S/tau) ) over each sample's candidate list.
    s_*: (B, M) scores; valid_mask: (B, M) bool. Padding gets ~zero teacher mass."""
    sS = (s_student / temperature).masked_fill(~valid_mask, -1e9)
    sT = (s_teacher / temperature).masked_fill(~valid_mask, -1e9)
    log_qS = F.log_softmax(sS, dim=-1)
    log_pT = F.log_softmax(sT, dim=-1)
    pT = log_pT.exp()
    kl = (pT * (log_pT - log_qS)).sum(-1)
    return kl.mean() * (temperature ** 2)


def prefix_kl_loss(s_student, s_teacher, cand_cb, valid_mask, temperature=1.0, w2=1.0, w3=0.0):  # s36
    """List-marginalized prefix-frontier KL at levels 1 and 2 (control for the
    'complete-code list scores do not constrain prefix survival' failure mode).
    Marginalizes both list distributions onto d1 groups and (d1,d2) groups and
    matches the group masses."""
    B, M = s_student.shape
    sS = (s_student / temperature).masked_fill(~valid_mask, -1e9)
    sT = (s_teacher / temperature).masked_fill(~valid_mask, -1e9)
    pS = F.softmax(sS, dim=-1)
    pT = F.softmax(sT, dim=-1)
    eps = 1e-9

    g1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    P_T1 = torch.zeros(B, CODEBOOK_SIZE, device=s_student.device, dtype=pT.dtype).scatter_add_(1, g1, pT)
    P_S1 = torch.zeros(B, CODEBOOK_SIZE, device=s_student.device, dtype=pS.dtype).scatter_add_(1, g1, pS)
    sup1 = P_T1 > eps
    kl1 = (P_T1 * (torch.log(P_T1.clamp_min(eps)) - torch.log(P_S1.clamp_min(eps)))).masked_fill(~sup1, 0.0).sum(-1)

    g2 = g1 * CODEBOOK_SIZE + cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    K2 = CODEBOOK_SIZE * CODEBOOK_SIZE
    P_T2 = torch.zeros(B, K2, device=s_student.device, dtype=pT.dtype).scatter_add_(1, g2, pT)
    P_S2 = torch.zeros(B, K2, device=s_student.device, dtype=pS.dtype).scatter_add_(1, g2, pS)
    sup2 = P_T2 > eps
    kl2 = (P_T2 * (torch.log(P_T2.clamp_min(eps)) - torch.log(P_S2.clamp_min(eps)))).masked_fill(~sup2, 0.0).sum(-1)

    kl3 = torch.zeros_like(kl2)  # s36
    if w3 and w3 > 0.0:
        # s36: d1d2d3-marginal KL via per-member group masses (M<=64 -> O(M^2) cheap);
        # each group's KL term is spread over members with 1/count weight so the group sums once.
        g3 = (g2 * CODEBOOK_SIZE + cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)).masked_fill(~valid_mask, -1)
        eq = (g3.unsqueeze(2) == g3.unsqueeze(1)) & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        eqf = eq.to(pT.dtype)
        mT = torch.bmm(eqf, pT.unsqueeze(-1)).squeeze(-1)
        mS = torch.bmm(eqf, pS.unsqueeze(-1)).squeeze(-1)
        cnt = eq.sum(-1).clamp_min(1).to(pT.dtype)
        term = (mT / cnt) * (torch.log(mT.clamp_min(eps)) - torch.log(mS.clamp_min(eps)))
        kl3 = term.masked_fill(~valid_mask | (mT <= eps), 0.0).sum(-1)
    return (kl1 + w2 * kl2 + w3 * kl3).mean()


def student_list_scores_jp(model, enc_seq, enc_mask, cand_cb, masks):
    """s24 arm C: deployed-equivalent scores for the JP variant — joint log-softmax over
    occupied prefixes + the same masked-renormalized suffix levels as the JP beam."""
    _m12, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted = masks
    B, M, _ = cand_cb.shape
    c1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    c2 = cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    c3 = cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)
    c4 = cand_cb[:, :, 3].clamp(0, CODEBOOK_SIZE - 1)
    v1 = (c1 + DIGIT_OFFSETS[0]).reshape(-1)
    v2 = (c2 + DIGIT_OFFSETS[1]).reshape(-1)
    v3 = (c3 + DIGIT_OFFSETS[2]).reshape(-1)

    ctx = model.encode(enc_seq, enc_mask)
    jp_all = F.log_softmax(model.headJP(ctx).float(), dim=-1)
    cls = model.jp_inv[c1 * CODEBOOK_SIZE + c2]
    lpJP = jp_all.gather(1, cls.clamp_min(0))
    lpJP = lpJP.masked_fill(cls < 0, -1e9)

    D = ctx.shape[1]
    ctx_f = ctx.unsqueeze(1).expand(-1, M, -1).reshape(B * M, D)
    e1 = model.embedding(v1)
    e2 = model.embedding(v2)
    e3 = model.embedding(v3)
    l3 = model.head3(torch.cat([ctx_f, e1, e2], 1)).float()
    l3 = l3.masked_fill(~mask_d1d2_d3[c1.reshape(-1), c2.reshape(-1)], float('-inf'))
    lp3 = F.log_softmax(l3, 1).gather(1, c3.reshape(-1, 1)).view(B, M)
    l4 = model.head4(torch.cat([ctx_f, e1, e2, e3], 1)).float()
    key3 = triplet_key_int(c1.reshape(-1), c2.reshape(-1), c3.reshape(-1))
    l4 = l4.masked_fill(~_gather_d4_mask_grad(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    lp4 = F.log_softmax(l4, 1).gather(1, c4.reshape(-1, 1)).view(B, M)
    return lpJP + lp3 + lp4


def student_list_scores_pd(model, enc_seq, enc_mask, cand_cb, masks, pd_depth=None):
    """s24 wave-3: deployed-equivalent scores for the PD (prefix-conditioned context) variant.
    Mirrors student_beam_pd: per-level context ctx_i = attn(q0 + pproj_i(prefix emb)).
    s23 refuted PD under GT-prefix digit-KD — the same wrong-target confound as the s12 cap;
    this scorer lets list-KL supervise PD's per-prefix capacity on beam-relevant prefixes."""
    mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted = masks
    B, M, _ = cand_cb.shape
    c1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    c2 = cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    c3 = cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)
    c4 = cand_cb[:, :, 3].clamp(0, CODEBOOK_SIZE - 1)
    v1 = (c1 + DIGIT_OFFSETS[0]).reshape(-1)
    v2 = (c2 + DIGIT_OFFSETS[1]).reshape(-1)
    v3 = (c3 + DIGIT_OFFSETS[2]).reshape(-1)

    pd_depth = int(pd_depth if pd_depth is not None else getattr(model, "pd_depth", 4))
    q0, k, v = model.q0_kv(enc_seq, enc_mask)
    ctx1 = model.ctx_from_q(q0, k, v, enc_mask)
    lp1_all = F.log_softmax(model.head1(ctx1).float(), dim=-1)
    lp1 = lp1_all.gather(1, c1)

    # expand kv/pad per candidate for the prefix-conditioned levels
    Bm = B * M
    nh, hd = k.shape[1], k.shape[3]
    S = k.shape[2]
    k_e = k.unsqueeze(1).expand(B, M, nh, S, hd).reshape(Bm, nh, S, hd)
    v_e = v.unsqueeze(1).expand(B, M, nh, S, hd).reshape(Bm, nh, S, hd)
    pad_e = enc_mask.unsqueeze(1).expand(B, M, S).reshape(Bm, S)
    q0_e = q0.unsqueeze(1).expand(B, M, -1).reshape(Bm, -1)
    ctx_f = ctx1.unsqueeze(1).expand(B, M, -1).reshape(Bm, -1)

    e1 = model.embedding(v1)
    e2 = model.embedding(v2)
    e3 = model.embedding(v3)

    ctx2 = model.prefix_ctx(q0_e, ctx_f, k_e, v_e, pad_e, model.pproj2(e1)) if pd_depth >= 2 else ctx_f
    l2 = model.head2(torch.cat([ctx2, e1], 1)).float()
    l2 = l2.masked_fill(~mask_d1_d2[c1.reshape(-1)], float('-inf'))
    lp2 = F.log_softmax(l2, 1).gather(1, c2.reshape(-1, 1)).view(B, M)

    ctx3 = model.prefix_ctx(q0_e, ctx_f, k_e, v_e, pad_e, model.pproj3(torch.cat([e1, e2], 1))) if pd_depth >= 3 else ctx_f
    l3 = model.head3(torch.cat([ctx3, e1, e2], 1)).float()
    l3 = l3.masked_fill(~mask_d1d2_d3[c1.reshape(-1), c2.reshape(-1)], float('-inf'))
    lp3 = F.log_softmax(l3, 1).gather(1, c3.reshape(-1, 1)).view(B, M)

    ctx4 = model.prefix_ctx(q0_e, ctx_f, k_e, v_e, pad_e, model.pproj4(torch.cat([e1, e2, e3], 1))) if pd_depth >= 4 else ctx_f
    l4 = model.head4(torch.cat([ctx4, e1, e2, e3], 1)).float()
    key3 = triplet_key_int(c1.reshape(-1), c2.reshape(-1), c3.reshape(-1))
    l4 = l4.masked_fill(~_gather_d4_mask_grad(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    lp4 = F.log_softmax(l4, 1).gather(1, c4.reshape(-1, 1)).view(B, M)
    return lp1 + lp2 + lp3 + lp4


def boundary_neg_indices(cand_cb, valid_mask, n_pos=10, n_neg_hi=40, n_neg_sample=0, seed=0):
    """Deterministically select outsider columns for a boundary-pair loss.

    The selection is a stable hash of the candidate SID, its teacher-rank column,
    and the recorded run seed. Invalid padded rows sort last. If a request has
    fewer than ``n_neg_sample`` valid outsiders, the returned tensor still has a
    fixed width; downstream valid masks remove the padded selections.
    """
    hi = min(int(n_neg_hi), int(cand_cb.shape[1]))
    lo = min(int(n_pos), hi)
    width = hi - lo
    sample = int(n_neg_sample)
    if sample <= 0:
        return None
    if sample >= width:
        return torch.arange(width, device=cand_cb.device).view(1, width).expand(cand_cb.shape[0], -1)
    neg_cb = cand_cb[:, lo:hi].long()
    neg_valid = valid_mask[:, lo:hi]
    code_hash = (((neg_cb[:, :, 0] * 257 + neg_cb[:, :, 1]) * 257
                  + neg_cb[:, :, 2]) * 257 + neg_cb[:, :, 3])
    rank = torch.arange(width, device=cand_cb.device, dtype=torch.long).view(1, width)
    priority = torch.remainder(
        code_hash * 1103515245 + (rank + 1) * 12345 + int(seed) * 2654435761,
        2147483647,
    )
    priority = priority.masked_fill(~neg_valid, torch.iinfo(torch.long).max)
    return priority.topk(sample, dim=1, largest=False).indices


def boundary_pair_loss(s_S, valid_mask, n_pos=10, n_neg_hi=40, gamma=0.05,
                       n_neg_sample=0, neg_idx=None):
    """s24 wave-4: rank-10 boundary pair loss. Dump rows are teacher-exact-sorted, so
    positives = rows 0..n_pos-1 (the teacher's served top-10), negatives = the near-boundary
    rows n_pos..n_neg_hi-1 (teacher ranks 11-40 — where the observed displacements live).
    gamma ~ the teacher's boundary margin (~0.035-0.05 nats): softplus((s- - s+)/gamma)."""
    import torch.nn.functional as _F
    sp = s_S[:, :n_pos]
    sn = s_S[:, n_pos:n_neg_hi]
    vp = valid_mask[:, :n_pos]
    vn = valid_mask[:, n_pos:n_neg_hi]
    if neg_idx is not None:
        if neg_idx.ndim != 2 or neg_idx.shape[0] != sn.shape[0]:
            raise ValueError("neg_idx must have shape (batch, sampled_negatives)")
        sn = sn.gather(1, neg_idx)
        vn = vn.gather(1, neg_idx)
    elif n_neg_sample and n_neg_sample < sn.shape[1]:
        # s48 (doc 2.3): hold the number of sampled negatives per user FIXED across B settings, so a
        # B sweep varies the candidate RANGE only -- not the pair count / total loss strength.
        # Random subset without replacement; invalid rows are pushed last so they are picked only
        # when a user genuinely has fewer than n_neg_sample valid outsiders (vn keeps masking them).
        _r = torch.rand(sn.shape, device=sn.device, dtype=torch.float32).masked_fill(~vn, 2.0)
        _idx = _r.topk(n_neg_sample, dim=1, largest=False).indices
        sn = sn.gather(1, _idx); vn = vn.gather(1, _idx)
    diff = (sn.unsqueeze(1) - sp.unsqueeze(2)) / gamma      # (B, n_pos, n_neg)
    pm = (vp.unsqueeze(2) & vn.unsqueeze(1)).float()
    loss = _F.softplus(diff) * pm
    return loss.sum() / pm.sum().clamp_min(1.0)


# s34: popularity-stratified list-KL (UnKD-style control). KL within popularity
# groups only — removes cross-group (popularity-axis) comparisons from the target.
def list_kl_group_loss(s_student, s_teacher, valid_mask, group_ids, n_groups=3, temperature=1.0):
    tot = None; cnt = 0
    for g in range(n_groups):
        m = valid_mask & (group_ids == g)
        rows = m.sum(-1) >= 2
        if not rows.any():
            continue
        sS = (s_student / temperature).masked_fill(~m, -1e9)[rows]
        sT = (s_teacher / temperature).masked_fill(~m, -1e9)[rows]
        log_qS = F.log_softmax(sS, dim=-1)
        log_pT = F.log_softmax(sT, dim=-1)
        kl = (log_pT.exp() * (log_pT - log_qS)).sum(-1).mean()
        tot = kl if tot is None else tot + kl
        cnt += 1
    if tot is None:
        return s_student.sum() * 0.0
    return tot / cnt * (temperature ** 2)


def student_list_scores_steps(model, enc_seq, enc_mask, cand_cb, masks):
    """s41 CDRS: per-DIGIT deployed-equivalent log-probs (B, M, 4). Body identical to
    student_list_scores; stack([lp1,lp2,lp3,lp4],-1).sum(-1) == student_list_scores() exactly."""
    mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted = masks
    B, M, _ = cand_cb.shape
    c1 = cand_cb[:, :, 0].clamp(0, CODEBOOK_SIZE - 1)
    c2 = cand_cb[:, :, 1].clamp(0, CODEBOOK_SIZE - 1)
    c3 = cand_cb[:, :, 2].clamp(0, CODEBOOK_SIZE - 1)
    c4 = cand_cb[:, :, 3].clamp(0, CODEBOOK_SIZE - 1)
    v1 = (c1 + DIGIT_OFFSETS[0]).reshape(-1)
    v2 = (c2 + DIGIT_OFFSETS[1]).reshape(-1)
    v3 = (c3 + DIGIT_OFFSETS[2]).reshape(-1)
    ctx = model.encode(enc_seq, enc_mask)
    lp1_all = F.log_softmax(model.head1(ctx).float(), dim=-1)
    lp1 = lp1_all.gather(1, c1)
    D = ctx.shape[1]
    ctx_f = ctx.unsqueeze(1).expand(-1, M, -1).reshape(B * M, D)
    e1 = model.embedding(v1); e2 = model.embedding(v2); e3 = model.embedding(v3)
    l2 = model.head2(torch.cat([ctx_f, e1], 1)).float()
    l2 = l2.masked_fill(~mask_d1_d2[c1.reshape(-1)], float('-inf'))
    lp2 = F.log_softmax(l2, 1).gather(1, c2.reshape(-1, 1)).view(B, M)
    l3 = model.head3(torch.cat([ctx_f, e1, e2], 1)).float()
    l3 = l3.masked_fill(~mask_d1d2_d3[c1.reshape(-1), c2.reshape(-1)], float('-inf'))
    lp3 = F.log_softmax(l3, 1).gather(1, c3.reshape(-1, 1)).view(B, M)
    l4 = model.head4(torch.cat([ctx_f, e1, e2, e3], 1)).float()
    key3 = triplet_key_int(c1.reshape(-1), c2.reshape(-1), c3.reshape(-1))
    l4 = l4.masked_fill(~_gather_d4_mask_grad(d4_keys_sorted, d4_masks_sorted, key3), float('-inf'))
    lp4 = F.log_softmax(l4, 1).gather(1, c4.reshape(-1, 1)).view(B, M)
    return torch.stack([lp1, lp2, lp3, lp4], dim=-1)  # (B, M, 4)


def cross_depth_relative_score_loss(t_steplp, s_steplp, cand_cb, valid_mask, score_scale,
                                    tau=1.0, n_pos=10, n_neg_hi=40):
    """s41 CDRS (delivery doc §3): align teacher & student PER-DEPTH cumulative score gaps on the
    rank-10 boundary pairs (pos rows 0..n_pos-1 x neg rows n_pos..n_neg_hi-1 — the paironly structure).
    Uniform depth weight AFTER first divergence (w=0 before); per-depth teacher-gap-std scale + global tau.
    KL(Bern[p_T]||Bern[p_S]) via BCE-with-logits(student_gap_scaled, teacher_prob)."""
    B, M, Dd = s_steplp.shape
    t_cum = t_steplp.float().cumsum(-1)                                   # (B,M,D)
    s_cum = s_steplp.float().cumsum(-1)
    tp = t_cum[:, :n_pos]; tn = t_cum[:, n_pos:n_neg_hi]                  # (B,np,D),(B,neg,D)
    sp = s_cum[:, :n_pos]; sn = s_cum[:, n_pos:n_neg_hi]
    t_gap = tp.unsqueeze(2) - tn.unsqueeze(1)                            # (B,np,neg,D)
    s_gap = sp.unsqueeze(2) - sn.unsqueeze(1)
    ci = cand_cb[:, :n_pos].unsqueeze(2); cj = cand_cb[:, n_pos:n_neg_hi].unsqueeze(1)
    diff = (ci != cj)                                                     # (B,np,neg,D)
    has_div = diff.any(-1)
    first_div = diff.to(torch.float32).argmax(-1)                        # first divergent depth
    depth = torch.arange(Dd, device=s_steplp.device).view(1, 1, 1, Dd)
    vp = valid_mask[:, :n_pos].unsqueeze(2); vn = valid_mask[:, n_pos:n_neg_hi].unsqueeze(1)
    active = (depth >= first_div.unsqueeze(-1)) & has_div.unsqueeze(-1) & (vp & vn).unsqueeze(-1)
    active = active.float()
    scale = score_scale.to(s_gap.device).view(1, 1, 1, Dd).clamp_min(1e-4)
    t_target = torch.sigmoid((t_gap / scale / tau).detach())
    s_logits = s_gap / scale / tau
    raw = F.binary_cross_entropy_with_logits(s_logits, t_target, reduction='none')
    return (raw * active).sum() / active.sum().clamp_min(1.0)


def serve_kl_loss(s_student, s_teacher, cand_cb, valid_mask, temperature=1.0,
                  w1=1.0, w2=1.0, w3=1.0, w4=1.0):
    """s43e E1: unified hierarchical serving-KL over the code-prefix tree. ONE loss = weighted sum
    of teacher/student KL at each prefix depth d=1..4, ALL taken from the SAME softmax over the M
    candidates (teacher total score s_teacher=bm_sc, student total s_S) -> needs only total scores,
    no per-step logp. The d4 (full-item) term IS list_kl_loss; d1/d2 == prefix_kl_loss; d3 is the
    depth the current ServeSID recipe leaves unsupervised. Uniform w -> the collaborator's single
    formula, no separate list/prefix ablation; w4>1 -> the 'ServeSID + d3' variant."""
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
    return (w1 * kl1 + w2 * kl2 + w3 * kl3 + w4 * kl4).mean() * (temperature ** 2)


def da_cdrs_loss(t_steplp, s_steplp, cand_cb, valid_mask, step_scale,
                 gamma=0.05, huber_delta=1.0, counterfactual=True, n_pos=10,
                 n_neg_hi=40, neg_idx=None):
    """s43e DA-CDRS: decision-attributed cross-depth INCREMENTAL score matching, added on top of the
    plain boundary-pair anchor (boundary_pair_loss, wired separately via pair_w). Per teacher rank-10
    boundary pair (pos rows 0..n_pos-1 x neg n_pos..n_neg_hi-1) and per shallow depth d<D: push the
    student per-depth increment gap dG_S^d toward the teacher's dG_T^d (Huber, per-depth increment-
    std-scaled), weighting depth d by its COUNTERFACTUAL decision impact
        A_d = [l_pair(G_S) - l_pair(G_S - dG_S^d + dG_T^d)]_+   (stop-grad, normalized over d<D),
    l_pair(G) = softplus(-G/gamma) == the anchor hinge. Fixes the s41 CDRS failure modes: INCREMENTAL
    (not cumulative) gaps -> no double-supervision; the pair loss is the anchor (not replaced); the
    attribution concentrates the correction on the depth(s) that actually flip THIS boundary.
    counterfactual=False -> uniform depth weights (the control that isolates the attribution)."""
    B, M, Dd = s_steplp.shape
    sp = s_steplp[:, :n_pos]
    sn_all = s_steplp[:, n_pos:n_neg_hi]
    tp = t_steplp[:, :n_pos].float()
    tn_all = t_steplp[:, n_pos:n_neg_hi].float()
    vp = valid_mask[:, :n_pos]
    vn_all = valid_mask[:, n_pos:n_neg_hi]
    if neg_idx is not None:
        if neg_idx.ndim != 2 or neg_idx.shape[0] != sn_all.shape[0]:
            raise ValueError("neg_idx must have shape (batch, sampled_negatives)")
        idx3 = neg_idx.unsqueeze(-1).expand(-1, -1, s_steplp.shape[-1])
        sn = sn_all.gather(1, idx3)
        tn = tn_all.gather(1, idx3)
        vn = vn_all.gather(1, neg_idx)
    else:
        sn, tn, vn = sn_all, tn_all, vn_all
    dG_S = sp.unsqueeze(2) - sn.unsqueeze(1)              # (B,np,neg,D) student per-depth incr gap
    dG_T = tp.unsqueeze(2) - tn.unsqueeze(1)             # teacher per-depth incr gap
    pm = (vp.unsqueeze(2) & vn.unsqueeze(1)).float()     # (B,np,neg) active pair
    Dc = Dd - 1                                          # corrected shallow depths d<D
    if counterfactual:
        G_S = dG_S.sum(-1)                               # (B,np,neg) total student gap
        G_tilde = G_S.unsqueeze(-1) - dG_S[..., :Dc] + dG_T[..., :Dc]   # replace depth-d increment
        A = F.relu(F.softplus(-G_S / gamma).unsqueeze(-1) - F.softplus(-G_tilde / gamma))
        w = (A / A.sum(-1, keepdim=True).clamp_min(1e-9)).detach()      # normalized over d<D, stop-grad
    else:
        w = torch.full((B, dG_S.shape[1], dG_S.shape[2], Dc), 1.0 / Dc,
                       device=s_steplp.device, dtype=dG_S.dtype)
    scale = step_scale.to(s_steplp.device).float().clamp_min(1e-4)[:Dc].view(1, 1, 1, Dc)
    resid = (dG_S[..., :Dc] - dG_T[..., :Dc]) / scale
    huber = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=huber_delta, reduction='none')
    term = (w * huber).sum(-1)                           # (B,np,neg)
    return (term * pm).sum() / pm.sum().clamp_min(1.0)
