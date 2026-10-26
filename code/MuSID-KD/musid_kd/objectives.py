"""MuSID-KD objectives (WWW 2027): HCDA and RIGD.

The candidate support is the same teacher-ranked list for both networks.
Scores exclude EOS and contain only the L SID-token log probabilities.
All these objectives run during training; no serving-time reranker is used.
"""
import torch
import torch.nn.functional as F


def hcda_loss(student_scores, teacher_scores, codes, valid_mask, temperature=1.0, eps=1e-9):
    """Sum of depth-wise prefix-marginal KL (paper Eqs. 4-5, 9).

    Item distributions are temperature-softmaxes of complete SID-path scores;
    each prefix marginal sums probabilities of candidate items sharing that prefix.
    No conventional KL temperature-square multiplier appears in the paper.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if codes.ndim != 3 or student_scores.shape != codes.shape[:2]:
        raise ValueError("expected scores [batch,candidates], codes [batch,candidates,depth]")
    if teacher_scores.shape != student_scores.shape or valid_mask.shape != student_scores.shape:
        raise ValueError("score/valid_mask shapes differ")
    mask = valid_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("each request needs at least one valid candidate")
    st = (student_scores.float() / temperature).masked_fill(~mask, -1e9)
    tt = (teacher_scores.float().detach() / temperature).masked_fill(~mask, -1e9)
    ps = F.softmax(st, -1) * mask
    pt = F.softmax(tt, -1) * mask
    total = student_scores.new_zeros((), dtype=torch.float32)
    for depth in range(1, codes.shape[-1] + 1):
        equal_prefix = (codes[:, :, None, :depth] == codes[:, None, :, :depth]).all(-1)
        equal_prefix = equal_prefix & mask[:, :, None] & mask[:, None, :]
        eqf = equal_prefix.to(pt.dtype)
        p_t = torch.bmm(eqf, pt.unsqueeze(-1)).squeeze(-1)
        p_s = torch.bmm(eqf, ps.unsqueeze(-1)).squeeze(-1)
        count = eqf.sum(-1).clamp_min(1)
        term = (p_t / count) * (p_t.clamp_min(eps).log() - p_s.clamp_min(eps).log())
        total = total + (term * mask).sum(-1).mean()
    return total


def _eligible_pairs(codes, valid_mask, top_k, negative_end):
    """Teacher rank [1,K] vs [K+1,B], depths first divergence through L-1."""
    batch, width, levels = codes.shape
    if not 0 < top_k < negative_end <= width:
        raise ValueError("require 0 < K < B <= candidate width")
    pos = codes[:, :top_k, :]
    neg = codes[:, top_k:negative_end, :]
    unequal = pos.unsqueeze(2) != neg.unsqueeze(1)
    first = unequal.long().argmax(-1)
    # If divergence occurs only at the last level, no RIGD-supervised level exists.
    has_eligible = unequal[..., :-1].any(-1)
    pair_valid = (valid_mask[:, :top_k, None] &
                  valid_mask[:, None, top_k:negative_end] & has_eligible)
    depth = torch.arange(levels, device=codes.device).view(1, 1, 1, levels)
    active = ((depth >= first.unsqueeze(-1)) & (depth < levels - 1)
              & pair_valid.unsqueeze(-1))
    return pair_valid, active


def rigd_loss(teacher_steps, student_steps, codes, valid_mask, scales,
              top_k=10, negative_end=40, margin=0.05, eps=1e-9, huber_beta=1.0):
    """Paper RIGD: positive replacement-gain weights and scaled Huber (Eqs. 10-16).

    Full pairwise gaps include the final SID position. RIGD supervises eligible
    depths only through L-1; zero-gain pairs still count in the denominator.
    Depth-weight calculation is stop-gradient.
    """
    if teacher_steps.shape != student_steps.shape or codes.shape != student_steps.shape:
        raise ValueError("teacher_steps, student_steps and codes require [B,M,L]")
    mask = valid_mask.bool()
    pair_valid, active = _eligible_pairs(codes, mask, top_k, negative_end)
    teacher = teacher_steps.float().detach()
    student = student_steps.float()
    g_t = teacher[:, :top_k].unsqueeze(2) - teacher[:, top_k:negative_end].unsqueeze(1)
    g_s = student[:, :top_k].unsqueeze(2) - student[:, top_k:negative_end].unsqueeze(1)
    global_gap = g_s.sum(-1, keepdim=True)
    with torch.no_grad():
        before = F.softplus(margin - global_gap)
        after = F.softplus(margin - (global_gap - g_s + g_t))
        gain = (before - after).clamp_min(0.0) * active
        weight = gain / gain.sum(-1, keepdim=True).clamp_min(eps)
    scale = torch.as_tensor(scales, dtype=student.dtype, device=student.device)
    if scale.ndim != 1 or scale.numel() != codes.shape[-1]:
        raise ValueError("scales must have one entry per SID position")
    normalized_delta = (g_s - g_t) / scale.clamp_min(1e-4).view(1, 1, 1, -1)
    per_level = F.smooth_l1_loss(
        normalized_delta, torch.zeros_like(normalized_delta),
        reduction="none", beta=huber_beta,
    )
    return ((weight * per_level).sum(-1) * pair_valid).sum() / pair_valid.sum().clamp_min(1)


@torch.no_grad()
def estimate_rigd_scales(codes, teacher_steps, nvalid, top_k=10,
                         negative_end=40, chunk_size=1024):
    """Training-only standard deviations of teacher pairwise increments."""
    if codes.shape != teacher_steps.shape:
        raise ValueError("codes and teacher_steps require matching [N,M,L]")
    levels = codes.shape[-1]
    sums = torch.zeros(levels, dtype=torch.float64, device=codes.device)
    squares = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    for lo in range(0, codes.shape[0], chunk_size):
        hi = min(codes.shape[0], lo + chunk_size)
        cb, step = codes[lo:hi], teacher_steps[lo:hi].float()
        mask = torch.arange(cb.shape[1], device=cb.device)[None, :] < nvalid[lo:hi, None]
        _, active = _eligible_pairs(cb, mask, top_k, negative_end)
        gap = step[:, :top_k].unsqueeze(2) - step[:, top_k:negative_end].unsqueeze(1)
        a = active.to(gap.dtype)
        sums += (gap * a).sum((0, 1, 2)).double()
        squares += (gap.square() * a).sum((0, 1, 2)).double()
        counts += active.sum((0, 1, 2)).double()
    mean = sums / counts.clamp_min(1)
    var = (squares / counts.clamp_min(1) - mean.square()).clamp_min(0)
    return var.sqrt().clamp_min(1e-4).float()
