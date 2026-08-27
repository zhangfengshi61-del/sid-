# s43e patch: additive + gated wiring of (E1) the unified hierarchical serving-KL and
# (DA-CDRS / E3) decision-attributed cross-depth INCREMENTAL score matching into the SID-MLP
# distill trainer. Layered ON TOP of the s41 CDRS patch (which must already be applied).
# Backs up each file to *.bak_s43e, idempotent (skips if already patched). Every edit is gated on
# serve_kl>0 / da_w>0 so all existing arms (r0base/lk4/lkpf/paironly/cdrs/...) stay bit-identical.
import io, os
R = os.environ.get(
    "SIDMLP_ROOT",
    os.path.join(os.environ.get("SERVESID_ROOT", "."), "repos", "SID-MLP"),
)


def edit(path, checks, repls, append=None, marker='s43e'):
    with io.open(path, 'r') as f:
        src = f.read()
    if marker in src:
        print(f'SKIP (already patched): {path}'); return
    bak = path + '.bak_s43e'
    if not os.path.exists(bak):
        with io.open(bak, 'w') as f: f.write(src)
    for anchor in checks:
        n = src.count(anchor)
        assert n == 1, f'anchor count {n} (want 1) in {path}: {anchor[:70]!r}'
    for old, new in repls:
        src = src.replace(old, new, 1)
    if append is not None:
        src = src.rstrip('\n') + '\n\n' + append
    with io.open(path, 'w') as f: f.write(src)
    print(f'PATCHED: {path}')


# ---------------- A. losses.py: append serve_kl_loss (E1) + da_cdrs_loss (DA-CDRS) ----------------
LOSSES_APPEND = r'''
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
                 gamma=0.05, huber_delta=1.0, counterfactual=True, n_pos=10, n_neg_hi=40):
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
    sp = s_steplp[:, :n_pos]; sn = s_steplp[:, n_pos:n_neg_hi]
    tp = t_steplp[:, :n_pos].float(); tn = t_steplp[:, n_pos:n_neg_hi].float()
    dG_S = sp.unsqueeze(2) - sn.unsqueeze(1)              # (B,np,neg,D) student per-depth incr gap
    dG_T = tp.unsqueeze(2) - tn.unsqueeze(1)             # teacher per-depth incr gap
    vp = valid_mask[:, :n_pos]; vn = valid_mask[:, n_pos:n_neg_hi]
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
'''
edit(os.path.join(R, 'sid_mlp/losses.py'), [], [], append=LOSSES_APPEND)

# ---------------- B. dataset.py: bm_step_scale (incremental-gap std) ----------------
edit(os.path.join(R, 'sid_mlp/dataset.py'),
     ["        self.bm_score_scale = None  # s41 CDRS: (4,) per-depth teacher-gap std",
      "    dataset.bm_score_scale = d['score_scale'].to(device) if 'score_scale' in d else None  # s41 CDRS"],
     [("        self.bm_score_scale = None  # s41 CDRS: (4,) per-depth teacher-gap std",
       "        self.bm_score_scale = None  # s41 CDRS: (4,) per-depth teacher-gap std\n"
       "        self.bm_step_scale = None    # s43e DA-CDRS: (4,) per-depth teacher INCREMENT-gap std"),
      ("    dataset.bm_score_scale = d['score_scale'].to(device) if 'score_scale' in d else None  # s41 CDRS",
       "    dataset.bm_score_scale = d['score_scale'].to(device) if 'score_scale' in d else None  # s41 CDRS\n"
       "    dataset.bm_step_scale = d['step_scale'].to(device) if 'step_scale' in d else None  # s43e DA-CDRS")])

# ---------------- C. cli/train.py: flags ----------------
edit(os.path.join(R, 'sid_mlp/cli/train.py'),
     ['    parser.add_argument("--cdrs_neg_end", type=int, default=40)     # s41 CDRS'],
     [('    parser.add_argument("--cdrs_neg_end", type=int, default=40)     # s41 CDRS',
       '    parser.add_argument("--cdrs_neg_end", type=int, default=40)     # s41 CDRS\n'
       '    parser.add_argument("--serve_kl", type=float, default=0.0)      # s43e E1 unified serving-KL\n'
       '    parser.add_argument("--serve_w1", type=float, default=1.0)      # s43e E1 depth weights\n'
       '    parser.add_argument("--serve_w2", type=float, default=1.0)\n'
       '    parser.add_argument("--serve_w3", type=float, default=1.0)\n'
       '    parser.add_argument("--serve_w4", type=float, default=1.0)\n'
       '    parser.add_argument("--da_w", type=float, default=0.0)          # s43e DA-CDRS correction weight\n'
       '    parser.add_argument("--da_mode", type=str, default="cf")        # s43e cf|uniform\n'
       '    parser.add_argument("--da_gamma", type=float, default=0.05)     # s43e counterfactual pair-hinge gamma\n'
       '    parser.add_argument("--da_huber", type=float, default=1.0)      # s43e Huber delta')])

# ---------------- D. train_decoder.py: import + lam + gate + steps-elif + branches + scale + need_masks ----
edit(os.path.join(R, 'sid_mlp/train_decoder.py'),
     ["    student_list_scores_steps,\n    cross_depth_relative_score_loss,",
      '    lam_cdrs = float(getattr(args, "cdrs_w", 0.0) or 0.0)  # s41 CDRS',
      "    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0 or lam_cdrs > 0):",
      "        elif lam_cdrs > 0:  # s41: per-digit scorer; sum == student_list_scores, reused for list/pair",
      '            loss = loss + lam_cdrs * l_cdrs\n            comps["cdrs"] = float(l_cdrs.item())\n        with torch.no_grad():',
      "        if float(getattr(args, 'cdrs_w', 0.0) or 0.0) > 0:  # s41 CDRS\n"
      "            args._cdrs_scale = getattr(train_ds, 'bm_score_scale', None)\n"
      "            if args._cdrs_scale is None:\n"
      "                raise SystemExit('--cdrs_w requires an augmented beam_dump (step_logp+score_scale)')",
      '        or float(getattr(args, "cdrs_w", 0.0) or 0.0) > 0))'],
     [("    student_list_scores_steps,\n    cross_depth_relative_score_loss,",
       "    student_list_scores_steps,\n    cross_depth_relative_score_loss,\n    serve_kl_loss,\n    da_cdrs_loss,"),
      ('    lam_cdrs = float(getattr(args, "cdrs_w", 0.0) or 0.0)  # s41 CDRS',
       '    lam_cdrs = float(getattr(args, "cdrs_w", 0.0) or 0.0)  # s41 CDRS\n'
       '    lam_serve = float(getattr(args, "serve_kl", 0.0) or 0.0)  # s43e E1\n'
       '    lam_da = float(getattr(args, "da_w", 0.0) or 0.0)         # s43e DA-CDRS'),
      ("    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0 or lam_cdrs > 0):",
       "    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0 or lam_cdrs > 0 or lam_serve > 0 or lam_da > 0):"),
      ("        elif lam_cdrs > 0:  # s41: per-digit scorer; sum == student_list_scores, reused for list/pair",
       "        elif lam_cdrs > 0 or lam_da > 0:  # s41/s43e: per-digit scorer; sum == student_list_scores"),
      ('            loss = loss + lam_cdrs * l_cdrs\n            comps["cdrs"] = float(l_cdrs.item())\n        with torch.no_grad():',
       '            loss = loss + lam_cdrs * l_cdrs\n            comps["cdrs"] = float(l_cdrs.item())\n'
       '        if lam_serve > 0:  # s43e E1 unified serving-KL\n'
       '            l_serve = serve_kl_loss(s_S, bm_sc, bm_cb, valid, temperature=args.list_temp,\n'
       '                                    w1=float(getattr(args, "serve_w1", 1.0)), w2=float(getattr(args, "serve_w2", 1.0)),\n'
       '                                    w3=float(getattr(args, "serve_w3", 1.0)), w4=float(getattr(args, "serve_w4", 1.0)))\n'
       '            loss = loss + lam_serve * l_serve\n'
       '            comps["serve"] = float(l_serve.item())\n'
       '        if lam_da > 0:  # s43e DA-CDRS correction (pair anchor added separately via pair_w)\n'
       '            _dascale = getattr(args, "_da_scale", None)\n'
       '            if _slp is None or bm_slp is None or _dascale is None:\n'
       '                raise SystemExit("--da_w requires an augmented dump with step_logp + step_scale (run add_step_scale_s43e)")\n'
       '            l_da = da_cdrs_loss(bm_slp, _slp, bm_cb, valid, _dascale,\n'
       '                                gamma=float(getattr(args, "da_gamma", 0.05)), huber_delta=float(getattr(args, "da_huber", 1.0)),\n'
       '                                counterfactual=(str(getattr(args, "da_mode", "cf")) != "uniform"),\n'
       '                                n_pos=int(getattr(args, "cdrs_pos_k", 10)), n_neg_hi=int(getattr(args, "cdrs_neg_end", 40)))\n'
       '            loss = loss + lam_da * l_da\n'
       '            comps["da"] = float(l_da.item())\n'
       '        with torch.no_grad():'),
      ("        if float(getattr(args, 'cdrs_w', 0.0) or 0.0) > 0:  # s41 CDRS\n"
       "            args._cdrs_scale = getattr(train_ds, 'bm_score_scale', None)\n"
       "            if args._cdrs_scale is None:\n"
       "                raise SystemExit('--cdrs_w requires an augmented beam_dump (step_logp+score_scale)')",
       "        if float(getattr(args, 'cdrs_w', 0.0) or 0.0) > 0:  # s41 CDRS\n"
       "            args._cdrs_scale = getattr(train_ds, 'bm_score_scale', None)\n"
       "            if args._cdrs_scale is None:\n"
       "                raise SystemExit('--cdrs_w requires an augmented beam_dump (step_logp+score_scale)')\n"
       "        if float(getattr(args, 'da_w', 0.0) or 0.0) > 0:  # s43e DA-CDRS\n"
       "            args._da_scale = getattr(train_ds, 'bm_step_scale', None)\n"
       "            if args._da_scale is None:\n"
       "                raise SystemExit('--da_w requires an augmented dump with step_scale (run add_step_scale_s43e)')"),
      ('        or float(getattr(args, "cdrs_w", 0.0) or 0.0) > 0))',
       '        or float(getattr(args, "cdrs_w", 0.0) or 0.0) > 0\n'
       '        or float(getattr(args, "serve_kl", 0.0) or 0.0) > 0\n'
       '        or float(getattr(args, "da_w", 0.0) or 0.0) > 0))  # s43e')])

# ---------------- E. scripts/train.sh: env passthrough ----------------
edit(os.path.join(R, 'scripts/train.sh'),
     ['        --cdrs_neg_end "${CDRS_NEG_END:-40}" \\'],
     [('        --cdrs_neg_end "${CDRS_NEG_END:-40}" \\',
       '        --cdrs_neg_end "${CDRS_NEG_END:-40}" \\\n'
       '        --serve_kl "${SERVE_KL:-0}" \\\n'
       '        --serve_w1 "${SERVE_W1:-1.0}" \\\n'
       '        --serve_w2 "${SERVE_W2:-1.0}" \\\n'
       '        --serve_w3 "${SERVE_W3:-1.0}" \\\n'
       '        --serve_w4 "${SERVE_W4:-1.0}" \\\n'
       '        --da_w "${DA_W:-0}" \\\n'
       '        --da_mode "${DA_MODE:-cf}" \\\n'
       '        --da_gamma "${DA_GAMMA:-0.05}" \\\n'
       '        --da_huber "${DA_HUBER:-1.0}" \\')])

print('ALL s43e PATCHES DONE')
