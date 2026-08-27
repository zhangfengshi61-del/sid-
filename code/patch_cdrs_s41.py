# s41 CDRS patch: additive + gated wiring of Cross-Depth Relative-Score Distillation into the
# SID-MLP distill trainer. Backs up each file to *.bak_cdrs_s41, idempotent (skips if already patched).
# All edits are gated on cdrs_w>0 so the existing arms (r0base/lk4/lkpf/paironly/...) are bit-identical.
import io, os, sys
R = os.environ.get(
    "SIDMLP_ROOT",
    os.path.join(os.environ.get("SERVESID_ROOT", "."), "repos", "SID-MLP"),
)

def edit(path, checks, repls, append=None):
    with io.open(path, 'r') as f:
        src = f.read()
    tag = 's41 CDRS' if append is None else 'cross_depth_relative_score_loss'
    if 'cdrs' in src.lower() and (tag in src or (append and 'cross_depth_relative_score_loss' in src)):
        print(f'SKIP (already patched): {path}'); return
    bak = path + '.bak_cdrs_s41'
    if not os.path.exists(bak):
        with io.open(bak, 'w') as f: f.write(src)
    for anchor in checks:
        n = src.count(anchor)
        assert n == 1, f'anchor count {n} (want 1) in {path}: {anchor[:60]!r}'
    for old, new in repls:
        src = src.replace(old, new, 1)
    if append is not None:
        src = src.rstrip('\n') + '\n\n' + append
    with io.open(path, 'w') as f: f.write(src)
    print(f'PATCHED: {path}')

# ---------------- A. losses.py: append student_list_scores_steps + cross_depth_relative_score_loss ----------------
LOSSES_APPEND = r'''
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
'''
edit(os.path.join(R, 'sid_mlp/losses.py'), [], [], append=LOSSES_APPEND)

# ---------------- B. dataset.py: init + attach + collate ----------------
edit(os.path.join(R, 'sid_mlp/dataset.py'),
     ["        self.bm_nvalid = None   # (N,) int16",
      "    dataset.bm_nvalid = d['nvalid'].to(device)",
      "    return base + (bm_cb, bm_sc, bm_nv)"],
     [("        self.bm_nvalid = None   # (N,) int16",
       "        self.bm_nvalid = None   # (N,) int16\n"
       "        self.bm_steplp = None       # s41 CDRS: (N, M, 4) teacher per-digit logp\n"
       "        self.bm_score_scale = None  # s41 CDRS: (4,) per-depth teacher-gap std"),
      ("    dataset.bm_nvalid = d['nvalid'].to(device)",
       "    dataset.bm_nvalid = d['nvalid'].to(device)\n"
       "    dataset.bm_steplp = d['step_logp'].to(device) if 'step_logp' in d else None  # s41 CDRS\n"
       "    dataset.bm_score_scale = d['score_scale'].to(device) if 'score_scale' in d else None  # s41 CDRS"),
      ("    return base + (bm_cb, bm_sc, bm_nv)",
       "    extras = (bm_cb, bm_sc, bm_nv)\n"
       "    if getattr(dataset, 'bm_steplp', None) is not None:  # s41 CDRS\n"
       "        extras = extras + (dataset.bm_steplp[idx].float(),)\n"
       "    return base + extras")])

# ---------------- C. cli/train.py: flags ----------------
edit(os.path.join(R, 'sid_mlp/cli/train.py'),
     ['    parser.add_argument("--pair_gamma", type=float, default=0.05)  # s24 wave-4'],
     [('    parser.add_argument("--pair_gamma", type=float, default=0.05)  # s24 wave-4',
       '    parser.add_argument("--pair_gamma", type=float, default=0.05)  # s24 wave-4\n'
       '    parser.add_argument("--cdrs_w", type=float, default=0.0)        # s41 CDRS\n'
       '    parser.add_argument("--cdrs_tau", type=float, default=0.0)      # s41 CDRS (0 -> reuse list_temp)\n'
       '    parser.add_argument("--cdrs_pos_k", type=int, default=10)       # s41 CDRS\n'
       '    parser.add_argument("--cdrs_neg_end", type=int, default=40)     # s41 CDRS')])

# ---------------- D. train_decoder.py: import + s_S/steps + cdrs branch + score_scale load ----------------
D_IMPORT_OLD = "    student_list_scores,\n    student_list_scores_jp,"
D_IMPORT_NEW = "    student_list_scores,\n    student_list_scores_steps,\n    cross_depth_relative_score_loss,\n    student_list_scores_jp,"

D_SS_OLD = '''    lam_pair = float(getattr(args, "pair_w", 0.0) or 0.0)
    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0):
        bm_cb, bm_sc, bm_nv = extras
        M = bm_cb.shape[1]
        valid = torch.arange(M, device=bm_cb.device).unsqueeze(0) < bm_nv.unsqueeze(1)
        if getattr(args, "variant", "base") == "pd":
            s_S = student_list_scores_pd(model, enc_seq, enc_mask, bm_cb, list_masks,
                                         pd_depth=int(getattr(args, "pd_depth", 4)))
        else:
            s_S = student_list_scores(model, enc_seq, enc_mask, bm_cb, list_masks)'''
D_SS_NEW = '''    lam_pair = float(getattr(args, "pair_w", 0.0) or 0.0)
    lam_cdrs = float(getattr(args, "cdrs_w", 0.0) or 0.0)  # s41 CDRS
    if extras is not None and (lam_list > 0 or lam_pf > 0 or lam_pair > 0 or lam_cdrs > 0):
        bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
        bm_slp = extras[3] if len(extras) > 3 else None  # s41 CDRS: teacher per-digit logp
        M = bm_cb.shape[1]
        valid = torch.arange(M, device=bm_cb.device).unsqueeze(0) < bm_nv.unsqueeze(1)
        _slp = None
        if getattr(args, "variant", "base") == "pd":
            s_S = student_list_scores_pd(model, enc_seq, enc_mask, bm_cb, list_masks,
                                         pd_depth=int(getattr(args, "pd_depth", 4)))
        elif lam_cdrs > 0:  # s41: per-digit scorer; sum == student_list_scores, reused for list/pair
            _slp = student_list_scores_steps(model, enc_seq, enc_mask, bm_cb, list_masks)
            s_S = _slp.sum(-1)
        else:
            s_S = student_list_scores(model, enc_seq, enc_mask, bm_cb, list_masks)'''

D_PAIR_OLD = '''        if lam_pair > 0:
            l_pair = boundary_pair_loss(s_S, valid, gamma=float(getattr(args, "pair_gamma", 0.05)))
            loss = loss + lam_pair * l_pair
            comps["pair"] = float(l_pair.item())
        with torch.no_grad():
            comps["ltop1"] = float((s_S.masked_fill(~valid, -1e9).argmax(1) == 0).float().mean().item())

    return loss, (logits1, logits2, logits3, logits4), (gd1, gd2, gd3, gd4), comps'''
D_PAIR_NEW = '''        if lam_pair > 0:
            l_pair = boundary_pair_loss(s_S, valid, gamma=float(getattr(args, "pair_gamma", 0.05)))
            loss = loss + lam_pair * l_pair
            comps["pair"] = float(l_pair.item())
        if lam_cdrs > 0:  # s41 CDRS
            _scale = getattr(args, "_cdrs_scale", None)
            if _slp is None or bm_slp is None or _scale is None:
                raise SystemExit("--cdrs_w requires variant=base + a beam_dump augmented with step_logp+score_scale")
            _tau = float(getattr(args, "cdrs_tau", 0.0) or 0.0) or float(getattr(args, "list_temp", 1.0))
            l_cdrs = cross_depth_relative_score_loss(
                bm_slp, _slp, bm_cb, valid, _scale, tau=_tau,
                n_pos=int(getattr(args, "cdrs_pos_k", 10)), n_neg_hi=int(getattr(args, "cdrs_neg_end", 40)))
            loss = loss + lam_cdrs * l_cdrs
            comps["cdrs"] = float(l_cdrs.item())
        with torch.no_grad():
            comps["ltop1"] = float((s_S.masked_fill(~valid, -1e9).argmax(1) == 0).float().mean().item())

    return loss, (logits1, logits2, logits3, logits4), (gd1, gd2, gd3, gd4), comps'''

D_ATTACH_OLD = "        attach_beam_dump(train_ds, args.beam_dump, device, topm=int(getattr(args, 'list_topm', 0) or 0))  # s36"
D_ATTACH_NEW = ("        attach_beam_dump(train_ds, args.beam_dump, device, topm=int(getattr(args, 'list_topm', 0) or 0))  # s36\n"
                "        if float(getattr(args, 'cdrs_w', 0.0) or 0.0) > 0:  # s41 CDRS\n"
                "            args._cdrs_scale = getattr(train_ds, 'bm_score_scale', None)\n"
                "            if args._cdrs_scale is None:\n"
                "                raise SystemExit('--cdrs_w requires an augmented beam_dump (step_logp+score_scale)')")

edit(os.path.join(R, 'sid_mlp/train_decoder.py'),
     [D_IMPORT_OLD, D_SS_OLD, D_PAIR_OLD, D_ATTACH_OLD],
     [(D_IMPORT_OLD, D_IMPORT_NEW), (D_SS_OLD, D_SS_NEW), (D_PAIR_OLD, D_PAIR_NEW), (D_ATTACH_OLD, D_ATTACH_NEW)])

# ---------------- E. scripts/train.sh: CDRS env passthrough ----------------
edit(os.path.join(R, 'scripts/train.sh'),
     ['        --list_topm "${LIST_TOPM:-0}" \\'],
     [('        --list_topm "${LIST_TOPM:-0}" \\',
       '        --list_topm "${LIST_TOPM:-0}" \\\n'
       '        --cdrs_w "${CDRS_W:-0}" \\\n'
       '        --cdrs_tau "${CDRS_TAU:-0}" \\\n'
       '        --cdrs_pos_k "${CDRS_POS_K:-10}" \\\n'
       '        --cdrs_neg_end "${CDRS_NEG_END:-40}" \\')])

print('ALL PATCHES DONE')
