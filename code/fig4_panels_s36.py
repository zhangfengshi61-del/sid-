"""s36: Fig.4b/4c + App B7(b) numbers from the s34 per-user npz (0-GPU, local).

Fig.4b  Prefix Deficit: normalized prefix log-mass deficit (review-approved form; raw
        score subtraction is banned). For candidate c at depth d within user u's shared
        set C_u = dedup({GT} + teacher-top10 + student-top10):
            n{T,S}_c(d) = cum lp_c(1..d) - logsumexp_{C_u} cum lp(1..d)
            deficit_c(d) = nT_c(d) - nS_c(d)   (>0: student under-weights c vs teacher)
        Aggregated over teacher-top-10 members by candidate pop bucket (tail/mid/head).
        Also emits the incremental per-level version (Delta_d - Delta_{d-1}); at d2 this
        IS the pi(d2|d1) term of the review's requested decomposition.
Fig.4c  Head Replacement Rate: among users whose GT is Mid (resp. Tail), the Head share
  of incoming items (student top-10 minus teacher top-10). Alternates emitted alongside.
B7(b)   Boundary swap: pairs (i in t10 & mid/tail, j in s10\t10 & head) where teacher
        orders i>j but student orders j>i (full-depth scores).

Buckets = the run's native train-count tertiles (cold excluded from panels).
"""
import json, sys
import numpy as np

RUNS = "runs"
CATS = ["Musical_Instruments", "Industrial_and_Scientific", "Video_Games"]
ARMS = ["r0base", "lkpf"]  # SID-MLP vs ServeSID
BN = ["cold", "tail", "mid", "head"]

def load(cat, arm):
    try:
        return np.load(f"{RUNS}/bucketdiag_pu_{cat}_{arm}.npz", allow_pickle=True)
    except FileNotFoundError:
        return None

def user_slices(off):
    return [(off[i], off[i + 1]) for i in range(len(off) - 1)]

def dedup_rows(tlp, slp, buck, flags):
    """Dedup identical candidates inside one user's slice (GT padding repeats)."""
    key = np.round(tlp.astype(np.float64), 6).tobytes(), np.round(slp.astype(np.float64), 6).tobytes()
    # row-wise dedup: build keys per row
    seen = {}
    keep = []
    for r in range(tlp.shape[0]):
        k = (tlp[r].tobytes(), slp[r].tobytes(), int(buck[r]))
        if k in seen:
            seen_idx = seen[k]
            # merge flags so t10/s10 membership survives dedup
            flags[seen_idx] |= flags[r]
            continue
        seen[k] = r
        keep.append(r)
    return np.array(keep, dtype=np.int64), flags

def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())

def panels(cat, arm):
    d = load(cat, arm)
    if d is None:
        return None
    off = d["cand_offsets"]
    tlp, slp = d["cand_tlp"].astype(np.float64), d["cand_slp"].astype(np.float64)
    buck, flags = d["cand_bucket"].copy(), d["cand_flags"].copy()
    gtb = d["gt_bucket"]
    dts, dst = d["disp_t_not_s"].astype(np.float64), d["disp_s_not_t"].astype(np.float64)

    # ---- Fig.4b deficit: per (bucket in {1,2,3}, depth 1..4) mean over teacher-top10 members
    sums = np.zeros((4, 4)); cnts = np.zeros((4, 4))
    # ---- B7(b) boundary swap
    swap_pairs = 0; swap_inv = 0
    for (lo, hi) in user_slices(off):
        t, s, b, f = tlp[lo:hi], slp[lo:hi], buck[lo:hi], flags[lo:hi].copy()
        keep, f = dedup_rows(t, s, b, f)
        t, s, b, f = t[keep], s[keep], b[keep], f[keep]
        ct = np.cumsum(t, axis=1); cs = np.cumsum(s, axis=1)
        in_t10 = (f & 2) > 0; in_s10 = (f & 4) > 0
        for dd in range(4):
            nT = ct[:, dd] - logsumexp(ct[:, dd]); nS = cs[:, dd] - logsumexp(cs[:, dd])
            dfc = nT - nS
            for bb in (1, 2, 3):
                m = in_t10 & (b == bb)
                if m.any():
                    sums[bb, dd] += dfc[m].mean(); cnts[bb, dd] += 1
        # boundary swap at full depth
        mi = in_t10 & ((b == 1) | (b == 2))
        hj = in_s10 & (~in_t10) & (b == 3)
        if mi.any() and hj.any():
            Ti, Tj = ct[mi, 3], ct[hj, 3]
            Si, Sj = cs[mi, 3], cs[hj, 3]
            better_t = Ti[:, None] > Tj[None, :]
            worse_s = Si[:, None] < Sj[None, :]
            swap_pairs += int(better_t.sum())
            swap_inv += int((better_t & worse_s).sum())

    heat = {}
    for bb, bn in ((3, "head"), (2, "mid"), (1, "tail")):
        heat[bn] = [round(sums[bb, dd] / max(cnts[bb, dd], 1), 5) for dd in range(4)]
    heat_inc = {bn: [round(v[0], 5)] + [round(v[i] - v[i - 1], 5) for i in range(1, 4)]
                for bn, v in heat.items()}

    # ---- Fig.4c HRR
    out_c = {}
    for qb, qn in ((2, "mid_gt"), (1, "tail_gt")):
        m = gtb == qb
        inc = dst[m]; outg = dts[m]
        inc_tot = inc.sum()
        out_c[qn] = {
            "primary_head_share_incoming": round(float(inc[:, 3].sum() / max(inc_tot.sum(), 1)), 4),
            "alt_ratio_inhead_over_outmidtail": round(float(inc[:, 3].sum() / max(outg[:, 1].sum() + outg[:, 2].sum(), 1)), 4),
            "per_user_mean_head_share": round(float(np.nanmean(
                np.where(inc.sum(1) > 0, inc[:, 3] / np.maximum(inc.sum(1), 1), np.nan))), 4),
            # rate versions (the informative axis: share is at ceiling for every arm;
            # the repair shows in VOLUME of replacement events, not their composition)
            "head_replacements_per_user": round(float(inc[:, 3].sum() / max(m.sum(), 1)), 4),
            "midtail_teacher_losses_per_user": round(float((outg[:, 1].sum() + outg[:, 2].sum()) / max(m.sum(), 1)), 4),
            "n_users": int(m.sum()),
        }
    m = np.ones(len(gtb), dtype=bool)
    out_c["all_users_altc"] = {
        "inhead_over_out_midtail": round(float(dst[:, 3].sum() / max(dts[:, 1].sum() + dts[:, 2].sum(), 1)), 4)}

    return {"deficit_cum": heat, "deficit_incremental": heat_inc,
            "hrr": out_c,
            "boundary_swap_rate": round(swap_inv / max(swap_pairs, 1), 4),
            "boundary_swap_pairs": swap_pairs}

res = {}
for cat in CATS:
    for arm in ARMS:
        r = panels(cat, arm)
        if r is None:
            print(f"skip {cat}/{arm} (npz missing)", file=sys.stderr)
            continue
        res[f"{cat}.{arm}"] = r
with open(f"{RUNS}/fig4_panels_s36.json", "w") as f:
    json.dump(res, f, indent=2)

# text rendering
for key, r in res.items():
    print(f"\n=== {key} ===")
    print("Fig.4b cumulative normalized prefix-mass deficit (teacher-top10 members; d1..d4):")
    for bn in ("head", "mid", "tail"):
        print(f"  {bn:>5s}: " + "  ".join(f"{v:+.4f}" for v in r["deficit_cum"][bn]))
    print("      incremental (level contribution; d2 col == pi(d2|d1) term):")
    for bn in ("head", "mid", "tail"):
        print(f"  {bn:>5s}: " + "  ".join(f"{v:+.4f}" for v in r["deficit_incremental"][bn]))
    print(f"Fig.4c HRR: mid-GT head-share-incoming={r['hrr']['mid_gt']['primary_head_share_incoming']}"
          f" tail-GT={r['hrr']['tail_gt']['primary_head_share_incoming']}"
          f" | B7b swap_rate={r['boundary_swap_rate']} (pairs={r['boundary_swap_pairs']})")
