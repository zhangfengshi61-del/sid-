#!/usr/bin/env python
"""Serving cost across BATCHING DISCIPLINES, grounded in real data:
 - per-query decode depth from saved beams.json (depth_q = max token-count over a query's 20 beams + 1 eos),
   full test set, no extra GPU.
 - measured uniform calibration t(depth) = A + B*depth (s per 64-batch) from the latency bench.
Regimes:
 - STATIC (measured): real bs=64 random-batch latency from bench_<name>.json (ground truth).
 - STATIC (from beams, cross-check): random bs=64 batches, cost = t(max depth_q in batch).
 - BUCKETED (oracle length grouping): group queries by depth_q -> homogeneous bs=64 batches, cost = sum_d ceil(n_d/64)*t(d).
 - CONTINUOUS (in-flight ideal): decoder always full, cost = (B/64)*sum_q depth_q + A*(N/64).
Shows: static makes avg-length irrelevant (batch-max rules); bucketed/continuous realize the avg-length benefit;
the benefit is exactly head->short, which the accuracy audit shows wrecks head identifiability."""
import json, glob, os
import numpy as np

ROOT = "./runs"
A, B = 0.1927, 0.1225          # t(depth) = A + B*depth, from bench uniform fit
BS = 64
RNG = np.random.RandomState(0)

POLICIES = {
    "fixed-6": f"{ROOT}/e2_sweep_BeautyRK_L6/beams.json",
    "vl_hs":   f"{ROOT}/e2_vl_vl_hs/beams.json",
    "vl_hl":   f"{ROOT}/e2_vl_vl_hl/beams.json",
}
BENCH = {"fixed-6": "uL6", "vl_hs": "vl_hs", "vl_hl": "vl_hl"}

def t(d): return A + B * np.asarray(d, float)

def measured_static(name):
    f = f"{ROOT}/bench/bench_{BENCH[name]}.json"
    if not os.path.exists(f):
        f2 = f"{ROOT}/bench/bench_{BENCH[name]}_a2.json"
        f = f2 if os.path.exists(f2) else f
    if not os.path.exists(f):
        return None
    recs = json.load(open(f))
    dt = np.array([r["dt"] for r in recs]); nq = np.array([r["nq"] for r in recs])
    return float(nq.mean() / np.median(dt))

print(f"calibration t(depth)={A}+{B}*depth s/64-batch; depth = max-beam-tokens+1\n")
print(f"{'policy':8s} {'Nq':>6} {'depth:mean':>10} {'p50':>4} {'p90':>4} {'max':>4} | "
      f"{'static_meas':>11} {'static_beam':>11} {'bucketed':>9} {'contin':>7}  (qps)")
rows = {}
for name, path in POLICIES.items():
    recs = json.load(open(path))
    dq = np.array([max((b.count("<") for b in r["beams"]), default=0) + 1 for r in recs])  # +1 eos
    N = len(dq)
    # static from beams: random bs64 batches, cost=t(batch-max depth)
    perm = RNG.permutation(N); nb = N // BS
    bm = dq[perm[:nb * BS]].reshape(nb, BS).max(axis=1)
    static_beam = N / float(t(bm).sum())
    # bucketed: group by depth
    vals, cnts = np.unique(dq, return_counts=True)
    bucket_time = float(sum(np.ceil(c / BS) * t(d) for d, c in zip(vals, cnts)))
    bucketed = N / bucket_time
    # continuous: decoder full, total steps=sum depth, plus per-batch overhead amortized
    cont_time = (B / BS) * float(dq.sum()) + A * (N / BS)
    cont = N / cont_time
    meas = measured_static(name)
    rows[name] = dict(N=N, dmean=dq.mean(), static_meas=meas, static_beam=static_beam,
                      bucketed=bucketed, cont=cont)
    print(f"{name:8s} {N:6d} {dq.mean():10.2f} {int(np.percentile(dq,50)):4d} {int(np.percentile(dq,90)):4d} "
          f"{int(dq.max()):4d} | {('%.1f'%meas) if meas else '   n/a':>11} {static_beam:11.1f} "
          f"{bucketed:9.1f} {cont:7.1f}")

print("\nRead:")
print(" - static_meas vs static_beam: cross-check (measured bs64 vs t(batch-max)-from-beams).")
print(" - static ~ same across policies (batch-MAX rules) => avg-length irrelevant under static batching.")
print(" - bucketed/continuous: vl_hs >> fixed-6 >> vl_hl (head-short shallow) => avg-length benefit only here.")
print(" - but vl_hs head-short is exactly what the accuracy audit shows destroys head identifiability.")
