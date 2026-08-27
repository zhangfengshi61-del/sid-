#!/usr/bin/env python
"""Efficiency-fork make/break PILOT (no varlen decode build). Question: does 'average SID length'
predict batched trie-beam serving cost, or does WHICH items get short codes matter -- and does any
realistic batching discipline dissolve the effect?

Grounded in MEASURED Beauty decode primitives (test_cce, beam20, bs64): seconds to decode one
64-item batch whose items have uniform length L: L2..L10 = 1.05/1.59/2.09/2.61/3.08 -> t(L)=0.544+0.254*L.
(Caveat: t(L) is the empirical per-batch cost at SID length L on this stack; it folds in both AR decode
steps and the L-dependent history-encoding. Good enough for a first-order serving probe.)

Length policies share an IDENTICAL length histogram (equal 20% popularity bins -> L in {2,4,6,8,10},
avg=6), so any serving-cost gap is the ASSIGNMENT, not the average length:
  fixed6     everyone L6 (matched-average fixed baseline)
  head_short most-popular->L2 ... least->L10  (VarLenRec style: short codes on hot items)
  head_long  most-popular->L10 ... least->L2  (our collision-corrected law)
  permuted   head_short lengths shuffled across items (length DECORRELATED from popularity = key control)
Traffic ~ train popularity. Three batching disciplines:
  static    arrival-order 64-batches; AR runs to the longest -> cost t(max L in batch)
  continuous slots refill on finish -> cost ~ sum of per-request lengths (avg-length rules)
  bucketed  group requests by length -> per-length 64-batches cost t(L) (short hot items help)"""
import json, numpy as np

POP = "./repos/LETTER/data/BeautyRK/BeautyRK.pop.json"
pop = json.load(open(POP)); N = len(pop)
p = np.array([pop[str(i)] for i in range(N)], dtype=np.float64)
w = p + 0.1; w = w / w.sum()                       # traffic weight ~ popularity (+eps so cold items possible)
order = np.argsort(-p, kind="stable")              # most popular first
Ls = np.array([2, 4, 6, 8, 10])
bin_of = np.empty(N, dtype=int); edges = np.linspace(0, N, 6).astype(int)
for b in range(5): bin_of[order[edges[b]:edges[b + 1]]] = b
len_head_short = Ls[bin_of]
len_head_long = Ls[4 - bin_of]
len_fixed6 = np.full(N, 6)
rng = np.random.RandomState(0)
len_permuted = len_head_short[rng.permutation(N)]
policies = {"fixed6": len_fixed6, "head_short": len_head_short, "head_long": len_head_long, "permuted": len_permuted}

def tb(L): return 0.544 + 0.254 * np.asarray(L)    # s per 64-batch, uniform length L
PERSTEP = 0.254                                     # marginal s per decode-step for a 64-batch
FIXED = 0.544                                       # fixed s per batch launch
B = 64; M = 1_000_000
reqs = rng.choice(N, size=M, p=w)
nb = M // B
# traffic share by length bin (diagnostic; same items so identical across policies that use bins)
share = np.bincount(bin_of[reqs], minlength=5) / M
print(f"N={N} M={M} B={B}; traffic share by pop-bin (top->bot) = {np.round(share,3).tolist()}", flush=True)
print(f"all binned policies: avg length = {len_head_short.mean():.2f} (identical histogram)\n", flush=True)
print(f"{'policy':11s} {'avgLen':>6} | {'static_s':>9} {'qps':>7} | {'cont_s':>8} {'qps':>7} | {'bucket_s':>9} {'qps':>7}", flush=True)
for name, Lmap in policies.items():
    lens = Lmap[reqs]
    lb = lens[:nb * B].reshape(nb, B)
    static = float(tb(lb.max(axis=1)).sum())
    cont = FIXED * (lens.sum() / lens.size) * (M / B) / 6.0 * 6.0  # placeholder, replaced below
    cont = FIXED * nb + PERSTEP * lens.sum() / B                    # decoder always full: total steps=sum(L)/B
    cnt = np.bincount(lens, minlength=11)
    bucket = sum(np.ceil(cnt[L] / B) * tb(L) for L in [2, 4, 6, 8, 10])
    print(f"{name:11s} {lens.mean():6.2f} | {static:9.1f} {M/static:7.0f} | {cont:8.1f} {M/cont:7.0f} | {bucket:9.1f} {M/bucket:7.0f}", flush=True)
print("\nRead: static==same across matched-avg policies => batch-MAX dominates (longest code rules, avg length irrelevant).", flush=True)
print("      bucketed head_short<<head_long => WHICH items get short codes matters, but only with length bucketing.", flush=True)
print("      cont identical across matched-avg policies => continuous batching makes only avg length matter.", flush=True)
