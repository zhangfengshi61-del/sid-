#!/usr/bin/env python
"""Model-free prefix geometry of a strong-base index: tests the 'identifiability != prefix-collapse'
objection for wide-net gather. For each H, report M_H = #distinct H-prefixes, fanout (items/prefix)
distribution, and the fanout-ranked cumulative coverage (how many top prefixes cover X% of items).
If M_3 ~ N and fanout ~1, the prefix pool does NOT collapse -> wide-net must gather MANY prefixes.
usage: prefix_geometry.py <DS> <TAG>   (reads index_<TAG>_L8.json)
"""
import sys, json
from collections import Counter
import numpy as np
DS, TAG = sys.argv[1], sys.argv[2]
idx = json.load(open(f"./repos/LETTER/data/{DS}/{DS}.index_{TAG}_L8.json"))
codes = [tuple(v) for v in idx.values()]
N = len(codes)
print(f"== {DS}/{TAG}  N={N} ==")
print(f"{'H':>2} {'M_H':>7} {'M_H/N':>6} {'fanout_p50':>10} {'fanout_p95':>10} {'fanout_max':>10} {'B_for_50pct':>11} {'B_for_90pct':>11}")
for H in range(1, 6):
    pref = [c[:H] for c in codes]
    cnt = Counter(pref)
    M = len(cnt)
    fan = np.array(sorted(cnt.values(), reverse=True))   # items per prefix, desc
    cov = np.cumsum(fan) / N
    b50 = int(np.searchsorted(cov, 0.5) + 1)
    b90 = int(np.searchsorted(cov, 0.9) + 1)
    print(f"{H:>2} {M:>7} {M/N:>6.2f} {np.percentile(fan,50):>10.1f} {np.percentile(fan,95):>10.1f} "
          f"{fan.max():>10d} {b50:>11} {b90:>11}")
print("  (B_for_X% = # top-fanout prefixes to cover X% of items; lower = more collapse = wide-net easier)")
