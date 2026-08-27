#!/usr/bin/env python
"""How does the identifiability horizon scale with catalog size N? (Addresses the #1 reviewer attack:
"L3 only suffices because your catalog is tiny.") Subsample items to increasing N, recompute min-unique-
prefix over the subset (codes fixed from the full RQ-VAE), average over 3 random subsamples. Compare to
the uniform-random floor log_K(N). Honest extrapolation to real-scale catalogs. CPU only."""
import json, math
import numpy as np
from collections import Counter

def minlen_arr(codes):
    n = len(codes); ml = [0] * n
    for L in range(1, 11):
        cnt = Counter(c[:L] for c in codes)
        for i in range(n):
            if ml[i] == 0 and cnt[codes[i][:L]] == 1:
                ml[i] = L
    for i in range(n):
        if ml[i] == 0:
            ml[i] = 10
    return np.array(ml)

for DS in ['SportsRK', 'BeautyRK', 'ToysRK']:
    idx = json.load(open(f'repos/LETTER/data/{DS}/{DS}.index_L10.json'))
    allc = [tuple(idx[str(i)]) for i in range(len(idx))]
    N = len(allc)
    print(f'=== {DS} (full N={N}, K=256) ===')
    print('   n      mean  p90  %>4   %>6   log256(n)')
    for n in [500, 1000, 2000, 4000, 8000, 16000, N]:
        if n > N:
            continue
        mls = []
        for s in range(3):
            rng = np.random.RandomState(s)
            sub = [allc[i] for i in rng.choice(N, size=n, replace=False)]
            mls.append(minlen_arr(sub))
        ml = np.concatenate(mls)
        print('  %6d  %.2f   %.0f   %4.1f  %4.2f   %.2f'
              % (n, ml.mean(), np.percentile(ml, 90), 100 * np.mean(ml > 4), 100 * np.mean(ml > 6), math.log(n, 256)))
    print()
