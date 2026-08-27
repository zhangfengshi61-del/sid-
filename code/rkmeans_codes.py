"""Residual K-Means codes from an embedding file, in ETEGRec's code format (semantic levels + 1 dedup
   token). Forced 256/level via faiss (fast). usage:
   python rkmeans_codes.py --emb <emb.npy> --levels <code_length-1> --out <codes.json>
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '32')
import argparse, json
import numpy as np
import faiss
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--emb', required=True)
ap.add_argument('--levels', type=int, required=True)   # semantic levels (= code_length - 1)
ap.add_argument('--code_num', type=int, default=256)
ap.add_argument('--out', required=True)
ap.add_argument('--niter', type=int, default=50)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

X = np.ascontiguousarray(np.load(a.emb).astype(np.float32))
N, D = X.shape
print(f'emb {X.shape} levels={a.levels} K={a.code_num}', flush=True)
res = X.copy()
codes = np.zeros((N, a.levels), dtype=int)
for l in range(a.levels):
    km = faiss.Kmeans(D, a.code_num, niter=a.niter, verbose=False, seed=a.seed)
    km.train(res)
    _, lab = km.index.search(res, 1)
    lab = lab.ravel()
    codes[:, l] = lab
    res = res - km.centroids[lab]
    print(f'  level {l}: used {len(set(lab.tolist()))}/{a.code_num} mean-resid {np.linalg.norm(res)/N:.5f}', flush=True)

g = defaultdict(int)
all_tokens = [[-1] * (a.levels + 1)]
maxc = 0
for i in range(N):
    pref = tuple(codes[i].tolist())
    d = g[pref]; g[pref] += 1
    maxc = max(maxc, d + 1)
    all_tokens.append(codes[i].tolist() + [min(d, a.code_num - 1)])
print(f'max conflict {maxc} ({"OK fits dedup" if maxc <= a.code_num else "CLIPPED"})', flush=True)
json.dump(all_tokens, open(a.out, 'w'))
print(f'saved {len(all_tokens)} codes code_length={a.levels+1} -> {a.out}', flush=True)
