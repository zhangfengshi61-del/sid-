#!/usr/bin/env python
"""Build collision-aware SID indices from an existing L10 index (post-hoc length allocation).
 - CA   : L_i = minimum unique-prefix length (identifiability only). Max efficiency.
 - CA-HB: L_i = max(min-unique-prefix, capacity_target(pop)) — identifiability FLOOR + popularity-graded
          capacity (head gets long codes for semantic-capacity accuracy, tail stays minimal). The "fused" method.
usage: build_ca.py <DATASET>  (reads <DS>.index_L10.json + <DS>.pop.json in repos/LETTER/data/<DS>)"""
import json, sys
from collections import Counter
import numpy as np

DS = sys.argv[1]
D = f"./repos/LETTER/data/{DS}"
idx10 = json.load(open(f"{D}/{DS}.index_L10.json"))
pop = json.load(open(f"{D}/{DS}.pop.json"))
ids = [str(i) for i in range(len(idx10))]
N = len(ids)
codes = {i: tuple(idx10[i]) for i in ids}

# min unique-prefix length per item
minlen = {}
for L in range(1, 11):
    cnt = Counter(codes[i][:L] for i in ids)
    for i in ids:
        if i not in minlen and cnt[codes[i][:L]] == 1:
            minlen[i] = L
for i in ids:
    minlen.setdefault(i, 10)

# popularity quintile bins (bin 0 = head)
p = np.array([pop[i] for i in ids]); order = np.argsort(-p, kind="stable")
bins = np.empty(N, int); edges = np.linspace(0, N, 6).astype(int)
for b in range(5):
    bins[order[edges[b]:edges[b + 1]]] = b
CAP = [10, 2, 2, 2, 2]  # capacity target by pop quintile: head->10 ... tail->2

ca = {i: idx10[i][:minlen[i]] for i in ids}
cahb = {i: idx10[i][:max(minlen[i], CAP[bins[int(i)]])] for i in ids}
json.dump(ca, open(f"{D}/{DS}.index_ca.json", "w"))
json.dump(cahb, open(f"{D}/{DS}.index_cahb.json", "w"))

head = set(str(i) for i in order[:N // 5])
for name, d in [("ca", ca), ("cahb", cahb), ("L6c-ref", {i: idx10[i][:6] for i in ids}),
                ("L10-ref", idx10)]:
    lens = [len(d[i]) for i in ids]
    sids = ["".join(d[i]) for i in ids]; c = Counter(sids)
    coll = sum(1 for s in sids if c[s] > 1) / N * 100
    hlen = sum(len(d[i]) for i in head) / len(head)
    print("%-8s avglen=%.3f headlen=%.3f maxlen=%d coll%%=%.2f" % (name, sum(lens) / N, hlen, max(lens), coll))
