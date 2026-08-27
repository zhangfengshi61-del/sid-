#!/usr/bin/env python
"""Collision-Corrected Evaluation (zhang-how 2605.25330 sec 3.1), per popularity tier.
From saved beam records (test_cce.py, shuffle=False) + dataset inter/index, computes per-tier
SID-level Hit/NDCG vs item-level ItemHit/ItemNDCG. Tests the audit: does the SID-level
'head-prefers-short' advantage shrink under collision correction?
"""
import argparse, json
import numpy as np
from collections import defaultdict


def tiers_of(pop):
    N = len(pop)
    order = np.argsort(-pop, kind="stable")
    rank = np.empty(N, dtype=np.int64); rank[order] = np.arange(N)
    q = rank / max(N - 1, 1)
    return ["Head" if q[i] < 0.2 else ("Tail" if q[i] >= 0.8 else "Body") for i in range(N)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index_file", required=True)
    ap.add_argument("--beams", required=True)
    ap.add_argument("--Ks", default="5,10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    inters = json.load(open(f"{a.data_dir}/{a.dataset}.inter.json"))
    index = json.load(open(f"{a.data_dir}/{a.dataset}{a.index_file}"))
    N = len(index)
    sid_of = {int(i): "".join(index[i]) for i in index}
    group = defaultdict(list)
    for i in range(N):
        group[sid_of[i]].append(i)
    pop = np.zeros(N, dtype=np.int64)
    users = list(inters.keys())
    for u in users:
        for it in (inters[u][:-2] if len(inters[u]) > 2 else []):
            pop[it] += 1
    tier = tiers_of(pop)
    target_items = [inters[u][-1] for u in users]
    recs = json.load(open(a.beams))
    assert len(recs) == len(target_items), (len(recs), len(target_items))
    Ks = [int(x) for x in a.Ks.split(",")]

    agg = defaultdict(lambda: defaultdict(float)); cnt = defaultdict(int)
    gsz = defaultdict(list)  # avg target-group size per tier (diagnostic)
    for idx, rec in enumerate(recs):
        ti = target_items[idx]; tr = tier[ti]
        cnt[tr] += 1; cnt["ALL"] += 1
        tsid = sid_of[ti]; g = len(group[tsid]); beams = rec["beams"]
        gsz[tr].append(g); gsz["ALL"].append(g)
        try:
            r = beams.index(tsid) + 1
        except ValueError:
            r = None
        for K in Ks:
            hit = 1.0 if (r is not None and r <= K) else 0.0
            ndcg = (1.0 / np.log2(r + 1)) if (r is not None and r <= K) else 0.0
            if r is None:
                ih = indcg = 0.0
            else:
                p = 1 + sum(len(group[beams[q]]) for q in range(r - 1))
                m = min(g, max(0, K - p + 1))
                ih = m / g
                indcg = sum(1.0 / np.log2(p + e) for e in range(1, m + 1)) / g
            for name, val in [(f"Hit@{K}", hit), (f"NDCG@{K}", ndcg),
                              (f"ItemHit@{K}", ih), (f"ItemNDCG@{K}", indcg)]:
                agg[tr][name] += val; agg["ALL"][name] += val

    out = {}
    for tr in ["Head", "Body", "Tail", "ALL"]:
        if cnt[tr] == 0:
            continue
        out[tr] = {m: round(agg[tr][m] / cnt[tr], 5) for m in agg[tr]}
        out[tr]["n"] = cnt[tr]
        out[tr]["avg_target_group_size"] = round(float(np.mean(gsz[tr])), 3)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
