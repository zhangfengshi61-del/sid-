#!/usr/bin/env python
"""E3 resolver-sensitivity (the strongest objection). zhang-how CCE divides credit by
collision-group size = assumes collided items are INDISTINGUISHABLE. But a deployed system
resolves a collided SID non-uniformly. Re-score saved beams under within-group resolvers and
recompute per-popularity-tier Hit/NDCG@K. Reuses test_cce beams + inter/index (CPU only).
  SID     : SID-level, no expansion -- what VarLenRec reported.
  uniform : expected credit over random within-group order == zhang-how CCE (sanity: == cce_eval).
  pop     : within-group order by train-popularity desc -> popular item wins its group (STEELMAN
            for "head-short"; if head-short survives this, the artifact claim must be qualified).
  oracle  : target placed first in its group (item-level best case / lower-bound on item-rank).
Verdict turns on the Head trend across L: if head-short only appears under SID/oracle but Head
*increases* with L under pop, the paradox is a metric artifact even against a charitable resolver."""
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
    ap.add_argument("--Ks", default="10")
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
    # within-group rank by popularity desc (ties: item id) -> 1-based position of each item
    pos_pop = {}
    for sid, members in group.items():
        for rk, it in enumerate(sorted(members, key=lambda x: (-int(pop[x]), x))):
            pos_pop[it] = rk + 1
    target_items = [inters[u][-1] for u in users]
    recs = json.load(open(a.beams))
    assert len(recs) == len(target_items), (len(recs), len(target_items))
    Ks = [int(x) for x in a.Ks.split(",")]

    rez = ["SID", "uniform", "pop", "oracle"]
    agg = {z: defaultdict(lambda: defaultdict(float)) for z in rez}
    cnt = defaultdict(int)
    for idx, rec in enumerate(recs):
        ti = target_items[idx]; tr = tier[ti]
        cnt[tr] += 1; cnt["ALL"] += 1
        tsid = sid_of[ti]; g = len(group[tsid]); beams = rec["beams"]
        try:
            r = beams.index(tsid) + 1
        except ValueError:
            r = None
        pre = sum(len(group[beams[q]]) for q in range(r - 1)) if r is not None else None
        for K in Ks:
            if r is None:
                vals = {z: (0.0, 0.0) for z in rez}
            else:
                # SID-level
                sid_h = 1.0 if r <= K else 0.0
                sid_n = (1.0 / np.log2(r + 1)) if r <= K else 0.0
                # uniform expected (== CCE)
                p = pre + 1; m = min(g, max(0, K - p + 1))
                u_h = m / g
                u_n = sum(1.0 / np.log2(p + e) for e in range(1, m + 1)) / g
                # popularity resolver (hard): target at its pop-rank within group
                ir = pre + pos_pop[ti]
                pp_h = 1.0 if ir <= K else 0.0
                pp_n = (1.0 / np.log2(ir + 1)) if ir <= K else 0.0
                # oracle: target first in its group
                oir = pre + 1
                o_h = 1.0 if oir <= K else 0.0
                o_n = (1.0 / np.log2(oir + 1)) if oir <= K else 0.0
                vals = {"SID": (sid_h, sid_n), "uniform": (u_h, u_n), "pop": (pp_h, pp_n), "oracle": (o_h, o_n)}
            for z in rez:
                h, n = vals[z]
                agg[z][tr][f"Hit@{K}"] += h; agg[z][tr][f"NDCG@{K}"] += n
                agg[z]["ALL"][f"Hit@{K}"] += h; agg[z]["ALL"][f"NDCG@{K}"] += n

    out = {}
    for z in rez:
        out[z] = {}
        for tr in ["Head", "Body", "Tail", "ALL"]:
            if cnt[tr] == 0:
                continue
            out[z][tr] = {m: round(agg[z][tr][m] / cnt[tr], 5) for m in agg[z][tr]}
            out[z][tr]["n"] = cnt[tr]
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
