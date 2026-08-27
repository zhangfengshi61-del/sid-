import json, os, sys
from collections import Counter

ROOT = "."

def load(p): return json.load(open(p))

def top10(records): return [r.get("beams", [])[:10] for r in records]

def cand_count(records):
    # how many distinct items the rescore actually ranked (proxy for candidate-pool size lower bound:
    # the stored beam list is truncated to k=20, so this is min(pool,20); use #empty as coverage-miss)
    return records

for ds in (sys.argv[1:] or ["BeautyRK", "ToysRK", "SportsRK"]):
    base = f"{ROOT}/runs/seed/{ds}_L8_s1"
    tags = {}
    for h in [1, 2, 3, 4, 5, 6]:
        p = f"{base}/beams_rescore_h{h}.json"
        if os.path.exists(p):
            tags[h] = top10(load(p))
    if not tags:
        print(ds, "no beams yet"); continue
    hs = sorted(tags)
    n = len(next(iter(tags.values())))
    ref_h = 4 if 4 in tags else max(hs)
    ref = tags[ref_h]
    # empties (candidate set was empty -> total miss)
    print(f"\n=== {ds}  n={n}  ref=H{ref_h} ===")
    for h in hs:
        t = tags[h]
        same_order = sum(1 for a, b in zip(t, ref) if a == b)
        same_set = sum(1 for a, b in zip(t, ref) if set(a) == set(b))
        empties = sum(1 for a in t if len(a) == 0)
        # how many queries where H{h} top1 differs from refs top1
        top1_diff = sum(1 for a, b in zip(t, ref) if (a[:1] != b[:1]))
        print(f"  H{h} vs H{ref_h}: identical_top10_order={same_order}/{n} ({same_order/n:.4f})  "
              f"same_set={same_set}/{n} ({same_set/n:.4f})  top1_diff={top1_diff}  empty_cand={empties}")
