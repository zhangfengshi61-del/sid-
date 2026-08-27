"""s38: NEZHA baseline NEW-METRICS (the collaborator's "baseline 全指标") for the NEZHA MTP student.

Same per-popularity-bucket metrics as ssm_bucketdiag.py (train-count tertiles: cold=0/tail/mid/head,
== bucket_diag_s34 convention ⇒ directly comparable to r0base/lkpf/GDN), but driving the NEZHA
self-drafting MTP head (nezha_baseline.nezha_beam) instead of the SSM decoder:
  - R@5/10, NDCG@5/10 (student).
  - PS@4 = GT SID still in the deployed beam-50 after digit 4 (== GT in student top-50).
  - PopGap@10 = mean train-pop of student top-10 items - GT item's train-pop.
Story-critical question: does the fast lossy NEZHA also lose MID/TAIL (like SID-MLP r0base + GDN),
or does its jointly-fine-tuned encoder preserve it? Lean (no teacher pass). Reuses NEZHA + its beam.

--lists_out (opt-in, additive): ALSO dump the per-request served lists in the shared s48 schema of
code/lists_s48.py (same dedup_items / npz names+dtypes / valid semantics / sanity block), scored
against the SAME cached TIGER teacher beam (runs/tbeam50_<CAT>.pt) every other method uses, so the
NEZHA row is directly comparable offline via code/metrics_s48.py. Without the flag this script runs
exactly as before.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from collections import defaultdict

R = os.environ.get("SERVESID_ROOT", ".")
sys.path.insert(0, f"{R}/repos/SID-MLP")
sys.path.insert(0, f"{R}/repos")
sys.path.insert(0, f"{R}/code")
os.chdir(f"{R}/repos")
for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[k] = "1"
from torch.utils.data import DataLoader
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS
from sid_mlp.beam import build_valid_tensors
from sid_mlp.inference import (build_tiger_pipeline, configure_precision, _torch_load,
                               find_valid_set)
from nezha_baseline import build_from_ckpt, nezha_beam

DEV = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--category", required=True)
ap.add_argument("--checkpoint", required=True)       # TIGER teacher (base arch + shared embedding)
ap.add_argument("--lens_path", required=True)        # NEZHA ckpt (fine-tuned encoder + MTP head)
ap.add_argument("--sem_ids", required=True)
ap.add_argument("--dataset_tag", required=True)
ap.add_argument("--graph_dir", required=True)
ap.add_argument("--dataset", default="AmazonReviews2023")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--beam_size", type=int, default=50)
ap.add_argument("--max_batches", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--lists_out", default="")       # s48 served-list dump, path WITHOUT extension
ap.add_argument("--teacher_cache", default="")   # runs/tbeam50_<CAT>.pt; REQUIRED with --lists_out
ap.add_argument("--pop_cache", default="")       # npz(items,counts,lo,hi); written if absent
ap.add_argument("--topk", type=int, default=20)  # dumped served-list depth
args = ap.parse_args()
args.encoder_ckpt = None; args.fp32 = False
configure_precision(args, DEV)

pipe = build_tiger_pipeline(args)
model = build_from_ckpt(args, pipe)                   # loads fine-tuned t5 + MTP head, .eval()
vs = find_valid_set(args.dataset_tag, explicit_path=None, lens_path=args.lens_path, graph_dir=args.graph_dir)
(m12, m123, d4k, d4m, k4, v4, _im, vi4) = build_valid_tensors(vs, DEV)
masks = (m12, m123, d4k, d4m)

def sid_to_item(tup):
    return vi4.get(tuple(int(x) for x in tup), -1)

def dedup_items(sid_list, cap):
    """s48 dump convention (verbatim lists_s48.dedup_items): item-level -- drop invalid SIDs, dedup
    preserving rank order, truncate. Returns (items, n_invalid, n_raw)."""
    seen = set(); out = []; ninv = 0
    for tup in sid_list:
        it = sid_to_item(tup)
        if it == -1:
            ninv += 1; continue
        if it in seen: continue
        seen.add(it); out.append(it)
        if len(out) >= cap: break
    return out, ninv, len(sid_list)

# ---- train popularity (== bucket_diag_s34 / ssm_bucketdiag): train-count tertiles ----
if args.pop_cache and os.path.exists(args.pop_cache):
    z = np.load(args.pop_cache)
    pop = {int(i): int(c) for i, c in zip(z["items"], z["counts"])}
    LO = int(z["lo"]); HI = int(z["hi"])
else:
    pop = defaultdict(int)
    tl = DataLoader(pipe.tokenized_datasets["train"], batch_size=256, shuffle=False,
                    collate_fn=pipe.tokenizer.collate_fn["train"])
    with torch.inference_mode():
        for b in tl:
            for row in b["labels"].tolist():
                it = sid_to_item(row[:4])
                if it != -1:
                    pop[it] += 1
    cs = sorted(pop.values())
    LO = cs[len(cs) // 3]; HI = cs[2 * len(cs) // 3]
    if args.pop_cache:
        its = np.array(sorted(pop.keys()), dtype=np.int64)
        np.savez_compressed(args.pop_cache, items=its,
                            counts=np.array([pop[int(i)] for i in its], dtype=np.int64), lo=LO, hi=HI)
def bucket(it):
    c = pop.get(it, 0)
    if c == 0: return 0
    if c <= LO: return 1
    if c <= HI: return 2
    return 3
BNAME = ["cold", "tail", "mid", "head"]
print(f"[pop] items={len(pop)} lo={LO} hi={HI}", flush=True)

ds = pipe.tokenized_datasets["test"]
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pipe.tokenizer.collate_fn["test"])
agg = defaultdict(float); bkt = {b: defaultdict(float) for b in range(4)}

# ---- s48 served-list dump (opt-in): shared cached TIGER teacher + this method, same schema ----
TOPK = args.topk; LW = args.beam_size          # W is shared by teacher AND method (lists_s48 protocol)
t_cache = None
GT, GP, GB, TT, ST, TN, SN, SRAW, SINV, VALID = [], [], [], [], [], [], [], [], [], []
row_ptr = 0
if args.lists_out:
    if not (args.teacher_cache and os.path.exists(args.teacher_cache)):
        raise SystemExit(f"--lists_out needs the shared teacher beam cache; "
                         f"--teacher_cache={args.teacher_cache!r} not found")
    t_cache = torch.load(args.teacher_cache, map_location="cpu")["codes"]   # (N,50,4) TOKEN-ID space
    print(f"[tcache] loaded {args.teacher_cache} rows={t_cache.shape[0]} beam={t_cache.shape[1]}",
          flush=True)
    if t_cache.shape[1] < LW:
        raise SystemExit(f"teacher cache beam {t_cache.shape[1]} < decode width {LW}")
    if LW < TOPK:
        print(f"!!! WARNING: decode width {LW} < topk {TOPK} -- served lists are SHORT and @{TOPK} "
              f"metrics are distorted (s_n/t_n record the truth) !!!", flush=True)

t0 = time.time()
with torch.inference_mode():
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        ii = batch["input_ids"].to(DEV); am = batch["attention_mask"].to(DEV)
        quad = nezha_beam(model, ii, am, args.beam_size, masks)   # (B, beam, 4) token-ids
        codes = quad.detach().cpu().tolist()
        labs = batch["labels"].tolist()
        if args.lists_out:
            TQ = t_cache[row_ptr:row_ptr + ii.shape[0], :LW].tolist()
            if len(TQ) != ii.shape[0]:
                raise SystemExit(f"teacher cache exhausted at row {row_ptr} "
                                 f"(rows={t_cache.shape[0]}, need {row_ptr + ii.shape[0]})")
            row_ptr += ii.shape[0]
        for r in range(ii.shape[0]):
            gt = tuple(int(x) for x in labs[r][:4]); gt_it = sid_to_item(gt)
            if args.lists_out:
                # every test row is dumped (even gt-unresolvable ones) so the row order stays
                # aligned with the teacher cache and with every other method's dump
                t_items, _tinv, _traw = dedup_items(TQ[r], TOPK)
                s_items, sinv, sraw = dedup_items(codes[r], TOPK)
                GT.append(gt_it); GP.append(pop.get(gt_it, 0) if gt_it != -1 else -1)
                GB.append(bucket(gt_it) if gt_it != -1 else -1)
                TT.append(t_items + [-1] * (TOPK - len(t_items)))
                ST.append(s_items + [-1] * (TOPK - len(s_items)))
                TN.append(len(t_items)); SN.append(len(s_items))
                SRAW.append(sraw); SINV.append(sinv)
                VALID.append(int(gt_it != -1 and len(t_items) > 0))
            if gt_it == -1:
                continue
            gb = bucket(gt_it); gt_pop = pop.get(gt_it, 0)
            seen = set(); items = []
            for c in codes[r]:
                it = sid_to_item(c)
                if it == -1 or it in seen:
                    continue
                seen.add(it); items.append(it)
            try:
                rk = items[:10].index(gt_it)
            except ValueError:
                rk = -1
            ps4 = int(gt_it in items[:args.beam_size])   # GT SID survives to beam-50 after d4
            top10 = items[:10]
            popgap = (float(np.mean([pop.get(x, 0) for x in top10])) - gt_pop) if top10 else 0.0
            hit10 = int(rk >= 0); nd10 = 1.0 / math.log2(rk + 2.0) if rk >= 0 else 0.0
            hit5 = int(0 <= rk < 5); nd5 = 1.0 / math.log2(rk + 2.0) if 0 <= rk < 5 else 0.0
            for tgt in (agg, bkt[gb]):
                tgt["n"] += 1; tgt["r10"] += hit10; tgt["n10"] += nd10
                tgt["r5"] += hit5; tgt["n5"] += nd5; tgt["ps4"] += ps4; tgt["popgap10"] += popgap
        if bi % 50 == 0:
            print(f"batch {bi} n={int(agg['n'])} elapsed={time.time()-t0:.0f}s", flush=True)

n = max(int(agg["n"]), 1)
out = {"tag": args.tag, "category": args.category, "n": int(agg["n"]),
       "bucket_def": f"train-count tertiles lo={LO} hi={HI} (cold=0)", "lens_path": args.lens_path,
       **{m: agg[m] / n for m in ("r5", "n5", "r10", "n10", "ps4", "popgap10")}, "buckets": {}}
for gb in range(4):
    bn = max(int(bkt[gb]["n"]), 1)
    out["buckets"][BNAME[gb]] = {"n": int(bkt[gb]["n"]),
        **{m: bkt[gb][m] / bn for m in ("r5", "n5", "r10", "n10", "ps4", "popgap10")}}
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

if args.lists_out:
    gt_a = np.array(GT, dtype=np.int32); gp_a = np.array(GP, dtype=np.int64)
    gb_a = np.array(GB, dtype=np.int8); valid_a = np.array(VALID, dtype=np.int8)
    tt_a = np.array(TT, dtype=np.int32); st_a = np.array(ST, dtype=np.int32)
    tn_a = np.array(TN, dtype=np.int16); sn_a = np.array(SN, dtype=np.int16)
    sraw_a = np.array(SRAW, dtype=np.int16); sinv_a = np.array(SINV, dtype=np.int16)
    np.savez_compressed(args.lists_out + ".npz", gt_item=gt_a, gt_pop=gp_a, gt_bucket=gb_a,
                        valid=valid_a, t_top=tt_a, s_top=st_a, t_n=tn_a, s_n=sn_a,
                        s_raw=sraw_a, s_invalid=sinv_a,
                        bucket_lo=LO, bucket_hi=HI, n_test=len(GT), W=LW, topk=TOPK)
    msk = valid_a.astype(bool)
    def _rank(lists, k):
        """1-based rank of gt in the first k entries, 0 if absent."""
        hit = (lists[:, :k] == gt_a[:, None])
        return np.where(hit.any(1), hit.argmax(1) + 1, 0)
    san = {"tag": args.tag, "category": args.category, "method": "nezha",
           "lens_path": args.lens_path, "W": LW, "topk": TOPK,
           "n_rows": int(len(GT)), "n_valid": int(msk.sum()), "bucket_lo": LO, "bucket_hi": HI,
           "mean_s_invalid": float(sinv_a[msk].mean()), "mean_s_n": float(sn_a[msk].mean()),
           "short_at_20": int((sn_a[msk] < 20).sum()), "short_at_10": int((sn_a[msk] < 10).sum())}
    for k in (5, 10, 20):
        tr = _rank(tt_a, k)[msk]; sr = _rank(st_a, k)[msk]
        san[f"teacher_r{k}"] = float((tr > 0).mean())
        san[f"teacher_n{k}"] = float(np.where(tr > 0, 1.0 / np.log2(np.maximum(tr, 1) + 1.0), 0.0).mean())
        san[f"student_r{k}"] = float((sr > 0).mean())
        san[f"student_n{k}"] = float(np.where(sr > 0, 1.0 / np.log2(np.maximum(sr, 1) + 1.0), 0.0).mean())
        inter = np.zeros(int(msk.sum())); union = np.zeros(int(msk.sum()))
        T = tt_a[msk][:, :k]; S = st_a[msk][:, :k]
        for i in range(T.shape[0]):
            a = set(int(x) for x in T[i] if x >= 0); b = set(int(x) for x in S[i] if x >= 0)
            inter[i] = len(a & b); union[i] = max(len(a | b), 1)
        san[f"jac{k}"] = float((inter / union).mean())
        san[f"overlap{k}"] = float((inter / k).mean())
    with open(args.lists_out + ".json", "w") as f:
        json.dump(san, f, indent=2)
    print("NEZHA_LISTS", json.dumps(san), flush=True)

print("NEZHA_BUCKETDIAG", json.dumps({k: v for k, v in out.items() if k != "buckets"}), flush=True)
print(json.dumps(out["buckets"]), flush=True)
print("NEZHA_BUCKETDIAG_DONE", flush=True)
