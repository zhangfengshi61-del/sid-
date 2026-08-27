#!/usr/bin/env python3
"""
metrics_s48.py -- all offline metrics for the s48 served-list deliverable.

Reads per-request served-list dumps and emits every offline number the paper needs,
with request-level paired bootstrap CIs.  numpy + stdlib only.  Reads only from
--runs_dir, writes only into --out_dir, no network.

================================================================================
INPUT
================================================================================
Files  <runs_dir>/lists_s48_<CAT>__<METHOD>__<SEED>.npz  (CAT may contain single
underscores; the field separator is a DOUBLE underscore).  Arrays, n = #requests,
row-aligned across every METHOD of a given CAT:

  gt_item   (n,)   int32  ground-truth next item id
  gt_pop    (n,)   int64  GT item's train-split interaction count
  gt_bucket (n,)   int8   0=cold(pop==0) 1=tail 2=mid 3=head (train-count tertiles
                          among items with count>=1; == the project's standard
                          bucket_diag/ssm_bucketdiag convention)
  t_top     (n,20) int32  TEACHER (TIGER) served list: valid, deduplicated item ids
                          in rank order, -1 padded
  s_top     (n,20) int32  THIS METHOD's served list, same convention
  t_n       (n,)   int16  count of real (non -1) entries of t_top
  s_n       (n,)   int16  count of real entries of s_top
  s_raw     (n,)   int16  raw beam list length before validity-filter/dedup
  s_invalid (n,)   int16  number of invalid SIDs in the raw beam
  scalars   bucket_lo, bucket_hi, n_test

-1 is padding EVERYWHERE and never enters a set, a rank, or a count.  The real
list of a row is the row's entries with -1 removed, order preserved; "the first K
real entries" is that list truncated to K.

================================================================================
METRIC DEFINITIONS  (K in {5,10,20}; L_M^K / L_T^K = first K real entries of the
method / teacher served list of that request)
================================================================================
1. AGREEMENT VS THE TEACHER REFERENCE (set semantics on real entries only)
     jac@K     = |set(L_M^K) & set(L_T^K)| / |set(L_M^K) | set(L_T^K)|
     overlap@K = |set(L_M^K) & set(L_T^K)| / K            (always divided by K)
   jac@K is UNDEFINED when the union is empty (both lists empty at K); such
   requests are excluded from the jac mean and counted in `jac_n_defined`.
   overlap@K is always defined.  A request whose method list or teacher list has
   fewer than K real entries is counted in short_list_count@K (broken out into
   short_list_method / short_list_teacher; short_list_count = "either is short").

2. TASK METRICS (computed for the method and, identically, for the teacher)
     rank    = 1-based position of gt_item in the real served list, 0 if absent
     R@K     = 1 if 0 < rank <= K else 0
     N@K     = 1/log2(rank+1) if 0 < rank <= K else 0        (NDCG@K, single
               relevant item, so N@K == DCG@K == the standard rec NDCG@K)

3. BUCKETS.  Requests are grouped by gt_bucket.  Rows reported:
     head (3), mid (2), tail (1)          -- the three tertile buckets, primary
     cold (0)                             -- separate 4th row
     tail_incl_cold (0 or 1)              -- tail with cold merged in
     all                                  -- every request
   Per bucket x K x metric in {R,N}: method value, teacher value,
   ratio_to_teacher = method/teacher (null when teacher == 0),
   delta = method - teacher, request count n, and integer hit_count =
   #{requests with R@K == 1} for method and teacher.

4. HIT-SWAP DECOMPOSITION (method vs teacher, per K, overall and per bucket)
     common_hit   gt in L_T^K and gt in L_M^K
     teacher_only gt in L_T^K, gt not in L_M^K
     method_only  gt not in L_T^K, gt in L_M^K
     common_miss  neither
   counts + proportions, and the identity (checked, residual emitted)
     R@K_M - R@K_T == P(method_only) - P(teacher_only)
   NDCG win/loss mass, with g = the per-request NDCG gain at K of eq. (2):
     G_T = mean(max(g_T - g_M, 0))      (mass the teacher wins)
     G_M = mean(max(g_M - g_T, 0))      (mass the method wins)
   and the identity (checked, residual emitted)
     N@K_M - N@K_T == G_M - G_T

5. CONDITIONAL AGREEMENT.  Requests are grouped by hit pattern at a grouping
   width group_k in {5,10,20}:
     both_hit      gt in L_M^gk and gt in L_T^gk
     same_rank_hit both_hit AND rank_M == rank_T   (a SUBSET of both_hit, not a
                   separate cell of the partition)
     both_miss / teacher_only / method_only  as in (4)
   For each group: jac@5/10/20, overlap@5/10/20, request count, and for the
   hit-in-both groups the mean/std/median/sign-split/histogram of
   (rank_M - rank_T)  (negative = the method ranks the GT higher than the
   teacher does).

6. PopGap@K needs the train popularity of every SERVED item, which is not in the
   list dump; it is read from the optional sidecar
       <pop_cache_dir>/pop_<CAT>.npz   arrays items (int64, sorted item ids) and
                                       counts (int64, train-interaction count),
                                       scalars lo, hi (the tertile cut points)
   Lookup is a vectorised sorted search (np.searchsorted) over `items`; an item
   ABSENT from `items` has train count 0 (it is a cold item) and is kept, not
   dropped.  With L^K the first K real entries as above,
       PopGap@K = mean over requests of ( mean_{j in L^K} pop(j) - gt_pop )
   reported for the method and for the teacher, overall and per bucket.  A
   request with zero real entries at K has an UNDEFINED PopGap@K: it is excluded
   from the mean and counted in method_n_defined / teacher_n_defined.  The
   method-minus-teacher delta is taken over the requests where BOTH sides are
   defined (count in delta_n_defined) so that it stays a paired per-request
   statistic.  ratio_to_teacher is deliberately NULL for popgap: it is a signed
   difference of counts, so a ratio is not interpretable.
   With no pop_<CAT>.npz for a category, PopGap is not invented: popgap10 stays
   null with popgap_note, no popgap rows are emitted, and every other output byte
   is unchanged.

================================================================================
STATISTICS
================================================================================
Request-level paired bootstrap, B = --boot (default 2000) replicates: request
indices are resampled WITH REPLACEMENT via np.random.default_rng(...).integers
and accumulated into per-replicate counts with np.bincount (algebraically
identical to materialising the resampled index list); every statistic here is a
ratio of two request sums, so replicate b evaluates
    stat_b = (sum_i w_bi * num_i) / (sum_i w_bi * den_i)
with den_i == 1 for plain means.  The SAME weight matrix is used for every
statistic of a given (cat, subset), so method-vs-teacher deltas, ratios and
method-A-vs-method-B deltas are PAIRED by construction.  CI = the 2.5 / 97.5
percentiles of the replicate distribution (null if fewer than 2 finite
replicates).  Per-bucket CIs resample within the bucket (i.e. conditional on the
bucket's request count).  The RNG for a subset is seeded from
(--boot_seed, sha256(cat|subset|n|sha1(index array))), so results are
reproducible and unchanged by --methods / --cats filtering.
--boot 0 disables the bootstrap (point estimates only, all CIs null).

Multi-seed: every (cat, method) cell is reported PER SEED; when >= 2 seeds exist
the same cell is additionally reported as seed_agg=mean and seed_agg=std (std =
sample std, ddof=1) over the per-seed values.  The grand mean is never reported
alone.  Aggregate rows carry null CIs (CIs are per-seed quantities).

Every division guards its denominator and emits null (JSON) / empty (CSV)
instead of NaN or a silent 0.

================================================================================
OUTPUTS  (--out_dir, both .json and .csv; the .csv is exactly the .json "rows"
array flattened, same field names; JSON also carries a "meta" block)
================================================================================
  reference_agreement.{json,csv}      per cat/method/seed/K: jac, overlap (+CIs),
                                      short-list counts, list-length / s_raw /
                                      s_invalid summaries, popgap10=null
  quality_buckets.{json,csv}          per cat/method/seed/bucket/K/metric, with
                                      metric in {R, N} and, when a pop cache is
                                      found, popgap: method + teacher absolute
                                      (+CIs), ratio, delta (+CIs), n, hit counts
                                      (popgap rows instead carry *_n_defined and
                                      leave ratio / hit counts null)
  serving_parity_blindspot.{json,csv} section=hit_swap: the (4) decomposition,
                                      NDCG win/loss mass, both identity residuals
                                      section=conditional_agreement: the (5) rows
  pair_deltas.{json,csv}              only when --pairs is given: paired A-B
                                      deltas of R/N (per bucket) and jac/overlap
                                      (overall) with bootstrap CIs on the delta.
                                      For jac the delta is taken over the requests
                                      where BOTH arms are defined, so it equals
                                      value_a_paired - value_b_paired (value_a /
                                      value_b are each arm's own marginal mean).
  run_manifest.{json,csv}             one row per npz consumed: sha256, n, mtime,
                                      scalars, bucket counts, alignment checks
Every row of every table carries cat / method / seed / bucket / K (empty where
the concept does not apply, e.g. bucket and K on manifest rows).

USAGE
  python3 metrics_s48.py --runs_dir runs/ --out_dir runs/s48_deliverables/ \
      [--boot 2000] [--boot_seed 12345] [--methods a,b,c] [--cats c1,c2] \
      [--pairs "A:B,C:D"] [--pop_cache_dir DIR]
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

SCRIPT_VERSION = "metrics_s48.py/1.0"
PREFIX = "lists_s48_"
KS = (5, 10, 20)
BUCKET_CODE = {"cold": 0, "tail": 1, "mid": 2, "head": 3}
BUCKET_ROWS = ("head", "mid", "tail", "cold", "tail_incl_cold", "all")
COND_GROUPS = ("both_hit", "same_rank_hit", "both_miss", "teacher_only", "method_only")
POPGAP_NOTE = "requires served-item popularity, not in this dump"
IDENTITY_TOL = 1e-12
REQUIRED_KEYS = ("gt_item", "gt_pop", "gt_bucket", "t_top", "s_top", "t_n", "s_n")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def safe_div(a, b):
    """a/b, or None when b == 0 (or either side is not finite)."""
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if b == 0.0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    v = a / b
    return v if math.isfinite(v) else None


def point_ratio(num_sub, den_sub, n_sub):
    """Point estimate of (sum num)/(sum den); den_sub None means den == 1 (mean)."""
    if n_sub == 0:
        return None
    num = float(np.sum(num_sub))
    den = float(n_sub) if den_sub is None else float(np.sum(den_sub))
    return safe_div(num, den)


def masked_mean(vals, mask, idx):
    """mean of vals over idx where mask != 0; None if no defined entry."""
    if len(idx) == 0:
        return None
    m = mask[idx]
    tot = float(np.sum(m))
    if tot == 0.0:
        return None
    return float(np.sum(vals[idx] * m)) / tot


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def array_fingerprint(arr):
    a = np.ascontiguousarray(arr, dtype=np.int64)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def f4(v, width=8):
    if v is None:
        return "n/a".rjust(width)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v).rjust(width)
    if not math.isfinite(fv):
        return "n/a".rjust(width)
    return ("%.4f" % fv).rjust(width)


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
class Bootstrap:
    """Collects (numerator, denominator) column specs per (cat, subset) and fills
    <key>, <key>_ci_lo, <key>_ci_hi on the target row dicts.

    The point estimate is computed immediately at add() time (one code path for
    every reported point value); only the CI is deferred to run()."""

    def __init__(self, n_boot, boot_seed, verbose=True):
        self.n_boot = int(n_boot)
        self.boot_seed = int(boot_seed)
        self.verbose = verbose
        self.jobs = OrderedDict()  # (cat, subset) -> {"idx": ndarray, "entries": [...]}

    def add(self, cat, subset, idx, target, key, num, den=None):
        """num/den are FULL-length per-request arrays; idx selects the subset."""
        n_sub = int(len(idx))
        target[key] = point_ratio(num[idx], None if den is None else den[idx], n_sub)
        target[key + "_ci_lo"] = None
        target[key + "_ci_hi"] = None
        if self.n_boot <= 0 or n_sub == 0:
            return
        job = self.jobs.get((cat, subset))
        if job is None:
            job = self.jobs[(cat, subset)] = {"idx": idx, "entries": []}
        elif len(job["idx"]) != n_sub:
            raise ValueError("subset %s/%s registered with two different sizes" % (cat, subset))
        job["entries"].append((target, key, num, den))

    def run(self):
        """Fill the CIs of every column registered so far, then drop the jobs (so
        the caller can release a category's per-request arrays before the next)."""
        if self.n_boot <= 0:
            self.jobs.clear()
            return
        for (cat, subset), job in self.jobs.items():
            idx = job["idx"]
            entries = job["entries"]
            n_sub = int(len(idx))
            m = len(entries)
            if n_sub == 0 or m == 0:
                continue
            t0 = time.time()
            num_mat = np.empty((n_sub, m), dtype=np.float64)
            den_cols = []
            den_list = []
            for j, (_t, _k, num, den) in enumerate(entries):
                num_mat[:, j] = num[idx]
                if den is not None:
                    den_cols.append(j)
                    den_list.append(np.asarray(den[idx], dtype=np.float64))
            den_mat = np.column_stack(den_list) if den_list else None
            den_cols = np.asarray(den_cols, dtype=np.int64)

            reps = np.empty((self.n_boot, m), dtype=np.float64)
            key = "%s|%s|%d|%s" % (cat, subset, n_sub, array_fingerprint(idx))
            rng = np.random.default_rng([self.boot_seed, stable_seed(key)])
            # replicates are drawn in chunks purely to cap memory; the RNG stream is
            # consumed sequentially, so the chunk size does not affect the result.
            chunk = max(1, min(256, int(1_000_000 // max(n_sub, 1))))
            done = 0
            while done < self.n_boot:
                c = min(chunk, self.n_boot - done)
                ridx = rng.integers(0, n_sub, size=(c, n_sub))
                flat = ridx + (np.arange(c, dtype=np.int64) * n_sub)[:, None]
                w = np.bincount(flat.ravel(), minlength=c * n_sub).reshape(c, n_sub)
                w = w.astype(np.float64)
                a = w @ num_mat
                d = np.full((c, m), float(n_sub), dtype=np.float64)
                if den_mat is not None:
                    d[:, den_cols] = w @ den_mat
                out = np.full((c, m), np.nan, dtype=np.float64)
                nz = d != 0.0
                np.divide(a, d, out=out, where=nz)
                reps[done:done + c, :] = out
                done += c

            for j, (target, k, _num, _den) in enumerate(entries):
                col = reps[:, j]
                col = col[np.isfinite(col)]
                if col.size >= 2:
                    lo, hi = np.percentile(col, [2.5, 97.5])
                    target[k + "_ci_lo"] = float(lo)
                    target[k + "_ci_hi"] = float(hi)
            if self.verbose:
                print("[boot] %-28s subset=%-15s n=%-7d cols=%-5d %.1fs"
                      % (cat, subset, n_sub, m, time.time() - t0), flush=True)
        self.jobs.clear()


# --------------------------------------------------------------------------- #
# loading + per-request derivation
# --------------------------------------------------------------------------- #
def parse_filename(path):
    base = os.path.basename(path)
    if not base.startswith(PREFIX) or not base.endswith(".npz"):
        return None
    stem = base[len(PREFIX):-len(".npz")]
    parts = stem.split("__")
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def load_dump(path, warns):
    cat, method, seed = parse_filename(path)
    with np.load(path, allow_pickle=False) as z:
        keys = sorted(z.files)
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise ValueError("%s: missing arrays %s" % (path, missing))
        d = {
            "path": os.path.abspath(path),
            "filename": os.path.basename(path),
            "cat": cat, "method": method, "seed": seed, "npz_keys": keys,
            "gt_item": np.asarray(z["gt_item"]).astype(np.int64).ravel(),
            "gt_pop": np.asarray(z["gt_pop"]).astype(np.int64).ravel(),
            "gt_bucket": np.asarray(z["gt_bucket"]).astype(np.int64).ravel(),
            "t_top": np.asarray(z["t_top"]).astype(np.int64),
            "s_top": np.asarray(z["s_top"]).astype(np.int64),
            "t_n": np.asarray(z["t_n"]).astype(np.int64).ravel(),
            "s_n": np.asarray(z["s_n"]).astype(np.int64).ravel(),
        }
        for opt in ("s_raw", "s_invalid"):
            d[opt] = np.asarray(z[opt]).astype(np.int64).ravel() if opt in z.files else None
        for sc in ("bucket_lo", "bucket_hi", "n_test"):
            d[sc] = int(np.asarray(z[sc]).ravel()[0]) if sc in z.files else None
    n = int(d["gt_item"].shape[0])
    d["n"] = n
    for k in ("gt_pop", "gt_bucket", "t_n", "s_n"):
        if d[k].shape[0] != n:
            raise ValueError("%s: %s has length %d, expected %d" % (path, k, d[k].shape[0], n))
    for k in ("t_top", "s_top"):
        if d[k].ndim != 2 or d[k].shape[0] != n:
            raise ValueError("%s: %s has shape %s, expected (%d, W)" % (path, k, d[k].shape, n))
    if d["n_test"] is not None and d["n_test"] != n:
        warns.append("%s: scalar n_test=%d != row count %d" % (d["filename"], d["n_test"], n))
    if int(np.sum(d["gt_item"] < 0)) > 0:
        warns.append("%s: %d rows have gt_item < 0 (they can never be hit)"
                     % (d["filename"], int(np.sum(d["gt_item"] < 0))))
    d["size_bytes"] = os.path.getsize(path)
    d["mtime_epoch"] = os.path.getmtime(path)
    d["mtime_iso"] = datetime.fromtimestamp(d["mtime_epoch"], tz=timezone.utc).isoformat()
    d["sha256"] = sha256_file(path)
    return d


def load_pop_cache(pop_dir, cat, warns):
    """<pop_dir>/pop_<CAT>.npz -> {items, counts, lo, hi, ...} or None if absent.
    A missing or malformed cache is never fatal: PopGap is then simply not
    computed for that category."""
    path = os.path.join(pop_dir, "pop_%s.npz" % cat)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if "items" not in z.files or "counts" not in z.files:
                warns.append("%s: no items/counts arrays -- PopGap skipped for %s"
                             % (os.path.basename(path), cat))
                return None
            items = np.asarray(z["items"]).astype(np.int64).ravel()
            counts = np.asarray(z["counts"]).astype(np.int64).ravel()
            lo = int(np.asarray(z["lo"]).ravel()[0]) if "lo" in z.files else None
            hi = int(np.asarray(z["hi"]).ravel()[0]) if "hi" in z.files else None
    except Exception as exc:                                    # unreadable sidecar
        warns.append("%s: unreadable (%s) -- PopGap skipped for %s"
                     % (os.path.basename(path), exc, cat))
        return None
    if items.shape != counts.shape:
        warns.append("%s: items %s vs counts %s -- PopGap skipped for %s"
                     % (os.path.basename(path), items.shape, counts.shape, cat))
        return None
    if items.size and np.any(np.diff(items) < 0):
        order = np.argsort(items, kind="stable")
        items = items[order]
        counts = counts[order]
        warns.append("%s: items were not sorted; sorted them before the lookup"
                     % os.path.basename(path))
    if items.size and int(np.sum(np.diff(items) == 0)):
        warns.append("%s: %d duplicate item ids; the first count of each wins"
                     % (os.path.basename(path), int(np.sum(np.diff(items) == 0))))
    return {"path": os.path.abspath(path), "filename": os.path.basename(path),
            "sha256": sha256_file(path), "items": items, "counts": counts,
            "lo": lo, "hi": hi, "n_items": int(items.size)}


def item_pop_lookup(ids, pop):
    """Train count of every id in `ids` (any shape), via sorted search.
    -1 padding -> 0; an id absent from the cache -> 0 (a cold item, kept)."""
    items = pop["items"]
    if items.size == 0:
        return np.zeros(np.shape(ids), dtype=np.float64)
    pos = np.searchsorted(items, ids)
    np.clip(pos, 0, items.size - 1, out=pos)
    hit = (items[pos] == ids) & (ids != -1)
    return np.where(hit, pop["counts"][pos], 0).astype(np.float64)


def compute_popgap(der, pop):
    """der['popgap'][side][K] = per-request PopGap@K (0 where undefined) and
    der['popgap'][side+'_def'][K] = 1.0 where it is defined."""
    dump = der["dump"]
    gt_pop = dump["gt_pop"].astype(np.float64)
    out = {"m": {}, "m_def": {}, "t": {}, "t_def": {}}
    for side, top in (("m", dump["s_top"]), ("t", dump["t_top"])):
        item_pop = item_pop_lookup(top, pop)
        valid = top != -1
        cum = np.cumsum(valid, axis=1)          # rank among REAL entries
        for K in KS:
            in_k = valid & (cum <= K)           # == the first K real entries
            cnt = in_k.sum(axis=1).astype(np.float64)
            tot = (item_pop * in_k).sum(axis=1)
            defined = cnt > 0
            mean_pop = np.divide(tot, cnt, out=np.zeros_like(tot), where=defined)
            out[side][K] = np.where(defined, mean_pop - gt_pop, 0.0)
            out[side + "_def"][K] = defined.astype(np.float64)
    der["popgap"] = out
    der["popgap_note"] = ("served-item popularity from %s (%d items)"
                          % (pop["filename"], pop["n_items"]))


def derive(dump, warns):
    """Per-request arrays. -1 is padding and is stripped before anything else."""
    n = dump["n"]
    gt = dump["gt_item"]
    s_rows = dump["s_top"].tolist()
    t_rows = dump["t_top"].tolist()

    s_len = np.zeros(n, dtype=np.int64)
    t_len = np.zeros(n, dtype=np.int64)
    s_rank = np.zeros(n, dtype=np.int64)
    t_rank = np.zeros(n, dtype=np.int64)
    jac = {K: np.zeros(n, dtype=np.float64) for K in KS}
    jac_def = {K: np.zeros(n, dtype=np.float64) for K in KS}
    ovl = {K: np.zeros(n, dtype=np.float64) for K in KS}
    dup_m = 0
    dup_t = 0

    for i in range(n):
        srow = [x for x in s_rows[i] if x != -1]
        trow = [x for x in t_rows[i] if x != -1]
        s_len[i] = len(srow)
        t_len[i] = len(trow)
        if len(set(srow)) != len(srow):
            dup_m += 1
        if len(set(trow)) != len(trow):
            dup_t += 1
        g = int(gt[i])
        if g != -1:
            if g in srow:
                s_rank[i] = srow.index(g) + 1
            if g in trow:
                t_rank[i] = trow.index(g) + 1
        for K in KS:
            sm = set(srow[:K])
            tm = set(trow[:K])
            inter = len(sm & tm)
            union = len(sm | tm)
            ovl[K][i] = inter / float(K)
            if union > 0:
                jac[K][i] = inter / float(union)
                jac_def[K][i] = 1.0

    if dup_m:
        warns.append("%s: %d rows have duplicate ids in the method list (contract says deduplicated)"
                     % (dump["filename"], dup_m))
    if dup_t:
        warns.append("%s: %d rows have duplicate ids in the teacher list (contract says deduplicated)"
                     % (dump["filename"], dup_t))
    mism_s = int(np.sum(s_len != dump["s_n"]))
    mism_t = int(np.sum(t_len != dump["t_n"]))
    if mism_s:
        warns.append("%s: %d rows where #real(s_top) != s_n (using the arrays, not s_n)"
                     % (dump["filename"], mism_s))
    if mism_t:
        warns.append("%s: %d rows where #real(t_top) != t_n (using the arrays, not t_n)"
                     % (dump["filename"], mism_t))

    der = {
        "dump": dump, "n": n,
        "s_len": s_len, "t_len": t_len, "s_rank": s_rank, "t_rank": t_rank,
        "jac": jac, "jac_def": jac_def, "ovl": ovl,
        "dup_m": dup_m, "dup_t": dup_t, "mism_s": mism_s, "mism_t": mism_t,
        "r_m": {}, "r_t": {}, "g_m": {}, "g_t": {},
        "common_hit": {}, "teacher_only": {}, "method_only": {}, "common_miss": {},
        "gain_m_excess": {}, "gain_t_excess": {},
    }
    s_safe = np.maximum(s_rank, 1).astype(np.float64)
    t_safe = np.maximum(t_rank, 1).astype(np.float64)
    for K in KS:
        hm = ((s_rank > 0) & (s_rank <= K))
        ht = ((t_rank > 0) & (t_rank <= K))
        r_m = hm.astype(np.float64)
        r_t = ht.astype(np.float64)
        g_m = np.where(hm, 1.0 / np.log2(s_safe + 1.0), 0.0)
        g_t = np.where(ht, 1.0 / np.log2(t_safe + 1.0), 0.0)
        der["r_m"][K] = r_m
        der["r_t"][K] = r_t
        der["g_m"][K] = g_m
        der["g_t"][K] = g_t
        der["common_hit"][K] = r_m * r_t
        der["teacher_only"][K] = r_t * (1.0 - r_m)
        der["method_only"][K] = r_m * (1.0 - r_t)
        der["common_miss"][K] = (1.0 - r_m) * (1.0 - r_t)
        der["gain_m_excess"][K] = np.maximum(g_m - g_t, 0.0)
        der["gain_t_excess"][K] = np.maximum(g_t - g_m, 0.0)
    return der


# --------------------------------------------------------------------------- #
# row builders
# --------------------------------------------------------------------------- #
def ident(dump, bucket, K):
    return OrderedDict([
        ("cat", dump["cat"]), ("method", dump["method"]), ("seed", dump["seed"]),
        ("seed_agg", "per_seed"), ("n_seeds", 1), ("bucket", bucket), ("K", K),
    ])


def build_reference_agreement(der, boot, idx_all):
    dump = der["dump"]
    cat = dump["cat"]
    n = der["n"]
    rows = []
    s_raw = dump["s_raw"]
    s_inv = dump["s_invalid"]
    for K in KS:
        r = ident(dump, "all", K)
        r["n"] = n
        boot.add(cat, "all", idx_all, r, "jac", der["jac"][K], der["jac_def"][K])
        r["jac_n_defined"] = int(np.sum(der["jac_def"][K]))
        boot.add(cat, "all", idx_all, r, "overlap", der["ovl"][K])
        short_m = int(np.sum(der["s_len"] < K))
        short_t = int(np.sum(der["t_len"] < K))
        r["short_list_count"] = int(np.sum((der["s_len"] < K) | (der["t_len"] < K)))
        r["short_list_method"] = short_m
        r["short_list_teacher"] = short_t
        r["mean_list_len_method_at_k"] = safe_div(float(np.sum(np.minimum(der["s_len"], K))), n)
        r["mean_list_len_teacher_at_k"] = safe_div(float(np.sum(np.minimum(der["t_len"], K))), n)
        r["mean_s_n"] = safe_div(float(np.sum(der["s_len"])), n)
        r["mean_t_n"] = safe_div(float(np.sum(der["t_len"])), n)
        r["mean_s_raw"] = safe_div(float(np.sum(s_raw)), n) if s_raw is not None else None
        r["mean_s_invalid"] = safe_div(float(np.sum(s_inv)), n) if s_inv is not None else None
        if s_raw is not None and s_inv is not None:
            r["mean_dedup_dropped"] = safe_div(float(np.sum(s_raw - s_inv - der["s_len"])), n)
            r["invalid_rate"] = safe_div(float(np.sum(s_inv)), float(np.sum(s_raw)))
            r["rows_with_invalid"] = int(np.sum(s_inv > 0))
        else:
            r["mean_dedup_dropped"] = None
            r["invalid_rate"] = None
            r["rows_with_invalid"] = None
        pg = der.get("popgap")
        r["popgap10"] = (point_ratio(pg["m"][10][idx_all], pg["m_def"][10][idx_all], n)
                         if pg is not None else None)
        r["popgap_note"] = der.get("popgap_note", POPGAP_NOTE)
        rows.append(r)
    return rows


def build_quality_buckets(der, boot, bucket_idx):
    dump = der["dump"]
    cat = dump["cat"]
    rows = []
    arrs = {}
    for K in KS:
        for metric, m_arr, t_arr in (("R", der["r_m"][K], der["r_t"][K]),
                                     ("N", der["g_m"][K], der["g_t"][K])):
            arrs[(K, metric)] = (m_arr, t_arr, m_arr - t_arr)   # once, reused by every bucket
    pg = der.get("popgap")
    if pg is not None:
        for K in KS:
            both = pg["m_def"][K] * pg["t_def"][K]
            arrs[(K, "popgap")] = (both, (pg["m"][K] - pg["t"][K]) * both)
    for bname in BUCKET_ROWS:
        idx = bucket_idx[bname]
        for K in KS:
            hits_m = int(np.sum(der["r_m"][K][idx])) if len(idx) else 0
            hits_t = int(np.sum(der["r_t"][K][idx])) if len(idx) else 0
            for metric in ("R", "N"):
                m_arr, t_arr, d_arr = arrs[(K, metric)]
                r = ident(dump, bname, K)
                r["metric"] = metric
                r["n"] = int(len(idx))
                boot.add(cat, bname, idx, r, "method_value", m_arr)
                boot.add(cat, bname, idx, r, "teacher_value", t_arr)
                boot.add(cat, bname, idx, r, "ratio_to_teacher", m_arr, t_arr)
                boot.add(cat, bname, idx, r, "delta", d_arr)
                r["method_hit_count"] = hits_m
                r["teacher_hit_count"] = hits_t
                rows.append(r)
            if pg is not None:
                both, d_arr = arrs[(K, "popgap")]
                r = ident(dump, bname, K)
                r["metric"] = "popgap"
                r["n"] = int(len(idx))
                boot.add(cat, bname, idx, r, "method_value", pg["m"][K], pg["m_def"][K])
                boot.add(cat, bname, idx, r, "teacher_value", pg["t"][K], pg["t_def"][K])
                r["ratio_to_teacher"] = None     # signed difference: a ratio is not meaningful
                boot.add(cat, bname, idx, r, "delta", d_arr, both)
                r["method_hit_count"] = None
                r["teacher_hit_count"] = None
                r["method_n_defined"] = int(np.sum(pg["m_def"][K][idx])) if len(idx) else 0
                r["teacher_n_defined"] = int(np.sum(pg["t_def"][K][idx])) if len(idx) else 0
                r["delta_n_defined"] = int(np.sum(both[idx])) if len(idx) else 0
                rows.append(r)
    return rows


def build_hit_swap(der, boot, bucket_idx, identity_checks):
    dump = der["dump"]
    cat = dump["cat"]
    rows = []
    for bname in BUCKET_ROWS:
        idx = bucket_idx[bname]
        nb = int(len(idx))
        for K in KS:
            r = ident(dump, bname, K)
            r["section"] = "hit_swap"
            r["group"] = ""
            r["group_k"] = ""
            r["n"] = nb
            for name in ("common_hit", "teacher_only", "method_only", "common_miss"):
                r[name] = int(np.sum(der[name][K][idx])) if nb else 0
                boot.add(cat, bname, idx, r, "p_" + name, der[name][K])
            r["recall_method"] = point_ratio(der["r_m"][K][idx], None, nb)
            r["recall_teacher"] = point_ratio(der["r_t"][K][idx], None, nb)
            r["ndcg_method"] = point_ratio(der["g_m"][K][idx], None, nb)
            r["ndcg_teacher"] = point_ratio(der["g_t"][K][idx], None, nb)
            boot.add(cat, bname, idx, r, "ndcg_mass_method", der["gain_m_excess"][K])
            boot.add(cat, bname, idx, r, "ndcg_mass_teacher", der["gain_t_excess"][K])

            swap = record_identity(
                identity_checks, "hit_swap", dump, bname, K,
                r["recall_method"], r["recall_teacher"], r["p_method_only"], r["p_teacher_only"])
            r["swap_identity_lhs"], r["swap_identity_rhs"] = swap[0], swap[1]
            r["swap_identity_residual"], r["swap_identity_ok"] = swap[2], swap[3]
            nd = record_identity(
                identity_checks, "ndcg_mass", dump, bname, K,
                r["ndcg_method"], r["ndcg_teacher"], r["ndcg_mass_method"], r["ndcg_mass_teacher"])
            r["ndcg_identity_lhs"], r["ndcg_identity_rhs"] = nd[0], nd[1]
            r["ndcg_identity_residual"], r["ndcg_identity_ok"] = nd[2], nd[3]
            rows.append(r)
    return rows


def record_identity(store, name, dump, bucket, K, a_m, a_t, b_m, b_t):
    """Check (a_m - a_t) == (b_m - b_t); returns (lhs, rhs, residual, ok)."""
    if a_m is None or a_t is None or b_m is None or b_t is None:
        return None, None, None, None
    lhs = float(a_m) - float(a_t)
    rhs = float(b_m) - float(b_t)
    res = lhs - rhs
    ok = bool(abs(res) <= IDENTITY_TOL)
    store.append(OrderedDict([
        ("identity", name), ("cat", dump["cat"]), ("method", dump["method"]),
        ("seed", dump["seed"]), ("bucket", bucket), ("K", K),
        ("lhs", lhs), ("rhs", rhs), ("residual", res), ("ok", ok),
    ]))
    return lhs, rhs, res, ok


def build_conditional_agreement(der):
    dump = der["dump"]
    rows = []
    s_rank = der["s_rank"]
    t_rank = der["t_rank"]
    for gk in KS:
        hm = der["r_m"][gk] > 0
        ht = der["r_t"][gk] > 0
        masks = OrderedDict([
            ("both_hit", hm & ht),
            ("same_rank_hit", hm & ht & (s_rank == t_rank)),
            ("both_miss", (~hm) & (~ht)),
            ("teacher_only", ht & (~hm)),
            ("method_only", hm & (~ht)),
        ])
        for gname in COND_GROUPS:
            gidx = np.flatnonzero(masks[gname])
            if gname in ("both_hit", "same_rank_hit") and len(gidx):
                delta = (s_rank[gidx] - t_rank[gidx]).astype(np.int64)
                uniq, cnt = np.unique(delta, return_counts=True)
                hist = "|".join("%d:%d" % (int(a), int(b)) for a, b in zip(uniq, cnt))
                d_mean = float(np.mean(delta))
                d_std = float(np.std(delta, ddof=1)) if len(delta) >= 2 else None
                d_med = float(np.median(delta))
                d_neg = int(np.sum(delta < 0))
                d_zero = int(np.sum(delta == 0))
                d_pos = int(np.sum(delta > 0))
            else:
                hist = ""
                d_mean = d_std = d_med = None
                d_neg = d_zero = d_pos = None
            for K in KS:
                r = ident(dump, "all", K)
                r["section"] = "conditional_agreement"
                r["group"] = gname
                r["group_k"] = gk
                r["n"] = int(len(gidx))
                r["jac"] = masked_mean(der["jac"][K], der["jac_def"][K], gidx)
                r["jac_n_defined"] = int(np.sum(der["jac_def"][K][gidx])) if len(gidx) else 0
                r["overlap"] = point_ratio(der["ovl"][K][gidx], None, len(gidx))
                r["rank_delta_mean"] = d_mean
                r["rank_delta_std"] = d_std
                r["rank_delta_median"] = d_med
                r["rank_delta_neg"] = d_neg
                r["rank_delta_zero"] = d_zero
                r["rank_delta_pos"] = d_pos
                r["rank_delta_hist"] = hist
                rows.append(r)
    return rows


def build_pair_deltas(cat, a_name, b_name, der_a, der_b, boot, bucket_idx):
    seed = der_a["dump"]["seed"]
    rows = []

    def base(bucket, K, metric):
        r = OrderedDict([
            ("cat", cat), ("method", "%s-vs-%s" % (a_name, b_name)),
            ("method_a", a_name), ("method_b", b_name), ("seed", seed),
            ("seed_agg", "per_seed"), ("n_seeds", 1),
            ("bucket", bucket), ("K", K), ("metric", metric),
        ])
        return r

    arrs = {}
    for K in KS:
        for metric, arr_a, arr_b in (("R", der_a["r_m"][K], der_b["r_m"][K]),
                                     ("N", der_a["g_m"][K], der_b["g_m"][K])):
            arrs[(K, metric)] = (arr_a, arr_b, arr_a - arr_b)   # once, reused by every bucket
    for bname in BUCKET_ROWS:
        idx = bucket_idx[bname]
        for K in KS:
            for metric in ("R", "N"):
                arr_a, arr_b, d_arr = arrs[(K, metric)]
                r = base(bname, K, metric)
                r["n"] = int(len(idx))
                r["value_a"] = point_ratio(arr_a[idx], None, len(idx))
                r["value_b"] = point_ratio(arr_b[idx], None, len(idx))
                boot.add(cat, bname, idx, r, "delta", d_arr)
                rows.append(r)
    idx = bucket_idx["all"]
    for K in KS:
        for metric, arr_a, arr_b, def_a, def_b in (
                ("jac", der_a["jac"][K], der_b["jac"][K], der_a["jac_def"][K], der_b["jac_def"][K]),
                ("overlap", der_a["ovl"][K], der_b["ovl"][K], None, None)):
            r = base("all", K, metric)
            r["n"] = int(len(idx))
            if def_a is None:
                r["value_a"] = point_ratio(arr_a[idx], None, len(idx))
                r["value_b"] = point_ratio(arr_b[idx], None, len(idx))
                boot.add(cat, "all", idx, r, "delta", arr_a - arr_b)
            else:
                r["value_a"] = masked_mean(arr_a, def_a, idx)
                r["value_b"] = masked_mean(arr_b, def_b, idx)
                # both-defined subset keeps the delta paired and well-defined
                both = def_a * def_b
                r["value_a_paired"] = masked_mean(arr_a, both, idx)
                r["value_b_paired"] = masked_mean(arr_b, both, idx)
                boot.add(cat, "all", idx, r, "delta", (arr_a - arr_b) * both, both)
            rows.append(r)
    return rows


def build_manifest_row(dump, der, ref, pop):
    gb = dump["gt_bucket"]
    r = OrderedDict([
        ("cat", dump["cat"]), ("method", dump["method"]), ("seed", dump["seed"]),
        ("bucket", ""), ("K", ""),
        ("filename", dump["filename"]), ("path", dump["path"]),
        ("sha256", dump["sha256"]), ("size_bytes", int(dump["size_bytes"])),
        ("n_rows", int(dump["n"])), ("n_test", dump["n_test"]),
        ("bucket_lo", dump["bucket_lo"]), ("bucket_hi", dump["bucket_hi"]),
        ("mtime_epoch", float(dump["mtime_epoch"])), ("mtime_iso", dump["mtime_iso"]),
        ("npz_keys", "|".join(dump["npz_keys"])),
        ("list_width", int(dump["s_top"].shape[1])),
    ])
    for bname in ("cold", "tail", "mid", "head"):
        r["n_" + bname] = int(np.sum(gb == BUCKET_CODE[bname]))
    r["n_bucket_other"] = int(np.sum(~np.isin(gb, [0, 1, 2, 3])))
    r["rows_len_mismatch_method"] = int(der["mism_s"])
    r["rows_len_mismatch_teacher"] = int(der["mism_t"])
    r["rows_dup_method"] = int(der["dup_m"])
    r["rows_dup_teacher"] = int(der["dup_t"])
    r["ref_file"] = ref["filename"]
    same_n = ref["n"] == dump["n"]
    r["gt_item_matches_ref"] = bool(same_n and np.array_equal(ref["gt_item"], dump["gt_item"]))
    r["gt_pop_matches_ref"] = bool(same_n and np.array_equal(ref["gt_pop"], dump["gt_pop"]))
    r["gt_bucket_matches_ref"] = bool(same_n and np.array_equal(ref["gt_bucket"], dump["gt_bucket"]))
    r["t_top_matches_ref"] = bool(same_n and ref["t_top"].shape == dump["t_top"].shape
                                  and np.array_equal(ref["t_top"], dump["t_top"]))
    if pop is not None:     # only when a pop cache was actually used, so that the
        # no-cache output stays byte-identical to the pre-PopGap version
        r["pop_cache_file"] = pop["filename"]
        r["pop_cache_sha256"] = pop["sha256"]
        r["pop_cache_n_items"] = pop["n_items"]
        r["pop_cache_lo"] = pop["lo"]
        r["pop_cache_hi"] = pop["hi"]
        r["pop_cache_cuts_match_dump"] = bool(pop["lo"] == dump["bucket_lo"]
                                              and pop["hi"] == dump["bucket_hi"])
        r["rows_gt_pop_ne_cache"] = int(np.sum(
            item_pop_lookup(dump["gt_item"], pop) != dump["gt_pop"].astype(np.float64)))
    return r


# --------------------------------------------------------------------------- #
# multi-seed aggregation
# --------------------------------------------------------------------------- #
def is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def non_aggregatable(field):
    return (field.endswith("_ci_lo") or field.endswith("_ci_hi")
            or field.endswith("_ok") or field.endswith("_note")
            or field.endswith("_hist") or field in ("seed", "seed_agg", "n_seeds"))


def merge_with_seed_aggregates(rows, key_fields):
    """Reorder rows so each (key_fields) group is contiguous, then append that
    group's mean/std-across-seeds rows (only when >= 2 seeds)."""
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(tuple(r[f] for f in key_fields), []).append(r)
    out = []
    for key, rs in groups.items():
        out.extend(rs)
        seeds = sorted({r["seed"] for r in rs})
        if len(seeds) < 2:
            continue
        fields = list(rs[0].keys())
        for stat in ("mean", "std"):
            agg = OrderedDict()
            for f in fields:
                if f in key_fields:
                    agg[f] = rs[0][f]
                elif f == "seed":
                    agg[f] = "*"
                elif f == "seed_agg":
                    agg[f] = stat
                elif f == "n_seeds":
                    agg[f] = len(seeds)
                elif non_aggregatable(f):
                    agg[f] = None
                else:
                    vals = [float(r[f]) for r in rs
                            if is_number(r.get(f)) and math.isfinite(float(r[f]))]
                    if not vals or (stat == "std" and len(vals) < 2):
                        agg[f] = None
                    elif stat == "mean":
                        agg[f] = sum(vals) / len(vals)
                    else:
                        mu = sum(vals) / len(vals)
                        agg[f] = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
            out.append(agg)
    return out


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % (type(o),))


def clean_value(v):
    """NaN/inf -> None; numpy scalars -> python scalars. Never emit NaN."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def csv_value(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        if v == int(v) and abs(v) < 1e15:
            return "%d" % int(v)
        return format(v, ".12g")
    if isinstance(v, int):
        return str(v)
    return str(v)


def write_table(out_dir, name, rows, meta):
    rows = [OrderedDict((k, clean_value(v)) for k, v in r.items()) for r in rows]
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    # every row carries every field of the table (null where it does not apply), so
    # the json rows and the csv columns are the same flat schema
    rows = [OrderedDict((k, r.get(k)) for k in fields) for r in rows]
    jpath = os.path.join(out_dir, name + ".json")
    cpath = os.path.join(out_dir, name + ".csv")
    with open(jpath, "w") as fh:
        json.dump({"meta": meta, "fields": fields, "rows": rows}, fh,
                  indent=2, default=json_default, allow_nan=False)
        fh.write("\n")
    with open(cpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([csv_value(r.get(f)) for f in fields])
    print("[write] %-32s %5d rows -> %s{.json,.csv}" % (name, len(rows), os.path.join(out_dir, name)))
    return jpath, cpath


# --------------------------------------------------------------------------- #
# stdout summary
# --------------------------------------------------------------------------- #
def print_summary(ref_rows, qual_rows, swap_rows, identity_checks, warns):
    print("")
    print("=" * 118)
    print("SUMMARY (bucket=all, K=10; 'ci' = request-level paired bootstrap 95% CI; per-seed rows plus mean/std)")
    print("=" * 118)
    ref = {(r["cat"], r["method"], r["seed"], r["seed_agg"]): r for r in ref_rows if r["K"] == 10}
    swap = {(r["cat"], r["method"], r["seed"], r["seed_agg"]): r
            for r in swap_rows if r.get("section") == "hit_swap" and r["bucket"] == "all" and r["K"] == 10}
    qual = {}
    for r in qual_rows:
        if r["bucket"] == "all" and r["K"] == 10:
            qual[(r["cat"], r["method"], r["seed"], r["seed_agg"], r["metric"])] = r
    hdr = ("%-26s %-14s %-7s %6s %8s %8s %8s %8s %8s %8s %7s %7s %10s"
           % ("cat", "method", "seed", "n", "R@10", "T_R@10", "ratio", "N@10", "jac@10", "ovl@10",
              "m_only", "t_only", "popgap@10"))
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(qual.keys(), key=lambda k: (k[0], k[1], {"per_seed": 0, "mean": 1, "std": 2}[k[3]], k[2])):
        if key[4] != "R":
            continue
        cat, method, seed, agg, _ = key
        rq = qual[key]
        rn = qual.get((cat, method, seed, agg, "N"))
        rr = ref.get((cat, method, seed, agg))
        rs = swap.get((cat, method, seed, agg))
        label = seed if agg == "per_seed" else agg.upper()
        print("%-26s %-14s %-7s %6s %s %s %s %s %s %s %7s %7s %s"
              % (cat[:26], method[:14], label[:7],
                 ("%d" % rq["n"]) if rq.get("n") is not None else "n/a",
                 f4(rq.get("method_value")), f4(rq.get("teacher_value")),
                 f4(rq.get("ratio_to_teacher")), f4(rn.get("method_value") if rn else None),
                 f4(rr.get("jac") if rr else None), f4(rr.get("overlap") if rr else None),
                 ("%d" % rs["method_only"]) if rs and rs.get("method_only") is not None else "n/a",
                 ("%d" % rs["teacher_only"]) if rs and rs.get("teacher_only") is not None else "n/a",
                 f4(rr.get("popgap10") if rr else None, width=10)))
    bad = [c for c in identity_checks if not c["ok"]]
    worst = max((abs(c["residual"]) for c in identity_checks), default=0.0)
    print("")
    print("identity checks: %d run, %d failed, max |residual| = %.3e (tolerance %.0e)"
          % (len(identity_checks), len(bad), worst, IDENTITY_TOL))
    for c in bad[:10]:
        print("  FAIL %s %s/%s/%s bucket=%s K=%s lhs=%.17g rhs=%.17g residual=%.3e"
              % (c["identity"], c["cat"], c["method"], c["seed"], c["bucket"], c["K"],
                 c["lhs"], c["rhs"], c["residual"]))
    n_pg = sum(1 for r in ref_rows if r["K"] == 10 and r.get("popgap10") is not None)
    if n_pg:
        print("PopGap: computed for %d (cat,method,seed) cells -- method AND teacher, per bucket, "
              "in quality_buckets metric=popgap" % n_pg)
    else:
        print("PopGap@10: not computed (%s)" % POPGAP_NOTE)
    if warns:
        print("")
        print("WARNINGS (%d):" % len(warns))
        for w in warns:
            print("  ! %s" % w)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="metrics_s48.py",
        description="Offline metrics for the s48 served-list deliverable (see module docstring).")
    ap.add_argument("--runs_dir", default="runs/")
    ap.add_argument("--out_dir", default="runs/s48_deliverables/")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (0 disables CIs)")
    ap.add_argument("--boot_seed", type=int, default=12345)
    ap.add_argument("--methods", default="", help="comma-separated method allow-list")
    ap.add_argument("--cats", default="", help="comma-separated category allow-list")
    ap.add_argument("--pairs", default="", help='comma-separated "A:B" method pairs for A-B deltas')
    ap.add_argument("--pop_cache_dir", default=None,
                    help="directory holding pop_<CAT>.npz for PopGap (default: --runs_dir)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    runs_dir = args.runs_dir
    out_dir = args.out_dir
    pop_cache_dir = args.pop_cache_dir if args.pop_cache_dir else runs_dir
    pop_caches = []
    keep_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    keep_cats = {c.strip() for c in args.cats.split(",") if c.strip()}
    pairs = []
    for spec in (p.strip() for p in args.pairs.split(",")):
        if not spec:
            continue
        if spec.count(":") != 1 or not all(spec.split(":")):
            raise SystemExit("bad --pairs entry %r (want A:B)" % spec)
        pairs.append(tuple(spec.split(":")))

    paths = sorted(glob.glob(os.path.join(runs_dir, PREFIX + "*.npz")))
    selected = []
    warns = []
    for p in paths:
        parsed = parse_filename(p)
        if parsed is None:
            warns.append("skipped unparseable filename: %s" % os.path.basename(p))
            continue
        cat, method, seed = parsed
        if keep_cats and cat not in keep_cats:
            continue
        if keep_methods and method not in keep_methods:
            continue
        selected.append(p)
    if not selected:
        raise SystemExit("no dumps matched %s/%s*.npz with the given --cats/--methods"
                         % (runs_dir, PREFIX))
    os.makedirs(out_dir, exist_ok=True)
    print("[scan] %d dump(s) selected from %s" % (len(selected), os.path.abspath(runs_dir)))

    boot = Bootstrap(args.boot, args.boot_seed)
    identity_checks = []
    ref_rows, qual_rows, blind_rows, pair_rows, manifest_rows = [], [], [], [], []

    by_cat = OrderedDict()
    for p in selected:
        cat = parse_filename(p)[0]
        by_cat.setdefault(cat, []).append(p)

    for cat in sorted(by_cat):
        dumps = [load_dump(p, warns) for p in sorted(by_cat[cat])]
        ref = dumps[0]
        n = ref["n"]
        for d in dumps[1:]:
            if d["n"] != n:
                warns.append("ALIGNMENT: %s has n=%d but reference %s has n=%d -- cross-method "
                             "comparisons for %s are NOT valid"
                             % (d["filename"], d["n"], ref["filename"], n, cat))
            elif not np.array_equal(d["gt_item"], ref["gt_item"]):
                warns.append("ALIGNMENT: %s gt_item differs from reference %s -- cross-method "
                             "comparisons for %s are NOT valid" % (d["filename"], ref["filename"], cat))
            elif not np.array_equal(d["gt_bucket"], ref["gt_bucket"]):
                warns.append("ALIGNMENT: %s gt_bucket differs from reference %s -- bucket rows use "
                             "the reference partition" % (d["filename"], ref["filename"]))
        gb = ref["gt_bucket"]
        idx_all = np.arange(n, dtype=np.int64)
        bucket_idx = OrderedDict()
        for bname in ("head", "mid", "tail", "cold"):
            bucket_idx[bname] = np.flatnonzero(gb == BUCKET_CODE[bname]).astype(np.int64)
        bucket_idx["tail_incl_cold"] = np.flatnonzero((gb == 1) | (gb == 0)).astype(np.int64)
        bucket_idx["all"] = idx_all
        covered = sum(len(bucket_idx[b]) for b in ("head", "mid", "tail", "cold"))
        if covered != n:
            warns.append("%s: %d/%d requests have a gt_bucket outside {0,1,2,3}" % (cat, n - covered, n))

        pop = load_pop_cache(pop_cache_dir, cat, warns)
        if pop is not None:
            pop_caches.append(OrderedDict([
                ("cat", cat), ("file", pop["path"]), ("sha256", pop["sha256"]),
                ("n_items", pop["n_items"]), ("lo", pop["lo"]), ("hi", pop["hi"])]))
            print("[pop] %s: %s (%d items, lo=%s hi=%s)"
                  % (cat, pop["filename"], pop["n_items"], pop["lo"], pop["hi"]), flush=True)

        ders = OrderedDict()
        for d in dumps:
            if d["n"] != n:
                warns.append("SKIPPED %s: row count %d != cat reference %d" % (d["filename"], d["n"], n))
                continue
            der = derive(d, warns)
            if pop is not None:
                compute_popgap(der, pop)
            ders[(d["method"], d["seed"])] = der
            manifest_rows.append(build_manifest_row(d, der, ref, pop))
            ref_rows.extend(build_reference_agreement(der, boot, idx_all))
            qual_rows.extend(build_quality_buckets(der, boot, bucket_idx))
            blind_rows.extend(build_hit_swap(der, boot, bucket_idx, identity_checks))
            blind_rows.extend(build_conditional_agreement(der))
            print("[derive] %s / %s / %s  n=%d" % (cat, d["method"], d["seed"], d["n"]), flush=True)

        for a_name, b_name in pairs:
            seeds_a = {s for (m, s) in ders if m == a_name}
            seeds_b = {s for (m, s) in ders if m == b_name}
            shared = sorted(seeds_a & seeds_b)
            if not shared:
                warns.append("pair %s:%s -- no shared seed present for %s (A seeds=%s, B seeds=%s)"
                             % (a_name, b_name, cat, sorted(seeds_a), sorted(seeds_b)))
                continue
            for seed in shared:
                pair_rows.extend(build_pair_deltas(cat, a_name, b_name, ders[(a_name, seed)],
                                                   ders[(b_name, seed)], boot, bucket_idx))

        # bootstrap this category now, then release its per-request arrays
        boot.run()
        ders.clear()

    boot.run()

    ref_rows = merge_with_seed_aggregates(ref_rows, ("cat", "method", "bucket", "K"))
    qual_rows = merge_with_seed_aggregates(qual_rows, ("cat", "method", "bucket", "K", "metric"))
    blind_rows = merge_with_seed_aggregates(
        blind_rows, ("cat", "method", "bucket", "K", "section", "group", "group_k"))
    if pair_rows:
        pair_rows = merge_with_seed_aggregates(
            pair_rows, ("cat", "method", "method_a", "method_b", "bucket", "K", "metric"))

    n_bad = sum(1 for c in identity_checks if not c["ok"])
    meta = OrderedDict([
        ("script", SCRIPT_VERSION),
        ("generated_utc", datetime.now(tz=timezone.utc).isoformat()),
        ("runs_dir", os.path.abspath(runs_dir)),
        ("out_dir", os.path.abspath(out_dir)),
        ("argv", list(sys.argv[1:] if argv is None else argv)),
        ("n_boot", args.boot), ("boot_seed", args.boot_seed),
        ("ci_percentiles", [2.5, 97.5]),
        ("ks", list(KS)),
        ("buckets", list(BUCKET_ROWS)),
        ("n_files", len(manifest_rows)),
        ("popgap10", None),
        ("popgap_note", POPGAP_NOTE if not pop_caches else
         "PopGap@K computed per row; see quality_buckets metric=popgap and "
         "reference_agreement.popgap10"),
        ("identity_checks_run", len(identity_checks)),
        ("identity_checks_failed", n_bad),
        ("identity_tolerance", IDENTITY_TOL),
        ("identity_max_abs_residual", max((abs(c["residual"]) for c in identity_checks), default=0.0)),
        ("numpy_version", np.__version__),
        ("warnings", list(warns)),
    ])
    if pop_caches:      # added only when a cache was used, so the no-cache output
        meta["pop_cache_dir"] = os.path.abspath(pop_cache_dir)   # is byte-identical
        meta["pop_caches"] = pop_caches

    write_table(out_dir, "reference_agreement", ref_rows, meta)
    write_table(out_dir, "quality_buckets", qual_rows, meta)
    blind_meta = OrderedDict(meta)
    blind_meta["identity_checks"] = identity_checks
    write_table(out_dir, "serving_parity_blindspot", blind_rows, blind_meta)
    if pair_rows:
        write_table(out_dir, "pair_deltas", pair_rows, meta)
    write_table(out_dir, "run_manifest", manifest_rows, meta)

    print_summary(ref_rows, qual_rows, blind_rows, identity_checks, warns)
    print("\n[done] %.1fs" % (time.time() - t_start))
    assert n_bad == 0, ("%d identity check(s) exceeded %g -- see serving_parity_blindspot.json "
                        "meta.identity_checks" % (n_bad, IDENTITY_TOL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
