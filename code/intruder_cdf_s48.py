#!/usr/bin/env python3
"""intruder_cdf_s48.py -- doc §2.3 deliverable: the teacher-rank CDF of the student's
Top-10 INTRUDERS, i.e. the empirical justification for the training hyper-parameter
B (the outsider rank upper bound of `boundary_pair_loss`).

================================================================================
DEFINITION
================================================================================
For one test request, with L_M = the METHOD's served top-10 and L_T = the TEACHER's
served top-10 (both item-level: invalid SIDs dropped, deduplicated, rank order kept):

    intruders(request) = { i in L_M : i not in L_T }

and for each intruder its TEACHER RANK = its 1-based position in the teacher's FULL
deduplicated ranking (from the shared cached teacher beam, width 50).  Bands:

    1-10    IMPOSSIBLE BY CONSTRUCTION (rank <= 10 => the item IS in the teacher's
            top-10 => it is not an intruder).  A non-zero count here means `t_top`
            in the npz and `tbeam50_<CAT>.pt` disagree -- a pipeline BUG, not a
            finding.  It is counted, reported, and makes the script exit 2.
    11-20 / 21-30 / 31-40 / 41-50          resolvable teacher ranks
    >50     NOT resolvable from the cached beam (see LIMITATION)

Reported per (cat, method, seed) and pooled over seeds: total intruders, intruders
per request, the band histogram (counts + fractions of all intruders), and the
CUMULATIVE fractions <=20, <=30, <=40, <=50, >50 with request-level bootstrap 95% CIs.
B=40 is justified iff the <=40 cumulative fraction is high.

================================================================================
LIMITATION -- READ BEFORE QUOTING ANY NUMBER (emitted as json["limitation"])
================================================================================
The doc also asks for the bands 51-60, 61-64 and "outside the teacher dump".  The
cached TEST-split teacher beam `runs/tbeam50_<CAT>.pt` is only WIDTH 50, so no rank
beyond 50 is resolvable from it: "rank 51-60", "rank 61-64" and "never appeared in
the teacher's beam at all" are INDISTINGUISHABLE here and collapse into the single
`>50` bucket.  Nothing is interpolated and no 51-60 / 61-64 split is invented.
Resolving 51-64 would require RE-RUNNING the teacher beam at width 64 on the TEST
split.  (The existing width-64 dump `runs/bmdump_AmazonReviews2023_<CAT>.pt` is a
TRAIN-split dump -- different rows, not aligned with these test dumps -- so it
cannot fill the gap.)

================================================================================
TENSOR SPACES (getting this wrong silently yields ZERO rank matches)
================================================================================
  runs/tbeam50_<CAT>.pt['codes']   (N_test, 50, 4) int32 -> TOKEN-ID (vocab) space,
                                   digit offsets already applied; row i == row i of
                                   the npz (both follow the same test DataLoader
                                   order; lists_s48.py slices this very cache with a
                                   running row pointer).
  valid_items_4 (build_valid_tensors) -> keyed in TOKEN-ID space, value = ITEM id.
  npz gt_item / t_top / s_top      -> ITEM-ID space already (lists_s48.py mapped
                                   them through the SAME sid_to_item).
So the only mapping this script performs is teacher quads (TOKEN-ID) -> item ids,
via the verbatim `sid_to_item` / `dedup_items` of code/lists_s48.py.

Two cross-source checks make a space/alignment mistake impossible to miss:
  (a) prefix check: t_top (npz) must equal the first 20 entries of the ranking
      re-derived here from tbeam50, and t_n must equal min(R, 20);
  (b) the 1-10 band above must be empty.
Either failing => outputs are still written (they are the diagnostic) and exit 2.

================================================================================
INPUTS / OUTPUTS
================================================================================
in : <runs_dir>/lists_s48_<CAT>__<METHOD>__<SEED>.npz   (field separator is a DOUBLE
     underscore; CAT may itself contain single underscores)
     arrays used: gt_item (n,), t_top (n,20), s_top (n,20), t_n (n,),
     valid (n,) int8 [optional -- if a dump lacks it the lists_s48 rule
     `gt_item >= 0 and t_n > 0` is recomputed, and which rule was used is recorded]
in : <runs_dir>/tbeam50_<CAT>.pt   dict with 'codes' (N_test, 50, 4) int32
in : the valid-set item mapping, obtained EXACTLY as code/lists_s48.py /
     code/bucket_diag_s34.py do it:
         vs = find_valid_set(dataset_tag, explicit_path=None, lens_path=..., graph_dir=...)
         (..., valid_items_4) = build_valid_tensors(vs, dev)
     No TIGER pipeline, no checkpoint and no GPU are needed (the run is 0-GPU);
     --sem_ids_fmt / --checkpoint_fmt are accepted for CLI parity with the s48 dump
     wave and are recorded in the json for provenance only.
out: <out>.json  full nested result + meta + limitation + warnings + input hashes
out: <out>.csv   flat: one row per cat/method/seed/scope x {summary, band, cumulative}
Nothing is written outside dirname(<out>).

USAGE
  python3 code/intruder_cdf_s48.py --runs_dir runs --cats MI,Sci --methods a,b \
      --graph_dir ./sidmlp-graph \
      --dataset_tag_fmt 'AmazonReviews2023_{cat}' \
      --out runs/s48_deliverables/kb_intruder_cdf [--boot 2000]

Relative path args are resolved against the INVOCATION cwd BEFORE this script
chdirs to $R/repos (the sid_mlp package expects that cwd) -- so `--runs_dir runs`
means `runs/` where you launched, not `$R/repos/runs`.
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

SCRIPT_VERSION = "intruder_cdf_s48.py/1.0"
R = os.environ.get("SERVESID_ROOT", ".")
PREFIX = "lists_s48_"
TEACHER_CACHE_FMT = "tbeam50_{cat}.pt"
RANK_DEPTH = 50            # width of the cached test-split teacher beam
TOPK = 10                  # the doc's "student Top-10"
BANDS = ("1-10", "11-20", "21-30", "31-40", "41-50", ">50")
BAND_HI = (10, 20, 30, 40, 50)                    # upper edge of each resolvable band
CUM_KEYS = ("le20", "le30", "le40", "le50", "gt50")
CUM_LABEL = {"le20": "<=20", "le30": "<=30", "le40": "<=40", "le50": "<=50", "gt50": ">50"}

LIMITATION = (
    "The cached TEST-split teacher beam runs/tbeam50_<CAT>.pt has WIDTH 50, so a teacher rank "
    "beyond 50 is not resolvable from it. The doc's requested bands 51-60, 61-64 and 'outside "
    "the teacher dump' are therefore NOT separable here: they are reported together as the "
    "single '>50' bucket. No 51-60 / 61-64 split is invented or interpolated. Resolving 51-64 "
    "would require re-running the teacher beam at width 64 on the TEST split; the existing "
    "width-64 dump runs/bmdump_AmazonReviews2023_<CAT>.pt is a TRAIN-split dump and is not "
    "row-aligned with these test dumps, so it cannot fill the gap."
)

def configure_bands(rank_depth, cache_fmt, cache_kind):
    """s49: re-derive the module-level band constants for a teacher cache of width
    `rank_depth`.  Called once from main() before any work; every consumer reads these
    as globals at call time, so no other code changes.  rank_depth=50 + the tbeam50 fmt
    reproduces the s48 constants byte for byte."""
    global TEACHER_CACHE_FMT, RANK_DEPTH, BANDS, BAND_HI, CUM_KEYS, CUM_LABEL, LIMITATION
    TEACHER_CACHE_FMT = cache_fmt
    RANK_DEPTH = int(rank_depth)
    edges = [e for e in range(10, RANK_DEPTH + 1, 10)]
    if not edges or edges[-1] != RANK_DEPTH:
        edges.append(RANK_DEPTH)
    BAND_HI = tuple(edges)
    bands, lo = [], 1
    for hi in BAND_HI:
        bands.append("%d-%d" % (lo, hi))
        lo = hi + 1
    bands.append(">%d" % RANK_DEPTH)
    BANDS = tuple(bands)
    CUM_KEYS = tuple(["le%d" % e for e in BAND_HI[1:]] + ["gt%d" % RANK_DEPTH])
    CUM_LABEL = {}
    for e in BAND_HI[1:]:
        CUM_LABEL["le%d" % e] = "<=%d" % e
    CUM_LABEL["gt%d" % RANK_DEPTH] = ">%d" % RANK_DEPTH
    if RANK_DEPTH >= 64:
        LIMITATION = (
            "Teacher ranks are resolved on a WIDTH-%d test-split teacher beam (%s), built by the "
            "same loader / valid set / dedup as every other s48 dump but with num_beams widened "
            "from the deployment 50 to %d. It is width-matched to the M=64 teacher TRAINING dump "
            "whose ranking the DA-CDRS `B` bound indexes, which is why 51-60 / 61-64 / >%d are "
            "separable here. CAVEAT, MEASURED: widening the beam changes beam search itself, so "
            "this ranking is NOT an extension of the width-50 one -- deep ranks reorder (see "
            "`beam_width_agreement` in this file). Ranks 1-10 agree ~100%%, top-10 sets 96-99%%, "
            "but whole top-50 lists agree on only 2-7%% of requests. Read the width-50 table as "
            "the DEPLOYMENT ladder and this one as the TRAINING-dump ladder; do not mix bands "
            "across the two." % (RANK_DEPTH, cache_kind, RANK_DEPTH, RANK_DEPTH))
    else:
        LIMITATION = (
            "The cached TEST-split teacher beam runs/tbeam50_<CAT>.pt has WIDTH 50, so a teacher "
            "rank beyond 50 is not resolvable from it. The doc's requested bands 51-60, 61-64 and "
            "'outside the teacher dump' are therefore NOT separable here: they are reported "
            "together as the single '>50' bucket. No 51-60 / 61-64 split is invented or "
            "interpolated. s49 resolves them in a SEPARATE run against the width-64 test beam "
            "runs/tbeam64_<CAT>.pt (--rank_depth 64); the two ladders are not interchangeable "
            "because beam width changes the deep ranking.")


CSV_FIELDS = ["cat", "method", "seed", "scope", "n_seeds", "kind", "band",
              "n_requests", "n_intruders", "intruders_per_request",
              "count", "fraction", "ci_lo", "ci_hi", "n_boot_finite"]


# --------------------------------------------------------------------------- #
# small helpers (guarded arithmetic -- None, never NaN, ever reaches the output)
# --------------------------------------------------------------------------- #
def safe_div(a, b):
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if b == 0.0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    v = a / b
    return v if math.isfinite(v) else None


def jsonable(v):
    """None stays None; non-finite floats become None."""
    if v is None:
        return None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def parse_filename(path):
    """lists_s48_<CAT>__<METHOD>__<SEED>.npz -> (cat, method, seed); CAT may contain
    single underscores, the field separator is a DOUBLE underscore (== metrics_s48)."""
    base = os.path.basename(path)
    if not base.startswith(PREFIX) or not base.endswith(".npz"):
        return None
    parts = base[len(PREFIX):-len(".npz")].split("__")
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


# --------------------------------------------------------------------------- #
# sid_mlp valid-set mapping -- imported EXACTLY as lists_s48.py / bucket_diag_s34.py do
# --------------------------------------------------------------------------- #
def load_valid_items(dataset_tag, graph_dir, lens_path, device):
    """-> (valid_items_4 dict {TOKEN-ID quad tuple: item id}, provenance dict).

    Same call, same args as code/lists_s48.py; only `valid_items_4` is used, but the
    call is left verbatim so the mapping is bit-identical to the one that produced
    the dumps.  Imports are function-local so --help works without torch."""
    import torch
    from sid_mlp.inference import find_valid_set
    from sid_mlp.beam import build_valid_tensors

    dev = torch.device(device)
    vs = find_valid_set(dataset_tag, explicit_path=None,
                        lens_path=(lens_path or None), graph_dir=graph_dir)
    (_m12, _m123, _dk, _dm, _k4, _v4, _im, valid_items_4) = build_valid_tensors(vs, dev)
    prov = {"dataset_tag": dataset_tag, "graph_dir": graph_dir,
            "lens_path": lens_path or None, "device": device,
            "n_valid_sids": int(len(valid_items_4))}
    return valid_items_4, prov


def load_teacher_codes(path):
    """runs/tbeam50_<CAT>.pt -> (N, W, 4) int64 numpy, TOKEN-ID space.  Loaded the
    same way lists_s48.py loads it."""
    import torch
    codes = torch.load(path, map_location="cpu")["codes"]
    return np.asarray(codes, dtype=np.int64)


def teacher_ranking(codes, valid_items_4, depth):
    """(N, W, 4) TOKEN-ID quads -> (T (N, depth) item ids rank-ordered with -1 pad,
    R (N,) number of real entries).  `sid_to_item` / `dedup_items` are verbatim from
    code/lists_s48.py (drop invalid SIDs, dedup preserving rank order, truncate)."""

    def sid_to_item(tup):
        return valid_items_4.get(tuple(int(x) for x in tup), -1)

    def dedup_items(sid_list, cap):
        seen = set()
        out = []
        ninv = 0
        for tup in sid_list:
            it = sid_to_item(tup)
            if it == -1:
                ninv += 1
                continue
            if it in seen:
                continue
            seen.add(it)
            out.append(it)
            if len(out) >= cap:
                break
        return out, ninv, len(sid_list)

    n = codes.shape[0]
    T = np.full((n, depth), -1, dtype=np.int64)
    Rn = np.zeros(n, dtype=np.int64)
    n_invalid = 0
    rows = codes.tolist()
    for i in range(n):
        items, ninv, _raw = dedup_items(rows[i], depth)
        Rn[i] = len(items)
        if items:
            T[i, :len(items)] = items
        n_invalid += ninv
    return T, Rn, int(n_invalid)


# --------------------------------------------------------------------------- #
# per-request derivation
# --------------------------------------------------------------------------- #
def first_k_real(arr, k):
    """(n, W) with -1 padding -> (n, k) = the first k REAL entries, order preserved,
    -1 padded.  General: tolerates -1 anywhere, not just trailing."""
    n = arr.shape[0]
    out = np.full((n, k), -1, dtype=np.int64)
    real = arr >= 0
    cum = np.cumsum(real, axis=1)
    sel = real & (cum <= k)
    rows, cols = np.nonzero(sel)
    out[rows, cum[rows, cols] - 1] = arr[rows, cols]
    return out


def band_counts(s_top, t_top, T):
    """-> (counts (n, 6) per-band intruder counts, n_intr (n,)).

    Teacher top-10 comes from the npz `t_top` (the very list every other s48 metric
    scores against); the RANK comes from the re-derived tbeam50 ranking `T`.  Using
    the two different sources on purpose is what makes the 1-10 band a real check."""
    S = first_k_real(s_top, TOPK)                       # (n, 10) method top-10
    T10 = first_k_real(t_top, TOPK)                     # (n, 10) teacher top-10
    s_real = S >= 0
    in_t10 = ((S[:, :, None] == T10[:, None, :]) & (T10[:, None, :] >= 0)).any(axis=2)
    intruder = s_real & ~in_t10                         # (n, 10)

    eq = (S[:, :, None] == T[:, None, :]) & (T[:, None, :] >= 0)     # (n, 10, depth)
    found = eq.any(axis=2)
    rank = np.where(found, eq.argmax(axis=2) + 1, 0)    # 0 == not resolvable (> depth)

    counts = np.zeros((s_top.shape[0], len(BANDS)), dtype=np.int64)
    lo = 1
    for j, hi in enumerate(BAND_HI):
        counts[:, j] = (intruder & (rank >= lo) & (rank <= hi)).sum(axis=1)
        lo = hi + 1
    counts[:, len(BAND_HI)] = (intruder & (rank == 0)).sum(axis=1)   # the ">50" bucket
    n_intr = intruder.sum(axis=1).astype(np.int64)
    if not np.array_equal(counts.sum(axis=1), n_intr):
        raise AssertionError("band counts do not partition the intruders")
    return counts, n_intr


# --------------------------------------------------------------------------- #
# request-level bootstrap (resamples REQUESTS: intruders within a request are dependent)
# --------------------------------------------------------------------------- #
def bootstrap_ci(nums, dens, n_boot, rng):
    """nums/dens: (n, k) per-request numerator/denominator columns; every statistic is
    (sum num)/(sum den) so replicate b is a plain weighted ratio.  ONE weight draw is
    shared by every column, so the columns of a cell are mutually paired.
    -> (lo, hi, n_finite) lists of length k, None where undefined."""
    n, k = nums.shape
    if n_boot <= 0 or n == 0:
        return [None] * k, [None] * k, [0] * k
    reps = np.full((n_boot, k), np.nan, dtype=np.float64)
    chunk = int(max(1, min(n_boot, 4_000_000 // max(n, 1))))
    done = 0
    while done < n_boot:
        c = int(min(chunk, n_boot - done))
        idx = rng.integers(0, n, size=(c, n))
        W = np.empty((c, n), dtype=np.float64)
        for r in range(c):
            W[r] = np.bincount(idx[r], minlength=n)
        num = W @ nums
        den = W @ dens
        ok = den > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            reps[done:done + c] = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
        done += c
    lo, hi, cnt = [], [], []
    for j in range(k):
        col = reps[:, j]
        col = col[np.isfinite(col)]
        cnt.append(int(col.size))
        if col.size < 2:
            lo.append(None)
            hi.append(None)
        else:
            lo.append(float(np.percentile(col, 2.5)))
            hi.append(float(np.percentile(col, 97.5)))
    return lo, hi, cnt


def cum_columns(counts, den_req):
    """counts (n, 6) -> (nums (n, 6), dens (n, 6)) for the 5 cumulative fractions plus
    intruders-per-request.  den_req = per-request denominator of the LAST column
    (1 per contributing (request, seed) pair)."""
    nb = len(BAND_HI)                       # resolvable bands; the last column is the ">depth" one
    n_intr = counts.sum(axis=1).astype(np.float64)
    cum = np.cumsum(counts[:, :nb], axis=1).astype(np.float64)      # <=e0, <=e1, ... <=e_{nb-1}
    cols = [cum[:, j] for j in range(1, nb)]                        # one per CUM_KEYS "le*" entry
    cols.append(counts[:, nb].astype(np.float64))                   # the ">depth" entry
    cols.append(n_intr)                                             # intruders-per-request numerator
    nums = np.column_stack(cols)
    dens = np.column_stack([n_intr] * (len(cols) - 1) + [den_req.astype(np.float64)])
    return nums, dens


def make_cell(cat, method, seed, scope, n_seeds, counts, den_req, n_boot, boot_seed):
    """counts (n_valid, 6) already restricted to the analysed requests; den_req (n_valid,)
    counts how many (request, seed) pairs that row contributes."""
    n_rows = int(counts.shape[0])
    tot_band = counts.sum(axis=0).astype(np.int64)
    total = int(tot_band.sum())
    n_req = int(den_req.sum())
    nb = len(BAND_HI)
    cum_tot = np.cumsum(tot_band[:nb])

    cell = {
        "cat": cat, "method": method, "seed": seed, "scope": scope, "n_seeds": int(n_seeds),
        "n_requests": n_req, "n_rows": n_rows, "n_intruders": total,
        "intruders_per_request": jsonable(safe_div(total, n_req)),
        "bands": OrderedDict((BANDS[j], int(tot_band[j])) for j in range(len(BANDS))),
        "band_fractions": OrderedDict(
            (BANDS[j], jsonable(safe_div(int(tot_band[j]), total))) for j in range(len(BANDS))),
        "cumulative": OrderedDict(),
    }
    cum_count = {}
    for j, key in enumerate(CUM_KEYS):
        cum_count[key] = int(cum_tot[j + 1]) if j < nb - 1 else int(tot_band[nb])
    point = {key: safe_div(cum_count[key], total) for key in CUM_KEYS}

    nums, dens = cum_columns(counts, den_req)
    rng = np.random.default_rng(
        [int(boot_seed), stable_seed("%s|%s|%s|%s|%d" % (cat, method, seed, scope, n_rows))])
    lo, hi, nfin = bootstrap_ci(nums, dens, n_boot, rng)
    for j, key in enumerate(CUM_KEYS):
        cell["cumulative"][key] = {
            "label": CUM_LABEL[key], "count": cum_count[key], "fraction": jsonable(point[key]),
            "ci_lo": jsonable(lo[j]), "ci_hi": jsonable(hi[j]), "n_boot_finite": nfin[j]}
    cell["intruders_per_request_ci_lo"] = jsonable(lo[nb])
    cell["intruders_per_request_ci_hi"] = jsonable(hi[nb])
    cell["intruders_per_request_n_boot_finite"] = nfin[nb]
    cell["impossible_band_1_10"] = int(tot_band[0])
    return cell


def cell_rows(cell):
    """cell -> flat csv rows."""
    base = {k: cell[k] for k in ("cat", "method", "seed", "scope", "n_seeds")}
    rows = [dict(base, kind="summary", band="", n_requests=cell["n_requests"],
                 n_intruders=cell["n_intruders"],
                 intruders_per_request=cell["intruders_per_request"],
                 count=cell["n_intruders"], fraction=None,
                 ci_lo=cell["intruders_per_request_ci_lo"],
                 ci_hi=cell["intruders_per_request_ci_hi"],
                 n_boot_finite=cell["intruders_per_request_n_boot_finite"])]
    for b in BANDS:
        rows.append(dict(base, kind="band", band=b, n_requests=cell["n_requests"],
                         n_intruders=cell["n_intruders"],
                         intruders_per_request=cell["intruders_per_request"],
                         count=cell["bands"][b], fraction=cell["band_fractions"][b],
                         ci_lo=None, ci_hi=None, n_boot_finite=None))
    for key in CUM_KEYS:
        c = cell["cumulative"][key]
        rows.append(dict(base, kind="cumulative", band=c["label"], n_requests=cell["n_requests"],
                         n_intruders=cell["n_intruders"],
                         intruders_per_request=cell["intruders_per_request"],
                         count=c["count"], fraction=c["fraction"],
                         ci_lo=c["ci_lo"], ci_hi=c["ci_hi"], n_boot_finite=c["n_boot_finite"]))
    return rows


# --------------------------------------------------------------------------- #
def load_dump(path):
    d = {"path": path, "filename": os.path.basename(path)}
    with np.load(path, allow_pickle=False) as z:
        for k in ("gt_item", "t_top", "s_top", "t_n"):
            if k not in z.files:
                raise SystemExit("%s: missing required array %r" % (path, k))
        d["gt_item"] = np.asarray(z["gt_item"]).astype(np.int64).ravel()
        d["t_top"] = np.asarray(z["t_top"]).astype(np.int64)
        d["s_top"] = np.asarray(z["s_top"]).astype(np.int64)
        d["t_n"] = np.asarray(z["t_n"]).astype(np.int64).ravel()
        d["valid_raw"] = np.asarray(z["valid"]).astype(np.int64).ravel() if "valid" in z.files else None
        d["npz_keys"] = sorted(z.files)
    n = int(d["gt_item"].shape[0])
    d["n"] = n
    for k in ("t_top", "s_top"):
        if d[k].ndim != 2 or d[k].shape[0] != n:
            raise SystemExit("%s: %s has shape %s, expected (%d, W)" % (path, k, d[k].shape, n))
    if d["t_n"].shape[0] != n:
        raise SystemExit("%s: t_n has length %d, expected %d" % (path, d["t_n"].shape[0], n))
    if d["valid_raw"] is not None:
        d["valid"] = d["valid_raw"] != 0
        d["valid_rule"] = "npz 'valid' array (lists_s48 protocol)"
    else:
        d["valid"] = (d["gt_item"] >= 0) & (d["t_n"] > 0)
        d["valid_rule"] = "recomputed: gt_item >= 0 and t_n > 0 (npz has no 'valid' array)"
    d["sha256"] = sha256_file(path)
    d["size_bytes"] = os.path.getsize(path)
    d["mtime_iso"] = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="s48 doc 2.3: teacher-rank CDF of the method's Top-10 intruders (0 GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs_dir", default="runs",
                    help="dir holding lists_s48_*.npz and tbeam50_<CAT>.pt")
    ap.add_argument("--cats", default="", help="comma-separated category allow-list (default: all)")
    ap.add_argument("--methods", default="", help="comma-separated method allow-list (default: all)")
    ap.add_argument("--graph_dir", required=True, help="passed straight to find_valid_set")
    ap.add_argument("--dataset_tag_fmt", default="AmazonReviews2023_{cat}")
    ap.add_argument("--sem_ids_fmt", default="",
                    help="recorded for provenance only; no pipeline is built")
    ap.add_argument("--checkpoint_fmt", default="",
                    help="recorded for provenance only; no pipeline is built")
    ap.add_argument("--lens_path_fmt", default="",
                    help="optional: passed as find_valid_set(lens_path=...) if the valid set "
                         "cannot be located from --graph_dir alone")
    ap.add_argument("--teacher_cache_fmt", default="tbeam50_{cat}.pt",
                    help="s49: teacher beam cache used to RESOLVE ranks; tbeam64_{cat}.pt "
                         "with --rank_depth 64 gives the doc's 51-60 / 61-64 bands")
    ap.add_argument("--teacher_top_from_cache", action="store_true",
                    help="s49: derive the teacher top-10 from the SAME cache the ranks come "
                         "from, instead of the npz t_top. REQUIRED with a cache whose width "
                         "differs from the one the dumps were built against (e.g. tbeam64), "
                         "otherwise the intruder definition and the rank ladder disagree.")
    ap.add_argument("--rank_depth", type=int, default=50,
                    help="s49: width of that cache; band edges are derived from it")
    ap.add_argument("--out", required=True, help="output path WITHOUT extension")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates (0 disables CIs)")
    ap.add_argument("--boot_seed", type=int, default=12345)
    ap.add_argument("--device", default="cpu", help="device for build_valid_tensors (cpu is enough)")
    args = ap.parse_args(argv)
    configure_bands(args.rank_depth, args.teacher_cache_fmt,
                    args.teacher_cache_fmt.format(cat="<CAT>"))

    t_start = time.time()
    # resolve BEFORE the chdir below, so relative args mean what the caller typed
    runs_dir = os.path.abspath(args.runs_dir)
    out_base = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_base)
    graph_dir = os.path.abspath(args.graph_dir)
    if not os.path.isdir(runs_dir):
        raise SystemExit("--runs_dir %r does not exist (resolved from cwd %r)"
                         % (runs_dir, os.getcwd()))
    if not out_dir:
        raise SystemExit("--out must include a directory")

    keep_cats = set(x for x in args.cats.split(",") if x)
    keep_methods = set(x for x in args.methods.split(",") if x)
    paths = sorted(glob.glob(os.path.join(runs_dir, PREFIX + "*.npz")))
    warnings = []
    selected = []
    for p in paths:
        parsed = parse_filename(p)
        if parsed is None:
            warnings.append("skipped unparseable filename: %s" % os.path.basename(p))
            continue
        cat, method, _seed = parsed
        if keep_cats and cat not in keep_cats:
            continue
        if keep_methods and method not in keep_methods:
            continue
        selected.append(p)
    if not selected:
        raise SystemExit("no dumps matched %s/%s*.npz with the given --cats/--methods"
                         % (runs_dir, PREFIX))
    by_cat = OrderedDict()
    for p in selected:
        by_cat.setdefault(parse_filename(p)[0], []).append(p)

    # the sid_mlp package expects this sys.path / cwd / offline env (== lists_s48.py)
    sys.path.insert(0, "%s/repos/SID-MLP" % R)
    for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[k] = "1"
    os.chdir("%s/repos" % R)

    print("[cfg] %s" % SCRIPT_VERSION)
    print("[cfg] runs_dir=%s" % runs_dir)
    print("[cfg] out=%s{.json,.csv}  boot=%d seed=%d" % (out_base, args.boot, args.boot_seed))
    print("[cfg] cwd=%s (chdir'd for sid_mlp)  graph_dir=%s" % (os.getcwd(), graph_dir))
    print("[scan] %d dump(s), %d category(ies)" % (len(selected), len(by_cat)))

    cells = []
    rows = []
    inputs = []
    checks = {"prefix_mismatch_rows": 0, "impossible_band_1_10": 0, "n_len_mismatch": 0}

    for cat in sorted(by_cat):
        dataset_tag = args.dataset_tag_fmt.format(cat=cat)
        lens_path = args.lens_path_fmt.format(cat=cat) if args.lens_path_fmt else ""
        tcache = os.path.join(runs_dir, TEACHER_CACHE_FMT.format(cat=cat))
        if not os.path.exists(tcache):
            raise SystemExit("teacher beam cache %r not found (needed for teacher ranks)" % tcache)

        try:
            valid_items_4, prov = load_valid_items(dataset_tag, graph_dir, lens_path, args.device)
        except Exception as exc:
            raise SystemExit(
                "find_valid_set/build_valid_tensors failed for cat=%s (dataset_tag=%s, graph_dir=%s,"
                " lens_path=%r, device=%s): %s\nHints: pass --lens_path_fmt if the valid set is only"
                " locatable from a student checkpoint; pass --device cuda if the mask build needs a"
                " GPU." % (cat, dataset_tag, graph_dir, lens_path or None, args.device, exc))
        codes = load_teacher_codes(tcache)
        if codes.ndim != 3 or codes.shape[2] != 4:
            raise SystemExit("%s: 'codes' has shape %s, expected (N, W, 4)" % (tcache, codes.shape))
        if codes.shape[1] < RANK_DEPTH:
            raise SystemExit("%s: teacher beam width %d < %d -- ranks 41-50 would be unresolvable"
                             % (tcache, codes.shape[1], RANK_DEPTH))
        if codes.shape[1] > RANK_DEPTH:
            warnings.append("%s: beam width %d > %d; only the first %d columns are used"
                            % (os.path.basename(tcache), codes.shape[1], RANK_DEPTH, RANK_DEPTH))
        t0 = time.time()
        T, Rn, n_inv = teacher_ranking(codes[:, :RANK_DEPTH], valid_items_4, RANK_DEPTH)
        print("[%s] teacher ranking: rows=%d valid_sids=%d mean_R=%.2f invalid_quads=%d (%.0fs)"
              % (cat, T.shape[0], prov["n_valid_sids"], float(Rn.mean()), n_inv, time.time() - t0))
        inputs.append({"kind": "teacher_beam", "cat": cat, "path": tcache,
                       "sha256": sha256_file(tcache), "rows": int(codes.shape[0]),
                       "beam_width": int(codes.shape[1]), "rank_depth_used": RANK_DEPTH,
                       "space": "TOKEN-ID (vocab)", "mean_dedup_depth": float(Rn.mean()),
                       "min_dedup_depth": int(Rn.min()), "invalid_quads": n_inv,
                       "valid_set": prov})

        per_method = OrderedDict()
        cat_idx = None                      # analysed request index of this category
        cat_idx_src = None
        for path in sorted(by_cat[cat]):
            _c, method, seed = parse_filename(path)
            d = load_dump(path)
            n = d["n"]
            if n > T.shape[0]:
                raise SystemExit("%s: %d rows but the teacher cache has only %d"
                                 % (d["filename"], n, T.shape[0]))
            if n != T.shape[0]:
                checks["n_len_mismatch"] += 1
                warnings.append("%s: n=%d != teacher cache rows=%d; using the first n rows "
                                "(row-aligned by construction, but the dump is truncated)"
                                % (d["filename"], n, T.shape[0]))
            Tn = T[:n]
            Rn_n = Rn[:n]

            # ---- cross-source check (a): npz t_top must be the prefix of the re-derived ranking
            wt = d["t_top"].shape[1]
            if wt > RANK_DEPTH:
                raise SystemExit("%s: t_top width %d > rank depth %d" % (d["filename"], wt, RANK_DEPTH))
            mism = np.any(d["t_top"] != Tn[:, :wt], axis=1) | (np.minimum(Rn_n, wt) != d["t_n"])
            if args.teacher_top_from_cache:
                # the check compares two DIFFERENT teacher widths on purpose here; keep the
                # measurement, drop its veto (see the module docstring for why).
                n_mism_measured = int(mism.sum())
                mism = np.zeros_like(mism)
            n_mism = int(mism.sum())
            if n_mism:
                checks["prefix_mismatch_rows"] += n_mism
                bad = int(np.flatnonzero(mism)[0])
                warnings.append("ALIGNMENT/SPACE BUG: %s: %d/%d rows where npz t_top != the "
                                "tbeam50-derived teacher ranking (first bad row %d: t_top=%s "
                                "derived=%s)" % (d["filename"], n_mism, n, bad,
                                                 d["t_top"][bad, :5].tolist(), Tn[bad, :5].tolist()))

            m = d["valid"]
            if cat_idx is None:
                cat_idx = np.flatnonzero(m).astype(np.int64)
                cat_idx_src = d["filename"]
            t_ref = first_k_real(Tn, TOPK) if args.teacher_top_from_cache else d["t_top"]
            counts, _n_intr = band_counts(d["s_top"], t_ref, Tn)

            same_mask = (cat_idx.shape[0] == int(m.sum())
                         and (cat_idx.size == 0 or int(cat_idx.max()) < n)
                         and bool(np.all(m[cat_idx])))
            if not same_mask:
                warnings.append("%s: its valid-request set differs from %s -- this cell uses its "
                                "own mask and is EXCLUDED from the pooled-over-seeds row"
                                % (d["filename"], cat_idx_src))
            idx = np.flatnonzero(m).astype(np.int64)
            c_sub = counts[idx]
            den = np.ones(c_sub.shape[0], dtype=np.float64)
            cell = make_cell(cat, method, seed, "seed", 1, c_sub, den, args.boot, args.boot_seed)
            cell["valid_rule"] = d["valid_rule"]
            cell["prefix_mismatch_rows"] = n_mism
            if args.teacher_top_from_cache:
                cell["teacher_top_source"] = "cache (%s)" % TEACHER_CACHE_FMT
                cell["prefix_check"] = "skipped_by_design"
                cell["npz_vs_cache_top20_disagreeing_rows"] = n_mism_measured
                cell["npz_vs_cache_note"] = (
                    "rows where the npz (width-50) teacher top-20 differs from this cache's "
                    "ranking; reported, not vetoed -- the two teacher widths are different "
                    "objects and this run deliberately uses the cache one throughout")
            else:
                cell["teacher_top_source"] = "npz t_top"
            checks["impossible_band_1_10"] += cell["impossible_band_1_10"]
            cells.append(cell)
            rows.extend(cell_rows(cell))
            print("  %-22s %-14s seed=%-6s n_req=%-7d intr=%-8d ipr=%s <=40=%s"
                  % (cat, method, seed, cell["n_requests"], cell["n_intruders"],
                     f4(cell["intruders_per_request"]), f4(cell["cumulative"]["le40"]["fraction"])))

            inputs.append({"kind": "dump", "cat": cat, "method": method, "seed": seed,
                           "path": path, "sha256": d["sha256"], "n": n,
                           "n_valid": int(m.sum()), "valid_rule": d["valid_rule"],
                           "size_bytes": d["size_bytes"], "mtime_iso": d["mtime_iso"],
                           "npz_keys": d["npz_keys"], "prefix_mismatch_rows": n_mism,
                           "space": "ITEM-ID (t_top / s_top / gt_item already mapped)"})
            if same_mask:
                per_method.setdefault(method, []).append((seed, counts[cat_idx]))

        # ---- pooled over seeds: clustered by REQUEST (the same request recurs per seed)
        for method, seedlist in per_method.items():
            if not seedlist:
                continue
            pooled = np.zeros_like(seedlist[0][1])
            for _s, c in seedlist:
                pooled = pooled + c
            den = np.full(pooled.shape[0], float(len(seedlist)), dtype=np.float64)
            cell = make_cell(cat, method, "pooled", "pooled", len(seedlist), pooled, den,
                             args.boot, args.boot_seed)
            cell["pooled_seeds"] = [s for s, _c in seedlist]
            cell["valid_rule"] = "intersection check passed: all pooled seeds share one valid set"
            cells.append(cell)
            rows.extend(cell_rows(cell))
            print("  %-22s %-14s seed=%-6s n_req=%-7d intr=%-8d ipr=%s <=40=%s  [%d seeds]"
                  % (cat, method, "POOLED", cell["n_requests"], cell["n_intruders"],
                     f4(cell["intruders_per_request"]), f4(cell["cumulative"]["le40"]["fraction"]),
                     len(seedlist)))

    checks["ok"] = (checks["prefix_mismatch_rows"] == 0 and checks["impossible_band_1_10"] == 0)
    result = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "limitation": LIMITATION,
        "meta": {
            "argv": sys.argv,
            "runs_dir": runs_dir, "out": out_base, "graph_dir": graph_dir,
            "dataset_tag_fmt": args.dataset_tag_fmt,
            "sem_ids_fmt": args.sem_ids_fmt or None,
            "checkpoint_fmt": args.checkpoint_fmt or None,
            "lens_path_fmt": args.lens_path_fmt or None,
            "provenance_only_args": ["sem_ids_fmt", "checkpoint_fmt"],
            "topk": TOPK, "rank_depth": RANK_DEPTH,
            "bands": list(BANDS), "cumulative_keys": list(CUM_KEYS),
            "boot": args.boot, "boot_seed": args.boot_seed,
            "bootstrap_unit": "request (intruders inside a request are dependent); pooled rows "
                              "resample requests as clusters across seeds",
            "intruder_def": "item in the method's top-10 that is not in the teacher's top-10 "
                            "(teacher top-10 from npz t_top; teacher rank from tbeam50)",
            "tensor_spaces": {
                "tbeam50_<CAT>.pt['codes']": "TOKEN-ID (vocab) space, (N,50,4)",
                "valid_items_4 keys": "TOKEN-ID (vocab) space -> ITEM id",
                "npz gt_item/t_top/s_top": "ITEM-ID space (already mapped by lists_s48.py)"},
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "checks": checks,
        "warnings": warnings,
        "inputs": inputs,
        "cells": cells,
        "rows": rows,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_base + ".json", "w") as fh:
        json.dump(result, fh, indent=2)
    with open(out_base + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})

    # ---- compact table ----
    print("")
    print("=== S48 INTRUDER TEACHER-RANK CDF (method top-10 \\ teacher top-10) ===")
    _cumfmt = " ".join(["%8s"] * len(CUM_KEYS))
    hdr = ("%-22s %-14s %-7s %8s %9s %6s | " + _cumfmt) % tuple(
        ["cat", "method", "seed", "n_req", "intruders", "ipr"]
        + [CUM_LABEL[_k] for _k in CUM_KEYS])
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        print(("%-22s %-14s %-7s %8d %9d %6s | " + _cumfmt) % tuple(
            [c["cat"], c["method"], c["seed"], c["n_requests"], c["n_intruders"],
             f4(c["intruders_per_request"], 6)]
            + [f4(c["cumulative"][_k]["fraction"]) for _k in CUM_KEYS]))
    print("-" * len(hdr))
    print("LIMITATION: %s" % LIMITATION)
    for w_ in warnings:
        print("[warn] %s" % w_)
    print("wrote %s.json / %s.csv (%d cells, %d rows, %.0fs)"
          % (out_base, out_base, len(cells), len(rows), time.time() - t_start))

    if not checks["ok"]:
        print("[BUG] cross-source checks FAILED: prefix_mismatch_rows=%d impossible_band_1_10=%d"
              % (checks["prefix_mismatch_rows"], checks["impossible_band_1_10"]))
        print("[BUG] the outputs above are the diagnostic; the CDF must NOT be quoted as-is.")
        print("S48_INTRUDER_CDF_FAILED", flush=True)
        return 2
    print("S48_INTRUDER_CDF_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
