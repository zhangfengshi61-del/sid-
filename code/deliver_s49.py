#!/usr/bin/env python3
"""s49: assemble the five 「待完成实验(3).md」 §5 deliverables that s48 left unbuilt.

s48 shipped  reference_agreement · quality_buckets · serving_parity_blindspot · kb_intruder_cdf ·
run_manifest(+_full).  This adds

  1. speed_summary        (doc §5.1)  <- runs/s48speed/*.json  + the reused prior-local runs
  2. final_da_quality     (doc §5.3)  <- quality_buckets + reference_agreement, subset to the
                                         final HServe+DA-CDRS checkpoint (+ HServe+plain pair,
                                         the doc §1.7B strong control)
  3. hserve_mechanism     (doc §5.4)  <- cov_s49_*.npz (Coverage@C + bootstrap CI) + mech A_hdrift
  4. da_mechanism         (doc §5.5)  <- mech B_fliprate (margin quintiles) + C_attribution
  5. kb_sensitivity       (doc §5.7)  <- reference_agreement (K_eval) + the K_train / B arms

NOTHING is recomputed from raw dumps here: every number is copied from an audited artifact and
carries its `source` file, so this file can never disagree with the s48 engine.  Where a number
is NOT comparable to its neighbours (different harness, different protocol) the row says so in
`source_kind` / `caveat` rather than being silently tabulated next to them.

House style follows metrics_s48.py: {meta, fields, rows} json + the same rows as csv.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone

SCRIPT_VERSION = "deliver_s49.py/1.0"

CATS = ("Industrial_and_Scientific", "Musical_Instruments", "Video_Games")
SHORT = {"Industrial_and_Scientific": "Sci", "Musical_Instruments": "MI", "Video_Games": "VG"}

# doc method name  <->  our arm label (established in notes/s48-experiment-wave.md §0)
DOC_NAME = {
    "tiger": "TIGER (teacher/reference)",
    "sidmlp": "SID-MLP (digit-CE only)",
    "lkonly": "d4-only list-KL",
    "serveW": "old weights + d3",
    "hserve": "HServe (full uniform)",
    "hserve_pair": "HServe + plain pair",
    "hserve_da": "HServe + DA-CDRS (DA_W=62.83)",
    "daw31": "HServe + DA-CDRS (DA_W=31.42)",
    "attroff31": "HServe + pair + uniform correction (DA_W=31.42)",
    "daonly31": "DA-CDRS only (w/o HServe, DA_W=31.42)",
    "attr_off": "HServe + pair + uniform correction (attribution-off, DA_W=62.83)",
    "da_only": "DA-CDRS only (w/o HServe, DA_W=62.83)",
    "lkpf": "old list-KL + prefix-KL",
    "nezha": "NEZHA", "gdn": "GDN", "mamba2": "Mamba2", "earn": "EARN",
    "atspeedS": "AtSpeed-S", "atspeedR": "AtSpeed-R", "padtiger": "PAD-TIGER",
}


def now_iso():
    return datetime.now(tz=timezone.utc).isoformat()


def jsonable(x):
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_table(out_dir, name, rows, meta, fields=None):
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    payload = OrderedDict([("meta", meta), ("fields", fields), ("rows", rows)])
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(payload, f, indent=1)
    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("[write] %-22s rows=%d -> %s.{json,csv}" % (name, len(rows), name))
    return len(rows)


def rel(path, root):
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


# --------------------------------------------------------------------------- #
# 1. speed_summary
# --------------------------------------------------------------------------- #
def build_speed(runs_dir, root):
    rows = []
    notes = []

    # (a) the s49/s48 clean single-harness sweep -- the ONLY mutually comparable block
    for path in sorted(glob.glob(os.path.join(runs_dir, "s48speed", "speed_s48_*.json"))):
        d = load_json(path)
        base = os.path.basename(path)
        cat = d.get("category")
        method = d.get("method")
        bs = int(d.get("batch_size", 0))
        caveat = ""
        if method == "gdn" and bs != 1:
            caveat = ("rep 1 is Triton-kernel-compile dominated and must be DISCARDED "
                      "(the fla kernel recompiles per seqlen bucket, --warmup cannot absorb it); "
                      "quote reps 2-4")
        for rep in d.get("reps_detail", []):
            rows.append(OrderedDict([
                ("scope", "per_rep"), ("method", method), ("doc_name", DOC_NAME.get(method, method)),
                ("cat", cat), ("cat_short", SHORT.get(cat, cat)), ("batch_size", bs),
                ("rep", rep.get("rep")), ("n_samples", rep.get("n_samples")),
                ("samples_per_s", jsonable(rep.get("samples_per_s"))),
                ("ms_per_sample", jsonable(rep.get("ms_per_sample"))),
                ("peak_mem_gb", jsonable(rep.get("peak_mem_gb"))),
                ("lat_p50_ms", jsonable(rep.get("lat_p50_ms"))),
                ("lat_p95_ms", jsonable(rep.get("lat_p95_ms"))),
                ("lat_p99_ms", jsonable(rep.get("lat_p99_ms"))),
                ("samples_per_s_std", None), ("beam", d.get("beam")), ("dtype", d.get("dtype")),
                ("gpu", d.get("gpu")), ("torch_version", d.get("torch_version")),
                ("cuda_version", d.get("cuda_version")),
                ("order_fingerprint", d.get("order_fingerprint")),
                ("source_kind", "s48_clean_harness"), ("source", rel(path, root)),
                ("checkpoint", d.get("checkpoint")), ("caveat", caveat),
            ]))
        if caveat:
            good = [r for r in d.get("reps_detail", []) if r.get("rep") != 1]
            if good:
                import statistics as _st
                sps = [r["samples_per_s"] for r in good]
                rows.append(OrderedDict([
                    ("scope", "mean_over_reps_excl_rep1"), ("method", method),
                    ("doc_name", DOC_NAME.get(method, method)),
                    ("cat", cat), ("cat_short", SHORT.get(cat, cat)), ("batch_size", bs),
                    ("rep", None), ("n_samples", d.get("n_samples")),
                    ("samples_per_s", float(_st.fmean(sps))),
                    ("ms_per_sample", float(_st.fmean([r["ms_per_sample"] for r in good]))),
                    ("peak_mem_gb", float(_st.fmean([r["peak_mem_gb"] for r in good]))),
                    ("lat_p50_ms", None), ("lat_p95_ms", None), ("lat_p99_ms", None),
                    ("samples_per_s_std", float(_st.pstdev(sps)) if len(sps) > 1 else 0.0),
                    ("beam", d.get("beam")), ("dtype", d.get("dtype")), ("gpu", d.get("gpu")),
                    ("torch_version", d.get("torch_version")),
                    ("cuda_version", d.get("cuda_version")),
                    ("order_fingerprint", d.get("order_fingerprint")),
                    ("source_kind", "s48_clean_harness"), ("source", rel(path, root)),
                    ("checkpoint", d.get("checkpoint")),
                    ("caveat", "THIS is the citable GDN throughput: rep 1 excluded because it "
                               "carries Triton kernel compilation that --warmup cannot absorb. "
                               "The plain mean_over_reps row for GDN is depressed by that "
                               "compile and must not be quoted."),
                ]))
        rows.append(OrderedDict([
            ("scope", "mean_over_reps"), ("method", method),
            ("doc_name", DOC_NAME.get(method, method)),
            ("cat", cat), ("cat_short", SHORT.get(cat, cat)), ("batch_size", bs),
            ("rep", None), ("n_samples", d.get("n_samples")),
            ("samples_per_s", jsonable(d.get("samples_per_s_mean"))),
            ("ms_per_sample", jsonable(d.get("ms_per_sample_mean"))),
            ("peak_mem_gb", jsonable(d.get("peak_mem_gb_mean"))),
            ("lat_p50_ms", jsonable(d.get("lat_p50_ms_mean"))),
            ("lat_p95_ms", jsonable(d.get("lat_p95_ms_mean"))),
            ("lat_p99_ms", jsonable(d.get("lat_p99_ms_mean"))),
            ("samples_per_s_std", jsonable(d.get("samples_per_s_std"))),
            ("beam", d.get("beam")), ("dtype", d.get("dtype")), ("gpu", d.get("gpu")),
            ("torch_version", d.get("torch_version")), ("cuda_version", d.get("cuda_version")),
            ("order_fingerprint", d.get("order_fingerprint")),
            ("source_kind", "s48_clean_harness"), ("source", rel(path, root)),
            ("checkpoint", d.get("checkpoint")), ("caveat", caveat),
        ]))
        del base

    # (b) reused PRIOR-LOCAL runs (different harness / conditions) -- listed but flagged
    reuse = [
        ("earn", "earntput_{cat}_{earn_k}.json", "throughput",
         "timed a bf16 UNCONSTRAINED stand-in decode, while EARN's quality eval is fp32 + "
         "constrained-Trie; NOT comparable to the s48 clean-harness block"),
        ("atspeedS", "atspeed_S_{short}.json", "throughput",
         "s40 run, separate harness/session; AtSpeed-S is lossless (jac=1.0) and ~0.70x TIGER-kv"),
        ("atspeedR", "atspeed_R_{short}.json", "throughput",
         "our vectorized reject-sampling runs 1.58x whereas the paper's serial implementation "
         "reports 0.22x -- cite the PAPER for AtSpeed-R speed, not this cell"),
        ("padtiger", "padinf_{short}_N100.json", "throughput",
         "s37 run, separate harness/session"),
        ("atspeed_kv", "atspeed_kv_{short}.json", "throughput",
         "the AtSpeed harness's own TIGER-kv reference point, for scaling within block (b) only"),
    ]
    for method, fmt, key, caveat in reuse:
        for cat in CATS:
            path = os.path.join(runs_dir, fmt.format(
                cat=cat, short=SHORT[cat],
                earn_k=("k3" if cat == "Video_Games" else "k2")))
            if not os.path.exists(path):
                notes.append("missing reused speed source: %s" % rel(path, root))
                continue
            d = load_json(path)
            rows.append(OrderedDict([
                ("scope", "single_run"), ("method", method),
                ("doc_name", DOC_NAME.get(method, method)),
                ("cat", cat), ("cat_short", SHORT[cat]),
                ("batch_size", d.get("bs") or d.get("batch_size")),
                ("rep", None), ("n_samples", d.get("total") or d.get("n_samples")),
                ("samples_per_s", jsonable(d.get(key))),
                ("ms_per_sample", jsonable(d.get("ms_per_sample"))),
                ("peak_mem_gb", jsonable(d.get("peak_mem_gb"))),
                ("lat_p50_ms", None), ("lat_p95_ms", None), ("lat_p99_ms", None),
                ("samples_per_s_std", None), ("beam", d.get("K") or d.get("beam")),
                ("dtype", d.get("dtype")), ("gpu", d.get("gpu")),
                ("torch_version", None), ("cuda_version", None),
                ("order_fingerprint", None),
                ("source_kind", "prior_local_run"), ("source", rel(path, root)),
                ("checkpoint", d.get("draft") or d.get("checkpoint")), ("caveat", caveat),
            ]))
    return rows, notes


# --------------------------------------------------------------------------- #
# 2. final_da_quality
# --------------------------------------------------------------------------- #
FINAL_METHODS = ("daw31", "hserve_pair")


def build_final_da(deliv_dir, root):
    rows = []
    qb_path = os.path.join(deliv_dir, "quality_buckets.json")
    ra_path = os.path.join(deliv_dir, "reference_agreement.json")
    qb = load_json(qb_path)
    ra = load_json(ra_path)
    for r in qb["rows"]:
        if r.get("method") in FINAL_METHODS:
            o = OrderedDict([("kind", "bucket_quality"),
                             ("doc_name", DOC_NAME.get(r["method"], r["method"]))])
            o.update(r)
            o["source"] = rel(qb_path, root)
            rows.append(o)
    for r in ra["rows"]:
        if r.get("method") in FINAL_METHODS:
            o = OrderedDict([("kind", "reference_agreement"),
                             ("doc_name", DOC_NAME.get(r["method"], r["method"]))])
            o.update(r)
            o["source"] = rel(ra_path, root)
            rows.append(o)
    return rows


# --------------------------------------------------------------------------- #
# 3. hserve_mechanism  (Coverage@C + per-depth HDrift)
# --------------------------------------------------------------------------- #
HDRIFT_ARMS = ("sidmlp", "lkonly", "lkpf", "hserve")


def bootstrap_mean_ci(vals, n_boot, seed_key):
    import numpy as np
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    mean = float(vals.mean())
    if n_boot <= 0:
        return mean, None, None, 0
    rng = np.random.default_rng([12345, seed_key])
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(n_boot)


def build_hserve_mech(runs_dir, root, n_boot):
    import numpy as np
    rows, notes = [], []

    # --- A. Coverage@C for SID-MLP, with the doc-required bootstrap CI
    cov_files = sorted(glob.glob(os.path.join(runs_dir, "cov_s49_*__sidmlp__*.npz")))
    if not cov_files:
        notes.append("no cov_s49_*.npz found -- Coverage@C section EMPTY (run runs/s49_cov.sh)")
    per_cell = {}
    for path in cov_files:
        stem = os.path.basename(path)[: -len(".npz")]
        _, cat_seed = stem.split("cov_s49_", 1)
        cat, _m, seed = cat_seed.split("__")
        z = np.load(path)
        den = z["den"].astype(np.float64)
        for C in z["Cs"].tolist():
            key = "inter_at_%d" % C
            if key not in z:
                continue
            per_req = z[key].astype(np.float64) / np.maximum(den, 1.0)
            mean, lo, hi, nb = bootstrap_mean_ci(per_req, n_boot, abs(hash((cat, seed, C))) % 10 ** 8)
            per_cell.setdefault((cat, C), []).append(mean)
            rows.append(OrderedDict([
                ("kind", "coverage_at_C"), ("cat", cat), ("cat_short", SHORT.get(cat, cat)),
                ("method", "sidmlp"), ("doc_name", DOC_NAME["sidmlp"]),
                ("seed", seed), ("seed_agg", "per_seed"), ("n_seeds", 1),
                ("C", int(C)), ("depth", None),
                ("value", jsonable(mean)), ("ci_lo", jsonable(lo)), ("ci_hi", jsonable(hi)),
                ("n_requests", int(len(per_req))),
                ("n_teacher_items", int(den.sum())), ("n_boot", nb),
                ("definition", "|L_T^10 cap C_S^C| / |L_T^10|  (doc 1.6A; denominator is the "
                               "teacher's ACTUAL served-list length, ==10 for all but a handful "
                               "of requests)"),
                ("source", rel(path, root)), ("caveat", ""),
            ]))
    for (cat, C), vals in sorted(per_cell.items()):
        a = np.asarray(vals, dtype=np.float64)
        rows.append(OrderedDict([
            ("kind", "coverage_at_C"), ("cat", cat), ("cat_short", SHORT.get(cat, cat)),
            ("method", "sidmlp"), ("doc_name", DOC_NAME["sidmlp"]),
            ("seed", "mean"), ("seed_agg", "mean"), ("n_seeds", len(vals)),
            ("C", int(C)), ("depth", None),
            ("value", float(a.mean())), ("ci_lo", None), ("ci_hi", None),
            ("n_requests", None), ("n_teacher_items", None), ("n_boot", 0),
            ("definition", "seed mean of the per-seed Coverage@C above"),
            ("source", "derived"), ("caveat", "std over seeds = %.6f" % float(a.std(ddof=0))),
        ]))

    # --- B. per-depth HDrift, and the full-HServe-vs-d4-only drop the doc asks for
    hd = {}
    for cat in CATS:
        for arm in HDRIFT_ARMS:
            path = os.path.join(runs_dir, "mech_s48_%s__%s.json" % (cat, arm))
            if not os.path.exists(path):
                notes.append("missing mechanism json: %s" % rel(path, root))
                continue
            d = load_json(path)
            a = d["A_hdrift"]
            for dep in ("d1", "d2", "d3", "d4"):
                v = a["per_depth"][dep]
                hd[(cat, arm, dep)] = v["mean"]
                rows.append(OrderedDict([
                    ("kind", "hdrift"), ("cat", cat), ("cat_short", SHORT.get(cat, cat)),
                    ("method", arm), ("doc_name", DOC_NAME.get(arm, arm)),
                    ("seed", d.get("seed_tag")), ("seed_agg", "per_seed"), ("n_seeds", 1),
                    ("C", None), ("depth", dep),
                    ("value", jsonable(v["mean"])), ("ci_lo", jsonable(v.get("ci_lo"))),
                    ("ci_hi", jsonable(v.get("ci_hi"))),
                    ("n_requests", d.get("counts", {}).get("n_used")),
                    ("n_teacher_items", None), ("n_boot", None),
                    ("definition", a["definition"]),
                    ("source", rel(path, root)),
                    ("caveat", "raw per-depth KL at temperature tau, NOT scaled by tau^2 "
                               "(serve_kl_loss applies tau^2 only for gradient balance); "
                               "temperature_sq_factor_applied=false"),
                ]))
            s = a["sum_d1_d4"]
            rows.append(OrderedDict([
                ("kind", "hdrift"), ("cat", cat), ("cat_short", SHORT.get(cat, cat)),
                ("method", arm), ("doc_name", DOC_NAME.get(arm, arm)),
                ("seed", d.get("seed_tag")), ("seed_agg", "per_seed"), ("n_seeds", 1),
                ("C", None), ("depth", "sum_d1_d4"),
                ("value", jsonable(s["mean"])), ("ci_lo", jsonable(s.get("ci_lo"))),
                ("ci_hi", jsonable(s.get("ci_hi"))),
                ("n_requests", d.get("counts", {}).get("n_used")),
                ("n_teacher_items", None), ("n_boot", None),
                ("definition", "sum of the four per-depth KLs"),
                ("source", rel(path, root)), ("caveat", ""),
            ]))
    for cat in CATS:
        for dep in ("d1", "d2", "d3", "d4"):
            a = hd.get((cat, "lkonly", dep))
            b = hd.get((cat, "hserve", dep))
            if a is None or b is None:
                continue
            rows.append(OrderedDict([
                ("kind", "hdrift_drop_vs_d4only"), ("cat", cat), ("cat_short", SHORT.get(cat, cat)),
                ("method", "hserve"), ("doc_name", DOC_NAME["hserve"]),
                ("seed", "s42"), ("seed_agg", "per_seed"), ("n_seeds", 1),
                ("C", None), ("depth", dep),
                ("value", jsonable(a - b)), ("ci_lo", None), ("ci_hi", None),
                ("n_requests", None), ("n_teacher_items", None), ("n_boot", None),
                ("definition", "HDrift(d4-only list-KL) - HDrift(full HServe) at this depth; "
                               "positive = full HServe reduces the drift (doc 1.6B)"),
                ("source", "derived from the two mech_s48 jsons above"),
                ("caveat", "point difference of two independently bootstrapped means; no CI on "
                           "the difference (the two arms are different checkpoints, not paired "
                           "per request in these dumps)"),
            ]))
    return rows, notes


# --------------------------------------------------------------------------- #
# 4. da_mechanism  (FlipRate by teacher-margin quintile + attribution faithfulness)
# --------------------------------------------------------------------------- #
FLIP_ARMS = ("sidmlp", "hserve", "hserve_pair", "hserve_da", "daw31")
ATTR_ARMS = ("hserve_da", "attr_off", "daw31", "attroff31")


def build_da_mech(runs_dir, root):
    rows, notes = [], []
    for cat in CATS:
        for arm in FLIP_ARMS:
            path = os.path.join(runs_dir, "mech_s48_%s__%s.json" % (cat, arm))
            if not os.path.exists(path):
                notes.append("missing mechanism json: %s" % rel(path, root))
                continue
            d = load_json(path)
            b = d["B_fliprate"]
            pr = d.get("protocol", {})
            common = OrderedDict([
                ("cat", cat), ("cat_short", SHORT.get(cat, cat)), ("method", arm),
                ("doc_name", DOC_NAME.get(arm, arm)), ("seed", d.get("seed_tag")),
                ("pair_k", pr.get("pair_k")), ("pair_b", pr.get("pair_b")),
            ])
            o = OrderedDict(common)
            o.update([("kind", "fliprate_overall"), ("bin", None),
                      ("edge_lo", None), ("edge_hi", None),
                      ("margin_mean", jsonable(b.get("overall_margin_mean"))),
                      ("n_pairs", d.get("counts", {}).get("n_pairs_total")),
                      ("n_requests", d.get("counts", {}).get("n_requests_with_pairs")),
                      ("value", jsonable(b["overall"]["mean"])),
                      ("ci_lo", jsonable(b["overall"].get("ci_lo"))),
                      ("ci_hi", jsonable(b["overall"].get("ci_hi"))),
                      ("definition", b["definition"]), ("source", rel(path, root)),
                      ("caveat", b.get("ci_note", ""))])
            rows.append(o)
            for q in b["quintiles"]:
                o = OrderedDict(common)
                o.update([("kind", "fliprate_by_margin_quintile"), ("bin", q.get("bin")),
                          ("edge_lo", jsonable(q.get("edge_lo"))),
                          ("edge_hi", jsonable(q.get("edge_hi"))),
                          ("margin_mean", jsonable(q.get("margin_mean"))),
                          ("n_pairs", q.get("n_pairs")), ("n_requests", q.get("n_requests")),
                          ("value", jsonable(q.get("fliprate"))),
                          ("ci_lo", jsonable(q.get("fliprate_ci_lo"))),
                          ("ci_hi", jsonable(q.get("fliprate_ci_hi"))),
                          ("definition", b["definition"]), ("source", rel(path, root)),
                          ("caveat", "bins are the SAME margin edges / requests / pairs across "
                                     "arms (edges_source=%s) so Fig.2 and Fig.4 can share them"
                                     % b.get("edges_source"))])
                rows.append(o)

        for arm in ATTR_ARMS:
            path = os.path.join(runs_dir, "mech_s48_%s__%s.json" % (cat, arm))
            if not os.path.exists(path):
                continue
            d = load_json(path)
            C = d.get("C_attribution")
            if not C:
                continue
            for view in ("Dc3", "Dc4"):
                blk = C.get(view)
                if not blk:
                    continue
                primary = (view == C.get("primary_variant"))
                common = OrderedDict([
                    ("cat", cat), ("cat_short", SHORT.get(cat, cat)), ("method", arm),
                    ("doc_name", DOC_NAME.get(arm, arm)), ("seed", d.get("seed_tag")),
                    ("pair_k", d.get("protocol", {}).get("pair_k")),
                    ("pair_b", d.get("protocol", {}).get("pair_b")),
                ])
                for dep, v in sorted(blk.get("attribution_mean", {}).items()):
                    o = OrderedDict(common)
                    o.update([("kind", "attribution_mean/%s%s" % (view, "" if primary else "_secondary")),
                              ("bin", dep), ("edge_lo", None), ("edge_hi", None),
                              ("margin_mean", None), ("n_pairs", blk.get("n_pairs")),
                              ("n_requests", None),
                              ("value", jsonable(v.get("mean") if isinstance(v, dict) else v)),
                              ("ci_lo", jsonable(v.get("ci_lo")) if isinstance(v, dict) else None),
                              ("ci_hi", jsonable(v.get("ci_hi")) if isinstance(v, dict) else None),
                              ("definition", C.get("definition")), ("source", rel(path, root)),
                              ("caveat", blk.get("note", ""))])
                    rows.append(o)
                for dep, v in sorted(blk.get("top_depth_fraction", {}).items()):
                    o = OrderedDict(common)
                    o.update([("kind", "top_depth_fraction/%s%s" % (view, "" if primary else "_secondary")),
                              ("bin", dep), ("edge_lo", None), ("edge_hi", None),
                              ("margin_mean", None), ("n_pairs", blk.get("n_pairs")),
                              ("n_requests", None), ("value", jsonable(v)),
                              ("ci_lo", None), ("ci_hi", None),
                              ("definition", "fraction of flipped pairs whose argmax_d A_d is d"),
                              ("source", rel(path, root)), ("caveat", "")])
                    rows.append(o)
                for grp in ("reduction", "increment", "unflip_rate"):
                    for variant, v in sorted((blk.get(grp) or {}).items()):
                        if not isinstance(v, dict) or "mean" not in v:
                            continue
                        caveat = ""
                        if grp in ("reduction", "increment") and "random" in variant:
                            caveat = ("TAUTOLOGICAL CONTROL -- A_d IS the counterfactual pair-loss "
                                      "reduction and 'top' is argmax_d A_d, so top >= random holds "
                                      "BY CONSTRUCTION however meaningless the attribution. The "
                                      "informative contrast is top vs uniform/all-depth.")
                        o = OrderedDict(common)
                        o.update([("kind", "%s/%s%s" % (grp, view, "" if primary else "_secondary")),
                                  ("bin", variant), ("edge_lo", None), ("edge_hi", None),
                                  ("margin_mean", None),
                                  ("n_pairs", blk.get("n_flipped_pairs") if grp == "unflip_rate"
                                   else blk.get("n_pairs")),
                                  ("n_requests", None), ("value", jsonable(v.get("mean"))),
                                  ("ci_lo", jsonable(v.get("ci_lo"))),
                                  ("ci_hi", jsonable(v.get("ci_hi"))),
                                  ("definition", C.get("definition")),
                                  ("source", rel(path, root)), ("caveat", caveat)])
                        rows.append(o)
    return rows, notes


# --------------------------------------------------------------------------- #
# 5. kb_sensitivity
# --------------------------------------------------------------------------- #
# arm label -> (family, K_train, B, base method, note)
KB_ARMS = OrderedDict([
    ("pk5",         ("K_train", 5,  35, "hserve_pair", "doc 2.2 row K=5")),
    ("hserve_pair", ("K_train", 10, 40, "hserve_pair", "doc 2.2 main config (reused 3-seed); "
                                                       "uses ALL negatives, so it is the main "
                                                       "result and NOT an in-sweep B cell")),
    ("pk20",        ("K_train", 20, 50, "hserve_pair", "doc 2.2 row K=20")),
    ("dak5",        ("K_train_DA", 5,  35, "hserve_da", "s49: doc 2.2 on the DA-CDRS method "
                                                        "(K/B set on BOTH the pair anchor and the "
                                                        "DA correction)")),
    ("hserve_da",   ("K_train_DA", 10, 40, "hserve_da", "s49: doc 2.2 main config for the DA method")),
    ("dak20",       ("K_train_DA", 20, 50, "hserve_da", "s49: doc 2.2 on the DA-CDRS method")),
    ("pb20", ("B", 10, 20, "hserve_pair", "doc 2.3, PAIR_NEGS=10 (negative COUNT held fixed)")),
    ("pb30", ("B", 10, 30, "hserve_pair", "doc 2.3, PAIR_NEGS=10")),
    ("pb40", ("B", 10, 40, "hserve_pair", "doc 2.3, PAIR_NEGS=10 -- the in-sweep B=40 cell, "
                                          "comparable to its neighbours (NOT the main result)")),
    ("pb50", ("B", 10, 50, "hserve_pair", "doc 2.3, PAIR_NEGS=10")),
    ("pb60", ("B", 10, 60, "hserve_pair", "doc 2.3, PAIR_NEGS=10")),
])
KEVAL_METHODS = ("hserve_pair", "hserve_da")


def build_kb(deliv_dir, root):
    rows, notes = [], []
    ra_path = os.path.join(deliv_dir, "reference_agreement.json")
    ra = load_json(ra_path)
    src = rel(ra_path, root)

    seen = set()
    for r in ra["rows"]:
        m = r.get("method")
        if m in KB_ARMS and r.get("bucket") == "all":
            fam, kt, b, base, note = KB_ARMS[m]
            seen.add(m)
            rows.append(OrderedDict([
                ("family", fam), ("arm", m), ("doc_name", DOC_NAME.get(base, base)),
                ("K_train", kt), ("B", b), ("K_eval", r.get("K")),
                ("cat", r.get("cat")), ("cat_short", SHORT.get(r.get("cat"), r.get("cat"))),
                ("seed", r.get("seed")), ("seed_agg", r.get("seed_agg")),
                ("n_seeds", r.get("n_seeds")), ("n", r.get("n")),
                ("jac", jsonable(r.get("jac"))), ("jac_ci_lo", jsonable(r.get("jac_ci_lo"))),
                ("jac_ci_hi", jsonable(r.get("jac_ci_hi"))),
                ("overlap", jsonable(r.get("overlap"))),
                ("overlap_ci_lo", jsonable(r.get("overlap_ci_lo"))),
                ("overlap_ci_hi", jsonable(r.get("overlap_ci_hi"))),
                ("note", note), ("source", src),
            ]))
        if m in KEVAL_METHODS and r.get("bucket") == "all":
            rows.append(OrderedDict([
                ("family", "K_eval"), ("arm", m), ("doc_name", DOC_NAME.get(m, m)),
                ("K_train", 10), ("B", 40), ("K_eval", r.get("K")),
                ("cat", r.get("cat")), ("cat_short", SHORT.get(r.get("cat"), r.get("cat"))),
                ("seed", r.get("seed")), ("seed_agg", r.get("seed_agg")),
                ("n_seeds", r.get("n_seeds")), ("n", r.get("n")),
                ("jac", jsonable(r.get("jac"))), ("jac_ci_lo", jsonable(r.get("jac_ci_lo"))),
                ("jac_ci_hi", jsonable(r.get("jac_ci_hi"))),
                ("overlap", jsonable(r.get("overlap"))),
                ("overlap_ci_lo", jsonable(r.get("overlap_ci_lo"))),
                ("overlap_ci_hi", jsonable(r.get("overlap_ci_hi"))),
                ("note", "doc 2.1: the SAME K_train=10,B=40 model evaluated at K_eval in {5,10,20} "
                         "-- generalisation across the metric cut, no retraining"),
                ("source", src),
            ]))
    for m in KB_ARMS:
        if m not in seen:
            notes.append("kb arm %r has no dump in reference_agreement (not trained/dumped yet)" % m)
    return rows, notes


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs_dir", default="")
    ap.add_argument("--deliv_dir", default="", help="dir holding the s48 engine outputs to subset")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--only", default="", help="comma-separated subset of the five tables")
    args = ap.parse_args(argv)
    root = args.root
    runs_dir = args.runs_dir or os.path.join(root, "runs")
    deliv_dir = args.deliv_dir or os.path.join(runs_dir, "s48_deliverables")
    out_dir = args.out_dir or os.path.join(runs_dir, "s49_deliverables")
    os.makedirs(out_dir, exist_ok=True)
    only = set(x.strip() for x in args.only.split(",") if x.strip())

    meta_base = OrderedDict([
        ("script", SCRIPT_VERSION), ("generated_utc", now_iso()),
        ("root", root), ("runs_dir", runs_dir), ("deliv_dir", deliv_dir), ("out_dir", out_dir),
        ("argv", sys.argv[1:]),
        ("provenance", "every value is COPIED from an audited artifact named in each row's "
                       "`source`; nothing is recomputed from raw dumps here"),
    ])

    counts = OrderedDict()
    if not only or "speed_summary" in only:
        rows, notes = build_speed(runs_dir, root)
        meta = OrderedDict(meta_base)
        meta["notes"] = notes
        meta["comparability"] = (
            "source_kind='s48_clean_harness' rows (TIGER-kv / SID-MLP / NEZHA / GDN / Mamba2) are "
            "the ONLY mutually comparable block: one single-harness sweep, one H20, node otherwise "
            "IDLE, identical timed region, bs=32 beam=50 warmup=20, 4 reps, full test split, all "
            "three datasets on one pod. source_kind='prior_local_run' rows come from earlier "
            "sessions and/or different harnesses -- read each row's `caveat` before placing it in "
            "a table. PUBLISHED paper speeds are deliberately NOT in this file (doc 4: published "
            "and H20-local numbers must not be mixed into one comparison table).")
        counts["speed_summary"] = write_table(out_dir, "speed_summary", rows, meta)

    if not only or "final_da_quality" in only:
        rows = build_final_da(deliv_dir, root)
        meta = OrderedDict(meta_base)
        meta["scope"] = ("doc 1.4: the final HServe+DA-CDRS checkpoint, full result set "
                         "(aggregate + Head/Mid/Tail R/N@5/10 absolute AND ratio-to-teacher, "
                         "Jac/Overlap@5/10/20, per-bucket request + hit counts, bootstrap CIs, "
                         "per-seed and mean/std). HServe+plain pair is included as the doc 1.7B "
                         "strong control, tagged by `method`.")
        meta["tail_caveat"] = ("Tail cells are reported with absolute values and CIs even where "
                               "the ratio is unstable; no unified Tail recovery claim is made.")
        counts["final_da_quality"] = write_table(out_dir, "final_da_quality", rows, meta)

    if not only or "hserve_mechanism" in only:
        rows, notes = build_hserve_mech(runs_dir, root, args.boot)
        meta = OrderedDict(meta_base)
        meta["notes"] = notes
        meta["scope"] = "doc 1.6A candidate Coverage@C (SID-MLP) + doc 1.6B per-depth HDrift"
        meta["coverage_vs_overlap_check"] = (
            "Coverage@10 here and overlap@10 in reference_agreement measure the same thing and "
            "AGREE to ~1e-3 (SID-MLP 3-seed mean: .684248 vs .685549 Sci, .760327 vs .761062 MI, "
            ".724495 vs .724995 VG). The residual is protocol, not error: (a) frontier_s23 "
            "proposes the student list at M=200 while lists_s48 uses the deployment W=50, so the "
            "top-10s differ on a small tail of requests -- consistent with the s48-measured "
            "student-width sensitivity of <=0.003 jac@10, unsigned; (b) Coverage divides by the "
            "teacher's ACTUAL served-list length, overlap divides by K. Quote either, not both, "
            "in one row of a table.")
        counts["hserve_mechanism"] = write_table(out_dir, "hserve_mechanism", rows, meta)

    if not only or "da_mechanism" in only:
        rows, notes = build_da_mech(runs_dir, root)
        meta = OrderedDict(meta_base)
        meta["notes"] = notes
        meta["scope"] = ("doc 1.7C boundary FlipRate by teacher-margin quintile + doc 1.7D "
                         "attribution faithfulness")
        meta["tautology_warning"] = (
            "The random-depth control is TAUTOLOGICAL and must not be reported as evidence: A_d IS "
            "the counterfactual pair-loss reduction and 'top' is argmax_d A_d, so "
            "reduction(top) >= reduction(random) holds by construction. Use top vs "
            "uniform/all-depth. Rows carry this in `caveat`.")
        counts["da_mechanism"] = write_table(out_dir, "da_mechanism", rows, meta)

    if not only or "kb_sensitivity" in only:
        rows, notes = build_kb(deliv_dir, root)
        meta = OrderedDict(meta_base)
        meta["notes"] = notes
        meta["scope"] = ("doc 2.1 K_eval generalisation (no retraining) + 2.2 K_train sensitivity "
                         "+ 2.3 outsider bound B sensitivity. No full K x B cartesian grid (doc 4).")
        meta["B_sweep_control"] = (
            "Every B cell fixes PAIR_NEGS=10, i.e. the number of sampled negatives per user is held "
            "constant so a B sweep varies the candidate RANGE only, not the pair count / total loss "
            "strength (doc 2.3). The 3-seed main result (arm `hserve_pair`) uses ALL negatives and "
            "is therefore NOT comparable inside the sweep -- `pb40` is its in-sweep counterpart.")
        meta["B_sweep_on_DA_note"] = (
            "The B sweep is run on the main method (HServe+plain pair), not on DA-CDRS: "
            "`da_cdrs_loss` has no fixed-negative-count knob, so a B sweep there would change the "
            "candidate range AND the loss strength together -- exactly the confound doc 2.3 "
            "forbids. The K_train sweep IS additionally run on the DA method (family "
            "`K_train_DA`), where no such control is required.")
        meta["intruder_cdf_pointer"] = (
            "The student-Top-10 intruder teacher-rank CDF that justifies B=40 lives in "
            "kb_intruder_cdf.{json,csv} (width-50 deployment ladder, canonical) and "
            "kb_intruder_cdf_w64.{json,csv} (width-64 ladder, matched to the M=64 teacher TRAINING "
            "dump that B actually indexes). Read each file's `limitation` first: the two ladders "
            "are NOT interchangeable.")
        counts["kb_sensitivity"] = write_table(out_dir, "kb_sensitivity", rows, meta)

    print("\n[deliver_s49] done -> %s" % out_dir)
    for k, v in counts.items():
        print("   %-20s %d rows" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
