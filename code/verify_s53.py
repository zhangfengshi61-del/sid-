#!/usr/bin/env python3
"""Independent per-request re-derivation of every s53 list-table metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def row_metrics(path: Path) -> dict[str, float]:
    z = np.load(path, allow_pickle=False)
    valid = z["valid"].astype(bool)
    gt = z["gt_item"][valid]
    ttop = z["t_top"][valid]
    stop = z["s_top"][valid]
    tn = z["t_n"][valid]
    sn = z["s_n"][valid]
    out = {}
    for k in (5, 10, 20):
        tr = []
        sr = []
        jac = []
        overlap = []
        for i in range(len(gt)):
            t = ttop[i, : min(k, int(tn[i]))].tolist()
            s = stop[i, : min(k, int(sn[i]))].tolist()
            ts = set(t)
            ss = set(s)
            tr.append(t.index(int(gt[i])) if int(gt[i]) in ts else -1)
            sr.append(s.index(int(gt[i])) if int(gt[i]) in ss else -1)
            inter = len(ts & ss)
            union = len(ts | ss)
            jac.append(inter / union if union else 1.0)
            overlap.append(inter / max(1, min(k, len(t), len(s))))
        tr = np.asarray(tr)
        sr = np.asarray(sr)
        out[f"teacher_r{k}"] = float(np.mean(tr >= 0))
        out[f"student_r{k}"] = float(np.mean(sr >= 0))
        tndcg = np.zeros(len(tr), dtype=np.float64)
        sndcg = np.zeros(len(sr), dtype=np.float64)
        tmask = tr >= 0
        smask = sr >= 0
        tndcg[tmask] = 1 / np.log2(tr[tmask] + 2)
        sndcg[smask] = 1 / np.log2(sr[smask] + 2)
        out[f"teacher_n{k}"] = float(np.mean(tndcg))
        out[f"student_n{k}"] = float(np.mean(sndcg))
        out[f"jac{k}"] = float(np.mean(jac))
        out[f"overlap{k}"] = float(np.mean(overlap))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/derived/s53",
        help="Directory containing one public-ID .json/.npz pair per experiment cell.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional JSON output path; the compact summary is always printed.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    summaries = sorted(
        path for path in data_dir.glob("*.json")
        if path.name != "index.json"
    )
    if not summaries:
        raise SystemExit(f"No JSON summaries found under {data_dir}")

    checks = []
    worst = 0.0
    for src in summaries:
        with src.open() as f:
            row = json.load(f)
        npz = src.with_suffix(".npz")
        if not npz.exists():
            raise SystemExit(f"Missing paired NPZ: {npz}")
        derived = row_metrics(npz)
        diffs = {key: abs(float(row[key]) - value) for key, value in derived.items()}
        cell_worst = max(diffs.values())
        worst = max(worst, cell_worst)
        checks.append({
            "cell_id": src.stem,
            "dataset": row["category"],
            "max_abs_error": cell_worst,
            "passed": cell_worst <= 1e-12,
            "diffs": diffs,
        })
    payload = {
        "implementation": "independent NumPy rank/set loop; does not import lists_s48.py",
        "n_cells": len(checks),
        "metrics_per_cell": 18,
        "max_abs_error": worst,
        "passed": bool(checks) and all(row["passed"] for row in checks),
        "checks": checks,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(payload, f, indent=1)
    print(json.dumps({
        key: payload[key]
        for key in ("n_cells", "metrics_per_cell", "max_abs_error", "passed")
    }))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
