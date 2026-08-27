#!/usr/bin/env python3
"""Validate syntax, derived-data safety, reproducibility, and anonymity."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from package_spec import ROOT, package_files


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_LITERALS = (
    "/" + "mnt/",
    "/" + "Users/",
    "/" + "root/",
    "/" + "home/",
    "shu" + "xiaoxie",
    "sha" + "7dow",
    "xie" + "Twim",
    "ten" + "cent",
    "." + "woa.",
    "mmb" + "iz",
    "wx" + "g",
    "ce" + "ph",
    "gem" + "ini",
    "gpu" + "q",
)

FORBIDDEN_REGEXES = (
    re.compile(r"\bmpi-[0-9]"),
    re.compile(r"\bworker-[0-9]"),
    re.compile(r"\bpod-[0-9]"),
    re.compile(r"\b[LR][0-9]{4}\b"),
    re.compile(r"\bsession[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\btask[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_ -]?key|password|credential)\s*[:=]", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9+/]{80,}={0,2}"),
)

NPZ_FIELDS = {
    "gt_item",
    "gt_pop",
    "gt_bucket",
    "valid",
    "t_top",
    "s_top",
    "t_n",
    "s_n",
    "s_raw",
    "s_invalid",
    "bucket_lo",
    "bucket_hi",
    "n_test",
    "W",
    "topk",
}


def scan_text() -> int:
    checked = 0
    failures: list[str] = []
    for path in package_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        checked += 1
        lower = text.lower()
        for literal in FORBIDDEN_LITERALS:
            if literal.lower() in lower:
                failures.append(f"{rel}: forbidden literal {literal!r}")
        for pattern in FORBIDDEN_REGEXES:
            if pattern.search(text):
                failures.append(f"{rel}: forbidden pattern {pattern.pattern!r}")
    if failures:
        raise SystemExit("\n".join(failures))
    return checked


def check_python_syntax() -> int:
    checked = 0
    for path in package_files():
        if path.suffix != ".py":
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checked += 1
    return checked


def check_derived_data() -> tuple[int, int]:
    data_dir = ROOT / "data/derived/s53"
    index = json.loads((data_dir / "index.json").read_text())
    expected = {row["cell_id"] for row in index["cells"]}
    json_cells = {path.stem for path in data_dir.glob("*.json")} - {"index"}
    npz_cells = {path.stem for path in data_dir.glob("*.npz")}
    if json_cells != expected or npz_cells != expected:
        raise SystemExit(
            f"Derived-data cell mismatch: expected={sorted(expected)}, "
            f"json={sorted(json_cells)}, npz={sorted(npz_cells)}"
        )

    rows = 0
    for cell_id in sorted(expected):
        summary = json.loads((data_dir / f"{cell_id}.json").read_text())
        forbidden_keys = {"tag", "seed_tag", "lens_path", "checkpoint", "source"}
        leaked = forbidden_keys.intersection(summary)
        if leaked:
            raise SystemExit(f"{cell_id}.json retains private metadata: {sorted(leaked)}")
        with np.load(data_dir / f"{cell_id}.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != NPZ_FIELDS:
                raise SystemExit(f"{cell_id}.npz field mismatch: {sorted(arrays.files)}")
            for key in arrays.files:
                if arrays[key].dtype.kind in {"O", "S", "U", "V"}:
                    raise SystemExit(f"{cell_id}.npz contains non-numeric field {key}")
            n_test = int(arrays["n_test"])
            if arrays["gt_item"].shape != (n_test,):
                raise SystemExit(f"{cell_id}.npz n_test mismatch")
            rows += n_test
    return len(expected), rows


def check_independent_metrics() -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "code/verify_s53.py",
            "--data-dir",
            "data/derived/s53",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    text_files = scan_text()
    python_files = check_python_syntax()
    cells, rows = check_derived_data()
    verification = check_independent_metrics()
    payload = {
        "passed": True,
        "text_files_scanned": text_files,
        "python_files_parsed": python_files,
        "derived_cells": cells,
        "derived_request_rows": rows,
        "metric_verification": verification,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
