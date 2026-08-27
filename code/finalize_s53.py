#!/usr/bin/env python3
"""Build the collaborator-facing s53 sensitivity, cost, and manifest tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
from datetime import datetime, timedelta
from pathlib import Path


CATS = {
    "Industrial_and_Scientific": ("Sci", 259_992),
    "Musical_Instruments": ("MI", 339_519),
    "Video_Games": ("VG", 530_300),
}
METRICS = [
    "teacher_r5", "teacher_n5", "student_r5", "student_n5", "jac5", "overlap5",
    "teacher_r10", "teacher_n10", "student_r10", "student_n10", "jac10", "overlap10",
    "teacher_r20", "teacher_n20", "student_r20", "student_n20", "jac20", "overlap20",
]
K_ARMS = {
    5: ("dak5daw31", 35),
    10: ("daw31", 40),
    20: ("dak20daw31", 50),
}
B_ARMS = {b: f"dab{b}daw31" for b in (20, 30, 40, 50, 60)}
RUN_TAGS = [
    "s53_dak5_daw31_Sci_s42",
    "s53_dak20_daw31_Sci_s42",
    "s53_dak5_daw31_MI_s42",
    "s53_dak20_daw31_MI_s42",
    "s53_dak5_daw31_VG_s42",
    "s53_dak20_daw31_VG_s42",
    "s53_r0base_cost_Sci_s42",
    *[f"s53_dab{b}_daw31_Sci_s42" for b in (20, 30, 40, 50, 60)],
]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def write_table(out: Path, name: str, rows: list[dict], meta: dict) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    payload = {"meta": meta, "fields": fields, "rows": rows}
    with (out / f"{name}.json").open("w") as f:
        json.dump(payload, f, indent=1)
    with (out / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_kv(path: Path) -> dict[str, str]:
    ans = {}
    if not path.exists():
        return ans
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            ans[key] = value
    return ans


def parse_recipe(text: str) -> dict[str, str]:
    ans = {}
    for token in shlex.split(text):
        if "=" in token:
            key, value = token.split("=", 1)
            ans[key] = value
    return ans


def last_epoch_fields(path: Path) -> dict[str, str]:
    last = {}
    if not path.exists():
        return last
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        if not line.startswith("epoch="):
            continue
        last = {
            key: value
            for token in line.split()
            if "=" in token
            for key, value in [token.split("=", 1)]
        }
    return last


def parse_datetime(text: str) -> datetime:
    return datetime.fromisoformat(text)


def timed_elapsed(path: Path) -> float | None:
    if not path.exists():
        return None
    marker = "Elapsed (wall clock) time (h:mm:ss or m:ss):"
    for line in path.read_text(errors="replace").splitlines():
        if marker not in line:
            continue
        value = line.split(marker, 1)[1].strip()
        parts = [float(x) for x in value.split(":")]
        if len(parts) == 2:
            return 60 * parts[0] + parts[1]
        if len(parts) == 3:
            return 3600 * parts[0] + 60 * parts[1] + parts[2]
    return None


def fresh_export_rows(root: Path) -> list[dict]:
    """Parse the isolated current-pod export wave, but only after its DONE gate."""
    runs = root / "runs"
    master = runs / "s53_cost" / "export" / "master.log"
    if not (runs / "S53_EXPORT_COST.DONE").exists() or not master.exists():
        return []

    rows = []
    for line in master.read_text(errors="replace").splitlines():
        tokens = shlex.split(line)
        if not tokens or tokens[0] not in {"base_end", "augment_end"}:
            continue
        fields = {
            key: value
            for token in tokens[1:]
            if "=" in token
            for key, value in [token.split("=", 1)]
        }
        cat = fields["cat"]
        is_base = tokens[0] == "base_end"
        if not is_base and fields.get("exit_code") != "0":
            continue
        cdir = master.parent / cat
        source = cdir / ("base_compare.txt" if is_base else "augment.time.txt")
        row = {
            "scope": "teacher_export",
            "dataset": cat,
            "dataset_short": CATS[cat][0],
            "item": "base_width64_dump" if is_base else "da_per_depth_augmentation",
            "wall_seconds": int(fields["wall_seconds"]),
            "gpus": 8 if is_base else 1,
            "source": rel(source, root),
            "master_log": rel(master, root),
            "caveat": (
                "fresh current-pod 8-way sharded export; merged tensors checked "
                "against the canonical width-64 dump"
                if is_base else
                "fresh current-pod timing; includes per-depth augmentation and "
                "step-scale backfill"
            ),
        }
        if is_base:
            checks = read_kv(source)
            equal_keys = [key for key in checks if key.endswith("_equal")]
            row["verified_canonical_equal"] = bool(equal_keys) and all(
                checks[key] == "1" for key in equal_keys
            )
        else:
            row["verified_canonical_equal"] = None
        rows.append(row)

    # Never mix a partially parsed fresh wave with historical timings.
    return rows if len(rows) == 2 * len(CATS) else []


def training_end(log: Path, start: datetime) -> datetime | None:
    if not log.exists():
        return None
    pattern = re.compile(r"^=== EVAL frontier .* @ (\d\d:\d\d:\d\d) ===$")
    for line in log.read_text(errors="replace").replace("\r", "\n").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        tod = datetime.strptime(match.group(1), "%H:%M:%S").time()
        end = datetime.combine(start.date(), tod, tzinfo=start.tzinfo)
        if end < start:
            end += timedelta(days=1)
        return end
    return None


def result_rows(root: Path) -> list[dict]:
    runs = root / "runs"
    rows = []
    for cat, (short, _) in CATS.items():
        loaded = {}
        for kval, (method, bval) in K_ARMS.items():
            path = runs / f"lists_s48_{cat}__{method}__s42.json"
            if not path.exists():
                continue
            loaded[kval] = (load_json(path), path, bval)
        ref = loaded.get(10, (None, None, None))[0]
        for kval in sorted(loaded):
            data, path, bval = loaded[kval]
            row = {
                "axis": "K_train",
                "dataset": cat,
                "dataset_short": short,
                "seed": 42,
                "K_train": kval,
                "B": bval,
                "DA_W": 31.42,
                "DA_NEGS": "all_outsiders",
                "n_rows": data["n_rows"],
                "n_valid": data["n_valid"],
                "mean_student_invalid": data["mean_s_invalid"],
                "short_at_10": data["short_at_10"],
                "short_at_20": data["short_at_20"],
                "checkpoint": os.path.basename(data["lens_path"]),
                "source": rel(path, root),
                "caveat": (
                    "K10 reuses the adopted final checkpoint; K5/K20 were retrained "
                    "at DA_W=31.42. One seed per cell."
                ),
            }
            for metric in METRICS:
                row[metric] = data[metric]
                row[f"delta_{metric}_vs_K10"] = (
                    data[metric] - ref[metric] if ref is not None else None
                )
            rows.append(row)

    cat = "Industrial_and_Scientific"
    short = CATS[cat][0]
    b_loaded = {}
    for bval, method in B_ARMS.items():
        path = runs / f"lists_s48_{cat}__{method}__s42.json"
        if path.exists():
            b_loaded[bval] = (load_json(path), path)
    b_ref = b_loaded.get(40, (None, None))[0]
    for bval in sorted(b_loaded):
        data, path = b_loaded[bval]
        tag = f"s53_dab{bval}_daw31_Sci_s42"
        train_fields = last_epoch_fields(runs / f"s43e_train_{tag}.log")
        row = {
            "axis": "B",
            "dataset": cat,
            "dataset_short": short,
            "seed": 42,
            "K_train": 10,
            "B": bval,
            "DA_W": 31.42,
            "DA_NEGS": 10,
            "n_rows": data["n_rows"],
            "n_valid": data["n_valid"],
            "mean_student_invalid": data["mean_s_invalid"],
            "short_at_10": data["short_at_10"],
            "short_at_20": data["short_at_20"],
            "checkpoint": os.path.basename(data["lens_path"]),
            "source": rel(path, root),
            "caveat": (
                "Controlled Sci-only B sweep: the same deterministic ten outsider "
                "indices are consumed by pair and DA. One seed per cell."
            ),
            "train_last_epoch": int(train_fields["epoch"]) if train_fields else None,
            "train_da_neg_valid_mean": (
                float(train_fields["da_neg_valid_mean"]) if train_fields else None
            ),
            "train_da_neg_sample_mean": (
                float(train_fields["da_neg_sample_mean"]) if train_fields else None
            ),
            "train_da_neg_short_frac": (
                float(train_fields["da_neg_short_frac"]) if train_fields else None
            ),
        }
        for metric in METRICS:
            row[metric] = data[metric]
            row[f"delta_{metric}_vs_B40"] = (
                data[metric] - b_ref[metric] if b_ref is not None else None
            )
        rows.append(row)
    return rows


def manifest_rows(root: Path) -> list[dict]:
    rows = []
    cost = root / "runs" / "s53_cost"
    ckpt_dir = root / "sidmlp-graph" / "SID-MLP" / "ckpt"
    for tag in RUN_TAGS:
        meta_path = cost / f"{tag}.meta.txt"
        meta = read_kv(meta_path)
        if not meta:
            continue
        cat = meta["category"]
        recipe = parse_recipe(meta.get("recipe", ""))
        ckpt = ckpt_dir / (
            f"{cat}_sidmlp_k0_xattn_h4_f1024_ad384_hhid512_hl1_lr5e-5_"
            f"{tag}_final.pth"
        )
        frontier = root / "runs" / f"frontier_s43e_{tag}.json"
        recovered = root / "runs" / f"S53_RECOVERED_{tag}.DONE"
        wrapper = "runs/s53_arm_p1.sh" if "_dab" in tag else "runs/s53_arm.sh"
        recipe_text = " ".join(f"{key}={value}" for key, value in recipe.items())
        row = {
            "tag": tag,
            "dataset": cat,
            "seed": int(meta["seed"]),
            "status": (
                "recovered_first_pass" if recovered.exists()
                else "complete" if (root / "runs" / f"S53_{tag}.DONE").exists()
                else "incomplete"
            ),
            "checkpoint": rel(ckpt, root) if ckpt.exists() else "",
            "frontier": rel(frontier, root) if frontier.exists() else "",
            "meta_source": rel(meta_path, root),
            "teacher_dump": rel(Path(meta["beam_dump"]), root),
            "launch_command": (
                f"env TAG={tag} CAT={cat} DUMP={rel(Path(meta['beam_dump']), root)} "
                f"{recipe_text} bash {wrapper}"
            ),
            "caveat": (
                "first-pass artifacts recovered after a post-completion duplicate-wrapper "
                "incident; see the per-tag recovery record"
                if recovered.exists() else ""
            ),
            "sidmlp_commit": meta["sidmlp_commit"],
            "losses_sha256": meta["losses_sha256"],
            "train_decoder_sha256": meta["train_decoder_sha256"],
            "train_cli_sha256": meta["train_cli_sha256"],
            "train_sh_sha256": meta["train_sh_sha256"],
        }
        row.update(recipe)
        rows.append(row)
    return rows


def cost_rows(root: Path) -> list[dict]:
    runs = root / "runs"
    cost = runs / "s53_cost"
    rows: list[dict] = []

    for cat, (short, _) in CATS.items():
        for kind, pattern in [
            ("served_list_eval_cache", f"tbeam50_{cat}.pt"),
            ("da_training_dump", f"bmdump_cdrs_{cat}.pt"),
        ]:
            path = runs / pattern
            if not path.exists():
                continue
            rows.append({
                "scope": "cache",
                "dataset": cat,
                "dataset_short": short,
                "item": kind,
                "bytes": path.stat().st_size,
                "gib": path.stat().st_size / 2**30,
                "source": rel(path, root),
                "caveat": "",
            })

    fresh_rows = fresh_export_rows(root)
    if fresh_rows:
        rows.extend(fresh_rows)
    else:
        historical_rows = [
            ("Industrial_and_Scientific", "base_width64_dump", 164, 8,
             "runs/bmdump_Industrial_and_Scientific.masterlog", ""),
            ("Musical_Instruments", "base_width64_dump", 198, 8,
             "runs/bmdump_Musical_Instruments.masterlog", ""),
            ("Musical_Instruments", "da_per_depth_augmentation", 542, 1,
             "runs/s43e_aug_Musical_Instruments.log", "includes step-scale backfill"),
            ("Video_Games", "da_per_depth_augmentation", 750, 1,
             "runs/s43e_aug_Video_Games.log", "includes step-scale backfill"),
        ]
        for cat, stage, seconds, gpus, source, caveat in historical_rows:
            rows.append({
                "scope": "teacher_export",
                "dataset": cat,
                "dataset_short": CATS[cat][0],
                "item": stage,
                "wall_seconds": seconds,
                "gpus": gpus,
                "source": source,
                "caveat": caveat + (
                    "; historical fallback used because the fresh isolated export "
                    "wave is not complete"
                ),
            })

    train_rows = []
    for tag in RUN_TAGS:
        meta_path = cost / f"{tag}.meta.txt"
        meta = read_kv(meta_path)
        if not meta or "end" not in meta:
            continue
        cat = meta["category"]
        recipe = parse_recipe(meta.get("recipe", ""))
        start = parse_datetime(meta["start"])
        end = parse_datetime(meta["end"])
        log = runs / f"s43e_train_{tag}.log"
        train_done = training_end(log, start)
        train_seconds = (train_done - start).total_seconds() if train_done else None
        time_path = cost / f"{tag}.time.txt"
        total_seconds = timed_elapsed(time_path) or (end - start).total_seconds()
        epochs = int(recipe.get("EPOCHS", 200))
        samples = CATS[cat][1]
        row = {
            "scope": "training",
            "dataset": cat,
            "dataset_short": CATS[cat][0],
            "item": tag,
            "train_wall_seconds": train_seconds,
            "seconds_per_epoch": train_seconds / epochs if train_seconds else None,
            "total_train_plus_frontier_seconds": total_seconds,
            "frontier_seconds": total_seconds - train_seconds if train_seconds else None,
            "train_samples_per_second": samples * epochs / train_seconds if train_seconds else None,
            "peak_gpu_memory_mib": int(meta.get("peak_gpu_memory_mib", 0)),
            "source": rel(meta_path, root),
            "time_source": rel(time_path, root),
            "log_source": rel(log, root),
            "caveat": (
                "single GPU; includes packed-cache load in train wall time"
                + (
                    "; first pass recovered after a post-completion duplicate-wrapper "
                    "incident; total comes from the exit-0 /usr/bin/time artifact"
                    if (runs / f"S53_RECOVERED_{tag}.DONE").exists() else ""
                )
            ),
            **recipe,
        }
        train_rows.append(row)

    baseline = next((r for r in train_rows if r["item"] == "s53_r0base_cost_Sci_s42"), None)
    for row in train_rows:
        if baseline and row["dataset"] == "Industrial_and_Scientific":
            row["throughput_vs_sidmlp"] = (
                row["train_samples_per_second"] / baseline["train_samples_per_second"]
            )
        else:
            row["throughput_vs_sidmlp"] = None
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "runs" / "delivery" / "2026-07-27"
    out.mkdir(parents=True, exist_ok=True)

    common = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "PROVISIONAL",
        "paper_trail_gate": (
            "Project ledger fails under paper-trail 0.13.1; these tables are not "
            "paper-facing until the ledger migration and coverage check are green."
        ),
    }
    write_table(out, "kb_sensitivity_da31", result_rows(root), {
        **common,
        "protocol": (
            "W=50, seed42. K sweep uses final DA_W=31.42; B sweep fixes "
            "DA_NEGS=PAIR_NEGS=10 and shares the exact indices."
        ),
    })
    write_table(out, "offline_cost_da31", cost_rows(root), {
        **common,
        "refresh_contract": (
            "Teacher change => refresh train/val/test hidden-state packed caches, "
            "width-64 train dump, DA augmentation, and width-50 eval cache. "
            "Student-only hyperparameter changes reuse all teacher caches."
        ),
    })
    manifests = manifest_rows(root)
    write_table(out, "run_manifest_da31", manifests, {
        **common,
        "complete_rows": sum(r["status"] in {"complete", "recovered_first_pass"} for r in manifests),
        "note": "Code hashes disambiguate the dirty live tree from the Git HEAD.",
    })
    print(out)


if __name__ == "__main__":
    main()
