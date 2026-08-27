# Source-code inventory

This directory is the anonymized, paper-relevant source snapshot used for the
AAAI Code & Data appendix. It contains every implementation that can affect the
reported method, evaluation tables, mechanisms, cost measurements, or
independent checks. Historical logs, checkpoints, one-off shell wrappers,
backups, and third-party repository copies are intentionally excluded.

Set `SERVESID_ROOT` to the repository root before running scripts that need
datasets or upstream repositories:

```bash
export SERVESID_ROOT="$(pwd)"
```

The expected optional layout is `repos/SID-MLP/` plus the public datasets and
teacher assets described in `aaai_artifact/ENVIRONMENT.md`. Every public script
uses relative paths or command-line arguments; no authoring-machine or cluster
path is embedded.

## Final method

`sid-mlp-distill/` is the source overlay for upstream SID-MLP commit
`02d9d8e9e8a8f7c92af63e108215174b5a60b72c`.

- `sid_mlp/losses.py`: deployed list scoring, uniform serving KL,
  boundary-pair loss, and depth-aware CDRS correction.
- `sid_mlp/train_decoder.py`: objective wiring, shared-negative controls, and
  validation.
- `sid_mlp/{models,dataset,inference}.py`: decoder, packed teacher-data loader,
  and deployed constrained inference.
- `sid_mlp/cli/train.py`, `scripts/train.sh`: CLI and portable launcher.
- `configs/servesid_final.yaml`: the adopted all-outsider operating point.

See `sid-mlp-distill/README.md` for the overlay procedure and objective.

## Paper pipeline

| Stage | Source |
|---|---|
| Teacher-list augmentation | `augment_dump_cdrs.py`, `add_step_scale_s43e.py` |
| Calibration | `cdrs_calibrate_s41.py`, `da_calibrate_s43e.py`, `da_calibrate_s45.py` |
| Main evaluation | `frontier_s23.py`, `lists_s48.py`, `bucket_diag_s34.py` |
| Aggregate tables | `metrics_s48.py`, `deliver_s49.py`, `finalize_s53.py` |
| Mechanism analyses | `mechanism_s48.py`, `intruder_cdf_s48.py`, `fig4_panels_s36.py`, `tcs_mine_s46.py` |
| Efficiency | `speed_s48.py` and `baselines/*throughput.py` |
| Provenance / verification | `manifest_s48.py`, `verify_s53.py` |

`verify_s53.py` is the zero-GPU, independent rank/set implementation used by
the packaged quick check. It does not import the scorer that generated the
tables.

## Baselines and earlier controls

- `baselines/`: AtSpeed-S/R, EARN, NEZHA, GatedDeltaNet/Mamba2, bucketed
  evaluation, and throughput aggregation.
- `*_microbench.py`, `serving_sim.py`, `cont_cost.py`, `contam_bench.py`:
  self-contained decode/serving primitives.
- `eteg_*.py`, `tiger_rescore_depth.py`, `rpg_*.py`, `sidmlp_*.py`: reference
  reproductions and mechanism checks against public upstream repositories.
- `prefix_geometry.py`, `scale_analysis.py`, `hsweep_analyze.py`,
  `cce_eval.py`, `resolver_eval.py`: synthetic and offline controls.

The two `patch_*.py` files are retained as provenance for the historical
development sequence. They are not required when using the final overlay in
`sid-mlp-distill/`.

## What “complete” means

Complete means all source needed to trace or rerun paper-facing results. It does
not mean vendoring unrelated upstream projects or retaining every abandoned
experiment. Excluded material is reproducibly regenerable or non-scientific:

- third-party repositories and their `.git` histories;
- model checkpoints, raw logs, environment directories, and caches;
- obsolete `.bak` files and duplicate wrappers containing only infrastructure
  launch details;
- raw Amazon review records, which remain governed by the public dataset's
  distribution terms.

The whitelist used for the reviewer ZIP is defined in
`aaai_artifact/build_zip.py`.
