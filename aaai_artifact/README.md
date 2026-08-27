# Anonymous AAAI-27 Code & Data Appendix

This directory is the entry point for the anonymous reviewer package. The
archive is designed to support three levels of inspection:

1. a zero-GPU integrity and independent-metric check;
2. table and mechanism regeneration from the included derived data;
3. full training/evaluation after obtaining the public Amazon Reviews 2023
   data and upstream model repositories.

## Quick check (zero GPU)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r aaai_artifact/requirements-review.txt
python aaai_artifact/check_artifact.py
python code/verify_s53.py --data-dir data/derived/s53
```

The second command independently recomputes 18 ranked-list metrics for each of
14 complete per-request cells. It does not import the table-generating scorer.

## Layout

| Path | Contents |
|---|---|
| `code/sid-mlp-distill/` | Final method overlay, launcher, and adopted configuration. |
| `code/baselines/` | Same-harness baseline implementations and speed evaluation. |
| `code/*.py` | Data preparation, evaluation, mechanisms, table assembly, and verification. |
| `data/derived/s53/` | De-identified per-request ranked lists for 14 sensitivity cells. |
| `delivery/2026-07-22/` | Complete cross-method aggregate tables and run manifests. |
| `delivery/2026-07-27/` | Supplemental aggregate tables and verification summary. |

`code/README.md` is the complete source inventory. `data/README.md` defines the
data schema and availability boundary.

## Full training

The final method is an overlay on public SID-MLP commit
`02d9d8e9e8a8f7c92af63e108215174b5a60b72c`. Apply the overlay as described in
`code/sid-mlp-distill/README.md`, prepare the public Amazon Reviews 2023
categories with the upstream TIGER/SID-MLP pipeline, then use:

```bash
export SERVESID_ROOT="$(pwd)"
export CONFIG="$SERVESID_ROOT/code/sid-mlp-distill/configs/servesid_final.yaml"
bash code/sid-mlp-distill/scripts/train.sh sidmlp \
  Industrial_and_Scientific 0
```

Paths for model assets, semantic IDs, caches, and output directories are
explicit environment variables in the launcher. No private infrastructure is
required by the source code.

## Build the OpenReview ZIP

From the repository root:

```bash
python aaai_artifact/build_zip.py --output /tmp/ServeSID-AAAI27-Code-Data.zip
```

The builder first runs the integrity/anonymization check, packages only a fixed
whitelist, normalizes ZIP timestamps, and writes a `SHA256SUMS` entry for every
file. Do not upload a ZIP of the whole Git working tree.

## Scope

The archive contains all code that can affect paper-facing results. It does not
vendor third-party repositories, raw reviews, checkpoints, logs, caches,
abandoned backups, or infrastructure wrappers. These exclusions are described
in `code/README.md`, `DATA_AND_LICENSES.md`, and `ANONYMIZATION.md`.
