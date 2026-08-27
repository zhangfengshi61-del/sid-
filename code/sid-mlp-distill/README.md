# ServeSID overlay for SID-MLP

This directory contains the anonymized source overlay for the final method. It
is applied to the public SID-MLP repository at commit
`02d9d8e9e8a8f7c92af63e108215174b5a60b72c`; unchanged upstream files are not
duplicated.

## Objective

The decoder is trained against the teacher's actually served candidate list,
rather than only teacher-forced token conditionals. The adopted objective adds
three terms to the upstream digit loss:

1. **Uniform serving KL** (`serve_kl_loss`) matches teacher and student mass at
   all four SID depths using deployed-equivalent constrained student scores.
2. **Boundary-pair anchor** (`boundary_pair_loss`) separates teacher top-K
   candidates from outsiders in the served boundary range.
3. **Depth-aware CDRS correction** (`da_cdrs_loss`) matches relative per-depth
   score contributions after normalizing teacher step scales.

The final all-outsider setting is recorded in
`configs/servesid_final.yaml`. The controlled fixed-negative sensitivity setting
uses the same negative indices for the pair and depth-aware terms; the training
loop checks this contract and fails closed when the ranges differ.

## Files

| File | Role |
|---|---|
| `sid_mlp/losses.py` | All paper objectives and deployed-equivalent list scoring. |
| `sid_mlp/train_decoder.py` | Objective wiring, training, validation, and control checks. |
| `sid_mlp/models.py` | Parallel prefix-conditioned SID decoder variants. |
| `sid_mlp/dataset.py` | Packed hidden states, teacher logits, and beam-dump attachment. |
| `sid_mlp/inference.py` | Valid-SID constrained factorized decoding and metrics. |
| `sid_mlp/cli/train.py` | Training CLI and configuration loading. |
| `scripts/train.sh` | Portable launcher using environment-variable paths. |
| `configs/*.yaml` | Upstream-compatible base and final operating points. |

## Use

1. Check out the upstream SID-MLP commit shown above.
2. Copy this directory's `sid_mlp/`, `scripts/`, and `configs/` over that
   checkout.
3. Place the checkout under `repos/SID-MLP/`, or set `SID_MLP_DIR`.
4. Provide the public Amazon Reviews 2023 data, semantic IDs, teacher
   checkpoint, packed teacher cache, and augmented beam dump.
5. From the artifact root:

```bash
export SERVESID_ROOT="$(pwd)"
export CONFIG="$SERVESID_ROOT/code/sid-mlp-distill/configs/servesid_final.yaml"
export BEAM_DUMP=/path/to/augmented_teacher_beam_dump.pt
export GRAPH_DIR=/path/to/packed_teacher_cache
export MODEL_DIR=/path/to/public_model_assets
bash code/sid-mlp-distill/scripts/train.sh sidmlp \
  Industrial_and_Scientific 0
```

The launcher accepts path and hyperparameter overrides through environment
variables; it contains no machine-specific defaults.
