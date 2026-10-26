#!/usr/bin/env bash
# MuSID-KD: same data/evaluation protocol, independent seeds (42,43,44).
set -euo pipefail
CATEGORY="${1:?Usage: run_musid_seeds.sh CATEGORY [CUDA_DEVICE]}"
CUDA_DEVICE="${2:-0}"
DATASET="${DATASET:-AmazonReviews2014}"
READOUT="${READOUT:-mlp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/code/sid-mlp-distill/configs/musid_kd.yaml}"
for SEED in 42 43 44; do
    echo "[MuSID-KD] dataset=$DATASET category=$CATEGORY readout=$READOUT seed=$SEED"
    bash "$ROOT/code/sid-mlp-distill/scripts/train.sh" sidmlp "$CATEGORY" "$CUDA_DEVICE" \
        --dataset "$DATASET" \
        --dataset_tag "${DATASET}_${CATEGORY}" \
        --val_dataset_tag "${DATASET}_${CATEGORY}_val" \
        --readout "$READOUT" --seed "$SEED"
done
