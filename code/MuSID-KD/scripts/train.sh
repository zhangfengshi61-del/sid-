#!/usr/bin/env bash
# Portable launcher: MuSID-KD default; ServeSID remains explicitly selectable.
#
# Usage:
#   train.sh [mode] [category] [cuda_device]
#
#   mode         musid-kd (default), sidmlp / sidmlp-pp-* (historical)
#   category     e.g. Industrial_and_Scientific
#   cuda_device  GPU index (default: 0)
#
# Paths are supplied through environment variables (no machine-specific
# defaults). The model assets directory is expected to contain, per category:
#   $MODEL_DIR/<category>/checkpoint.pth   (TIGER teacher checkpoint)
#   $MODEL_DIR/<category>/sem_ids.pt       (semantic IDs)
# Override either with CHECKPOINT / SEM_IDS directly.

set -euo pipefail

MODE="${1:-musid-kd}"
CATEGORY="${2:-}"
CUDA_DEVICE="${3:-${CUDA_DEVICE:-0}}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MUSID_KD_ROOT="${MUSID_KD_ROOT:-$ROOT}"

SID_MLP_DIR="${SID_MLP_DIR:-$MUSID_KD_ROOT/repos/SID-MLP}"
CONFIG="${CONFIG:-$MUSID_KD_ROOT/code/MuSID-KD/configs/musid_kd.yaml}"
GRAPH_DIR="${GRAPH_DIR:-${SID_MLP_GRAPH_DIR:-}}"
MODEL_DIR="${MODEL_DIR:-}"
BEAM_DUMP="${BEAM_DUMP:-}"

: "${CATEGORY:?Set CATEGORY via the second argument (e.g. Industrial_and_Scientific).}"
: "${GRAPH_DIR:?Set GRAPH_DIR to the packed teacher cache directory.}"
: "${MODEL_DIR:?Set MODEL_DIR to the public model assets directory.}"
: "${BEAM_DUMP:?Set BEAM_DUMP to the augmented teacher beam dump (.pt).}"

CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/$CATEGORY/checkpoint.pth}"
SEM_IDS="${SEM_IDS:-$MODEL_DIR/$CATEGORY/sem_ids.pt}"

for path in "$CONFIG" "$CHECKPOINT" "$SEM_IDS" "$BEAM_DUMP"; do
    if [[ ! -e "$path" ]]; then
        echo "error: required path does not exist: $path" >&2
        exit 1
    fi
done

export PYTHONPATH="$MUSID_KD_ROOT/code/MuSID-KD:$SID_MLP_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_DIR="${PROJECT_DIR:-$SID_MLP_DIR}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

ARGS=( "$MODE" --config "$CONFIG" --category "$CATEGORY" \
       --checkpoint "$CHECKPOINT" --sem_ids "$SEM_IDS" \
       --graph_dir "$GRAPH_DIR" --beam_dump "$BEAM_DUMP" )

exec python -m musid_kd.cli.train "${ARGS[@]}" "${@:4}"
