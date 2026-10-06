#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py}
WORK_DIR=${2:-./work_dirs/your_folder_name}

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/occ_train.py "$CONFIG" --work-dir "$WORK_DIR" "${@:3}"
