#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py}
CHECKPOINT=${2:-./ckpts/viewformer_res50_704x256_depthpretrain_90e.pth}

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/occ_test.py "$CONFIG" "$CHECKPOINT" --eval occ "${@:3}"
