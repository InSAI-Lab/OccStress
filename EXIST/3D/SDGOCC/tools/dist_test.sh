#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
NATTEN_LIB="$(find "$REPO_ROOT/projects/natten/src/build" -maxdepth 1 -type d -name 'lib.*' | head -n 1)"
PYTHONPATH_PREFIX="$REPO_ROOT/mmdetection3d:$REPO_ROOT/projects:$REPO_ROOT"
if [[ -n "$NATTEN_LIB" ]]; then
  PYTHONPATH_PREFIX="$NATTEN_LIB:$PYTHONPATH_PREFIX"
fi

PYTHONPATH="$PYTHONPATH_PREFIX:${PYTHONPATH:-}" \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$SCRIPT_DIR/test.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --launcher pytorch \
    ${@:4}
