#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29100}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
DEBUGPY_PORT=${DEBUGPY_PORT:-}
DEBUGPY_WAIT_FOR_CLIENT=${DEBUGPY_WAIT_FOR_CLIENT:-1}

if [[ -n "$DEBUGPY_PORT" ]]; then
    if [[ "$GPUS" != "1" || "$NNODES" != "1" ]]; then
        echo "DEBUGPY_PORT is only supported with GPUS=1 and NNODES=1." >&2
        exit 1
    fi

    WAIT_ARGS=()
    if [[ "$DEBUGPY_WAIT_FOR_CLIENT" == "1" ]]; then
        WAIT_ARGS+=(--wait-for-client)
    fi

    PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$PORT \
    WORLD_SIZE=1 \
    RANK=0 \
    LOCAL_RANK=0 \
    python -m debugpy --listen "$DEBUGPY_PORT" "${WAIT_ARGS[@]}" \
        $(dirname "$0")/test.py \
        $CONFIG \
        $CHECKPOINT \
        --launcher pytorch \
        --eval bbox \
        --no-gpu-collect \
        ${@:4}
else
    PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/test.py \
        $CONFIG \
        $CHECKPOINT \
        --launcher pytorch \
        --eval bbox \
        --no-gpu-collect \
        ${@:4}
fi
