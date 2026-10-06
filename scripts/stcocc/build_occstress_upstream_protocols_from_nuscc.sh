#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$REPO_ROOT}
PYTHON=${PYTHON:-python}
OCCSTRESS_DATA_ROOT=${OCCSTRESS_DATA_ROOT:-$ROOT/data/OccStress}
GENERATOR=$REPO_ROOT/scripts/build_upstream_occstress_protocol.py
BACKBONE_PROTOCOL=${BACKBONE_PROTOCOL:-$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl}
UPSTREAM_OCC_ROOT=${UPSTREAM_OCC_ROOT:-$OCCSTRESS_DATA_ROOT/occ/upstream/OccStress-nuScenes/camera_only/stcocc}
TARGET_OCC_ROOT=${TARGET_OCC_ROOT:-}
OVERWRITE=${OVERWRITE:-0}
CLEAN_ONLY=${CLEAN_ONLY:-0}

CORRUPTIONS=(
  Brightness
  CameraCrash
  ColorQuant
  Fog
  FrameLost
  LowLight
  MotionBlur
  Snow
)
SEVERITIES=(easy mid hard)
FRAME_PROTOCOLS=(current history_k1 all_frame)

common_args=(
  --root "$ROOT"
  --dataset nuscenes
  --occstress-root "$OCCSTRESS_DATA_ROOT"
  --backbone-protocol "$BACKBONE_PROTOCOL"
  --subtrack camera_only
  --source-model stcocc
)

if [[ -n "$TARGET_OCC_ROOT" ]]; then
  common_args+=(--target-root "$TARGET_OCC_ROOT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  common_args+=(--overwrite)
fi

clean_root="$UPSTREAM_OCC_ROOT/clean"
if [[ ! -d "$clean_root" ]]; then
  echo "missing clean upstream root: $clean_root" >&2
  exit 1
fi

if [[ "$CLEAN_ONLY" != "1" ]]; then
  for corruption in "${CORRUPTIONS[@]}"; do
    for severity in "${SEVERITIES[@]}"; do
      if [[ ! -d "$UPSTREAM_OCC_ROOT/$corruption/$severity" ]]; then
        echo "missing export: $UPSTREAM_OCC_ROOT/$corruption/$severity" >&2
        exit 1
      fi
    done
  done
fi

"$PYTHON" "$GENERATOR" \
  "${common_args[@]}" \
  --corruption clean \
  --clean-input-root "$clean_root"

if [[ "$CLEAN_ONLY" == "1" ]]; then
  exit 0
fi

for corruption in "${CORRUPTIONS[@]}"; do
  for severity in "${SEVERITIES[@]}"; do
    corrupted_root="$UPSTREAM_OCC_ROOT/$corruption/$severity"
    for frame_protocol in "${FRAME_PROTOCOLS[@]}"; do
      "$PYTHON" "$GENERATOR" \
        "${common_args[@]}" \
        --corruption "$corruption" \
        --severity "$severity" \
        --frame-protocol "$frame_protocol" \
        --history-k 1 \
        --clean-input-root "$clean_root" \
        --corrupted-input-root "$corrupted_root"
    done
  done
done
