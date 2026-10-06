#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OCCSTRESS_CODE_ROOT:-$(realpath "$SCRIPT_DIR/../..")}"
DATA_ROOT="${SDGOCC_DATA_ROOT:-$ROOT/data/nuscenes}"

declare -A MAP=(
  ["$DATA_ROOT/fusionocc-nuscenes_infos_train.pkl"]="$DATA_ROOT/bevdetv2-nuscenes_infos_train.pkl"
  ["$DATA_ROOT/fusionocc-nuscenes_infos_val.pkl"]="$DATA_ROOT/bevdetv2-nuscenes_infos_val.pkl"
)

for src in "${!MAP[@]}"; do
  dst="${MAP[$src]}"
  if [[ ! -f "$src" ]]; then
    echo "missing source info file: $src" >&2
    exit 1
  fi
  rm -f "$dst"
  ln -s "$src" "$dst"
  printf 'linked %s -> %s\n' "$dst" "$(readlink -f "$dst")"
done
