# OccStress Adapter Notes

This directory keeps the STCOcc source layout used for the `upstream/camera_only/stcocc` track.

## Paths

```bash
export OCCSTRESS_CODE_ROOT=$(cd ../../.. && pwd)
export OCCSTRESS_DATA_ROOT=$OCCSTRESS_CODE_ROOT/data/OccStress
```

## Common Output Layout

```text
data/OccStress/occ/upstream/OccStress-nuScenes/camera_only/stcocc/<corruption>/<severity>/<scene>/<token>/labels.npz
data/OccStress/protocols/upstream/OccStress-nuScenes/camera_only/stcocc/<corruption>/<severity>/<protocol>.pkl
```

## Helper Scripts

Build the clean upstream protocol with explicit release-layout inputs:

```bash
python ../../../scripts/build_upstream_occstress_protocol.py \
  --root "$OCCSTRESS_CODE_ROOT" \
  --backbone-protocol "$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl" \
  --clean-input-root "$OCCSTRESS_DATA_ROOT/occ/upstream/OccStress-nuScenes/camera_only/stcocc/clean" \
  --subtrack camera_only --source-model stcocc --corruption clean
```

For a corrupted protocol, also specify `--corrupted-input-root`, `--corruption`,
`--severity` and `--frame-protocol`. Clean has no severity directory. The old
batch shell helper retains legacy defaults; do not run it against a finalized
release root without explicitly adapting its inputs. For native export settings
and the completed smoke gate, see [upstream exports](../../../docs/UPSTREAM_EXPORTS.md).

Create annotation subsets if a STCOcc evaluation wrapper needs smaller info files:

```bash
python ../../../scripts/stcocc/create_info_subset.py --help
```

Do not commit datasets, checkpoints, generated predictions, or local build artifacts with this method directory.
