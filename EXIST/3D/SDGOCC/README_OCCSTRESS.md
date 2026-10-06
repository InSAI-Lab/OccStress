# OccStress Adapter Notes

This directory keeps the original SDGOcc method layout.

SDGOcc is used for the `upstream/pointcloud_fusion/sdgocc` track.

## Paths

```bash
export OCCSTRESS_CODE_ROOT=$(cd ../../.. && pwd)
export OCCSTRESS_DATA_ROOT=$OCCSTRESS_CODE_ROOT/data/OccStress
```

## Common Output Layout

```text
data/OccStress/occ/upstream/OccStress-nuScenes/pointcloud_fusion/sdgocc/<corruption>/<severity>/<scene>/<token>/labels.npz
data/OccStress/protocols/upstream/OccStress-nuScenes/pointcloud_fusion/sdgocc/<corruption>/<severity>/<protocol>.pkl
```

## Helper Scripts

Convert SDGOcc `results.pkl` into OccStress `labels.npz` files:

```bash
python ../../../scripts/sdgocc/occstress_upstream/convert_sdgocc_results_to_occstress_upstream.py \
  --root "$OCCSTRESS_CODE_ROOT" \
  --results-root /path/to/sdgocc/results \
  --upstream-root "$OCCSTRESS_DATA_ROOT/occ/upstream/OccStress-nuScenes/pointcloud_fusion/sdgocc"
```

Build a pointcloud-fusion clean protocol using the shared builder:

```bash
python ../../../scripts/build_upstream_occstress_protocol.py \
  --root "$OCCSTRESS_CODE_ROOT" \
  --backbone-protocol "$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl" \
  --clean-input-root "$OCCSTRESS_DATA_ROOT/occ/upstream/OccStress-nuScenes/pointcloud_fusion/sdgocc/clean" \
  --subtrack pointcloud_fusion --source-model sdgocc --corruption clean
```

Clean has no severity directory. For corrupted protocols, add the explicit
corrupted input root, corruption/severity and temporal pattern. The old SDGOcc
batch builder retains legacy layout defaults; it is not the release-layout
entrypoint. Use separate construction scratch space, not a finalized dataset.
Small clean/beam-missing export-to-GenieDrive gates passed using an isolated
legacy environment; see [upstream exports](../../../docs/UPSTREAM_EXPORTS.md).

Run sanity checks before downstream 4D evaluation:

```bash
python ../../../scripts/sdgocc/occstress_upstream/sanity_check_sdgocc_results.py \
  --results /path/to/results.pkl \
  --ann /path/to/annotation.pkl \
  --gt-root /path/to/nuscenes/gts
```
