# Dataset Construction

Use downloaded protocols for evaluation. Rebuilding assets is optional and
requires authorized source GT, metadata and (for upstream tracks) raw sensors.
These commands write a **new** shared OccStress root. They do not change
corruption algorithms, seeds, label mappings, temporal masks or control policies.
Do not point reconstruction at an existing downloaded release.

## nuScenes Manual

Install `pip install -e '.[construction]'` and prepare the
[original dependencies](EXTERNAL_DATA_PREPARATION.md). The original nuScenes
root below contains `v1.0-trainval/` and `gts/`.

```bash
export OCCSTRESS_DATA_ROOT="/path/to/new/OccStress"
python scripts/nuscenes/build_clean_backbone.py \
  --data-root /path/to/nuscenes --ann-file /path/to/trusted_val_infos.pkl

for family in semantic hole dropout; do
  for severity in easy mid hard; do
    python scripts/nuscenes/generate_manual.py "$family" \
      --data-root /path/to/nuscenes --severity "$severity"
  done
done
python scripts/nuscenes/generate_manual.py traffic --data-root /path/to/nuscenes

python scripts/nuscenes/generate_protocols.py \
  --data-root /path/to/nuscenes \
  --backbone-path "$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl"
```

The clean backbone fixes the validation anchor set. The protocol builder
requires all backbone anchors and builds the other 37 protocols, including
misalignment matrices. Optional misalignment event/cache generation is separate;
those cached transforms are not required by the protocol builder.
Traffic needs mirrored historical, current **and future** frames; do not use an
input-only token filter for it.

Outputs are `occ/manual/OccStress-nuScenes/`,
`events/manual/OccStress-nuScenes/`, `meta/OccStress-nuScenes/manual/`
and `protocols/manual/OccStress-nuScenes/`. Clean references use
`external/OccStress-nuScenes/gts/`; mount that GT location before evaluating.
All records remain movable with the shared root. Native CLI output-root flags
refer to that shared root, not a dataset or family subdirectory.

Protocol builders refuse existing PKLs unless `--overwrite` is supplied.
Generators retain their existing skip/overwrite behavior. An incomplete build
is not a formal benchmark: validate cohort size, targets and referenced files.

## Waymo EFFOcc

After [canonical export](UPSTREAM_EXPORTS.md), build either upstream track:

```bash
python scripts/waymo/build_effocc_waymo_upstream_protocols.py \
  --occstress-root "$OCCSTRESS_DATA_ROOT" --index-root /path/to/lidar/frame_index
python scripts/waymo/build_effocc_waymo_camera_upstream_protocols.py \
  --occstress-root "$OCCSTRESS_DATA_ROOT" --index-root /path/to/camera/frame_index
```

Each validates export receipts and emits 73 protocols below
`protocols/upstream/OccStress-Waymo/{pointcloud_fusion,camera_fusion}/effocc/`.
Default occupancy roots use the matching `occ/upstream/` branches. The clean
backbone is `protocols/manual/OccStress-Waymo/clean/H4_F6_val_backbone.pkl`.
Defaults require 202 scenes, 7,998 selected frames and 5,978 anchors; test-only
count overrides must not be used to claim full evaluation.

`--spec-only` writes a separate `protocol-specs.json`, never a completion
receipt. `--skip-file-validation` is an explicit diagnostic shortcut, not a
release-validation pass. Full construction preserves future GT, trajectories
and commands from the portable clean backbone.
