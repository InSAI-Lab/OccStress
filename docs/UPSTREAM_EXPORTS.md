# Extended Upstream Exports

**Source availability:** all prepared source/export integrations are included
in the full-source candidate, including FusionOcc, SDGOcc and CVT-Occ.
Use the matching environment, data and checkpoint.

`configs/upstream_methods.json` records the source/config/exporter/builder pairs.
These are code entrypoints, not GPU validation or asset-completion certificates.

| Source | Dataset | Model Inputs | Corrupted Sensor | Source Tree |
| --- | --- | --- | --- | --- |
| ALOcc | nuScenes | six cameras | camera | EXIST/3D/ALOcc |
| STCOcc | nuScenes | six cameras | camera | EXIST/3D/STCOcc |
| FusionOcc | nuScenes | cameras + LiDAR | camera | EXIST/3D/FusionOcc |
| SDGOcc | nuScenes | cameras + LiDAR | LiDAR | EXIST/3D/SDGOCC |
| CVT-Occ | Waymo | five cameras | camera | EXIST/3D/CVT-Occ |
| EFFOcc point-stress | Waymo | cameras + LiDAR | LiDAR | EXIST/3D/EFFOcc |
| EFFOcc camera-stress | Waymo | cameras + LiDAR | camera | EXIST/3D/EFFOcc |
| FlashOcc | CARLA | four cameras | camera | EXIST/3D/EFFOcc |
| EFFOcc | CARLA | cameras + LiDAR | LiDAR | EXIST/3D/EFFOcc |

FlashOcc is implemented in the EFFOcc tree with its own CARLA config/checkpoint;
it is not a second copy of the same fusion checkpoint. The Waymo camera-stress
EFFOcc export retains clean LiDAR and lives under `upstream/camera_fusion/effocc`.
Do not call it camera-only. CARLA also supports the optional `fusion_camera`
export mode; it is separate from FlashOcc's camera-only experiment.

## Source Dependencies

Use one isolated environment and the appropriate working directory. For EFFOcc
or FlashOcc:

```bash
export OCCSTRESS_CODE_ROOT=/path/to/OccStress-code
export PYTHONPATH="$OCCSTRESS_CODE_ROOT/EXIST/3D/EFFOcc:$OCCSTRESS_CODE_ROOT/scripts/waymo:$OCCSTRESS_CODE_ROOT/scripts/carla"
cd "$OCCSTRESS_CODE_ROOT/EXIST/3D/EFFOcc"
```

For CVT-Occ, use its dedicated environment, set `CVTOCC_WAYMO_ONLY=1`, and use
`EXIST/3D/CVT-Occ:EXIST/3D/CVT-Occ/dependencies/mmdetection3d:scripts/waymo`
as absolute PYTHONPATH entries. The dependency is the v1.0.0rc4 source snapshot
from the actual CVT runtime, with the MMCV upper-bound adjustment. It is used
directly on PYTHONPATH; do not install its obsolete transitive requirements.
Do not add the STCOcc, II-World or EFFOcc model packages as well.
The unused, syntactically broken upstream `dts_transformer.py` prototype is
excluded; it is not imported by the CVT-Occ Waymo module/config.

Shared camera operators are in `scripts/waymo/waymo_corruptions.py` and
`effocc_waymo_camera_corruptions.py`. They need imagecorruptions and Pillow.
Point operators are implemented in `waymo_sdgocc_corruptions.py` and the CARLA
adapter. Robo3D's fog lookup tables are an **external asset**, not source code.
The corruption sources have non-commercial/share-alike declarations and
operator-specific exceptions; see [the license audit](LICENSES_AND_CITATIONS.md#corruption-implementations).
Set `ROBO3D_ROOT` to the official repository snapshot at
`481a3b8634b2d291b84736cab1b546ed266efafa`; table files are expected below
`create/nuscenes_c/fog/`. No tables are bundled or downloaded in this code pass.
Raw Waymo TFRecord preparation additionally needs the Waymo/TensorFlow data
environment; keep that out of inference environments.

SDGOcc's native loader expects `point_label/LIDAR_TOP/*.npy` alongside
`samples/LIDAR_TOP/*.pcd.bin`, including for the selected corrupted setting.
Use `scripts/sdgocc/generate_point_labels.py` and
`scripts/sdgocc/build_nusc_c_point_labels.py` for those external annotations;
relocating `samples` alone is insufficient. Keep generated auxiliary labels
separate from predictions. Its export converter can apply the official camera
visibility mask; preserve the chosen mask policy in comparisons.

For CARLA native inference, build the sensor/calibration index separately
from the canonical occupancy benchmark. For validation only:

```bash
export UNIOCC_CARLA_LOCAL_ROOT=/path/to/model-runtime
python tools/data_converter/uniocc_carla_converter.py \
  --data-root /path/to/raw/Carla-2Hz-val \
  --output-root "$UNIOCC_CARLA_LOCAL_ROOT/prepared/Carla-2Hz-val" \
  --split val --workers 4
```

Run from the EFFOcc source root, in its model environment. This produces
`uniocc_carla_val_infos.pkl` and the native model view. It is not a second
application of the canonical occupancy coordinate conversion. The CARLA
spconv fallback used for the small A100 gate is documented in
[environments](ENVIRONMENTS.md).

## Export And Protocol Construction

The shared upstream builder and ALOcc/STCOcc/CARLA batch builders use the
namespaced layout in [datasets](DATASETS.md). Set `OCCSTRESS_DATA_ROOT` to the
shared `OccStress` root, not a per-dataset folder. They consume already exported
canonical occupancy and a trusted, release-normalized clean backbone. They do
not generate raw annotations, rerun 3D models, transform CARLA coordinates again
or rebuild the global dataset catalog/checksum manifests.

```bash
export OCCSTRESS_DATA_ROOT=/datasets/OccStress
export OCCSTRESS_EXTERNAL_ROOT=/datasets/OccStress-external

python scripts/alocc/build_nuscenes_occstress_protocols.py
bash scripts/stcocc/build_occstress_upstream_protocols_from_nuscc.sh
python scripts/carla/build_carla_upstream_protocols.py --track both --jobs 2
```

Use these commands only after all selected upstream export settings are ready.
Input defaults are `occ/upstream/<dataset>/<subtrack>/<source>/...` and
`protocols/manual/<dataset>/clean/H4_F6_val_backbone.pkl`. Protocols are written
under `protocols/upstream/<dataset>/...`; metadata goes to
`meta/<dataset>/upstream/...`. STCOcc keeps the published filename prefixes
`current_only`, `recent_burst` and `history_only`; the CLI temporal modes remain
`current`, `history_k1` and `all_frame`, respectively. ALOcc accepts both native export receipts and
release `payload_validated` markers; absent checkpoint provenance is reported
as unavailable, never invented.

Python builders accept `--occstress-root` to override the environment. STCOcc
also accepts `PYTHON`, `BACKBONE_PROTOCOL` and `UPSTREAM_OCC_ROOT` environment
overrides, and `CLEAN_ONLY=1` for the initial clean gate. Complete builds fail on
missing settings instead of silently skipping them. Existing protocols are not
overwritten unless `--overwrite` (or STCOcc `OVERWRITE=1`) is explicitly supplied.
Use a separate staging root for rebuilding; modifying a published dataset also
requires refreshing its separate catalog and integrity manifests.

The shared single-protocol CLI additionally accepts `--dataset`,
`--backbone-protocol`, `--clean-input-root`, `--corrupted-input-root` and
`--external-root`. All builders use `--occstress-root` for the shared dataset
root. Old unnamespaced layouts are not auto-detected.
Stored occupancy references are relative and can be remounted on another
machine. Future GT and control metadata are preserved from the backbone;
CARLA targets resolve through `external/OccStress-CARLA/canonical_gt`, not an
upstream prediction directory. Files are checked for existence; this is not an
exhaustive voxel-content validation.

First export clean frames, validate shape/class mapping/order, then export
corrupted settings. Exporters write canonical 200x200x16 Occ3D-18 NPZs and
completion/provenance records; builders form the temporal protocols from them.
Waymo uses 7,998 selected frames and 5,978 H4/F6 anchors; CARLA uses 360 frames
and 330 anchors. Upstream 3D visibility-mask reproduction and downstream
full-grid metrics must remain separate.

The full native CLI is available with each entrypoint's `--help`, in its model
environment. Required paths are intentionally explicit:

- STCOcc nuScenes: run `EXIST/3D/STCOcc/tools/test.py CONFIG CHECKPOINT` from
  its source root with `STCOCC_PRED_DUMP_ROOT=/path/to/output/clean` for
  dump-only inference. Supply its native `stcocc-nuscenes_infos_val.pkl` through
  `--cfg-options data.test.ann_file=...`. For camera stress, set
  `data.test.use_corner_case_data=/path/to/Brightness` and
  `data.test.corner_case_degree=hard`, and use a separate output root.
  This retains the native temporal sampler/state memory: process each scene
  from its start in chronological order. A single-scene smoke test needs
  `data.test_dataloader.samples_per_gpu=1`. Build downstream protocols with
  `scripts/stcocc/build_occstress_upstream_protocols_from_nuscc.sh` or the shared
  protocol builder. The validated local checkpoint's filename/provenance must
  be recorded; do not assume every checkpoint in an experiment directory is
  an official model-zoo release.
- CVT-Occ: `export_cvtocc_waymo_upstream.py CONFIG CHECKPOINT Clean` with
  `--output-root` and `--output-json`; set `OCCSTRESS_WAYMO_ROOT` and
  `CVTOCC_RUNTIME_ROOT` for the 2 Hz metadata/config. Then use
  `build_cvtocc_waymo_upstream_protocols.py` with explicit backbone/index/output
  roots, starting with `--clean-only`.
- EFFOcc Waymo point: `export_effocc_waymo_upstream.py CONFIG CHECKPOINT clean`
  requires annotation, frame index, sensor root, point alignment, Robo3D root
  and output paths. Build alignment with `build_effocc_waymo_point_alignment.py`.
  Finish with `build_effocc_waymo_upstream_protocols.py`.
- EFFOcc Waymo camera: `export_effocc_waymo_camera_upstream.py` requires
  annotation, data root, pose file, Occ3D GT and output paths. Its builder is
  `build_effocc_waymo_camera_upstream_protocols.py`. Point `--occ-gt-root` to
  the directory containing the validation scenes (`voxel04/validation-data`),
  not its parent `voxel04`.
- CARLA: `export_effocc_carla_clean.py` and `export_effocc_carla_upstream.py`
  accept the native camera/fusion configs. The latter selects `--track camera`
  for FlashOcc and `--track lidar` for EFFOcc; supply canonical clean, raw and
  output/manifest roots. Use `build_carla_upstream_protocols.py` afterward.

Use small scene/sample limits when validating a new setup. Export once per source,
corruption and severity, not once per downstream model/temporal pattern. Keep
scene-local temporary files on node-local scratch; copy validated scene shards
back with bounded I/O concurrency. This release does not launch cluster jobs.

## Checked Scope

Small A100 gates exercised the native exporters for ALOcc, STCOcc, FusionOcc, SDGOcc,
CVT-Occ, EFFOcc and FlashOcc, followed by a GenieDrive H4/F6 forecast using the
exported states. Each uses five exported states per clean/corrupted setting
and one downstream anchor. STCOcc's native sampler includes finite padding;
the five states are unique frame tokens, not its padded iteration count.
These are export/interface checks, not upstream accuracy reproduction or an
exhaustive corruption-suite test. See [code status](CODE_RELEASE_STATUS.md).

SparseWorld-TC is a separate camera-to-forecast path, not a 3D state exporter.
Its native nuScenes loader also requires the original `admlp` and `occworld`
auxiliary metadata under its working directory, and a scene-local `frame_idx`
in annotation records. These external inputs are not replaced by occupancy
protocol files; a missing auxiliary file must not be silently ignored.
