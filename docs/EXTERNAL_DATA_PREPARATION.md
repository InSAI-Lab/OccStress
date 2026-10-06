# Prepare External Data

These commands prepare original dependencies for **already exported** OccStress
occupancy. They do not regenerate corruptions or start training. Accept source
access terms first; see [distribution scope](ASSET_DISTRIBUTION.md).

Run from the checkout with Python 3.10+, NumPy and the released OccStress
protocol/metadata packages installed. The designated repository is
[insailab/OccStress](https://huggingface.co/datasets/insailab/OccStress).
Use the [download guide](RESOURCES.md) to select and verify the required archives
before preparing these external dependencies.

```bash
python -m pip install -e .
export OCCSTRESS_DATA_ROOT=/datasets/OccStress
export OCCSTRESS_EXTERNAL_ROOT=/datasets/OccStress-external
```

## nuScenes

Obtain Occ3D-nuScenes validation GT through the
[official instructions](https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-nuscenes).
Select the directory containing `scene-*/<token>/labels.npz`, not its parent.
Do not substitute OpenOccupancy or OpenOcc labels. Obtain the world-info package
linked by [II-World](https://github.com/lzzzzzm/II-World), and the scene-organized
validation info linked by [DOME](https://github.com/gusongen/DOME).
The two info PKLs are not interchangeable.

```bash
python tools/prepare_external_data.py mount --dataset nuscenes \
  --kind gts --source /datasets/raw/occ3d-nuscenes/gts
python tools/prepare_external_data.py mount --dataset nuscenes \
  --kind world-nuscenes_infos_val.pkl --source /datasets/raw/infos/world-nuscenes_infos_val.pkl
python tools/prepare_external_data.py mount --dataset nuscenes \
  --kind nuscenes_infos_val_temporal_v3_scene.pkl \
  --source /datasets/raw/infos/nuscenes_infos_val_temporal_v3_scene.pkl
python tools/prepare_external_data.py check --dataset nuscenes --trust-pickle
```

Both info files are checked so the setup serves all included forecasters.
Occupancy-input evaluation does not require the original camera/LiDAR streams.
Native source re-export does require those sensors, calibration and its model's
own annotation index; see [upstream exports](UPSTREAM_EXPORTS.md).

## Waymo

Register with Waymo and obtain **Occ3D-Waymo validation voxel04** through
[Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-waymo).
Point to the directory containing `validation-data`:

```bash
python tools/prepare_external_data.py mount --dataset waymo \
  --kind native_gt --source /datasets/raw/occ3d-waymo/voxel04
python tools/prepare_external_data.py check --dataset waymo --trust-pickle
```

The release provides the selected frame identities and
`meta/OccStress-Waymo/controls.json`. Preserve the 7,998 frames / 202 scenes /
5,978 anchors; do not reconstruct a different 2 Hz split. Only native
`voxel_label` needs loader remapping. Released `semantics` is already Occ3D-18.

For source re-export, use the matching Perception sensor segments referenced by
Occ3D, not an arbitrary Motion dataset version. The native sensor view, camera
info and point alignment are separate inputs described in
[upstream exports](UPSTREAM_EXPORTS.md). CVT-Occ source and its dedicated
environment profile are included. No additional raw sensor download is needed to
consume already exported occupancy.

## CARLA

Use `tasl-lab/uniocc`, dataset revision
`e81775b36e376f591a1145ca054b72fd541a67eb`, folder `Carla-2Hz-val`.
Its actual inventory has `scene_Town05`, `scene_Town05_Opt` and `scene_Town07`,
120 frames each. The upstream README summary lists different counts; the
fixed file inventory and source checksum define this benchmark.

```bash
python -m pip install huggingface_hub
hf download tasl-lab/uniocc --repo-type dataset \
  --revision e81775b36e376f591a1145ca054b72fd541a67eb \
  --include 'Carla-2Hz-val/*' --local-dir /datasets/raw/uniocc
python tools/prepare_external_data.py mount --dataset carla \
  --kind native_dataset --source /datasets/raw/uniocc/Carla-2Hz-val
python tools/prepare_external_data.py carla-gt \
  --source-root /datasets/raw/uniocc/Carla-2Hz-val
python tools/prepare_external_data.py check --dataset carla --trust-pickle
```

`carla-gt` reads the published JSON frame index and numeric native NPZ arrays,
maps UniOcc labels to Occ3D and flips grid axis 1 exactly once. Output is
`$OCCSTRESS_EXTERNAL_ROOT/OccStress-CARLA/canonical_gt/<scene>/<token>/labels.npz`.
The published right-handed poses, trajectories and commands are not rewritten.
Only 360 clean frames are processed, not the corrupted assets. The tool checks
the small `scene_infos.pkl` checksum without unpickling it.

Do not run historical `prepare_carla_model_view.py` on the shared release tree:
that script expects a legacy intermediate layout. The new command needs neither
that intermediate tree nor the manual-corruption generator.
Existing output is reused only if its arrays match exactly; mismatches fail.
`--max-frames 2` is for isolated partial smoke tests and cannot satisfy the
complete dependency check. Never pass already converted GT as the raw source.

## Validation Boundary

`check` validates the full clean-backbone anchor count and existence/nonzero
size of all referenced GT and required info/control files. It does not load
official info PKLs, certify their semantic equivalence, rehash all voxels or
run a GPU model. `--trust-pickle` explicitly trusts the OccStress backbone;
use only known release files. Follow with a small clean/corrupted model gate.

Mounts are symlinks, not OS-enforced read-only filesystems. Existing destinations
are never replaced. Keep the external root, its links and generated GT outside
public upload archives.
