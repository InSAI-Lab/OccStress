# 3D Method Adapters

OccStress keeps 3D occupancy method repositories under `EXIST/3D`. These methods serve two roles:

- direct 3D occupancy robustness evaluation;
- upstream occupancy generation for 4D forecasting evaluation.

## Common Contract

All 3D adapters should use the same path contract:

```bash
export OCCSTRESS_CODE_ROOT=/path/to/OccStress-code
export OCCSTRESS_DATA_ROOT=$OCCSTRESS_CODE_ROOT/data/OccStress
```

Inputs:

- authorized original data under native model mounts or `OCCSTRESS_EXTERNAL_ROOT`;
- protocols under `$OCCSTRESS_DATA_ROOT/protocols/upstream/<dataset>/...`;
- predicted occupancy under `$OCCSTRESS_DATA_ROOT/occ/upstream/<dataset>/...`.

Dataset namespaces are `OccStress-nuScenes`, `OccStress-Waymo` and
`OccStress-CARLA`. See [datasets](DATASETS.md) for the current layout. Use
explicit roots for historical construction helpers; their old batch defaults
are not a release-layout interface.

Outputs:

- semantic occupancy predictions saved as `labels.npz`;
- each `labels.npz` must contain `semantics`;
- if a method uses visibility masks, keep mask handling explicit in the method wrapper.

## 3D Source Availability

| Method | Release Location | Notes |
| --- | --- | --- |
| ALOcc | `EXIST/3D/ALOcc` | Export flags and protocol helpers included; small upstream gate passed. |
| BEVFormer | `EXIST/3D/BEVFormer` | Original method layout retained; add method-specific wrapper if used in OccStress evaluation. |
| COTR | `EXIST/3D/COTR` | Original method layout retained; add method-specific wrapper if used in OccStress evaluation. |
| FBBEV | `EXIST/3D/FBBEV` | Original method layout retained; helper scripts exist under `scripts/fbocc`. |
| FusionOcc | `EXIST/3D/FusionOcc` | Included source and dedicated export helpers. |
| OccFusion | `EXIST/3D/OccFusion` | Included source and existing helpers; not part of the seven-source runtime gate. |
| PanoOcc | `EXIST/3D/PanoOcc` | Original method layout retained. |
| SDGOcc | `EXIST/3D/SDGOCC` | Included source and dedicated export helpers. |
| SparseOcc | `EXIST/3D/SparseOcc` | Original method layout retained. |
| STCOcc | `EXIST/3D/STCOcc` | Source model for `upstream/camera_only/stcocc`. |
| ViewFormerOcc | `EXIST/3D/ViewFormerOcc` | Original method layout retained; helper scripts exist under `scripts/viewformerocc`. |
| CVT-Occ | `EXIST/3D/CVT-Occ` | Included Waymo camera export integration; small runtime gate passed. |
| EFFOcc | `EXIST/3D/EFFOcc` | Waymo/CARLA fusion exports; small gates passed. |
| FlashOcc | `EXIST/3D/EFFOcc` | Separate CARLA camera-only config/checkpoint; small gate passed. |

Seven upstream sources passed the checks in [upstream exports](UPSTREAM_EXPORTS.md).
Other retained 3D trees are not implicitly covered by those GPU gates.

## Upstream Tracks

### Camera-Only: STCOcc

Expected output:

```text
data/OccStress/occ/upstream/OccStress-nuScenes/camera_only/stcocc/<corruption>/<severity>/<scene>/<token>/labels.npz
data/OccStress/protocols/upstream/OccStress-nuScenes/camera_only/stcocc/<corruption>/<severity>/<protocol>.pkl
```

Helper scripts:

- `scripts/build_upstream_occstress_protocol.py` with explicit backbone/input roots
- `scripts/stcocc/create_info_subset.py`

### Pointcloud-Fusion: SDGOcc

The following documents the export/data contract. SDGOcc source and dedicated
helpers are included. Canonical exports can also be consumed without
installing the upstream model.

Expected output:

```text
data/OccStress/occ/upstream/OccStress-nuScenes/pointcloud_fusion/sdgocc/<corruption>/<severity>/<scene>/<token>/labels.npz
data/OccStress/protocols/upstream/OccStress-nuScenes/pointcloud_fusion/sdgocc/<corruption>/<severity>/<protocol>.pkl
```

Helper scripts:

- `scripts/sdgocc/occstress_upstream/convert_sdgocc_results_to_occstress_upstream.py`
- `scripts/build_upstream_occstress_protocol.py` with explicit backbone/input roots
- `scripts/sdgocc/occstress_upstream/sanity_check_sdgocc_results.py`

## Shared Utilities

Use the shared resolver under `occstress/datasets/` and the canonical layout in
[DATASETS.md](DATASETS.md). Clean upstream outputs omit the severity directory.
The shared builder and ALOcc/STCOcc/CARLA batch builders support that layout;
see [upstream construction](UPSTREAM_EXPORTS.md#export-and-protocol-construction).
The older `tools/adapters/occstress_3d.py` is a legacy helper, not the publication
path authority.
