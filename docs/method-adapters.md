# Method Adapters

The full-source candidate includes all prepared integrations. Native execution
requires the matching environment, datasets and checkpoints.

Keep each method in an isolated environment: several repositories define
incompatible `mmdet3d` modules with the same import name. Run from the method
directory and set `OCCSTRESS_CODE_ROOT` to the release root.

| Method | Observation times (s) | Evaluation entry | Code integration |
| --- | --- | --- | --- |
| OccWorld | -2, -1.5, -1, -0.5, 0 | `EXIST/4D/OccWorld/eval_metric_stp3.py` | Future-aligned loader/config updated; cross-dataset wrapper imported |
| I2-World | -2, -1.5, -1, -0.5, 0 | `EXIST/4D/II-World/tools/test.py` | nuScenes retained; Waymo/CARLA scene-sharded wrapper and streaming metric callback merged |
| COME | -1.5, -1, -0.5, 0 | `EXIST/4D/COME/tools/test_diffusion_control.py` | Future-aligned loader/config; Waymo/CARLA wrappers imported |
| GenieDrive | -1.5, -1, -0.5, 0 | `EXIST/4D/GenieDrive/occ_gen/tools/test_occstress.py` | Protocol evaluator and Waymo/CARLA configs imported |
| DOME | -1.5, -1, -0.5, 0 | `EXIST/4D/DOME/tools/eval_occstress.py` | Protocol evaluator, shard dispatcher and merger imported |
| SparseWorld-TC | Five camera states | `EXIST/4D/SparseWorld/tools/waymo_occstress_eval.py` | Camera-direct adapter; dataset-specific entrypoints in the evaluation contract |

All six forecasters have passed small A100 checks on nuScenes, Waymo and CARLA,
using new environment prefixes. Strict migration comparisons and their limits
are recorded in the release status; SparseWorld-TC has smoke-test evidence,
not GPU prediction-parity evidence. This table describes interfaces, not
completion of every full evaluation combination or paper-score reproduction.
Read `configs/methods.json` and [release status](CODE_RELEASE_STATUS.md) before
using the package to recreate a paper table.

## Scientific Boundaries

- Score six genuine future frames, never current reconstruction plus five futures.
- OccWorld/COME release configs default to `future_aligned=True`; the explicit
  legacy switch is retained for historical debugging, not comparable reporting.
- Keep the original checkpoints fixed for cross-dataset zero-shot evaluation.
- Preserve each model's control policy. Action-conditioned evaluations use the
  prescribed GT future controls; do not describe them as control-free forecasting.
- Traffic mirroring changes inputs, targets and relevant poses/controls together.
- Report the t=-2.0 position of four-state models as not observed, not a robustness gain.
- SparseWorld-TC is direct camera forecasting. Its manifest is not a manual-state
  protocol and it must not be assigned a point-upstream result.

## Upstream Export

The prepared source inventory includes the following implementations/helpers.
- ALOcc export flags in `tools/test.py` and the detector, plus protocol build/validation in `scripts/alocc/`.
- FusionOcc sharded export, protocol build and validation in `scripts/fusionocc/`.
- EFFOcc/FlashOcc model sources, Waymo/CARLA configs, corruption adapters and exporters.
- CVT-Occ Waymo source/config/exporter and its actual runtime mmdet3d dependency snapshot.

The canonical export is `scene/token/labels.npz` with `semantics`, the required
voxel shape and class mapping. See `configs/upstream_methods.json` for source
modality, corrupted sensor and config/export pairing. Seven upstream sources
have passed small clean/corruption export-to-GenieDrive gates. Checkpoint
provenance is retained with private validation receipts; some checkpoint
downloads remain unavailable. The released occupancy
archives are available through the [download guide](RESOURCES.md) and do not
require upstream checkpoints to consume. A camera stressor does not imply
camera-only input. See [the checked scope](UPSTREAM_EXPORTS.md#checked-scope).

## Diagnostics

`scripts/statistics/` includes scene confusion accumulation and paired
scene-bootstrap analysis. The imported bootstrap aggregators retain the original
nuScenes/STCOcc/SDGOcc protocol assumptions; they are not universal cross-dataset
summarizers. `scripts/position_sweep/` includes the final summary and editable SVG
renderer, including explicit unobserved-position placeholders.
