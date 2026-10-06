# Release Status

Updated: 2026-10-06. This page describes the public source package, its tested
scope and its remaining resource restrictions. Historical execution logs and
machine-specific package inventories are not part of the release.

## Source And Resources

| Component | Status |
| --- | --- |
| Core package | CPU-only protocol validation, count-based F6 metrics, result merge and summaries |
| Included forecasters | II-World, GenieDrive, OccWorld, COME, DOME and SparseWorld-TC |
| Included upstream integrations | ALOcc, STCOcc, FusionOcc, SDGOcc, CVT-Occ, EFFOcc and FlashOcc |
| Additional source | OccFusion source and existing helpers; not part of the seven-source runtime gate |
| Dataset assets | All 75 archives published and remote size/checksum verified; all three datasets pin the verified index revision in the resource registry |
| Checkpoints | Exact file guidance and available runtime/official digests recorded; GenieDrive identity matched; remaining unverified references are explicit |
| Citation | NeurIPS 2026; [arXiv:2512.15621](https://arxiv.org/abs/2512.15621) |

The root MIT license covers original OccStress code only. Retained third-party
components have their own terms; see [licenses](LICENSES_AND_CITATIONS.md).
Data and checkpoint availability are separate from source availability.

## Validation Scope

CPU regressions cover protocol construction, dataset paths, metric accumulation,
result validation, launch/config guards, packaging and source availability.
The download tool pins a dataset commit, rejects incomplete selections, resumes
through the Hub client and verifies archive sizes/checksums. Its offline checks
do not require a model framework or a GPU.
The full-source profile runs checks for all restored implementations. The
optional filtered profile still skips tests requiring its omitted source.

Small A100 runtime gates used real checkpoints and data: clean plus one
corruption on each dataset, two anchors per setting and six future horizons.
They validate installation and code migration, **not full-paper reproduction**.

| Forecaster | Completed small-sample checks | Limitations |
| --- | --- | --- |
| II-World | Six metric comparisons matched; Waymo/CARLA confusion matrices matched | nuScenes native path does not return confusion matrices |
| GenieDrive | Six metric comparisons matched; Waymo/CARLA confusion matrices matched | No matrix-parity claim for nuScenes |
| DOME | Six deterministic comparisons matched in metrics and matrices | Default autotuning showed about 0.03 pp variation |
| COME | Six deterministic comparisons matched in metrics and matrices | Default autotuning showed about 0.094 pp variation |
| OccWorld | Six deterministic comparisons matched in metrics and matrices | Default mode showed about 0.007-0.023 pp variation |
| SparseWorld-TC | Six GPU smoke checks; Waymo/CARLA preprocessing matched | Not a GPU-prediction parity claim |

COME/OccWorld comparisons used old and release code in the same new environment,
not old-versus-new PyTorch environments. Deterministic options are opt-in and do
not change historical defaults. The training entrypoints inherited from model
repositories are not covered by these evaluation gates.

Upstream gates exported five states per clean/corrupted setting and passed them
through one GenieDrive H4/F6 anchor, checking `200x200x16` grids, 18-class labels
and finite final metrics. Tested sources/settings were:

- ALOcc and STCOcc on nuScenes: camera Brightness-hard.
- FusionOcc and SDGOcc on nuScenes: camera Brightness-hard and LiDAR
  beam_missing-heavy, respectively.
- CVT-Occ on Waymo: brightness-hard.
- EFFOcc on Waymo: camera Brightness-hard and LiDAR beam_missing-heavy.
- FlashOcc/EFFOcc on CARLA: camera Brightness-hard and LiDAR beam_missing-heavy;
  the optional fusion-model camera stress was checked separately.

STCOcc used the existing experiment checkpoint `iter_63288.pth`, not an asserted
official model-zoo download. FusionOcc/SDGOcc used isolated clones of a working
legacy environment with rebuilt extensions, not fresh dependency solves.
Other listed gates used new environment prefixes. A100 checks do not establish
H100 or RTX 5090 compatibility; see [environment recipes](ENVIRONMENTS.md).

## Dataset And Metric Boundaries

[Distribution scope](ASSET_DISTRIBUTION.md) records per-source conditions.
The [external preparation guide](EXTERNAL_DATA_PREPARATION.md) includes a direct
360-frame CARLA GT converter and safe mount/reference checks.

- Set `OCCSTRESS_DATA_ROOT` for the shared benchmark layout. Raw sensors,
  official GT and auxiliary resources use `OCCSTRESS_EXTERNAL_ROOT` and are
  external dependencies, not bundled source assets.
- Manual generators, the shared upstream builder, ALOcc/STCOcc/CARLA wrappers
  and both Waymo EFFOcc builders use the namespaced release layout and relative
  protocol/event references. Upstream builders consume existing canonical exports, not
  the raw sensor data or global release catalog. See
  [construction boundaries](UPSTREAM_EXPORTS.md#export-and-protocol-construction).
- The published release includes 975 protocols, 326 asset conditions and 1,654,143
  asset files. Its 75 archives total 40.95 GB compressed, including metadata.
  All remote archive sizes/digests matched; anonymous CARLA/STCOcc downloads
  passed archive checks. Loader checks sampled completed assets; no exhaustive
  payload rehash is implied.
- Unified results use `occstress-present-gt-v1`. Historical paper metrics may
  handle absent classes and zero IoU differently; do not relabel them as the
  unified convention. Full-paper scores are reported in the paper.
- Main temporal regimes combine position and duration. Position sweeps control
  a single affected state; an unobserved `t=-2.0` state is not robustness gain.
- SparseWorld-TC is camera-direct and has no manual/point-source interface.
  EFFOcc camera stress retains clean LiDAR and is not a camera-only model.

## Before Publishing

1. Retain all applicable third-party notices and operator-specific terms.
2. Check the live dataset package index for the selected track. Keep unresolved
   weight identities and optional experiment-weight mirrors explicit.
   Robo3D fog lookup tables remain external assets.
3. Build a clean public candidate, run the checks below, review its diff and
   freeze the release commit. GitHub CI and tagging happen when publishing.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python tools/check_release.py --public --syntax
python tools/check_documentation.py
python tools/check_formal_configs.py
python tools/update_release_metadata.py
python tools/release_status.py
```

These checks do not download data, run GPU jobs or certify legal compliance.
`tools/release_status.py --require-downloads` intentionally fails while any
resource is uploading, pending or identity-unverified. This includes optional
upstream re-export weights and is stricter than a code-only release.
Full-paper reruns and filling `results/reference` are not required
for this source release. Runtime trees with extensions, caches or data mounts
must not be published as source archives.
