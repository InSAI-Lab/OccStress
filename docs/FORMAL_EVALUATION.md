# Formal Evaluation

The whitelist is `configs/evaluation_contract.json`. Config inheritance and
declared entrypoints are hashed in `configs/formal_configs.lock.json`. This is
internal interface coverage, **not completed experiment or asset coverage**.
The full-source profile includes all prepared model integrations. Only the
optional filtered profile omits selected sources and refuses their launch.
See [source packaging](CODE_LAYOUT.md#source-packaging).

```bash
python tools/check_formal_configs.py
python tools/release_status.py
```

The first checks drift without importing frameworks. Only maintainers should
use `--write-lock`, after reviewing deliberate changes. This lock is not a full
Python import graph; also retain the release commit and source-import ledger.

## Fixed Contract

- 2 Hz H4/current/F6: observations -2.0 through 0 s; targets +0.5 through +3.0 s.
  Current reconstruction cannot substitute for the first future frame.
- OccWorld/II-World observe five states; COME/DOME/GenieDrive the final four.
  SparseWorld-TC takes cameras directly, with no manual/point-input track.
- Current-only: slot 0; Recent-burst: -1 and 0; History-only: -4 through -1.
  Legacy filenames are respectively `current`, `history_k1`, `all_frame`.
  Do not infer their meaning from the old names.
- The position sweep is a separate fixed-budget diagnostic. Four-state models
  do not observe slot -4: mark it unobserved, not robust. Do not mix sample512
  with full-anchor sweeps.
- nuScenes: 4,519 anchors; Waymo: 5,978; CARLA: 330. CARLA uses the converted
  right-handed OccStress-CARLA coordinates, not raw left-handed simulator grids.
- Keep native controls: II-World Waymo uses fixed command [1,0,0], zero LCF and
  official GT poses; COME/GenieDrive use GT trajectory/command; OccWorld uses
  GT commands but predicts ego motion; DOME uses GT future ego-motion poses.
- Nontraffic futures stay clean. Traffic consistently transforms occupancy,
  targets, poses, trajectories and commands and is reported separately.
- Downstream Waymo/CARLA use fixed nuScenes checkpoints zero-shot. Dataset-
  specific **upstream** training is a separate setting.

The JSON also fixes class maps, masks and aggregation. Actual runs must record
checkpoint, protocol and class-map SHA256, plus the pinned dataset revision.
Select available assets through the [download guide](RESOURCES.md).

## Metrics And Legacy Results

Keep six horizons; paper tables average **1, 2 and 3 seconds**, not a wrapper's
six-horizon `average_miou`. Sum confusion counts before computing IoU; never
average shard mIoUs or select class support independently per worker.

Cross-dataset summaries use present-GT occupied classes and occupied binary IoU.
Keep zero-IoU classes. The released II-World metric no longer removes zeros.
GenieDrive's legacy native logger can still exclude zero classes: recompute
explicit present-class summaries from raw counts and do not relabel historical
metrics as if these policies were identical. Final results/provenance migration
is a separate audit, not a silent change in this code pass.

Official **upstream 3D reproduction** visibility masks are not downstream OccStress
masks. The canonical downstream grid is evaluated without camera/lidar visibility
masks. Label any other mask policy separately.

## Launch Selection

Set roots and use absolute asset paths:

```bash
export OCCSTRESS_CODE_ROOT="$PWD"
export OCCSTRESS_DATA_ROOT=/path/to/OccStress
python tools/run_formal.py --model iiworld --dataset waymo \
  --python /path/to/envs/occstress-iiworld/bin/python -- \
  --tokenizer-checkpoint /path/to/tokenizer.pth \
  --world-checkpoint /path/to/world_model.pth \
  --protocol /path/to/OccStress/protocols/manual/OccStress-Waymo/clean/H4_F6_val_backbone.pkl \
  --base-info /path/to/OccStress/meta/OccStress-Waymo/controls.json \
  --scene-shard 000 --clean-token-root /path/to/scratch/future-tokens \
  --token-work-root /path/to/scratch/current-tokens \
  --output-json /path/to/results/clean-scene000.json --max-samples 32
```

Add `--execute` before `--` only after gates. For CARLA change the dataset and
paths; the selector supplies both CARLA configs. Other methods similarly take
their native checkpoint/data/output arguments after `--`. Formal config and
dataset overrides are rejected; use native entrypoints for exploratory runs.

II-World nuScenes still uses separate `--stage tokenizer` and `--stage world`
commands with checkpoint as first native positional argument. Set protocol/base/
token environment variables from its README. The cross-dataset wrapper runs
both stages, streams forecast metrics, and uses bounded per-scene current tokens
plus a reusable future cache. Traffic has a separate future-cache namespace.
This is **not** a zero-disk-cache evaluator.

## Legacy Isolation

Unlisted configs are references, experiments or ablations, not formal entries.
They remain for history but are not selected by the launcher.
`OCCSTRESS_OCCWORLD_FUTURE_ALIGNED=0` and `OCCSTRESS_COME_FUTURE_ALIGNED=0` are rejected;
keep legacy outputs separate. No combination is claimed completed merely
because its code path is listed.
# Lightweight Core and Compatibility

The model-native selectors below remain supported. See [Core workflow](CORE_WORKFLOW.md)
for explicit task manifests, verified resume, versioned count normalization and
suite summaries. The new core policy does not overwrite legacy/native paper scores.
