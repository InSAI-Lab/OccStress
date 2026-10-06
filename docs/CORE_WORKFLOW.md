# Core Workflow

## Install and Try Without a GPU

From the repository root, in a lightweight Python 3.10+ environment:

```bash
python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 python examples/minimal_eval.py --output-dir /tmp/occstress-demo
python tools/validate_protocol.py examples/fixtures/protocol.json --expected-anchors 1
python tools/summarize.py suite --suite /tmp/occstress-demo/suite.json \
  /tmp/occstress-demo/clean.json /tmp/occstress-demo/synthetic_error.json \
  --output /tmp/occstress-demo/resummary.json --csv /tmp/occstress-demo/summary.csv
```

These are **synthetic** scores, not any paper model's results. The core only
requires NumPy. `pip install -e '.[construction]'` adds Pillow/SciPy for manual
operators. Full dataset builders still need their documented SDK/MMCV environment.
Installing the package does not install any forecasting model.

## Build and Validate Protocols

```bash
python scripts/nuscenes/generate_manual.py semantic --help
python scripts/nuscenes/build_clean_backbone.py --help
python scripts/nuscenes/generate_protocols.py --help
python tools/validate_protocol.py /datasets/OccStress/protocols/manual/OccStress-Waymo/clean/H4_F6_val_backbone.pkl \
  --trust-pickle --expected-anchors 5978
```

Only use `--trust-pickle` for files you trust: Python pickles can execute code.
The lightweight validator checks canonical H4/current/F6 tokens and unique anchors;
it does not certify voxel assets, controls, masks or calibration. Use the existing
dataset-specific validators for those checks.

## Launch and Resume

The single-task `tools/run_formal.py` interface is unchanged. For multiple
protocols or scene shards, prepare an explicit task manifest following
`configs/run-manifest.example.json`:

```bash
python tools/run_formal.py --manifest configs/run-manifest.example.json
python tools/run_formal.py --manifest /runs/my-tasks.json --execute
```

The first command only prints commands. Each task selects an isolated Python
interpreter and retains native flags, because tokenizer caches, controls and
scene-shard selectors differ between methods. `native_output` must match the
native output flag. `expected_records` must match the actual shard, not the full
dataset. The wrapper does not fabricate missing assets or translate unsupported
native arguments.

Declare every consumed checkpoint component, protocol, base-info and extra
configuration asset in `inputs`; the runner hashes the declared files. Voxel asset
integrity still relies on the external dataset manifest. Run output paths must
be absolute, unique and separate from input files.

After successful execution, the runner checks F6 raw counts and the expected
record count, then writes an atomic `.done.json` receipt. Resume verifies the
task/input/locked-source fingerprint and output SHA256. A mismatched existing
output is not overwritten or accepted: archive it or select a fresh path.
An interrupted task may leave a `.lock`; verify that its process is gone before
removing that lock. Failed tasks never receive a success receipt.

Tasks within one manifest run sequentially. For GPU concurrency, submit disjoint
manifests/output paths through your scheduler and set `CUDA_VISIBLE_DEVICES` per
worker. Native scene-sharding support is model-specific; this runner neither
allocates GPUs nor claims a common sharding API across all six models.

## Normalize and Summarize

`tools/summarize.py normalize` reads raw sufficient statistics from II-World,
GenieDrive, COME, OccWorld, DOME or SparseWorld-TC. Metrics-only logs cannot be
converted reliably. Supply metadata with `identity`, `anchor_ids` and
`future_horizons_seconds`, following the synthetic demo's canonical result:

```bash
python tools/summarize.py normalize --native /runs/clean.native.json \
  --metadata /runs/clean.metadata.json --output /runs/clean.canonical.json
python tools/summarize.py merge /runs/clean.scene-a.json /runs/clean.scene-b.json \
  --output /runs/clean.canonical.json
python tools/summarize.py suite --suite configs/suites/waymo.json \
  /runs/canonical/*.json --output /runs/manual-summary.json --csv /runs/manual-summary.csv
```

Metadata must describe **exactly the evaluated anchors**, using
`[scene_name, anchor_token]` pairs. Never attach the full protocol's IDs to a
partial/native subset. The normalizer checks the count and F6 declaration but
cannot independently recover missing anchor identity from old aggregate logs.
Record model/dataset/track/source/protocol, code revision, all checkpoint SHA256s,
protocol/base-info/class-map SHA256s, input offsets, control policy, mask and seed.

The explicit new policy is `occstress-present-gt-v1`: globally present GT occupied
classes (0..16), including zero IoU; binary occupied IoU; scores in percent.
Confusion rows are GT and columns are predictions, with binary occupied label 1.
Counts are merged **before** scoring. All six true future horizons are retained;
paper averages use 1/2/3 seconds. No current reconstruction is substituted.
Empty support is JSON `null`, never NaN.

This policy is not a relabeling of legacy native logs or historical paper tables.
Keep those artifacts unchanged and identify any newly recomputed scores explicitly.
Inconsistent semantic/binary masks are rejected rather than silently repaired.

The three shipped suites define the 38 manual conditions. Upstream summaries
use an explicit source-specific `conditions` list with one clean condition and
the declared corruptions; do not mix sensor sources into one clean baseline.
Robust mean excludes clean and traffic. Missing protocols normally fail;
`--allow-partial` reports coverage with full-suite robustness left null. Completed
conditions must have matching model/checkpoint/control identity and anchor cohorts.

## Optional Analysis

Bootstrap and other statistical scripts remain in `scripts/statistics/`, outside
the default package workflow and CI model gates. Their historical score policies
must be interpreted as documented, not substituted for the core policy.
