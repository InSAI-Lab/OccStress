# Quick Start

**Source availability:** the full-source candidate includes all prepared model
integrations and helpers. Data, weights and isolated environments are separate
prerequisites. See [model adapters](method-adapters.md).

This is a code-first preparation snapshot. Small A100 tests have passed for six
forecasters and seven upstream sources; see [the exact scope](CODE_RELEASE_STATUS.md).
Dataset packages are selected through the published index; checkpoint identities
and official links are recorded in [the resource registry](../configs/resources.json).
See [download selection](RESOURCES.md) for per-shard availability.
Commands below use separately prepared assets. Small-sample checks do not
replace full-paper reproduction. Component terms are listed in the
[license inventory](LICENSES_AND_CITATIONS.md).

## 1. Code-Only Checks

From the repository root, using Python 3.10+:

```bash
export OCCSTRESS_CODE_ROOT="$PWD"
python -m pip install -e '.[construction]'
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python tools/check_release.py --syntax
python tools/release_status.py
python tools/check_formal_configs.py
python tools/update_release_metadata.py
```

Try the [synthetic CPU demo and result workflow](CORE_WORKFLOW.md) before
configuring a model environment. These checks do not need real datasets or weights.

## 2. Set the Shared Dataset Root

First [download and verify the selected packages](RESOURCES.md), then prepare
authorized GT and metadata with the
[external dependency guide](EXTERNAL_DATA_PREPARATION.md). Do not mount native
left-handed CARLA arrays as canonical GT.

```bash
export OCCSTRESS_DATA_ROOT="/path/to/OccStress"
export OCCSTRESS_EXTERNAL_ROOT="/path/to/OccStress-external"
```

Existing authorized originals can be mounted without copying. See [resources](RESOURCES.md) for default mounts
and per-shard availability. Historical PKLs with another machine's
absolute paths must be regenerated or explicitly relocated before evaluation.

For example, resolve a protocol without loading any PKL:

```bash
python scripts/occstress_layout.py resolve-manual-protocol clean_H4_F6_val_backbone
python scripts/occstress_layout.py resolve-manual-protocol clean_H4_F6_val_backbone --dataset waymo
```

## 3. Use an Isolated Model Environment

See [environments](ENVIRONMENTS.md). Run each method from its own directory.
Do not combine multiple model-specific `mmdet3d` packages on one PYTHONPATH.
The explicit CVT-Occ/base-package pairing is documented separately.
Use `tools/run_formal.py` for whitelisted config selection and
`tools/install_environment.py` to print per-model installation commands.
Both default to dry-run; see [formal evaluation](FORMAL_EVALUATION.md).

| Method | Instructions |
| --- | --- |
| OccWorld | [README_OCCSTRESS.md](../EXIST/4D/OccWorld/README_OCCSTRESS.md) |
| I2-World | [README_OCCSTRESS.md](../EXIST/4D/II-World/README_OCCSTRESS.md) |
| COME | [README_OCCSTRESS.md](../EXIST/4D/COME/README_OCCSTRESS.md) |
| GenieDrive | [README_OCCSTRESS.md](../EXIST/4D/GenieDrive/README_OCCSTRESS.md) |
| DOME | [README_OCCSTRESS.md](../EXIST/4D/DOME/README_OCCSTRESS.md) |
| SparseWorld-TC | [README_OCCSTRESS.md](../EXIST/4D/SparseWorld/README_OCCSTRESS.md) |

Start with clean and a small `max_samples` limit before a full protocol.
Use a new output directory to avoid accidentally resuming from an unrelated
checkpoint. Record exact weights, code revision, protocol, label mapping,
physical horizons and control policy.

## 4. Construction and Analysis

- Manual generators: `scripts/nuscenes/generate_manual.py FAMILY`; the original
  `scripts/generate_*_subset.py` commands remain compatible. Semantic/dropout/hole
  accept `--realization-seed`. Omit it for the original deterministic realization.
  See [manual reconstruction](DATA_CONSTRUCTION.md) for the shared-root commands.
- Dataset builders: `scripts/waymo/` and `scripts/carla/`. CARLA's model view
  requires the explicit class/coordinate transformation step.
- Upstream exports: ALOcc, STCOcc, FusionOcc, SDGOcc, CVT-Occ, EFFOcc and FlashOcc have
  [explicit config/exporter/builder pairings](UPSTREAM_EXPORTS.md).
  Keep each source's sensor inputs, class map and native environment distinct.
- Optional statistics: `scripts/statistics/`, outside the default evaluation workflow.
- Position plots: `scripts/position_sweep/render_isometric_position_sweep_svg.py`
  with `--input`, `--output-dir`, and optional `--include-tminus4`.

No construction, GPU scheduling, download, or training is launched by setup.
