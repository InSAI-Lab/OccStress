# Code Layout

The lightweight package is separate from model code and model environments.
No model, dataset or checkpoint is downloaded by installing the core package.

```text
occstress/
  protocols/     Temporal masks, canonical layout, suite expansion, record checks
  datasets/      Portable roots and explicit label remapping
  corruptions/   Existing nuScenes manual operators, with their original seeds
  metrics/       Versioned count-based F6 scoring
  results/       Strict JSON, validation, shard merge and suite summaries
  adapters/      Native command selection, raw-count readers, source descriptors
configs/
  evaluation_contract.json   Authoritative timeline, model and dataset registry
  methods.json               Generated compatibility inventory, not another authority
  suites/                    nuScenes/Waymo/CARLA manual38 definitions
  resources.json             Dataset downloads and checkpoint references
  distribution.json          Source profiles and optional exclusions
  run-manifest.example.json  Explicit, resumable native tasks
EXIST/{3D,4D}/                Model sources, adaptations and upstream notices
environments/                Isolated installation profiles for each model
scripts/
  nuscenes/                  nuScenes construction entrypoints
  waymo/                     Waymo-specific construction and upstream export
  carla/                     CARLA-specific construction and upstream export
  statistics/                Optional research analyses, not required for evaluation
  position_sweep/            Diagnostic summaries and plots
tools/                       Launch, summarize, validate and release maintenance
examples/fixtures/           Small synthetic CPU fixtures, no benchmark data
tests/                       CPU regression tests
results/reference/           Reserved for audited, compact result summaries
docs/                        Setup, interfaces, extension guide and provenance
```

## Compatibility

Old `scripts/generate_*_subset.py`, `scripts/generate_protocols.py`,
`scripts/build_clean_backbone.py`, `scripts/occstress_layout.py`, and
`tools/adapters/occstress_paths.py` remain import/CLI shims. Importing a shim
returns the implementation module, including mutable realization-seed state.
Existing dataset paths and included model-native entrypoints remain valid.
The maintainer tree retains all internal model sources. Build the independent
full-source candidate using `tools/build_public_release.py --profile full_source_candidate`; do not publish the
maintainer Git history. See [source packaging](#source-packaging).
Historical analysis notes, machine-specific package dumps and migration logs
are excluded from that candidate. Public setup instructions use configurable
paths and model-specific installation recipes, not a particular cluster.

`occstress` extends its namespace for SparseWorld's existing dataset adapters.
It does not import any model registry into the core process. Native models are
still launched in separate interpreters; do not combine their MMDetection stacks.

Waymo/CARLA corruption implementations stay dataset-specific. This migration
does not silently substitute nuScenes operators, change random seeds, or regenerate
any assets. The shared label-remapping API is explicit and rejects unmapped labels.

## Registry Maintenance

Edit `configs/evaluation_contract.json` for formal model interfaces. Then review
the diff and deliberately refresh derived files:

```bash
python tools/update_release_metadata.py --write
python tools/check_formal_configs.py --write-lock
```

Without the write flags these tools only check for drift. Source-import hashes
document selected source snapshots.
The lock covers core scoring, suites, launchers, declared model entrypoints,
formal config inheritance and environment recipes, not every third-party file.

## Source Packaging

Build a new source-only candidate outside the maintainer checkout. Do not push
the internal working tree or its existing Git history as the release.

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/build_public_release.py --profile full_source_candidate --output ../OccStress-public
cd ../OccStress-public
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python tools/check_release.py --public --syntax
PYTHONDONTWRITEBYTECODE=1 python tools/check_documentation.py
PYTHONDONTWRITEBYTECODE=1 python tools/update_release_metadata.py
```

The builder never overwrites an existing output. It excludes `.git`, model
assets, runtime caches, historical research notes, machine-specific inventories
and external mounts. Safe relative model links to the checkout's shared `data/`
directory are retained; that directory ships only its README, not datasets.
The builder regenerates the config lock and source-availability registry.

`full_source_candidate` includes all prepared integrations. The optional
`public_candidate` profile omits the components and helpers listed in
`configs/distribution.json`, keeps source-availability placeholders and rejects
their launch. An unmodified upstream clone does not restore the omitted
OccStress adapters. Packaging does not download assets or upload the repository.

Retain upstream LICENSE files, copyright notices and component-specific terms;
see [third-party notices](../THIRD_PARTY_NOTICES.md).
