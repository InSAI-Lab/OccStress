# Datasets

All datasets share **one OccStress root**. Resource type comes first, then track,
then dataset. Public dataset names are `OccStress-nuScenes`, `OccStress-Waymo`
and `OccStress-CARLA`.

```text
OccStress/
  meta/
    OccStress-nuScenes/{dataset.json,class_mapping.json}
    OccStress-Waymo/{dataset.json,class_mapping.json,controls.json}
    OccStress-CARLA/{dataset.json,class_mapping.json,controls.json}
  protocols/
    manual/<dataset>/<family>/<severity>/*.pkl
    upstream/<dataset>/<subtrack>/<source>/<condition>/*.pkl
    position_sweep/OccStress-nuScenes/<setting>/*.pkl
  occ/
    manual/<dataset>/<family>/<severity>/<scene>/<frame>/labels.npz
    upstream/<dataset>/<subtrack>/<source>/<condition>/<scene>/<frame>/labels.npz
  events/manual/<dataset>/...
  manifests/
    protocols.json
    assets.jsonl
    checksums/
    <dataset>.payload_progress.json
  external/<dataset>/...
```

Clean manual protocols use `clean/H4_F6_val_backbone.pkl`; traffic has no severity
directory. Misalignment is represented by transforms, not duplicated occupancy.
Upstream `condition` is `clean` or `<corruption>/<severity>`. Temporal patterns
and position sweeps reuse assets; they do not add copies per model or protocol.

## Mounts

From the checkout root:

```bash
export OCCSTRESS_CODE_ROOT="$PWD"
export OCCSTRESS_DATA_ROOT="/datasets/OccStress"
```

Alternatively, create `data/OccStress` as a local symlink **only if that path does
not already exist**. Do not delete or overwrite an existing dataset to create it.
The `--occstress-root` argument, where a native model exposes it, means this
shared root, not a per-dataset directory.

Protocol paths are relative to this root, independent of the model process's
working directory. Dataset names prevent collisions between identical scene or
frame IDs. The resolver rejects mismatched dataset namespaces and parent traversal.

## External Dependencies

Follow the [step-by-step preparation commands](EXTERNAL_DATA_PREPARATION.md).
They cover safe mounts, raw-CARLA-to-canonical-GT conversion, and full
clean-reference checks without changing released controls.

Original images, point clouds and official clean GT are not bundled in staging.
Provide them under the following logical namespaces using authorized local mounts:

- `external/OccStress-nuScenes/gts/<scene>/<token>/labels.npz`
- `external/OccStress-Waymo/native_gt/validation-data/...`
- `external/OccStress-CARLA/canonical_gt/<scene>/<frame>/labels.npz`
- `external/OccStress-CARLA/native_dataset/...` for raw sensor dependencies.

`OCCSTRESS_EXTERNAL_ROOT` optionally relocates the `external/` directory; it must
contain the same `OccStress-*` subdirectories. nuScenes also requires official
world-info metadata. Waymo/CARLA use `meta/<dataset>/controls.json`. Control arrays
are preserved; loading JSON must not recompute poses, trajectories or commands.
Canonical `semantics` is already Occ3D-18. Only native Waymo `voxel_label` needs
mapping, exactly once. CARLA assets use the right-handed canonical view.

## Validation and Availability

```bash
python tools/validate_occstress_dataset.py "$OCCSTRESS_DATA_ROOT"
python tools/validate_occstress_dataset.py "$OCCSTRESS_DATA_ROOT" \
  --dataset waymo --require-payloads
```

The default check reads the catalog and metadata, not all voxel files. During
assembly, per-dataset progress and condition `.done.json` markers determine
payload readiness. After finalization, `manifests/status.json` points to the
consolidated `finalization.json`; an earlier `stage=protocol_catalog` snapshot
is not a live completion signal. `validation_coverage.json` records sampled
loader coverage separately. A completed payload is still **not** proof of full
model-loader or GPU reproduction equivalence.

Finalized datasets include `EXTERNAL_DATA.md` and
`meta/external_dependencies.json`. These describe authorized external mounts,
not bundled originals or guaranteed availability on a new machine. CARLA command
order is right/left/straight in the canonical x-forward/y-left frame; its control
audit and any diagnostic metadata corrections are recorded under `manifests/`.

The current internal staging and sampled runtime checks are complete as recorded
in [release status](CODE_RELEASE_STATUS.md). Published archives and their
availability are listed in the [download guide](RESOURCES.md). Original clean
GT remains an external dependency; publication does not replace source terms.
Do not infer public availability from staging or model smoke tests.
