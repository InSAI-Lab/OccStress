# SparseWorld-TC OccStress Adapter

This is a one-stage camera forecaster. It does not accept manual occupancy-state
protocols or a point-upstream occupancy source. Its camera manifest contains
clean plus 8 corruptions x 3 severities x 3 temporal regimes.

From an isolated SparseWorld environment with locally built CUDA extensions:

```bash
cd "$OCCSTRESS_CODE_ROOT/EXIST/4D/SparseWorld"
python tools/build_occstress_manifest.py
python tools/occstress_eval.py \
  --checkpoint /path/to/sparseworld_tc.pth \
  --ann-file /path/to/bevdetv2-nuscenes_infos_val.pkl \
  --camera-root /path/to/nuScenes-c \
  --protocol-index 0 --anchor-mode occstress --max-samples 32 \
  --output-json "$OCCSTRESS_CODE_ROOT/outputs/sparseworld/clean-smoke.json"
```

Use `--anchor-mode official` for the original reproduction population; formal
OccStress uses its canonical anchor set. Do not mix these populations.

Waymo and CARLA entrypoints are `tools/waymo_occstress_eval.py` and
`tools/carla_occstress_eval.py`. These need camera/pose metadata in addition to
occupancy targets. New-environment A100 smoke tests passed for clean and one
corruption on all three datasets, two anchors per setting. Waymo/CARLA image
preprocessing matches the earlier implementation; GPU prediction parity and
full-paper reproduction are not claimed. See
[release status](../../../docs/CODE_RELEASE_STATUS.md).

The original repository snapshot has no top-level license file. Resolve
redistribution terms before publicly publishing this vendored tree or weights.
The root OccStress license does not relicense third-party code.
