# GenieDrive OccStress Adapter

Only the occupancy-generation branch is included, not the video pipeline.
The base snapshot precedes unrelated multi-agent work. Source revisions and
selected working-tree updates are recorded in `docs/source-imports.json`.

After setting `OCCSTRESS_CODE_ROOT` at the release root and installing a separate
GenieDrive environment:

```bash
cd "$OCCSTRESS_CODE_ROOT/EXIST/4D/GenieDrive/occ_gen"
export GENIEDRIVE_NUSCENES_ROOT="/path/to/nuscenes/"
python tools/test_occstress.py \
  --checkpoint /path/to/genie_occ.pth \
  --protocol "$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl" \
  --occstress-root "$OCCSTRESS_DATA_ROOT" \
  --base-info /path/to/world-nuscenes_infos_val.pkl \
  --output-json "$OCCSTRESS_CODE_ROOT/outputs/geniedrive/clean-smoke.json" \
  --max-samples 32
```

For Waymo/CARLA choose `configs/world_model/vae_e2e_occstress_waymo.py` or
`vae_e2e_occstress_carla.py` with the corresponding protocol, canonical base-info
and root. Their class-map manifest must accompany base-info. Small fresh-environment
A100 comparisons passed on all three datasets; see
[release status](../../../docs/CODE_RELEASE_STATUS.md) for evidence and limits.
Do not reuse a nuScenes base-info PKL with a different dataset.

The adapter observes -1.5, -1, -0.5, 0 seconds, uses prescribed GT future controls,
and scores six actual future frames. Prefer the official checkpoint reference;
see the shared resource registry rather than assuming bundled weights.
