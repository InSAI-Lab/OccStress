# DOME OccStress Adapter

Set `OCCSTRESS_CODE_ROOT` from the release root and activate an isolated DOME
environment. Data/weights are placeholders; this is an entrypoint template.

```bash
cd "$OCCSTRESS_CODE_ROOT/EXIST/4D/DOME"
python tools/eval_occstress.py \
  --config config/train_dome.py \
  --checkpoint /path/to/dome.pth \
  --vae-checkpoint /path/to/occvae.pth \
  --protocol "$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl" \
  --base-info /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --occstress-root "$OCCSTRESS_DATA_ROOT" \
  --nuscenes-root /path/to/nuscenes \
  --output-json "$OCCSTRESS_CODE_ROOT/outputs/dome/clean-smoke.json" \
  --max-samples 32 --num-workers 2
```

`tools/dispatch_occstress_shards.py` and `tools/merge_occstress_shards.py`
provide shard-based evaluation and count aggregation. Inspect their arguments;
do not choose concurrency based on another machine's GPU memory.

The adapter uses four observed states and six genuine future targets. Token-based
random seeds preserve per-anchor sampling across sharding. New-environment A100
small-sample comparisons passed on all three datasets with `--deterministic`;
see [release status](../../../docs/CODE_RELEASE_STATUS.md) for default-policy
differences and scope. Optional qualitative voxel export is not packaged.
