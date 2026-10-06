# OccWorld OccStress Adapter

Set the shared paths at the release root, then activate an isolated environment.
The release config defaults to six genuine future frames.

```bash
cd "$OCCSTRESS_CODE_ROOT/EXIST/4D/OccWorld"
export OCCWORLD_CKPT=/path/to/occworld.pth
export OCCSTRESS_OCCWORLD_DATA_PATH=/path/to/nuscenes
export OCCSTRESS_OCCWORLD_IMAGESET=/path/to/nuscenes_infos_val_temporal_v3_scene.pkl
export OCCSTRESS_OCCWORLD_PROTOCOL_PATH="$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl"
export OCCSTRESS_OCCWORLD_MAX_SAMPLES=32
export CUDA_VISIBLE_DEVICES=0
python eval_metric_stp3.py --py-config config/occworld_occstress.py \
  --work-dir "$OCCSTRESS_CODE_ROOT/outputs/occworld/clean-smoke" \
  --metrics-json "$OCCSTRESS_CODE_ROOT/outputs/occworld/clean-smoke.json"
```

Unset `OCCSTRESS_OCCWORLD_MAX_SAMPLES` for a full protocol. `OCCSTRESS_OCCWORLD_FUTURE_ALIGNED=0`
is only for reproducing historical misaligned outputs, not current benchmark
reporting. Use an empty work directory to avoid implicit checkpoint resumption.

Small fresh-environment A100 deterministic comparisons passed on nuScenes,
Waymo and CARLA; see [release status](../../../docs/CODE_RELEASE_STATUS.md).
This is not full-paper reproduction. Downloads remain pending in the registry.
