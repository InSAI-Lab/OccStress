# COME OccStress Adapter

COME needs VAE, scene-centric U-Net, world-model and ControlNet weights.
All four resources are explicitly listed as pending in the shared registry.

```bash
cd "$OCCSTRESS_CODE_ROOT/EXIST/4D/COME"
export COME_STAGE_ONE_CKPT=/path/to/unet.pth
export COME_WORLD_MODEL_CKPT=/path/to/world_model.pth
export COME_VAE_CKPT=/path/to/occvae.pth
export OCCSTRESS_COME_DATA_PATH=/path/to/nuscenes
export OCCSTRESS_COME_IMAGESET=/path/to/nuscenes_infos_val_temporal_v3_scene.pkl
export OCCSTRESS_COME_PROTOCOL_PATH="$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl"
export OCCSTRESS_COME_MAX_SAMPLES=32
export CUDA_VISIBLE_DEVICES=0
python tools/test_diffusion_control.py \
  --py-config configs/local_eval_controlnet_occstress.py \
  --resume-from /path/to/controlnet.pth \
  --work-dir "$OCCSTRESS_CODE_ROOT/outputs/come/clean-smoke" \
  --metrics-json "$OCCSTRESS_CODE_ROOT/outputs/come/clean-smoke.json"
```

Use an empty work directory, and unset `OCCSTRESS_COME_MAX_SAMPLES` for a full protocol.
The release config defaults to `OCCSTRESS_COME_FUTURE_ALIGNED=1`: four observations
through t=0 and targets +0.5 through +3.0 seconds. The legacy switch is for
historical debugging only.

Waymo/CARLA wrappers and configs are included. Small new-environment A100
deterministic comparisons passed on all three datasets; this does not certify
paper-score reproduction or every GPU architecture. Supply matching canonical
metadata, controls and class maps. See
[release status](../../../docs/CODE_RELEASE_STATUS.md). The optional qualitative
voxel exporter is not packaged.
