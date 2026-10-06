# I2-World OccStress Adapter

This snapshot contains the two-stage nuScenes adapter and the Waymo/CARLA
scene-shard wrapper. A fresh environment passed small A100 tests on all three
datasets. Metrics match the pre-migration snapshot; Waymo/CARLA confusion
matrices also match. This is not full-paper reproduction. See
[release status](../../../docs/CODE_RELEASE_STATUS.md).

The existing interface exports tokens before forecasting. In an isolated
I2-World environment, set the shared roots at the release root, then:

```bash
cd "$OCCSTRESS_CODE_ROOT/EXIST/4D/II-World"
export PROTOCOL_NAME=clean_H4_F6_val_backbone
export PROTOCOL_PATH="$OCCSTRESS_DATA_ROOT/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl"
export IIWORLD_BASE_INFO=/path/to/world-nuscenes_infos_trainval.pkl
export OCCSTRESS_IIWORLD_SAVE_ROOT=/path/to/temporary/protocol-tokens
export OCCSTRESS_IIWORLD_TOKEN_ROOT="$OCCSTRESS_IIWORLD_SAVE_ROOT/token_4f"
export IIWORLD_CLEAN_TOKEN_ROOT=/path/to/clean-token_4f
python tools/test.py configs/scene_tokenizer/ii_scene_tokenizer_occstress.py \
  /path/to/tokenizer.pth --eval mIoU
python tools/test.py configs/world_model/ii_generate_world_occstress.py \
  /path/to/world_model.pth --eval mIoU
```

Clean tokens must be prepared with the matching tokenizer/config. Token caches
are not distributed as checkpoint or source code assets. The two configs now
consume the environment variables above, and their dataset import names match
the packaged module.

For Waymo/CARLA use `tools/test_waymo_occstress.py` through the formal selector in
`docs/FORMAL_EVALUATION.md`. It exports current tokens per protocol/scene shard,
uses a shared future-token cache, streams confusion matrices instead of retaining
predictions, and removes temporary current tokens after success. Traffic future
tokens are isolated from unmirrored targets; config, adapter, base and checkpoint
hashes participate in the future cache identity.

The explicit `--dataset carla` must be paired with both CARLA configs. The formal
selector supplies them. `--keep-token-cache` is debugging-only. Use a distinct
work root per concurrent shard, and merge confusion counts before computing scores.
The correction to zero-IoU class handling is described in the formal guide;
do not silently mix historical native log scores with present-class summaries.

The MIT notice is restored from experimental base `2b487f8`; later upstream
HEAD deleted it. See [the license audit](../../../docs/LICENSES_AND_CITATIONS.md)
for the version-specific evidence, not a blanket grant for future updates.
