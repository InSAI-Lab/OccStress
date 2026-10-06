# EFFOcc / FlashOcc OccStress Integration

This tree contains EFFOcc Waymo camera+LiDAR adapters and CARLA EFFOcc/FlashOcc
configs. FlashOcc uses a distinct camera-only model and checkpoint. Waymo
EFFOcc camera-stress corrupts cameras **within a fusion model**.

See `docs/UPSTREAM_EXPORTS.md`, `configs/upstream_methods.json` and
`docs/ENVIRONMENTS.md` at the release root for the reviewed pairings and isolated
installation recipe. Small new-environment A100 export-to-GenieDrive gates have
passed for the documented Waymo and CARLA pairings. These are interface checks,
not full 3D accuracy reproduction. See
[release status](../../../docs/CODE_RELEASE_STATUS.md).
