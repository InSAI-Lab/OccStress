# OccStress adapter/portability modifications; see docs/source-imports.json.
"""II-World zero-shot evaluation on the right-handed OccStress-CARLA view."""

import os

_base_ = ["./ii_generate_world_waymo_occstress.py"]

samples_per_gpu = int(os.environ.get("OCCSTRESS_CARLA_BATCH_SIZE", "16"))
workers_per_gpu = int(os.environ.get("OCCSTRESS_CARLA_NUM_WORKERS", "2"))

# OccStress-CARLA is converted into the nuScenes/Occ3D coordinate convention. Keep
# the official nuScenes control path instead of Waymo's synthetic command.
model = dict(
    dataset_type="occ3d",
    eval_metric="forecasting_miou",
    eval_all_future=True,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(dataset_name="OccStress-CARLA"),
    val=dict(dataset_name="OccStress-CARLA"),
    test=dict(dataset_name="OccStress-CARLA"),
)

del os
