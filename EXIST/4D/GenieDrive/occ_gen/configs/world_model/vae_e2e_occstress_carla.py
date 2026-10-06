# OccStress adapter/portability modifications; see docs/source-imports.json.
"""GenieDrive zero-shot evaluation on the right-handed OccStress-CARLA view."""

import os

_base_ = ["./vae_e2e_occstress_waymo.py"]

samples_per_gpu = int(os.environ.get("OCCSTRESS_CARLA_BATCH_SIZE", "1"))
workers_per_gpu = int(os.environ.get("OCCSTRESS_CARLA_NUM_WORKERS", "2"))

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(dataset_name="OccStress-CARLA"),
    val=dict(dataset_name="OccStress-CARLA"),
    test=dict(dataset_name="OccStress-CARLA"),
)

del os
