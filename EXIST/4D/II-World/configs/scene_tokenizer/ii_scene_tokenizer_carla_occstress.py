# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Export protocol-conditioned II-World tokens for OccStress-CARLA."""

import os

_base_ = ["./ii_scene_tokenizer_waymo_occstress.py"]

workers_per_gpu = int(os.environ.get("OCCSTRESS_CARLA_NUM_WORKERS", "2"))
model = dict(results_type="occ3d")
data = dict(
    workers_per_gpu=workers_per_gpu,
    train=dict(dataset_name="OccStress-CARLA"),
    val=dict(dataset_name="OccStress-CARLA"),
    test=dict(dataset_name="OccStress-CARLA"),
)

del os
