# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
from pathlib import Path

_base_ = ['./vae_e2e_occstress.py']

custom_imports = dict(
    imports=['mmdet3d.datasets.waymo_occstress_world_dataset'],
    allow_failed_imports=False,
)

protocol_path = str(Path(os.environ['OCCSTRESS_WAYMO_PROTOCOL']).resolve())
base_info = str(Path(os.environ['OCCSTRESS_WAYMO_BASE_INFO']).resolve())
scene_shard = os.environ.get('OCCSTRESS_WAYMO_SCENE_SHARD')
max_samples_value = os.environ.get('OCCSTRESS_WAYMO_MAX_SAMPLES')
max_samples = int(max_samples_value) if max_samples_value else None

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

dataset_type = 'OccStressWaymoWorldDataset'
samples_per_gpu = int(os.environ.get('OCCSTRESS_WAYMO_BATCH_SIZE', '1'))
workers_per_gpu = int(os.environ.get('OCCSTRESS_WAYMO_NUM_WORKERS', '2'))

test_pipeline = [
    dict(type='LoadStreamOcc3D', dataset_type='waymo_occstress', to_long=True),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Collect3D', keys=['voxel_semantics']),
]

share_data_config = dict(
    type=dataset_type,
    base_info=base_info,
    protocol_path=protocol_path,
    ann_file=base_info,
    pose_file=None,
    classes=occ_class_names,
    use_sequence_group_flag=False,
    dataset_name='occ3d',
    eval_metric='forecasting_miou',
    eval_time=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    load_previous_data=True,
    load_interval=1,
    scene_shard=scene_shard,
    max_samples=max_samples,
)

test_data_config = dict(
    data_root='',
    split='validation',
    pipeline=test_pipeline,
    load_future_frame_number=6,
    load_previous_frame_number=3,
    native_history_frames=3,
    test_mode=True,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='EpochBasedRunner'),
    train=dict(**test_data_config, **share_data_config),
    val=dict(**test_data_config, **share_data_config),
    test=dict(**test_data_config, **share_data_config),
)

del os, Path
