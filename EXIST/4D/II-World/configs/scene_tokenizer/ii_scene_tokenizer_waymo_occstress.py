# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
from pathlib import Path

_base_ = ['./ii_scene_tokenizer_waymo_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.waymo_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_occstress',
    ],
    allow_failed_imports=False,
)

protocol_path = str(Path(os.environ['OCCSTRESS_WAYMO_PROTOCOL']).resolve())
base_info = str(Path(os.environ['OCCSTRESS_WAYMO_BASE_INFO']).resolve())
save_root = str(Path(os.environ['OCCSTRESS_IIWORLD_WAYMO_TOKEN_ROOT']).resolve())
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

token_mode = os.environ.get('IIWORLD_WAYMO_TOKEN_MODE', 'current')
dataset_type = (
    'OccStressWaymoFutureTokenizerDataset'
    if token_mode == 'future' else 'OccStressWaymoTokenizerDataset')
workers_per_gpu = int(os.environ.get('OCCSTRESS_WAYMO_NUM_WORKERS', '2'))

collect_meta_keys = (
    'sample_idx', 'index', 'sequence_group_idx', 'curr_to_prev_ego_rt',
    'start_of_sequence', 'bda_mat', 'scene_name', 'occ_path', 'protocol_sample_id',
    'occstress_save_token', 'anchor_token', 'occstress_corruption',
)

model = dict(
    type='OccStressIISceneTokenizer',
    save_root=save_root,
    save_only_anchor=True,
)

test_pipeline = [
    dict(type='OccStressLoadStreamOcc3D', dataset_type='waymo_occstress', to_long=True),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Collect3D', keys=['voxel_semantics'], meta_keys=collect_meta_keys),
]
train_pipeline = test_pipeline

share_data_config = dict(
    type=dataset_type,
    protocol_path=protocol_path,
    base_info=base_info,
    ann_file=base_info,
    classes=occ_class_names,
    dataset_name='waymo-Occ3D',
    eval_metric='miou',
    scene_shard=scene_shard,
    max_samples=max_samples,
)

test_data_config = dict(
    data_root='',
    pipeline=test_pipeline,
    test_mode=True,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root='',
        pipeline=train_pipeline,
        test_mode=True,
        **share_data_config,
    ),
    val=dict(**test_data_config, **share_data_config),
    test=dict(**test_data_config, **share_data_config),
)

del os, Path
