# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
from pathlib import Path

_base_ = ['./ii_generate_world_waymo.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.waymo_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
    ],
    allow_failed_imports=False,
)

protocol_path = str(Path(os.environ['OCCSTRESS_WAYMO_PROTOCOL']).resolve())
base_info = str(Path(os.environ['OCCSTRESS_WAYMO_BASE_INFO']).resolve())
current_token_root = str(
    Path(os.environ['OCCSTRESS_IIWORLD_WAYMO_TOKEN_ROOT']).resolve() / 'token_4f')
clean_token_root = str(
    Path(os.environ.get(
        'IIWORLD_WAYMO_CLEAN_TOKEN_ROOT',
        'data/waymo/save_dir/token_4f')).resolve())
scene_shard = os.environ.get('OCCSTRESS_WAYMO_SCENE_SHARD')
max_samples_value = os.environ.get('OCCSTRESS_WAYMO_MAX_SAMPLES')
max_samples = int(max_samples_value) if max_samples_value else None

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]

dataset_type = 'OccStressWaymoWorldDataset'
dataset_name = 'waymo-Occ3D'
eval_metric = 'forecasting_miou'
samples_per_gpu = int(os.environ.get('OCCSTRESS_WAYMO_BATCH_SIZE', '1'))
workers_per_gpu = int(os.environ.get('OCCSTRESS_WAYMO_NUM_WORKERS', '2'))

model = dict(
    dataset_type='waymo',
    eval_metric=eval_metric,
    eval_all_future=True,
    vqvae_checkpoint=str(Path(
        os.environ['IIWORLD_TOKENIZER_CHECKPOINT']).resolve()),
)

collect_meta_keys = (
    'sample_idx', 'index', 'sequence_group_idx', 'curr_to_prev_ego_rt',
    'start_of_sequence', 'scene_name', 'occ_index', 'occ_path',
    'previous_curr_to_prev_ego_rt', 'curr_to_future_ego_rt',
    'gt_ego_fut_trajs', 'gt_ego_fut_cmd', 'curr_ego_to_global',
    'gt_ego_lcf_feat', 'valid_frame', 'ego_to_global_rotation',
    'ego_to_global_translation', 'protocol_sample_id', 'history_tokens',
    'future_tokens', 'future_latent_names', 'future_length', 'anchor_token',
    'current_latent_name', 'occstress_protocol',
)

test_pipeline = [
    dict(
        type='OccStressLoadStreamLatentToken',
        current_data_path=current_token_root,
        future_data_path=clean_token_root,
        dataset_type='waymo_occstress',
    ),
    dict(type='OccStressLoadStreamOcc3D', dataset_type='waymo_occstress'),
    dict(
        type='Collect3D',
        keys=['voxel_semantics', 'latent'],
        meta_keys=collect_meta_keys,
    ),
]
train_pipeline = test_pipeline

share_data_config = dict(
    type=dataset_type,
    protocol_path=protocol_path,
    base_info=base_info,
    ann_file=base_info,
    pose_file=None,
    classes=occ_class_names,
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    load_previous_data=True,
    use_sequence_group_flag=False,
    load_interval=1,
    scene_shard=scene_shard,
    max_samples=max_samples,
)

test_data_config = dict(
    data_root='',
    split='validation',
    pipeline=test_pipeline,
    load_future_frame_number=6,
    load_previous_frame_number=4,
    native_history_frames=4,
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
