_base_ = ['./ii_generate_world_4gpu_bs16.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.nuscenes_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
    ],
    allow_failed_imports=False)

from configs.world_model.stage2_token_roots import baseline_stage2_lookup_token_roots

import os
import sys
from pathlib import Path

code_root = str(Path(os.environ.get('OCCSTRESS_CODE_ROOT', '../../..')).resolve())
if code_root not in sys.path:
    sys.path.insert(0, code_root)
from scripts.occstress_layout import dataset_occstress_root, resolve_manual_protocol_path

occstress_root = str(dataset_occstress_root(root=code_root))
protocol_name = os.environ.get('PROTOCOL_NAME', 'clean_H4_F6_val_backbone')
protocol_path = os.environ.get('PROTOCOL_PATH') or str(
    resolve_manual_protocol_path(protocol_name, root=code_root))
occstress_token_root = os.environ.get('OCCSTRESS_IIWORLD_TOKEN_ROOT', str(Path(occstress_root) / 'cache/tokens' / protocol_name / 'token_4f'))
clean_token_roots = [os.environ['IIWORLD_CLEAN_TOKEN_ROOT']] if os.environ.get('IIWORLD_CLEAN_TOKEN_ROOT') else baseline_stage2_lookup_token_roots()
merged_ann_file = os.environ.get('IIWORLD_BASE_INFO', str(
    Path(code_root) / 'data/nuscenes/world-nuscenes_infos_trainval.pkl'))
del os, sys, Path, dataset_occstress_root, resolve_manual_protocol_path
dataset_name = 'occ3d'
eval_metric = 'forecasting_miou'
occ_class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                   'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                   'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free']
train_load_future_frame_number = 6
test_load_future_frame_number = 6
test_load_previous_frame_number = 4
samples_per_gpu = 16
workers_per_gpu = 4

dataset_type = 'OccStressNuScenesWorldDataset'
data_root = 'data/nuscenes/'

collect_meta_keys = (
    'sample_idx', 'pts_filename', 'index', 'sequence_group_idx',
    'curr_to_prev_ego_rt', 'start_of_sequence', 'can_bus', 'scene_name',
    'occ_index', 'occ_path', 'previous_curr_to_prev_ego_rt',
    'curr_to_future_ego_rt', 'gt_ego_fut_trajs', 'gt_ego_fut_cmd',
    'curr_ego_to_global', 'gt_ego_lcf_feat', 'valid_frame',
    'ego_to_global_rotation', 'ego_to_global_translation',
    'ego_from_sensor', 'sample_weight', 'protocol_sample_id',
    'history_tokens', 'future_tokens', 'future_length', 'anchor_token',
    'occstress_protocol', 'occstress_corruption', 'occstress_misalignment')

train_pipeline = [
    dict(
        type='OccStressLoadStreamLatentToken',
        current_data_path=occstress_token_root,
        future_data_path=clean_token_roots),
    dict(type='Collect3D', keys=['latent'], meta_keys=collect_meta_keys)
]

test_pipeline = [
    dict(
        type='OccStressLoadStreamLatentToken',
        current_data_path=occstress_token_root,
        future_data_path=clean_token_roots),
    dict(type='OccStressLoadStreamOcc3D'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent'], meta_keys=collect_meta_keys)
]

share_data_config = dict(
    type=dataset_type,
    protocol_path=protocol_path,
    ann_file=merged_ann_file,
    classes=occ_class_names,
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    load_previous_data=True,
    use_sequence_group_flag=False,
)

test_data_config = dict(
    data_root=data_root,
    pipeline=test_pipeline,
    load_future_frame_number=test_load_future_frame_number,
    load_previous_frame_number=test_load_previous_frame_number,
    test_mode=True,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        protocol_path=protocol_path,
        ann_file=merged_ann_file,
        pipeline=train_pipeline,
        classes=occ_class_names,
        test_mode=False,
        load_future_frame_number=train_load_future_frame_number,
        load_previous_frame_number=test_load_previous_frame_number,
        use_sequence_group_flag=False,
        load_previous_data=True,
        dataset_name=dataset_name,
        eval_metric=eval_metric,
    ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)
