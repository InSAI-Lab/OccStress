_base_ = ['./ii_generate_world_stage2_history_norm_calib.py']

import os

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.nuscenes_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
        'mmdet3d.models.ii_world.world_model.ii_world_history_norm_calib',
    ],
    allow_failed_imports=False,
)

from configs.world_model.stage2_token_roots import baseline_stage2_lookup_token_roots

protocol_name = os.environ.get('PROTOCOL_NAME', 'dropout_hard_current_H4_F6_val_backbone')
protocol_path = os.environ.get(
    'PROTOCOL_PATH',
    'data/OccStress/protocols/manual/dropout/hard/current_H4_F6_val_backbone.pkl')
occstress_token_root = os.environ.get(
    'CURRENT_TOKEN_ROOT',
    f'data/OccStress/save_dir_stage2_vote_prior_baseline_occstress_hard/{protocol_name}/token_4f')
clean_token_roots = baseline_stage2_lookup_token_roots()

merged_ann_file = 'data/nuscenes/world-nuscenes_infos_trainval.pkl'
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

model = dict(
    type='II_WorldHistoryNormCalib',
    previous_frame_exist=True,
    previous_frame=4,
    test_previous_frame=4,
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    norm_calib=dict(
        enabled=True,
        history_index=-1,
        strength=1.0,
        deadband=0.03,
        min_scale=0.90,
        max_scale=1.15,
    ),
)

train_pipeline = [
    dict(
        type='OccStressLoadStreamLatentHistoryToken',
        current_data_path=[occstress_token_root] + clean_token_roots,
        future_data_path=clean_token_roots,
        history_data_path=clean_token_roots,
        history_frame_number=4,
    ),
    dict(type='Collect3D', keys=['latent', 'history_latent'], meta_keys=collect_meta_keys)
]

test_pipeline = [
    dict(
        type='OccStressLoadStreamLatentHistoryToken',
        current_data_path=[occstress_token_root] + clean_token_roots,
        future_data_path=clean_token_roots,
        history_data_path=clean_token_roots,
        history_frame_number=4,
    ),
    dict(type='OccStressLoadStreamOcc3D'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'history_latent'], meta_keys=collect_meta_keys)
]

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
    val=dict(
        data_root=data_root,
        pipeline=test_pipeline,
        load_future_frame_number=test_load_future_frame_number,
        load_previous_frame_number=test_load_previous_frame_number,
        test_mode=True,
    ),
    test=dict(
        data_root=data_root,
        pipeline=test_pipeline,
        load_future_frame_number=test_load_future_frame_number,
        load_previous_frame_number=test_load_previous_frame_number,
        test_mode=True,
    ),
)

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

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)
