_base_ = ['./ii_generate_world_occstress.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.nuscenes_occstress_world_dataset',
        'mmdet3d.datasets.pipelines.loading_occstress',
        'mmdet3d.datasets.pipelines.loading_vote_prior',
        'mmdet3d.models.ii_world.world_model.ii_world_vote_fusion',
    ],
    allow_failed_imports=False,
)

from configs.world_model.stage2_token_roots import baseline_stage2_lookup_token_roots

protocol_name = 'dropout_hard_current_H4_F6_val_backbone'
protocol_path = 'data/OccStress/protocols/manual/dropout/hard/current_H4_F6_val_backbone.pkl'
occstress_token_root = f'data/OccStress/save_dir_stage2_vote_prior_baseline_occstress_hard/{protocol_name}/token_4f'
vote_token_root = f'data/OccStress/save_dir_vote_prior_baseline_occstress_hard/{protocol_name}/token_4f'
clean_token_roots = baseline_stage2_lookup_token_roots()

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
    type='II_WorldVoteFusion',
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    vote_fusion=dict(
        enabled=True,
        corr_ratio=0.5,
        seed=3407,
        gate_bias=-4.0,
        identity_weight=0.02,
        gate_weight=0.001,
        noise_std=0.02,
        channel_drop_prob=0.08,
        block_drop_prob=0.65,
        min_block=4,
        max_block=14,
    ),
)

train_pipeline = [
    dict(
        type='OccStressLoadStreamLatentVoteToken',
        current_data_path=occstress_token_root,
        future_data_path=clean_token_roots,
        vote_data_path=vote_token_root,
        load_vote_confidence=True,
    ),
    dict(type='Collect3D', keys=['latent', 'vote_latent', 'vote_confidence'], meta_keys=collect_meta_keys)
]

test_pipeline = [
    dict(
        type='OccStressLoadStreamLatentVoteToken',
        current_data_path=occstress_token_root,
        future_data_path=clean_token_roots,
        vote_data_path=vote_token_root,
        load_vote_confidence=True,
    ),
    dict(type='OccStressLoadStreamOcc3D'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'vote_latent', 'vote_confidence'], meta_keys=collect_meta_keys)
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

load_from = 'work_dirs/ii_generate_world_baseline_original_stage2_pytorchms_8gpu_job3993720/iter_95952_ema.pth'

freeze_dict = dict(
    pose_encoder=True,
    transformer=True,
)
