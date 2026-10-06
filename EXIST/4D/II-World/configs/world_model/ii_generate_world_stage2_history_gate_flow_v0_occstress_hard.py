_base_ = ['./ii_generate_world_stage2_history_gate_noflow_v0_occstress_hard.py']

from configs.world_model.stage2_token_roots import baseline_stage2_lookup_token_roots

protocol_name = 'dropout_hard_current_H4_F6_val_backbone'
occstress_token_root = f'data/OccStress/save_dir_stage2_vote_prior_baseline_occstress_hard/{protocol_name}/token_4f'
clean_token_roots = baseline_stage2_lookup_token_roots()
flow_root = 'data/nuscenes/occ_flow_gt'

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
    history_gate=dict(
        use_flow_summary=True,
        flow_summary_dim=5,
    ),
)

train_pipeline = [
    dict(
        type='OccStressLoadStreamLatentHistoryToken',
        current_data_path=occstress_token_root,
        future_data_path=clean_token_roots,
        history_data_path=clean_token_roots,
        history_frame_number=4,
        load_flow_summary=True,
        flow_root=flow_root),
    dict(type='Collect3D', keys=['latent', 'history_latent', 'history_flow_summary'], meta_keys=collect_meta_keys)
]

test_pipeline = [
    dict(
        type='OccStressLoadStreamLatentHistoryToken',
        current_data_path=occstress_token_root,
        future_data_path=clean_token_roots,
        history_data_path=clean_token_roots,
        history_frame_number=4,
        load_flow_summary=True,
        flow_root=flow_root),
    dict(type='OccStressLoadStreamOcc3D'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'history_latent', 'history_flow_summary'], meta_keys=collect_meta_keys)
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
