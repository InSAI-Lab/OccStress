_base_ = ['./ii_generate_world_stage2_history_selective_gate_v1.py']

from configs.world_model.stage2_token_roots import baseline_stage2_token_root

flow_root = 'data/nuscenes/occ_flow_gt'
train_token_root = baseline_stage2_token_root('train')
test_token_root = baseline_stage2_token_root('test')

train_pipeline = [
    dict(
        type='LoadStreamLatentHistoryToken',
        data_path=train_token_root,
        history_frame_number=4,
        load_flow_summary=True,
        flow_root=flow_root,
    ),
    dict(type='Collect3D', keys=['latent', 'history_latent', 'history_flow_summary'])
]

test_pipeline = [
    dict(
        type='LoadStreamLatentHistoryToken',
        data_path=test_token_root,
        history_frame_number=4,
        load_flow_summary=True,
        flow_root=flow_root,
    ),
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'history_latent', 'history_flow_summary'])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
