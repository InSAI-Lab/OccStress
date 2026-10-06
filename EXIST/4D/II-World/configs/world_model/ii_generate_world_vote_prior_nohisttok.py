_base_ = ['./ii_generate_world_baseline_nohisttok.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_vote_prior',
        'mmdet3d.models.ii_world.world_model.ii_world_vote_fusion',
    ],
    allow_failed_imports=False,
)

token_root = 'data/nuscenes/save_dir_baseline_nohisttok/token_4f'
vote_token_root = 'data/nuscenes/save_dir_vote_prior_nohisttok/token_4f'

model = dict(
    type='II_WorldVoteFusion',
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
        type='LoadStreamLatentVoteToken',
        data_path=token_root,
        vote_data_path=vote_token_root,
        load_vote_confidence=True,
    ),
    dict(type='Collect3D', keys=['latent', 'vote_latent', 'vote_confidence'])
]

test_pipeline = [
    dict(
        type='LoadStreamLatentVoteToken',
        data_path=token_root,
        vote_data_path=vote_token_root,
        load_vote_confidence=True,
    ),
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'vote_latent', 'vote_confidence'])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

load_from = 'work_dirs/ii_generate_world_baseline_nohisttok/iter_95952_ema.pth'
