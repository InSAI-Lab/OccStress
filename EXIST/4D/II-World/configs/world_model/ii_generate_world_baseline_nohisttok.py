_base_ = ['./ii_generate_world.py']

token_root = 'data/nuscenes/save_dir_baseline_nohisttok/token_4f'
tokenizer_ckpt = 'work_dirs/ii_scene_tokenizer_4f/iter_47976_ema.pth'

model = dict(
    vqvae_checkpoint=tokenizer_ckpt,
    vqvae=dict(
        save_results=False,
        save_root_override='data/nuscenes/save_dir_baseline_nohisttok',
        disable_history_at_test=True,
    ),
)

train_pipeline = [
    dict(type='LoadStreamLatentToken', data_path=token_root),
    dict(type='Collect3D', keys=['latent'])
]

test_pipeline = [
    dict(type='LoadStreamLatentToken', data_path=token_root),
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent'])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
