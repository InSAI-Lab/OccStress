_base_ = ['./ii_generate_world.py']

from configs.world_model.stage2_token_roots import baseline_stage2_token_root

train_token_root = baseline_stage2_token_root('train')
test_token_root = baseline_stage2_token_root('test')

model = dict(
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
)

train_pipeline = [
    dict(type='LoadStreamLatentToken', data_path=train_token_root),
    dict(type='Collect3D', keys=['latent'])
]

test_pipeline = [
    dict(type='LoadStreamLatentToken', data_path=test_token_root),
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent'])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
