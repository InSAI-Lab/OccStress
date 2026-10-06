_base_ = ['./ii_generate_world.py']

token_root = 'data/nuscenes/save_dir_dynstatic_round4/token_4f'
tokenizer_ckpt = 'work_dirs/history_fusion_dynstatic_round4/latest.pth'

model = dict(
    vqvae_checkpoint=tokenizer_ckpt,
    vqvae=dict(
        save_results=False,
        save_root_override='data/nuscenes/save_dir_dynstatic_round4',
        vq=dict(
            type='IntraInterVectorQuantizerDynStaticGateFusion',
            gate_hidden=128,
            dynamic_gate_scale=0.25,
        ),
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
