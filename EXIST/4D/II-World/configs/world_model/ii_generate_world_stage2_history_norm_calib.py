_base_ = ['./ii_generate_world.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_history_norm_calib',
    ],
    allow_failed_imports=False,
)

from configs.world_model.stage2_token_roots import baseline_stage2_token_root

train_load_previous_frame_number = 4
test_load_previous_frame_number = 4

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
        type='LoadStreamLatentHistoryToken',
        data_path=baseline_stage2_token_root('train'),
        history_frame_number=4,
    ),
    dict(type='Collect3D', keys=['latent', 'history_latent'])
]

test_pipeline = [
    dict(
        type='LoadStreamLatentHistoryToken',
        data_path=baseline_stage2_token_root('test'),
        history_frame_number=4,
    ),
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'history_latent'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        load_previous_frame_number=train_load_previous_frame_number,
    ),
    val=dict(
        pipeline=test_pipeline,
        load_previous_frame_number=test_load_previous_frame_number,
    ),
    test=dict(
        pipeline=test_pipeline,
        load_previous_frame_number=test_load_previous_frame_number,
    ),
)
