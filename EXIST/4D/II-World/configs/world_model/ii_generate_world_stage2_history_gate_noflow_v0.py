_base_ = ['./ii_generate_world.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_history_gate',
    ],
    allow_failed_imports=False,
)

from configs.world_model.stage2_token_roots import baseline_stage2_token_root

train_load_previous_frame_number = 4
test_load_previous_frame_number = 4

model = dict(
    type='II_WorldHistoryGate',
    previous_frame_exist=True,
    previous_frame=4,
    test_previous_frame=4,
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    history_gate=dict(
        enabled=True,
        history_frame_number=4,
        use_flow_summary=False,
        corr_ratio=0.5,
        slot_corr_prob=0.5,
        seed=3407,
        gate_bias=-4.0,
        consistency_weight=0.05,
        gate_supervision_weight=0.02,
        gate_clean_target=0.7,
        gate_corrupt_target=0.0,
        noise_std=0.03,
        channel_drop_prob=0.08,
        block_drop_prob=0.65,
        min_block=4,
        max_block=14,
        spatial_shift_prob=0.35,
        max_shift=4,
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

load_from = 'work_dirs/ii_generate_world_baseline_original_stage2_pytorchms_8gpu_job3993720/iter_95952_ema.pth'
freeze_dict = dict(
    pose_encoder=True,
    transformer=True,
)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(interval=2000)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[16000],
)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=2000,
        priority='NORMAL',
        interval=2000,
    ),
    dict(
        type='ScheduledSampling',
        total_iter=20000,
        loss_iter=None,
    ),
]
