_base_ = ['./ii_generate_world.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_history_selective_gate',
    ],
    allow_failed_imports=False,
)

from configs.world_model.stage2_token_roots import baseline_stage2_token_root

flow_root = 'data/nuscenes/occ_flow_gt'
train_load_previous_frame_number = 4
test_load_previous_frame_number = 4

model = dict(
    type='II_WorldHistorySelectiveGate',
    previous_frame_exist=True,
    previous_frame=4,
    test_previous_frame=4,
    vqvae_checkpoint='ckpts/ii_scene_tokenizer_4f.pth',
    history_gate=dict(
        enabled=True,
        history_frame_number=4,
        use_flow_summary=True,
        flow_summary_dim=5,
        current_corr_ratio=0.5,
        history_corr_ratio=0.5,
        slot_corr_prob=0.5,
        seed=3407,
        history_strength_init=0.0,
        current_strength_init=0.0,
        history_gate_bias=0.0,
        current_gate_bias=-4.0,
        current_identity_weight=0.05,
        current_denoise_weight=0.1,
        current_gate_supervision_weight=0.02,
        history_gate_supervision_weight=0.02,
        strength_reg_weight=0.001,
        clean_consistency_weight=0.0,
        history_gate_clean_target=0.85,
        history_gate_corrupt_target=0.0,
        current_gate_clean_target=0.0,
        current_gate_corrupt_target=0.85,
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
        load_flow_summary=True,
        flow_root=flow_root,
    ),
    dict(type='Collect3D', keys=['latent', 'history_latent', 'history_flow_summary'])
]

test_pipeline = [
    dict(
        type='LoadStreamLatentHistoryToken',
        data_path=baseline_stage2_token_root('test'),
        history_frame_number=4,
        load_flow_summary=True,
        flow_root=flow_root,
    ),
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent', 'history_latent', 'history_flow_summary'])
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
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

lr = 1e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
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
