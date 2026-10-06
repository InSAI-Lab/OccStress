_base_ = ['./ii_generate_world.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_robust_finetune',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='II_WorldRobustFinetune',
    robust_finetune=dict(
        enabled=True,
        corr_ratio=0.5,
        seed=3407,
        gate_bias=-4.0,
        identity_weight=0.05,
        denoise_weight=0.1,
        gate_weight=0.0005,
        noise_std=0.03,
        channel_drop_prob=0.08,
        block_drop_prob=0.65,
        min_block=4,
        max_block=14,
        spatial_shift_prob=0.35,
        max_shift=4,
    ),
)

# Full-stage2 robust finetune: initialize from the locked baseline but let the
# prediction model adapt to clean/corrupted current-token inputs.
load_from = 'work_dirs/ii_generate_world_baseline_original_stage2_pytorchms_8gpu_job3993720/iter_95952_ema.pth'

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
