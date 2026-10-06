_base_ = ['./ii_generate_world.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.ii_world.world_model.ii_world_current_scale_aug',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='II_WorldCurrentScaleAug',
    scale_aug=dict(
        enabled=True,
        prob=0.6,
        up_prob=0.45,
        down_range=(0.90, 0.96),
        up_range=(1.03, 1.08),
        apply_at_test=False,
    ),
)

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

lr = 5e-5
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
