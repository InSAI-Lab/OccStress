_base_ = ['../scene_tokenizer/ii_scene_tokenizer_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_stage3',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_stage3_pred_only',
        'mmdet3d.models.losses.focal_loss_torch',
    ],
    allow_failed_imports=False,
)

pred_latent_root = 'data/nuscenes/stage3_pred_latents/baseline_stage2_0p5s_fp32'

model = dict(
    type='IISceneTokenizerStage3PredOnly',
    target_frame_index=1,
    target_valid_index=0,
    freeze_encoder=True,
    freeze_vq=True,
    freeze_class_embeds=True,
    focal_loss=dict(type='CustomFocalLossTorch', loss_weight=10.0),
)

train_load_future_frame_number = 1
test_load_future_frame_number = 1

train_pipeline = [
    dict(type='LoadStage3PredLatent', data_path=pred_latent_root, to_float32=True),
    dict(type='LoadStreamOcc3D', to_long=True, corruption_type='origin'),
    dict(type='Collect3D', keys=['pred_latent', 'voxel_semantics', 'valid_frame']),
]

test_pipeline = [
    dict(type='LoadStage3PredLatent', data_path=pred_latent_root, to_float32=True),
    dict(type='LoadStreamOcc3D', to_long=True, corruption_type='origin'),
    dict(type='Collect3D', keys=['pred_latent', 'voxel_semantics', 'valid_frame']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
        load_future_frame_number=train_load_future_frame_number,
        load_previous_frame_number=0,
        load_previous_data=False,
    ),
    val=dict(
        pipeline=test_pipeline,
        load_future_frame_number=test_load_future_frame_number,
        load_previous_frame_number=0,
        load_previous_data=False,
    ),
    test=dict(
        pipeline=test_pipeline,
        load_future_frame_number=test_load_future_frame_number,
        load_previous_frame_number=0,
        load_previous_data=False,
    ),
)

# Decoder-only finetune from the locked baseline tokenizer checkpoint.
load_from = 'ckpts/ii_scene_tokenizer_4f.pth'
resume_from = None

lr = 5e-5
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[3500],
)
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(interval=1000)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=1000,
    )
]

work_dir = 'work_dirs/ii_stage3_decoder_pred_only_0p5s_baseline_fp32'
