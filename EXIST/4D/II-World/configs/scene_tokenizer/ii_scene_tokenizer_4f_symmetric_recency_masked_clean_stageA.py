_base_ = ['./ii_scene_tokenizer_4f_symmetric_recency_fusion.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.nuscenes_world_dataset',
        'mmdet3d.datasets.pipelines.loading',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_vector_quantizer_symmetric_recency_fusion',
    ],
    allow_failed_imports=False)

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

freeze_dict = dict(
    encoder=True,
    decoder=True,
)

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=1e-2)

train_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True),
    dict(
        type='RandomMaskStreamOcc3D',
        masked_pass_prob=0.3,
        free_class_idx=17,
        current_frame_only=True,
        block_mask_prob=1.0,
        block_mask_area_range=(0.10, 0.20),
        block_mask_num_range=(1, 2),
        sector_mask_prob=0.15,
        sector_width_range=(0.35, 0.55),
    ),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Collect3D', keys=['voxel_semantics']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
    ),
)

# Stage-A style short warm start. The submission script overrides these again,
# but keeping them here makes ad-hoc local runs safer and more reproducible.
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(interval=2000)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[3200],
)

