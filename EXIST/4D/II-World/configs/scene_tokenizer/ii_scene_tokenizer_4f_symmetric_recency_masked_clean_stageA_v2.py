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
    flip_dy_ratio=0.5,
)

freeze_dict = dict(
    encoder=True,
    decoder=True,
    class_embeds=True,
)

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            **{
                'vq.score_head': dict(lr_mult=10.0),
                'vq.context_proj': dict(lr_mult=10.0),
                'vq.recency_logits': dict(lr_mult=10.0, decay_mult=0.0),
            }))
)

train_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True),
    dict(
        type='RandomMaskStreamOcc3D',
        masked_pass_prob=0.10,
        free_class_idx=17,
        current_frame_only=True,
        block_mask_prob=1.0,
        block_mask_area_range=(0.04, 0.08),
        block_mask_num_range=(1, 1),
        sector_mask_prob=0.0,
        sector_width_range=(0.30, 0.45),
    ),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Collect3D', keys=['voxel_semantics']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
)

runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(interval=1000)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[1600],
)
custom_hooks = []
