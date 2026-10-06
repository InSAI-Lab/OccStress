_base_ = ['./ii_generate_world_stage2_current_scale_aug_v0.py']

model = dict(
    scale_aug=dict(
        prob=0.4,
        up_prob=0.45,
        down_range=(0.93, 0.97),
        up_range=(1.02, 1.06),
    ),
)

lr = 2e-5
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
