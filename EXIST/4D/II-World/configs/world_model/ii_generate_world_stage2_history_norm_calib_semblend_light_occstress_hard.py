_base_ = ['./ii_generate_world_stage2_history_norm_calib_occstress_hard.py']

model = dict(
    norm_calib=dict(
        deadband=0.03,
        min_scale=0.85,
        max_scale=1.15,
        down_mode='blend',
        down_blend_strength=0.05,
    ),
)
