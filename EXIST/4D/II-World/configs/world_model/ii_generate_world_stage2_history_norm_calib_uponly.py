_base_ = ['./ii_generate_world_stage2_history_norm_calib.py']

model = dict(
    norm_calib=dict(
        deadband=0.03,
        min_scale=1.0,
        max_scale=1.15,
        down_mode='none',
    ),
)
