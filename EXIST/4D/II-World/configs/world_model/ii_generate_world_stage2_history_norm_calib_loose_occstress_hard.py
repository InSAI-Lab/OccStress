_base_ = ['./ii_generate_world_stage2_history_norm_calib_occstress_hard.py']

model = dict(
    norm_calib=dict(
        deadband=0.02,
        min_scale=0.88,
        max_scale=1.18,
    ),
)
