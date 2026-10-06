_base_ = ['./ii_scene_tokenizer_4f_currbev_kalman_prior_slot_frozen.py']

model = dict(
    kalman_posterior_aux_loss_weight=1.0,
)

optimizer = dict(
    lr=5e-5,
    paramwise_cfg=dict(
        custom_keys=dict(
            kalman_q_head=dict(lr_mult=1.0, decay_mult=1.0),
            kalman_r_head=dict(lr_mult=1.0, decay_mult=1.0),
            kalman_log_p0=dict(lr_mult=1.0, decay_mult=0.0),
        )))
