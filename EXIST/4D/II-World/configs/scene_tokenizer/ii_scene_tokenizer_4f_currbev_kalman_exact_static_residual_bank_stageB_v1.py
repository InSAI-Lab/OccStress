_base_ = ['./ii_scene_tokenizer_4f_currbev_kalman_exact_static_residual_bank_stageA_v1.py']

# Stage B: once Stage A has adapted VQ+decoder to the fixed slot path,
# optionally unfreeze the encoder and continue low-LR finetuning end-to-end.
# This is only needed if Stage A clean/corr curves plateau below expectations.

freeze_dict = dict(
    encoder=False,
    decoder=False,
    class_embeds=True,
    vq=False,
)

optimizer = dict(
    type='AdamW',
    lr=5e-6,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            kalman_q_head=dict(lr_mult=5.0, decay_mult=1.0),
            kalman_r_head=dict(lr_mult=5.0, decay_mult=1.0),
            kalman_log_p0=dict(lr_mult=5.0, decay_mult=0.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=24000)
checkpoint_config = dict(interval=4000)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[18000],
)
