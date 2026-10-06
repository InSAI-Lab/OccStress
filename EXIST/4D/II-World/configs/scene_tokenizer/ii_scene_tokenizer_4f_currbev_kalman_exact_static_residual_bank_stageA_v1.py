_base_ = ['./ii_scene_tokenizer_4f_currbev_kalman_exact_static_residual_bank_frozen_v1.py']

# Stage A: adapt the readout path to the fixed slot geometry.
# Keep the encoder latent manifold stable, and finetune:
#   - VQ temporal recovery
#   - decoder
#   - Kalman auxiliary heads
# This is the most direct way to "repair decoder/VQ" after fixing the slot warp.

freeze_dict = dict(
    encoder=True,
    decoder=False,
    class_embeds=True,
    vq=False,
)

# No pattern-based unfreeze is needed in Stage A because decoder/VQ stay trainable.
unfreeze_patterns = []

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            kalman_q_head=dict(lr_mult=5.0, decay_mult=1.0),
            kalman_r_head=dict(lr_mult=5.0, decay_mult=1.0),
            kalman_log_p0=dict(lr_mult=5.0, decay_mult=0.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(interval=2000)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[6400],
)
