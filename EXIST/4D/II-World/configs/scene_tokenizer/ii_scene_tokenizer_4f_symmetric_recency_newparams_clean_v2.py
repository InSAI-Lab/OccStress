_base_ = ['./ii_scene_tokenizer_4f_symmetric_recency_newparams_clean_v1.py']

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            **{
                'vq.recency_logits': dict(decay_mult=0.0),
            }))
)

runner = dict(type='IterBasedRunner', max_iters=500)
checkpoint_config = dict(interval=500)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[400],
)
