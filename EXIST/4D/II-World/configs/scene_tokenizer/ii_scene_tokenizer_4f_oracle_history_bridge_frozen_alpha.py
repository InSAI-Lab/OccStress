_base_ = ['./ii_scene_tokenizer_4f_oracle_history_bridge.py']

freeze_dict = dict(
    encoder=True,
    decoder=True,
    class_embeds=True,
    vq=True,
)

unfreeze_patterns = [
    'oracle_bridge_alpha_head*',
]

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            oracle_bridge_alpha_head=dict(lr_mult=1.0, decay_mult=1.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(interval=2000)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[3200],
)
