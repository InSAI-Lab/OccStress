_base_ = ['./ii_scene_tokenizer_4f_oracle_history_bridge.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_recovery_mlp',
    ],
    allow_failed_imports=False)

model = dict(
    type='IISceneTokenizerCurrBevRecoveryMLP',
    save_results=False,
    recovery_hidden=256,
    recovery_use_clean_cache=True,
    recovery_use_flow_mag=True,
    recovery_use_valid_ratio=True,
    recovery_use_dynamic_ratio=True,
    recovery_use_reliability=True,
    recovery_latent_repair_loss_weight=1.0,
    recovery_latent_keep_loss_weight=0.25,
    recovery_dynamic_boost=1.0,
    recovery_bad_threshold=0.5,
)

freeze_dict = dict(
    encoder=True,
    decoder=True,
    class_embeds=True,
    vq=True,
)

unfreeze_patterns = [
    'recovery_mlp*',
]

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            recovery_mlp=dict(lr_mult=1.0, decay_mult=1.0),
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
