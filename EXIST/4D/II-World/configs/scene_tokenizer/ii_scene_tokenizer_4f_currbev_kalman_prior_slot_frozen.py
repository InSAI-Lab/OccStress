_base_ = ['./ii_scene_tokenizer_4f_oracle_history_bridge.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_prior_slot',
    ],
    allow_failed_imports=False)

model = dict(
    type='IISceneTokenizerCurrBevKalmanPriorSlot',
    save_results=False,
    kalman_slot=0,
    kalman_cov_groups=8,
    kalman_init_cov=0.10,
    kalman_min_cov=1e-4,
    kalman_use_clean_cache=True,
    kalman_use_flow_mag=True,
    kalman_use_valid_ratio=True,
    kalman_use_dynamic_ratio=True,
    kalman_use_reliability=True,
    kalman_debug_decode_state=False,
)

freeze_dict = dict(
    encoder=True,
    decoder=True,
    class_embeds=True,
    vq=True,
)

unfreeze_patterns = [
    'kalman_*',
]

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            kalman_q_head=dict(lr_mult=1.0, decay_mult=1.0),
            kalman_r_head=dict(lr_mult=1.0, decay_mult=1.0),
            kalman_log_p0=dict(lr_mult=1.0, decay_mult=0.0),
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
