_base_ = ['./ii_scene_tokenizer_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_exact_static_residual_bank',
    ],
    allow_failed_imports=False)

flow_root = 'data/nuscenes/occ_flow_gt'

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_load_previous_frame_number = 4
test_load_previous_frame_number = 4
train_load_future_frame_number = 0
test_load_future_frame_number = 0

model = dict(
    type='IISceneTokenizerCurrBevKalmanExactStaticResidualBank',
    unified_motion_dynamic_transport='scatter',
    save_results=False,
    kalman_cov_groups=8,
    kalman_init_cov=0.10,
    kalman_min_cov=1e-4,
    kalman_use_clean_cache=False,
    kalman_use_flow_mag=True,
    kalman_use_valid_ratio=True,
    kalman_use_dynamic_ratio=True,
    kalman_use_reliability=True,
    unified_motion_use_dynamic_gate=True,
    kalman_posterior_aux_loss_weight=0.25,
    kalman_posterior_dynamic_boost=1.0,
    kalman_bad_threshold=0.5,
    kalman_debug_decode_state=False,
)

freeze_dict = dict(
    encoder=True,
    decoder=True,
    class_embeds=True,
    vq=True,
)

unfreeze_patterns = [
    'kalman_q_head*',
    'kalman_r_head*',
    'kalman_log_p0',
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

train_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True, corruption_type='origin'),
    dict(
        type='RandomDisagreementStreamOcc3D',
        perturb_prob=0.5,
        free_class_idx=17,
        current_frame_only=True,
        block_mask_prob=0.7,
        block_mask_area_range=(0.04, 0.10),
        block_mask_num_range=(1, 1),
        semantic_swap_prob=0.3,
        semantic_swap_area_range=(0.02, 0.06),
    ),
    dict(type='LoadOracleFlowGTSequence', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(
        type='Collect3D',
        keys=[
            'voxel_semantics',
            'voxel_semantics_clean',
            'oracle_static_flow_seq',
            'oracle_static_flow_valid_seq',
            'oracle_dynamic_residual_flow_seq',
            'oracle_dynamic_residual_flow_valid_seq',
            'oracle_dynamic_mask_seq',
            'oracle_dynamic_residual_flow_forward_seq',
            'oracle_dynamic_residual_flow_valid_forward_seq',
            'oracle_dynamic_source_mask_seq',
            'frame_reliability_map',
        ],
    ),
]

test_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True, corruption_type='origin'),
    dict(type='LoadOracleFlowGTSequence', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(
        type='Collect3D',
        keys=[
            'voxel_semantics',
            'voxel_semantics_clean',
            'oracle_static_flow_seq',
            'oracle_static_flow_valid_seq',
            'oracle_dynamic_residual_flow_seq',
            'oracle_dynamic_residual_flow_valid_seq',
            'oracle_dynamic_mask_seq',
            'oracle_dynamic_residual_flow_forward_seq',
            'oracle_dynamic_residual_flow_valid_forward_seq',
            'oracle_dynamic_source_mask_seq',
        ],
    ),
]

data = dict(
    train=dict(
        load_previous_frame_number=train_load_previous_frame_number,
        load_previous_data=True,
        load_future_frame_number=train_load_future_frame_number,
        pipeline=train_pipeline,
    ),
    val=dict(
        load_previous_frame_number=test_load_previous_frame_number,
        load_previous_data=True,
        load_future_frame_number=test_load_future_frame_number,
        pipeline=test_pipeline,
    ),
    test=dict(
        load_previous_frame_number=test_load_previous_frame_number,
        load_previous_data=True,
        load_future_frame_number=test_load_future_frame_number,
        pipeline=test_pipeline,
    ),
)
