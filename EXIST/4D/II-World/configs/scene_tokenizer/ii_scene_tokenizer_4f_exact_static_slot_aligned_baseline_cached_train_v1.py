_base_ = ['./ii_scene_tokenizer_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_exact_static_residual_bank_cached',
    ],
    allow_failed_imports=False)

flow_root = 'data/nuscenes/occ_flow_gt'

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_load_previous_frame_number = 1
test_load_previous_frame_number = 1
train_load_future_frame_number = 0
test_load_future_frame_number = 0

model = dict(
    type='IISceneTokenizerCurrBevKalmanExactStaticResidualBankCached',
    freeze_kalman_modules=True,
    unified_motion_dynamic_transport='scatter',
    save_results=False,
    kalman_cov_groups=8,
    kalman_init_cov=0.10,
    kalman_min_cov=1e-4,
    kalman_use_clean_cache=False,
    kalman_use_flow_mag=True,
    kalman_use_valid_ratio=True,
    kalman_use_dynamic_ratio=True,
    kalman_use_reliability=False,
    unified_motion_use_dynamic_gate=False,
    kalman_posterior_aux_loss_weight=0.0,
    kalman_posterior_dynamic_boost=0.0,
    kalman_bad_threshold=0.5,
    kalman_debug_decode_state=False,
)

freeze_dict = dict()
unfreeze_patterns = []

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys=dict(
            kalman_q_head=dict(lr_mult=0.0, decay_mult=0.0),
            kalman_r_head=dict(lr_mult=0.0, decay_mult=0.0),
            kalman_log_p0=dict(lr_mult=0.0, decay_mult=0.0),
        )))

train_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True),
    dict(type='LoadOracleFlowGT', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(
        type='Collect3D',
        keys=[
            'voxel_semantics',
            'oracle_static_flow',
            'oracle_static_flow_valid',
            'oracle_dynamic_residual_flow',
            'oracle_dynamic_residual_flow_valid',
            'oracle_dynamic_residual_flow_forward',
            'oracle_dynamic_residual_flow_valid_forward',
            'oracle_dynamic_mask',
            'oracle_dynamic_mask_source',
        ],
    ),
]

test_pipeline = [
    dict(type='LoadStreamOcc3D', corruption_type='origin', corruption_path='data/val_random_corruptions_v3.pkl'),
    dict(type='LoadOracleFlowGT', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(
        type='Collect3D',
        keys=[
            'voxel_semantics',
            'oracle_static_flow',
            'oracle_static_flow_valid',
            'oracle_dynamic_residual_flow',
            'oracle_dynamic_residual_flow_valid',
            'oracle_dynamic_residual_flow_forward',
            'oracle_dynamic_residual_flow_valid_forward',
            'oracle_dynamic_mask',
            'oracle_dynamic_mask_source',
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
