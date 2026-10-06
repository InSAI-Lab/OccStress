_base_ = ['./ii_scene_tokenizer_4f_currbev_kalman_exact_static_residual_bank_frozen_v1.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_currbev_kalman_exact_static_residual_bank_cached_occ_slots',
    ],
    allow_failed_imports=False)

model = dict(
    type='IISceneTokenizerCurrBevKalmanExactStaticResidualBankCachedOccSlots',
    freeze_kalman_modules=True,
    unified_motion_dynamic_transport='scatter',
    occ_slot_use_clean_cache=False,
    unified_motion_use_dynamic_gate=False,
)

flow_root = 'data/nuscenes/occ_flow_gt'

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

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
            'oracle_static_flow_forward_seq',
            'oracle_static_flow_valid_forward_seq',
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
    val=dict(
        load_previous_frame_number=4,
        load_previous_data=True,
        load_future_frame_number=0,
        pipeline=test_pipeline,
    ),
    test=dict(
        load_previous_frame_number=4,
        load_previous_data=True,
        load_future_frame_number=0,
        pipeline=test_pipeline,
    ),
)
