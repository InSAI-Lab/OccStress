_base_ = ['./ii_scene_tokenizer_4f_v0_original.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_v0_original',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_vector_quantizer_v0_original',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_v0_occ_slot_invalidmask',
    ],
    allow_failed_imports=False,
)

flow_root = 'data/nuscenes/occ_flow_gt'

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_load_previous_frame_number = 0
test_load_previous_frame_number = 0
train_load_future_frame_number = 0
test_load_future_frame_number = 0

model = dict(
    type='IISceneTokenizerV0OccSlotInvalidMask',
    unified_motion_dynamic_transport='scatter',
    occ_slot_use_clean_cache=False,
    occ_slot_apply_latent_valid=True,
    occ_slot_latent_valid_threshold=0.0,
    profile_runtime=False,
    profile_runtime_interval=10,
    save_results=False,
)

occ_flow_keys = [
    'oracle_static_flow',
    'oracle_static_flow_valid',
    'oracle_static_flow_forward',
    'oracle_static_flow_valid_forward',
    'oracle_dynamic_residual_flow',
    'oracle_dynamic_residual_flow_valid',
    'oracle_dynamic_residual_flow_forward',
    'oracle_dynamic_residual_flow_valid_forward',
    'oracle_dynamic_mask',
    'oracle_dynamic_mask_source',
]

train_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True),
    dict(type='LoadOracleFlowGT', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Collect3D', keys=['voxel_semantics'] + occ_flow_keys),
]

test_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True, corruption_type='origin'),
    dict(type='LoadOracleFlowGT', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Collect3D', keys=['voxel_semantics'] + occ_flow_keys),
]

data = dict(
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    train=dict(
        load_previous_frame_number=train_load_previous_frame_number,
        load_previous_data=False,
        load_future_frame_number=train_load_future_frame_number,
        pipeline=train_pipeline,
    ),
    val=dict(
        load_previous_frame_number=test_load_previous_frame_number,
        load_previous_data=False,
        load_future_frame_number=test_load_future_frame_number,
        pipeline=test_pipeline,
    ),
    test=dict(
        load_previous_frame_number=test_load_previous_frame_number,
        load_previous_data=False,
        load_future_frame_number=test_load_future_frame_number,
        pipeline=test_pipeline,
    ),
)

log_config = dict(interval=10)
