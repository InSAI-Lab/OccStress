_base_ = ['./ii_scene_tokenizer_4f.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pipelines.loading_oracle_flow',
        'mmdet3d.models.ii_world.scene_tokenizer.ii_tokenizer_oracle_history_bridge',
    ],
    allow_failed_imports=False)

flow_root = 'data/nuscenes/occ_flow_gt'

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_load_previous_frame_number = 1
test_load_previous_frame_number = 1
train_load_future_frame_number = 0
test_load_future_frame_number = 0

model = dict(
    type='IISceneTokenizerOracleHistoryBridge',
    save_results=False,
    oracle_bridge_use_clean_cache=True,
    oracle_bridge_use_flow_mag=True,
    oracle_bridge_use_valid_ratio=True,
    oracle_bridge_use_dynamic_ratio=True,
    oracle_bridge_use_reliability=True,
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
    dict(type='LoadOracleFlowGT', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(
        type='Collect3D',
        keys=[
            'voxel_semantics',
            'oracle_flow',
            'oracle_flow_valid',
            'oracle_dynamic_mask',
            'frame_reliability_map',
        ],
    ),
]

test_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True, corruption_type='origin'),
    dict(type='LoadOracleFlowGT', flow_root=flow_root),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(
        type='Collect3D',
        keys=[
            'voxel_semantics',
            'oracle_flow',
            'oracle_flow_valid',
            'oracle_dynamic_mask',
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
