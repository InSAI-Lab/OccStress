_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
        6,
    'input_size': (512, 1408),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

use_mask = True  # Set to False when not using the mask.

voxel_size = [0.05, 0.05, 0.05]

img_backbone_out_channel = 256
feature_channel = 32
lidar_out_channel = 32
img_channels = feature_channel
numC_Trans = img_channels + lidar_out_channel
num_classes = 18

multi_adj_frame_id_cfg = (1, 1 + 1, 1)
multi_adj_frame_id_cfg_lidar = (1, 7 + 1, 1)

model = dict(
    type='FusionOCC',
    lidar_in_channel=5,
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    lidar_out_channel=lidar_out_channel,
    align_after_view_transformation=True,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    fuse_loss_weight=0.1,
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 1024,
        out_channels=img_backbone_out_channel,
        # with_cp=False,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2),
    img_view_transformer=dict(
        type='CrossModalLSS',
        feature_channels=feature_channel,
        seg_num_classes=num_classes,
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=img_backbone_out_channel,
        mid_channels=128,
        depth_channels=88,
        is_train=True,  # set to False during inference
        out_channels=img_channels,
        sid=False,
        collapse_z=False,
        depthnet_cfg=dict(aspp_mid_channels=96, ),
        downsample=16),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=img_channels,
        with_cp=False,
        num_layer=[1, ],
        num_channels=[img_channels, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=img_channels * (len(range(*multi_adj_frame_id_cfg)) + 1) + lidar_out_channel,
        num_layer=[1, 2, 3],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    occ_encoder_neck=dict(type='LSSFPN3D',
                          in_channels=numC_Trans * 7,
                          out_channels=numC_Trans),
    out_dim=numC_Trans,
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=use_mask,
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
img_seg_dir = 'data/nuscenes/imgseg/samples'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageSeg',
        downsample=1,
        is_train=True,
        data_config=data_config,
        sequential=True,
        img_seg_dir=img_seg_dir
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='FuseAdjacentSweeps',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointsLidar2Ego'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='LoadAnnotationsAll',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', "points", 'sparse_depth', 'segs', 'voxel_semantics', 'mask_camera'])
]

test_pipeline = [
    dict(
        type='PrepareImageSeg',
        restore_upsample=8,  # Consistent with downsampling when generating image segmentation
        downsample=1,
        data_config=data_config,
        sequential=True,
        img_seg_dir=img_seg_dir
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='FuseAdjacentSweeps',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointsLidar2Ego'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='LoadAnnotationsAll',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'points', 'sparse_depth'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    use_mask=use_mask,
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='fusionocc',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    multi_adj_frame_id_cfg_lidar=multi_adj_frame_id_cfg_lidar,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'fusionocc-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'fusionocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config
)
for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

evaluation = dict(interval=1, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

# load_from = "../../ckpt/fusion_occ_mask.pth"
