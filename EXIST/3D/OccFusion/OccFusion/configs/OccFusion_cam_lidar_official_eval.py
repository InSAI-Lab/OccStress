_base_ = ['./OccFusion.py']

load_from = None
backend_args = None

use_lidar = True
use_radar = False
use_occ3d = False
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]

model = dict(
    use_occ3d=use_occ3d,
    use_lidar=use_lidar,
    use_radar=use_radar,
    view_transformer=dict(
        enable_fix=True,
        use_lidar=use_lidar,
        use_radar=use_radar),
)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOccupancy'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    dict(
        type='MultiViewWrapper',
        transforms=dict(type='PhotoMetricDistortion3D')),
    dict(type='SegLabelMapping'),
    dict(
        type='Custom3DPack',
        keys=['img', 'points', 'pts_semantic_mask', 'occ_200'],
        meta_keys=['lidar2img', 'ego2img'])
]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOccupancy'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    dict(type='SegLabelMapping'),
    dict(
        type='Custom3DPack',
        keys=['img', 'points', 'pts_semantic_mask', 'occ_200'],
        meta_keys=['lidar2img', 'ego2img'])
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        ann_file='nuscenes_infos_occfusion_train.pkl',
        pipeline=train_pipeline,
        use_occ3d=use_occ3d))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        ann_file='nuscenes_infos_occfusion_val.pkl',
        pipeline=val_pipeline,
        use_occ3d=use_occ3d))

test_dataloader = val_dataloader
