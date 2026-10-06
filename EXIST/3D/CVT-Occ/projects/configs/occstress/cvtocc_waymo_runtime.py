# OccStress portable runtime base; see docs/source-imports.json.
import os

_base_ = ["../cvtocc/cvtocc_waymo.py"]

runtime_root = os.environ.get("CVTOCC_RUNTIME_ROOT", "runtime/CVT-Occ")
data_root = f"{runtime_root}/kitti-format/"
occ_data_root = f"{runtime_root}/occ3d"
ann_file = f"{occ_data_root}/waymo_infos_val.pkl"
pose_file = f"{occ_data_root}/cam_infos_vali.pkl"
occ_gt_root = f"{occ_data_root}/voxel04/validation-data/"

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

test_pipeline = [
    dict(
        type="MyLoadMultiViewImageFromFiles",
        to_float32=True,
        img_scale=(1280, 1920),
        decode_backend="opencv",
    ),
    dict(
        type="LoadOccGTFromFileWaymo",
        data_root=occ_gt_root,
        use_larger=True,
        crop_x=False,
        use_infov_mask=True,
        use_camera_mask=True,
        use_lidar_mask=False,
        FREE_LABEL=23,
        num_classes=16,
    ),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D",
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type="CustomCollect3D",
                keys=["img", "voxel_semantics", "valid_mask"],
                meta_keys=[
                    "filename",
                    "pts_filename",
                    "sample_idx",
                    "scene_token",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "lidar2img",
                    "sensor2ego",
                    "cam_intrinsic",
                    "ego2global",
                ],
            ),
        ],
    ),
]

data = dict(
    workers_per_gpu=2,
    val=dict(
        data_root=data_root,
        ann_file=ann_file,
        pose_file=pose_file,
        pipeline=test_pipeline,
    ),
    test=dict(
        data_root=data_root,
        ann_file=ann_file,
        pose_file=pose_file,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(pipeline=test_pipeline)
work_dir = f"{runtime_root}/outputs/cvtocc-clean"
load_from = None

del os
