# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
from pathlib import Path

_base_ = ["./cvtocc_waymo_runtime.py"]

runtime_root = os.environ.get("OCCSTRESS_CODE_ROOT") or str(Path.cwd().resolve().parents[2])
cvt_runtime = os.environ.get("CVTOCC_RUNTIME_ROOT", f"{runtime_root}/runtime/CVT-Occ")
shared_root = os.environ.get("OCCSTRESS_DATA_ROOT", f"{runtime_root}/data/OccStress")
external_root = os.environ.get("OCCSTRESS_EXTERNAL_ROOT", f"{shared_root}/external")
waymo_occstress_root = os.environ.get("OCCSTRESS_WAYMO_ROOT", f"{external_root}/OccStress-Waymo")
data_root = os.environ.get("CVTOCC_SENSOR_ROOT", f"{waymo_occstress_root}/sensors/camera_2hz/")
ann_file = f"{cvt_runtime}/metadata/waymo_infos_val_2hz.pkl"
frame_index_root = os.environ.get(
    "CVTOCC_FRAME_INDEX_ROOT",
    f"{shared_root}/meta/OccStress-Waymo/upstream/camera_only/cvtocc/frame_index",
)
occ_gt_root = os.environ.get("CVTOCC_GT_ROOT", f"{waymo_occstress_root}/gts/")

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

model = dict(
    queue_length=7,
    sampled_queue_length=7,
    sample_num=[6, 5, 4, 3, 2, 1, 0],
    pts_bbox_head=dict(
        transformer=dict(
            queue_length=7,
            sampled_queue_length=7,
        ),
    ),
)

local_dataset = dict(
    data_root=data_root,
    ann_file=ann_file,
    pose_file=None,
    frame_index_root=frame_index_root,
    sensor_root=data_root,
    load_interval=1,
    pipeline=test_pipeline,
)

data = dict(
    workers_per_gpu=2,
    val=local_dataset,
    test=local_dataset,
)

evaluation = dict(pipeline=test_pipeline)
work_dir = f"{cvt_runtime}/outputs/local-2hz"
load_from = None

del os, Path
