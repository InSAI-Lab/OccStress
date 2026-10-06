# OccStress adapter/portability modifications; see docs/source-imports.json.
import os

_base_ = [
    "../effocc_fusion_r18_data_scales/flashocc_fusion_r18_base_100%_seqs.py"
]

native_classes = [
    "undefined", "car", "bicycle", "motorcycle", "pedestrian",
    "traffic_cone", "vegetation", "road", "terrain", "building", "free",
]
data_config = dict(
    cams=["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK"],
    Ncams=4,
    input_size=(256, 704),
    src_size=(800, 800),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.25, 0.25),
    resize_test=0.0,
)
grid_config = dict(
    x=[-40.0, 40.0, 0.4],
    y=[-40.0, 40.0, 0.4],
    z=[-1.0, 5.4, 6.4],
    depth=[1.0, 45.0, 0.5],
)
file_client_args = dict(backend="disk")
bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

model = dict(
    img_backbone=dict(pretrained=None),
    img_view_transformer=dict(
        input_size=data_config["input_size"],
        grid_config=grid_config,
    ),
    occ_head=dict(num_classes=11),
)

train_pipeline = [
    dict(type="PrepareImageInputs", is_train=True,
         data_config=data_config, sequential=False),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5,
         use_dim=5, file_client_args=file_client_args),
    dict(type="LoadOccGTFromFile"),
    dict(type="BEVAug", bda_aug_conf=bda_aug_conf, classes=native_classes),
    dict(type="PointToMultiViewDepthFusion", downsample=1,
         grid_config=grid_config),
    dict(type="DefaultFormatBundle3D", class_names=native_classes),
    dict(type="Collect3D", keys=[
        "points", "img_inputs", "gt_depth", "voxel_semantics",
        "mask_lidar", "mask_camera",
    ]),
]
test_pipeline = [
    dict(type="PrepareImageInputs", data_config=data_config, sequential=False),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5,
         use_dim=5, file_client_args=file_client_args),
    dict(type="BEVAug", bda_aug_conf=bda_aug_conf,
         classes=native_classes, is_train=False),
    dict(type="PointToMultiViewDepthFusion", downsample=1,
         grid_config=grid_config),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="DefaultFormatBundle3D", class_names=native_classes,
                 with_label=False),
            dict(type="Collect3D", keys=["points", "img_inputs", "gt_depth"]),
        ],
    ),
]

local_root = os.environ.get(
    "UNIOCC_CARLA_LOCAL_ROOT", "data/UniOcc-CARLA"
)
train_info = os.path.join(
    local_root, "prepared/Carla-10Hz-train/uniocc_carla_train_infos.pkl"
)
val_info = os.path.join(
    local_root, "prepared/Carla-2Hz-val/uniocc_carla_val_infos.pkl"
)
dataset_common = dict(
    type="UniOccCarlaDataset",
    data_root="/",
    classes=native_classes,
    modality=dict(use_lidar=True, use_camera=True, use_radar=False,
                  use_map=False, use_external=False),
    box_type_3d="LiDAR",
    filter_empty_gt=False,
)
data = dict(
    _delete_=True,
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        **dataset_common,
        ann_file=train_info,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        **dataset_common,
        ann_file=val_info,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        **dataset_common,
        ann_file=val_info,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[18, 22],
)
runner = dict(type="EpochBasedRunner", max_epochs=24)
custom_hooks = [
    dict(type="MEGVIIEMAHook", init_updates=0, save_interval=4,
         max_keep_ckpts=3, priority="NORMAL"),
]
checkpoint_config = dict(interval=4, max_keep_ckpts=3)
evaluation = dict(interval=4, start=12, pipeline=test_pipeline)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
load_from = os.environ.get("EFFOCC_CARLA_INIT") or None
fp16 = dict(loss_scale="dynamic")
