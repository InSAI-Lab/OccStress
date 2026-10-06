# OccStress adapter/portability modifications; see docs/source-imports.json.
_base_ = []

import os
import sys

ROBUSTOCC_ROOT = os.path.abspath(
    os.environ.get('OCCSTRESS_CODE_ROOT', '../../..'))
if ROBUSTOCC_ROOT not in sys.path:
    sys.path.insert(0, ROBUSTOCC_ROOT)
from scripts.occstress_layout import resolve_manual_protocol_path

eval_with_pose = True
start_frame = 0
mid_frame = 5
end_frame = 11

plan_return_last = True
eval_length = end_frame - mid_frame

grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 50
return_len_ = end_frame
return_len_train = end_frame
num_frames_ = 15
load_from = os.environ.get('OCCWORLD_CKPT', 'out/occworld/latest.pth')
port = int(os.environ.get('OCCSTRESS_OCCWORLD_PORT', '25120'))
revise_ckpt = 3
eval_every_epochs = 1
save_every_epochs = 1
multisteplr = False
multisteplr_config = dict(
    decay_rate=0.1,
    decay_t=[43500],
    t_in_epochs=False,
    warmup_lr_init=1e-6,
    warmup_t=warmup_iters,
)
freeze_dict = dict(
    vae=True,
    transformer=False,
    pose_encoder=False,
    pose_decoder=False,
)
optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
)

data_path = os.environ.get('OCCSTRESS_OCCWORLD_DATA_PATH', os.path.join(ROBUSTOCC_ROOT, 'data/nuscenes'))
carla_protocol_path = os.environ.get('OCCSTRESS_CARLA_PROTOCOL')
waymo_protocol_path = os.environ.get('OCCSTRESS_WAYMO_PROTOCOL')
occupancy_protocol_path = waymo_protocol_path or carla_protocol_path
protocol_name = os.environ.get('OCCSTRESS_OCCWORLD_PROTOCOL_NAME', 'clean_H4_F6_val_backbone')
protocol_path = occupancy_protocol_path or os.environ.get(
    'OCCSTRESS_OCCWORLD_PROTOCOL_PATH')
future_aligned = os.environ.get(
    'OCCSTRESS_OCCWORLD_FUTURE_ALIGNED', '1').lower() not in ('0', 'false', 'no')
if occupancy_protocol_path:
    future_aligned = True
max_samples_value = os.environ.get('OCCSTRESS_OCCWORLD_MAX_SAMPLES')
max_samples = int(max_samples_value) if max_samples_value else None
if not protocol_path:
    protocol_path = str(resolve_manual_protocol_path(protocol_name, root=ROBUSTOCC_ROOT))
imageset = os.environ.get(
    'OCCSTRESS_OCCWORLD_IMAGESET',
    'data/nuscenes_infos_val_temporal_v3_scene.pkl')
if occupancy_protocol_path:
    imageset = os.environ.get(
        'OCCSTRESS_WAYMO_BASE_INFO', os.environ.get('OCCSTRESS_CARLA_BASE_INFO'))
dataset_type = (
    'OccStressWaymoSceneDataset' if waymo_protocol_path else
    'OccStressCARLASceneDataset' if carla_protocol_path else
    'OccStressNuScenesSceneDatasetLidarTraverse')
occupancy_only = bool(occupancy_protocol_path)

train_dataset_config = dict(
    type=dataset_type,
    data_path=data_path,
    return_len=return_len_train if future_aligned else return_len_train + 1,
    offset=0,
    imageset=imageset,
    protocol_path=protocol_path,
    test_mode=True,
    future_aligned=future_aligned,
    max_samples=max_samples,
)

val_dataset_config = dict(
    type=dataset_type,
    data_path=data_path,
    return_len=return_len_ if future_aligned else return_len_ + 1,
    offset=0,
    imageset=imageset,
    protocol_path=protocol_path,
    test_mode=True,
    future_aligned=future_aligned,
    max_samples=max_samples,
)

train_wrapper_config = dict(type='tpvformer_dataset_nuscenes', phase='train')
val_wrapper_config = dict(type='tpvformer_dataset_nuscenes', phase='val')

_loader_batch_size = int(os.environ.get('OCCSTRESS_OCCWORLD_BATCH_SIZE', '1'))
_loader_num_workers = int(os.environ.get('OCCSTRESS_OCCWORLD_NUM_WORKERS', '1'))
train_loader = dict(
    batch_size=_loader_batch_size,
    shuffle=False,
    num_workers=_loader_num_workers)
val_loader = dict(
    batch_size=_loader_batch_size,
    shuffle=False,
    num_workers=_loader_num_workers)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='CeLoss',
            weight=1.0,
            input_dict=dict(
                ce_inputs='ce_inputs',
                ce_labels='ce_labels',
            )),
        dict(
            type='PlanRegLossLidar',
            weight=0.1,
            loss_type='l2',
            num_modes=3,
            input_dict=dict(
                rel_pose='rel_pose',
                metas='metas',
            )),
    ],
)

loss_input_convertion = dict(
    ce_inputs='ce_inputs',
    ce_labels='ce_labels',
    rel_pose='pose_decoded',
    metas='output_metas',
)

base_channel = 64
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type='TransVQVAE',
    num_frames=num_frames_,
    delta_input=False,
    offset=1,
    vae=dict(
        type='VAERes2D',
        encoder_cfg=dict(
            type='Encoder2D',
            ch=base_channel,
            out_ch=base_channel,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(50,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=_dim_ * expansion,
            resolution=200,
            z_channels=base_channel * 2,
            double_z=False,
        ),
        decoder_cfg=dict(
            type='Decoder2D',
            ch=base_channel,
            out_ch=_dim_ * expansion,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(50,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=_dim_ * expansion,
            resolution=200,
            z_channels=base_channel * 2,
            give_pre_end=False,
        ),
        num_classes=18,
        expansion=expansion,
        vqvae_cfg=dict(
            type='VectorQuantizer',
            sane_index_shape=True,
            n_e=n_e_,
            e_dim=base_channel * 2,
            beta=1.0,
            z_channels=base_channel * 2,
            use_voxel=False,
        )),
    transformer=dict(
        type='PlanUAutoRegTransformer',
        num_tokens=1,
        num_frames=num_frames_,
        num_layers=2,
        img_shape=(base_channel * 2, 50, 50),
        pose_shape=(1, base_channel * 2),
        pose_attn_layers=2,
        pose_output_channel=base_channel * 2,
        tpe_dim=base_channel * 2,
        channels=(base_channel * 2, base_channel * 4, base_channel * 8),
        temporal_attn_layers=6,
        output_channel=n_e_,
        learnable_queries=False,
    ),
    pose_encoder=dict(
        type='PoseEncoder',
        in_channels=5,
        out_channels=base_channel * 2,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    pose_decoder=dict(
        type='PoseDecoder',
        in_channels=base_channel * 2,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
)

shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]
unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = './config/label_mapping/nuscenes-occ.yaml'

del os, sys, resolve_manual_protocol_path
