# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
import sys
from pathlib import Path

ROBUSTOCC_ROOT = str(
    Path(os.environ.get('OCCSTRESS_CODE_ROOT', '../../..')).resolve())
if ROBUSTOCC_ROOT not in sys.path:
    sys.path.insert(0, ROBUSTOCC_ROOT)
from scripts.occstress_layout import resolve_manual_protocol_path

work_dir = './work_dir/dome'

stage_one_config = os.environ.get(
    'COME_STAGE_ONE_CONFIG',
    'configs/unet/unet_aligned_past2s_future_3s.py')
stage_one_ckpt = os.environ.get(
    'COME_STAGE_ONE_CKPT',
    'work_dir/unet/unet_past2s_future3s.pth')

world_model_config = os.environ.get(
    'COME_WORLD_MODEL_CONFIG',
    'configs/train_dome_v5_invisible_fixed_mask.py')
world_model_ckpt = os.environ.get(
    'COME_WORLD_MODEL_CKPT',
    'work_dir/dome_v2/best_miou_world_model.pth')

start_frame = 0
mid_frame = 4
end_frame = 10
eval_length = end_frame - mid_frame

return_len_train = 10
return_len_ = 10
grad_max_norm = 1
print_freq = 20
max_epochs = 1000
warmup_iters = 50
ema = True
load_from = ''
vae_load_from = os.environ.get('COME_VAE_CKPT', 'ckpts/occvae_latest.pth')
port = int(os.environ.get('OCCSTRESS_COME_PORT', '25130'))
revise_ckpt = 3
eval_every_epochs = 10
save_every_epochs = 200

multisteplr = True
multisteplr_config = dict(
    decay_rate=1,
    decay_t=[0],
    t_in_epochs=False,
    warmup_lr_init=1e-06,
    warmup_t=0,
)
optimizer = dict(optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001))

schedule = dict(
    beta_end=0.02,
    beta_schedule='linear',
    beta_start=0.0001,
    variance_type='learned_range',
)

sample = dict(
    enable_temporal_attentions=True,
    enable_vae_temporal_decoder=True,
    guidance_scale=7.5,
    n_conds=4,
    num_sampling_steps=20,
    run_time=0,
    sample_method='ddpm',
    seed=None,
)
p_use_pose_condition = 0.9

replace_cond_frames = True
cond_frames_choices = [
    [],
    [0],
    [0, 1],
    [0, 1, 2],
    [0, 1, 2, 3],
]

data_path = os.environ.get(
    'OCCSTRESS_COME_DATA_PATH',
    str(Path(ROBUSTOCC_ROOT) / 'data/nuscenes'))
protocol_name = os.environ.get('OCCSTRESS_COME_PROTOCOL_NAME', 'clean_H4_F6_val_backbone')
waymo_protocol_path = os.environ.get('OCCSTRESS_WAYMO_PROTOCOL')
future_aligned = os.environ.get(
    'OCCSTRESS_COME_FUTURE_ALIGNED', '1').lower() not in ('0', 'false', 'no')
max_samples_value = os.environ.get('OCCSTRESS_COME_MAX_SAMPLES')
max_samples = int(max_samples_value) if max_samples_value else None
anchor_deterministic_seed = bool(waymo_protocol_path) and os.environ.get(
    'COME_ANCHOR_DETERMINISTIC', '1').lower() not in ('0', 'false', 'no')
anchor_seed_base = int(os.environ.get('COME_ANCHOR_SEED_BASE', '42'))
protocol_path = os.environ.get('OCCSTRESS_COME_PROTOCOL_PATH') or waymo_protocol_path
if not protocol_path:
    protocol_path = str(resolve_manual_protocol_path(protocol_name, root=ROBUSTOCC_ROOT))
imageset = os.environ.get(
    'OCCSTRESS_COME_IMAGESET',
    'data/nuscenes_infos_val_temporal_v3_scene.pkl',
)

train_dataset_config = dict(
    type='OccStressNuScenesSceneDatasetLidarTraverse',
    data_path=data_path,
    return_len=return_len_train,
    offset=0,
    imageset=imageset,
    protocol_path=protocol_path,
    test_mode=True,
    new_rel_pose=True,
    future_aligned=future_aligned,
    max_samples=max_samples,
)

val_dataset_config = dict(
    type='OccStressNuScenesSceneDatasetLidarTraverse',
    data_path=data_path,
    return_len=return_len_,
    offset=0,
    imageset=imageset,
    protocol_path=protocol_path,
    test_mode=True,
    new_rel_pose=True,
    future_aligned=future_aligned,
    max_samples=max_samples,
)

if waymo_protocol_path:
    waymo_imageset = str(
        Path(os.environ['OCCSTRESS_WAYMO_BASE_INFO']).resolve())
    waymo_scene_shard = os.environ.get('OCCSTRESS_WAYMO_SCENE_SHARD')
    waymo_max_samples_value = os.environ.get('OCCSTRESS_WAYMO_MAX_SAMPLES')
    waymo_max_samples = (
        int(waymo_max_samples_value) if waymo_max_samples_value else None)
    waymo_dataset_config = dict(
        type='OccStressWaymoSceneDataset',
        data_path='',
        return_len=10,
        offset=0,
        imageset=waymo_imageset,
        protocol_path=str(Path(waymo_protocol_path).resolve()),
        test_mode=True,
        new_rel_pose=True,
        scene_shard=waymo_scene_shard,
        max_samples=waymo_max_samples,
    )
    train_dataset_config = waymo_dataset_config
    val_dataset_config = waymo_dataset_config

train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuscenes')
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuscenes')
_loader_batch_size = int(os.environ.get('OCCSTRESS_COME_BATCH_SIZE', '1'))
_loader_num_workers = int(os.environ.get('OCCSTRESS_COME_NUM_WORKERS', '1'))
if anchor_deterministic_seed and _loader_batch_size != 1:
    raise ValueError(
        'COME deterministic OccStress-Waymo evaluation requires batch_size=1')
train_loader = dict(batch_size=_loader_batch_size, num_workers=_loader_num_workers, shuffle=False)
val_loader = dict(batch_size=_loader_batch_size, num_workers=_loader_num_workers, shuffle=False)

loss = dict(
    loss_cfgs=[
        dict(
            input_dict=dict(ce_inputs='ce_inputs', ce_labels='ce_labels'),
            type='CeLoss',
            weight=1.0,
        ),
    ],
    type='MultiLoss',
)
loss_input_convertion = dict()

_dim_ = 16
base_channel = 64
expansion = 8
n_e_ = 512
num_heads = 12
hidden_size = 768

model = dict(
    delta_input=False,
    world_model=dict(
        attention_mode='xformers',
        class_dropout_prob=0.1,
        extras=1,
        hidden_size=hidden_size,
        in_channels=64,
        input_size=25,
        learn_sigma=True,
        mlp_ratio=4.0,
        num_classes=1000,
        num_frames=return_len_train,
        num_heads=num_heads,
        patch_size=1,
        pose_encoder=dict(
            do_proj=True,
            in_channels=2,
            num_fut_ts=1,
            num_layers=2,
            num_modes=3,
            out_channels=hidden_size,
            type='PoseEncoder_fourier',
            zero_init=False,
        ),
        type='DomeControlNet',
    ),
    sampling_method='SAMPLE',
    topk=10,
    vae=dict(
        encoder_cfg=dict(
            attn_resolutions=(50,),
            ch=base_channel,
            ch_mult=(1, 2, 4, 8),
            double_z=False,
            dropout=0.0,
            in_channels=128,
            num_res_blocks=2,
            out_ch=base_channel,
            resamp_with_conv=True,
            resolution=200,
            type='Encoder2D',
            z_channels=base_channel * 2,
        ),
        decoder_cfg=dict(
            attn_resolutions=(50,),
            ch=base_channel,
            ch_mult=(1, 2, 4, 8),
            dropout=0.0,
            give_pre_end=False,
            in_channels=_dim_ * expansion,
            num_res_blocks=2,
            out_ch=_dim_ * expansion,
            resamp_with_conv=True,
            resolution=200,
            type='Decoder3D',
            z_channels=base_channel,
        ),
        expansion=expansion,
        num_classes=18,
        scaling_factor=0.18215,
        type='VAERes3D',
    ),
)
shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = './configs/label_mapping/nuscenes-occ.yaml'

find_unused_parameters = True

evaluators = [
    dict(type='MIoU', ignore_label=-1),
    dict(
        type='SeqMIoU',
        ignore_label=-1,
        timestamps=['0.5s', '1s', '1.5s', '2s', '2.5s', '3s'],
    ),
]

compute_generative_metrics = not bool(waymo_protocol_path)
use_post_fusion = False

del os, sys, Path, resolve_manual_protocol_path
