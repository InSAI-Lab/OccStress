_base_ = ['./OccFusion_occ3d_cam_lidar_debug.py']

load_from = 'ckpt/r101_dcn_fcos3d_pretrain.pth'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
)

test_dataloader = val_dataloader

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_begin=1, val_interval=1)
