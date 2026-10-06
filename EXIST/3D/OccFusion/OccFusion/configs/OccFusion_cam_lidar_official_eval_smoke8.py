_base_ = ['./OccFusion_cam_lidar_official_eval.py']

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file='nuscenes_infos_occfusion_val_official_smoke8.pkl',
        use_occ3d=False))

test_dataloader = val_dataloader
