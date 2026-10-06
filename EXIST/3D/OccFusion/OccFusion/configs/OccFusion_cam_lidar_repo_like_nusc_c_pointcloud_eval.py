import os

_base_ = ['./OccFusion_cam_lidar_repo_like_eval.py']

corruption = os.environ.get('CORRUPTION')
severity = os.environ.get('SEVERITY')

if not corruption or not severity:
    raise RuntimeError('CORRUPTION and SEVERITY must be set for nuScenes-C pointcloud eval')

_pts_root = f'../nuScenes-C/raw/pointcloud/nuScenes-C/{corruption}/{severity}'

data_prefix = dict(
    pts=f'{_pts_root}/samples/LIDAR_TOP',
    pts_semantic_mask=f'{_pts_root}/lidarseg/v1.0-trainval',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

val_dataloader = dict(
    dataset=dict(
        data_prefix=data_prefix))

test_dataloader = val_dataloader
