import os

_base_ = ['./OccFusion_cam_lidar_repo_like_eval.py']

corruption = os.environ.get('CORRUPTION')
severity = os.environ.get('SEVERITY')

if not corruption or not severity:
    raise RuntimeError('CORRUPTION and SEVERITY must be set for nuScenes-C image eval')

_cam_root = f'../nuScenes-C/raw/image/nuScenes-c/{corruption}/{severity}'

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    CAM_FRONT=f'{_cam_root}/CAM_FRONT',
    CAM_FRONT_LEFT=f'{_cam_root}/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT=f'{_cam_root}/CAM_FRONT_RIGHT',
    CAM_BACK=f'{_cam_root}/CAM_BACK',
    CAM_BACK_RIGHT=f'{_cam_root}/CAM_BACK_RIGHT',
    CAM_BACK_LEFT=f'{_cam_root}/CAM_BACK_LEFT')

val_dataloader = dict(
    dataset=dict(
        data_prefix=data_prefix))

test_dataloader = val_dataloader
