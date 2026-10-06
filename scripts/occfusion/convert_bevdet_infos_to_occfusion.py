import argparse
import os
import pickle
from pathlib import Path

import numpy as np


CAMERA_TYPES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def quat_to_mat(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
                    dtype=np.float32)


def pose_from_quat_trans(quat, trans):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = quat_to_mat(quat)
    pose[:3, 3] = np.asarray(trans, dtype=np.float32)
    return pose.tolist()


def lidar2sensor_from_sensor2lidar(rot, trans):
    lidar2sensor = np.eye(4, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    lidar2sensor[:3, :3] = rot.T
    lidar2sensor[:3, 3:4] = -np.matmul(rot.T, trans.reshape(3, 1))
    return lidar2sensor.tolist()


def convert_info(info, sample_idx):
    data_info = {
        'sample_idx': sample_idx,
        'token': info['token'],
        'timestamp': info['timestamp'] / 1e6,
        'scene_token': info.get('scene_token'),
        'scene_name': info.get('scene_name'),
        'ego2global': pose_from_quat_trans(
            info['ego2global_rotation'], info['ego2global_translation']),
        'lidar_points': {
            'num_pts_feats': info.get('num_features', 5),
            'lidar_path': Path(info['lidar_path']).name,
            'lidar2ego': pose_from_quat_trans(
                info['lidar2ego_rotation'], info['lidar2ego_translation']),
        },
        'lidar_sweeps': [],
        'images': {},
        'pts_semantic_mask_path': f"{info['lidar_token']}_lidarseg.bin",
        'occ_path': info.get('occ_path'),
    }

    for sweep in info.get('sweeps', []):
        data_info['lidar_sweeps'].append({
            'timestamp':
            sweep['timestamp'] / 1e6,
            'sample_data_token':
            sweep['sample_data_token'],
            'lidar_points': {
                'lidar_path': sweep['data_path'],
                'lidar2ego': pose_from_quat_trans(
                    sweep['sensor2ego_rotation'], sweep['sensor2ego_translation']),
                'lidar2sensor': lidar2sensor_from_sensor2lidar(
                    sweep['sensor2lidar_rotation'], sweep['sensor2lidar_translation']),
            },
            'ego2global':
            pose_from_quat_trans(sweep['ego2global_rotation'],
                                 sweep['ego2global_translation']),
        })

    for cam in CAMERA_TYPES:
        cam_info = info['cams'][cam]
        data_info['images'][cam] = {
            'img_path': Path(cam_info['data_path']).name,
            'cam2img': np.asarray(cam_info['cam_intrinsic'],
                                  dtype=np.float32).tolist(),
            'sample_data_token': cam_info['sample_data_token'],
            'timestamp': cam_info['timestamp'] / 1e6,
            'cam2ego': pose_from_quat_trans(cam_info['sensor2ego_rotation'],
                                            cam_info['sensor2ego_translation']),
            'lidar2cam': lidar2sensor_from_sensor2lidar(
                cam_info['sensor2lidar_rotation'],
                cam_info['sensor2lidar_translation']),
        }

    return data_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-version', default='v1.0-trainval')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        payload = pickle.load(f)

    infos = payload['infos']
    converted = {
        'metainfo': {
            'dataset': 'nuscenes',
            'version': payload.get('metadata', {}).get('version',
                                                       args.dataset_version),
            'info_version': 'occfusion-local-v1',
        },
        'data_list': [convert_info(info, idx) for idx, info in enumerate(infos)],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"converted {len(infos)} infos -> {args.output}")


if __name__ == '__main__':
    main()
