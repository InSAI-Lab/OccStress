import copy
import os

from pyquaternion import Quaternion
import numpy as np
import cv2

seg_color_dict = {
    0: (255, 255, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (0, 255, 255),
    5: (255, 0, 255),
    6: (255, 255, 0),
    7: (202, 235, 216),
    8: (251, 255, 242),
    9: (255, 235, 205),
    10: (255, 153, 18),
    11: (255, 215, 0),
    12: (218, 112, 214),
    13: (3, 168, 158),
    14: (176, 23, 31),
    15: (255, 192, 203),
    16: (252, 230, 202),
    17: (0, 0, 0)
}


def project_lidar2image(nusc, img, lidar_pts, lidar_file, camera_data, down_sample, lidar_seg=None, save_path=None):
    img_h, img_w, _ = img.shape
    points = copy.deepcopy(lidar_pts.transpose())

    # step1: lidar frame -> ego frame
    calib_data = nusc.get('calibrated_sensor', lidar_file['calibrated_sensor_token'])
    rot_matrix = Quaternion(calib_data['rotation']).rotation_matrix
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    for i in range(3):
        points[i, :] += calib_data['translation'][i]

    # step2: ego frame -> global frame
    ego_data = nusc.get('ego_pose', lidar_file['ego_pose_token'])
    rot_matrix = Quaternion(ego_data['rotation']).rotation_matrix
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    for i in range(3):
        points[i, :] += ego_data['translation'][i]

    # step3: global frame -> ego frame
    ego_data = nusc.get('ego_pose', camera_data['ego_pose_token'])
    for i in range(3):
        points[i, :] -= ego_data['translation'][i]
    rot_matrix = Quaternion(ego_data['rotation']).rotation_matrix.T
    points[:3, :] = np.dot(rot_matrix, points[:3, :])

    # step4: ego frame -> cam frame
    calib_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    for i in range(3):
        points[i, :] -= calib_data['translation'][i]
    rot_matrix = Quaternion(calib_data['rotation']).rotation_matrix.T
    points[:3, :] = np.dot(rot_matrix, points[:3, :])

    # step5: cam frame -> uv pixel
    visible = points[2, :] > 0.1
    # colors = get_rgb_by_distance(points[2, :], min_val=0, max_val=50)
    intrinsic = calib_data['camera_intrinsic']
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = np.array(intrinsic)
    points = np.concatenate((points[:3, :], np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(trans_mat, points)[:3, :]
    points /= points[2, :]
    points = points[:2, :]
    lidar_seg = lidar_seg.reshape(1, -1)
    points = np.concatenate([points, lidar_seg], axis=0)

    visible = np.logical_and(visible, points[0, :] >= 0)
    visible = np.logical_and(visible, points[0, :] < img_w)
    visible = np.logical_and(visible, points[1, :] >= 0)
    visible = np.logical_and(visible, points[1, :] < img_h)
    points = points[:, np.where(visible == 1)[0]].astype(int)
    # colors = colors[np.where(visible == 1)[0], :]

    h, w, d = img.shape
    back_img = np.ones([h, w], dtype=np.uint8) * 17
    for i in range(points.shape[1]):
        x_r = 4
        y_r = 8
        x_s, y_s = max(0, points[0, i] - x_r), max(0, points[1, i] - y_r)
        x_e, y_e = min(points[0, i] + x_r, w - 1), min(points[1, i] + y_r, h - 1)
        back_img[y_s:y_e, x_s:x_e] = int(points[2, i])
    back_img = back_img[::down_sample, ::down_sample]
    if save_path is not None:
        np.save(save_path, back_img)


def process_one_sample(nusc,
                       sample,
                       down_sample,
                       lidar_seg=None,
                       proj_lidar=False,
                       save_dir=None):
    data_root = nusc.dataroot
    camera_file = dict()
    lidar_file = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    for key in sample['data']:
        if key.startswith('CAM'):
            sample_data = nusc.get('sample_data', sample['data'][key])
            camera_file[sample_data['channel']] = sample_data

    lidar_path = os.path.join(data_root, lidar_file['filename'])
    lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))[:, :4]

    ori_img_size = (1600, 900)
    for camera_type in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']:
        camera_data = camera_file['CAM_{}'.format(camera_type)]
        img_path = os.path.join(data_root, camera_data['filename'])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        assert ori_img_size == (w, h)
        save_path = os.path.join(save_dir, camera_data['filename'].replace(".jpg", ""))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if proj_lidar:
            project_lidar2image(nusc, img, lidar_pts, lidar_file, camera_data, down_sample,
                                lidar_seg=lidar_seg,
                                save_path=save_path)
