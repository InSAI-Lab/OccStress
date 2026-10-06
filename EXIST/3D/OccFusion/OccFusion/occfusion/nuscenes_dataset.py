# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from nuscenes.nuscenes import NuScenes

from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesSegDataset(BaseDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    METAINFO = {
        'classes':
        ('IoU', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'),
        'ignore_index':
        255,
        'label_mapping':
        dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
              (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (41, 0),
              (42, 0), (43, 0), (9, 1), (14, 2),
              (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
              (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
              (26, 13), (27, 14), (28, 15), (30, 16)]),
        'palette': [
            [0, 0, 0],  # noise                                         0
            [255, 120, 50],  # barrier              orange              1
            [255, 192, 203],  # bicycle              pink               2
            [255, 255, 0],  # bus                  yellow               3
            [0, 150, 245],  # car                  blue                 4
            [0, 255, 255],  # construction_vehicle cyan                 5
            [255, 127, 0],  # motorcycle           dark orange          6
            [255, 0, 0],  # pedestrian           red                    7
            [255, 240, 150],  # traffic_cone         light yellow       8
            [135, 60, 0],  # trailer              brown                 9 
            [160, 32, 240],  # truck                purple              10
            [255, 0, 255],  # driveable_surface    dark pink            11
            [139, 137, 137],  # other_flat           dark red           12
            [75, 0, 75],  # sidewalk             dard purple            13
            [150, 240, 80],  # terrain              light green         14
            [230, 230, 250],  # manmade              white              15
            [0, 175, 0],  # vegetation           green                  16
                          # empty?                                      17
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 use_occ3d=False,
                 **kwargs) -> None:
        self.use_occ3d = use_occ3d
        self._nusc = None
        self._sample_to_lidarseg = {}
        if self.use_occ3d:
            self.METAINFO = {
                'classes':
                    ('IoU', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation','others'),
                'ignore_index': 255,
                'label_mapping':
                dict([(1, 17), (5, 17), (7, 17), (8, 17), (10, 17), (11, 17), (13, 17),
                    (19, 17), (20, 17), (0, 17), (29, 17), (31, 17), (41, 17),
                    (42, 17), (43, 17), (9, 1), (14, 2),
                    (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
                    (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
                    (26, 13), (27, 14), (28, 15), (30, 16)])
                }
        metainfo = dict(label2cat={
            i: cat_name
            for i, cat_name in enumerate(self.METAINFO['classes'])
        })
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def _get_nusc(self) -> NuScenes:
        if self._nusc is None:
            version = getattr(self, '_old_info_version', None) or 'v1.0-trainval'
            self._nusc = NuScenes(
                version=version, dataroot=self.data_root, verbose=False)
        return self._nusc

    def _get_lidarseg_filename(self, sample_token: str) -> str:
        if sample_token not in self._sample_to_lidarseg:
            sample = self._get_nusc().get('sample', sample_token)
            self._sample_to_lidarseg[sample_token] = \
                f"{sample['data']['LIDAR_TOP']}_lidarseg.bin"
        return self._sample_to_lidarseg[sample_token]

    @staticmethod
    def _build_lidar2sensor(rot, trans):
        lidar2sensor = np.eye(4, dtype=np.float32)
        rot = np.asarray(rot)
        trans = np.asarray(trans)
        lidar2sensor[:3, :3] = rot.T
        lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
        return lidar2sensor.tolist()

    def _convert_old_info(self, info: dict, sample_idx: int) -> dict:
        data_info = dict(
            sample_idx=sample_idx,
            token=info['token'],
            timestamp=info['timestamp'] / 1e6,
            ego2global=convert_quaternion_to_matrix(
                info['ego2global_rotation'], info['ego2global_translation']),
            lidar_points=dict(
                num_pts_feats=info.get('num_features', 5),
                lidar_path=Path(info['lidar_path']).name,
                lidar2ego=convert_quaternion_to_matrix(
                    info['lidar2ego_rotation'], info['lidar2ego_translation']),
            ),
            lidar_sweeps=[],
            images={},
            instances=[],
            pts_semantic_mask_path=self._get_lidarseg_filename(info['token']),
        )

        for sweep in info.get('sweeps', []):
            data_info['lidar_sweeps'].append(
                dict(
                    timestamp=sweep['timestamp'] / 1e6,
                    ego2global=convert_quaternion_to_matrix(
                        sweep['ego2global_rotation'],
                        sweep['ego2global_translation']),
                    lidar_points=dict(
                        lidar_path=sweep['data_path'],
                        lidar2ego=convert_quaternion_to_matrix(
                            sweep['sensor2ego_rotation'],
                            sweep['sensor2ego_translation']),
                        lidar2sensor=self._build_lidar2sensor(
                            sweep['sensor2lidar_rotation'],
                            sweep['sensor2lidar_translation']),
                    ),
                ))

        for cam, cam_info in info['cams'].items():
            data_info['images'][cam] = dict(
                img_path=Path(cam_info['data_path']).name,
                cam2img=np.asarray(cam_info['cam_intrinsic']).tolist(),
                cam2ego=convert_quaternion_to_matrix(
                    cam_info['sensor2ego_rotation'],
                    cam_info['sensor2ego_translation']),
                lidar2cam=self._build_lidar2sensor(
                    cam_info['sensor2lidar_rotation'],
                    cam_info['sensor2lidar_translation']),
                sample_data_token=cam_info.get('sample_data_token'),
                timestamp=cam_info['timestamp'] / 1e6,
            )

        return data_info

    def load_data_list(self) -> List[dict]:
        annotations = load(self.ann_file)
        if not (isinstance(annotations, dict) and 'infos' in annotations):
            return super().load_data_list()

        metadata = annotations.get('metadata', {})
        self._old_info_version = metadata.get('version', 'v1.0-trainval')
        data_list = []
        for idx, raw_info in enumerate(annotations['infos']):
            parsed = self.parse_data_info(self._convert_old_info(raw_info, idx))
            if isinstance(parsed, list):
                data_list.extend(parsed)
            else:
                data_list.append(parsed)
        return data_list

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        data_list = []
        info['lidar_points']['lidar_path'] = \
            osp.join(
                self.data_prefix.get('pts', ''),
                info['lidar_points']['lidar_path'])

        for cam_id, img_info in info['images'].items():
            if 'img_path' in img_info:
                if cam_id in self.data_prefix:
                    cam_prefix = self.data_prefix[cam_id]
                else:
                    cam_prefix = self.data_prefix.get('img', '')
                img_info['img_path'] = osp.join(cam_prefix,
                                                img_info['img_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.metainfo['label_mapping']

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode:
            info['eval_ann_info'] = dict()

        data_list.append(info)
        return data_list
