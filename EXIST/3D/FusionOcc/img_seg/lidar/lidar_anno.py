import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils import data
from nuscenes import NuScenes
from nuscenes.utils import splits


class nuScenesLidarSeg():
    def __init__(self,
                 nusc, data_path,
                 version='v1.0-trainval', imageset='train', num_vote=1,
                 label_mapping_path=os.path.join(
                     os.path.dirname(os.path.abspath(__file__)), "config/label_mapping/nuscenes.yaml")
                 ):
        scenes = splits.train
        self.sample_token2lidar_token = {}
        self.split = imageset
        with open(label_mapping_path, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.num_vote = num_vote
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']

        self.nusc = nusc
        self.data_path = data_path
        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)
        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))

    def loadDataByIndex(self, index, is_token=False):
        if is_token:
            lidar_sample_token = index
        else:
            lidar_sample_token = self.token_list[index]['lidar_token']

        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token, lidar_path

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break
            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []
        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            sample_token = sample['token']
            self.sample_token2lidar_token[sample_token] = lidar_token

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    cam_token = []
                    for i in self.img_view:
                        cam_token.append(sample['data'][i])
                    self.token_list.append(
                        {'lidar_token': lidar_token,
                         'cam_token': cam_token}
                    )

    def get_lidar_seg(self, token, is_sample_idx=True):
        if is_sample_idx:
            lidar_token = self.sample_token2lidar_token[token]
        else:
            lidar_token = token
        lidar_seg_path = os.path.join(self.data_path, self.nusc.get('lidarseg', lidar_token)['filename'])
        sem_label = np.fromfile(lidar_seg_path, dtype=np.uint8).reshape((-1, 1))  # label
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)
        return sem_label

    def get_sample_data(self, index, is_token=False):
        if is_token:
            lidar_token = self.sample_token2lidar_token[index]
            index = lidar_token
        pointcloud, sem_label, instance_label, lidar_sample_token, lidar_path = \
            self.loadDataByIndex(index, is_token=is_token)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)
        return pointcloud, sem_label.astype(np.uint8), lidar_path, lidar_sample_token
