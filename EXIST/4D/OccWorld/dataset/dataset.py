# OccStress adapter/portability modifications; see docs/source-imports.json.
import os, numpy as np, pickle
from pathlib import Path
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET
import torch


class _FallbackLiDARInstance3DBoxes:
    """Minimal box wrapper for eval paths that only need `.tensor`."""

    def __init__(self, tensor, box_dim=None, origin=None):
        del box_dim, origin
        self.tensor = torch.as_tensor(tensor, dtype=torch.float32)

    def convert_to(self, box_mode):
        del box_mode
        return self


class _FallbackBox3DMode:
    LIDAR = 'LIDAR'


def _get_mmdet3d_box_types():
    try:
        from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
        return LiDARInstance3DBoxes, Box3DMode
    except ModuleNotFoundError:
        return _FallbackLiDARInstance3DBoxes, _FallbackBox3DMode


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
            self, 
            data_path,
            return_len, 
            offset,
            imageset='train', 
            nusc=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs, dtype=np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs, dtype=np.int64)
        metas = {}
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        metas.update(self.get_meta_data(scene_name, idx))
        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))
        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict
        
@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarTraverse(nuScenesSceneDatasetLidar):
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
    ):
        super().__init__(data_path, return_len, offset, imageset, nusc, times, test_mode, input_dataset, output_dataset)
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        _, Box3DMode = _get_mmdet3d_box_types()
        self.box_mode_3d = Box3DMode.LIDAR
        
    def __len__(self):
        'Denotes the total number of samples'
        return sum(self.scene_lens)
    
    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs, dtype=np.int64)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs, dtype=np.int64)
        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        # import pdb; pdb.set_trace()
        return input_occs[:self.return_len], output_occs[self.offset:], metas
    
    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        '''gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
                print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)
        '''
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        LiDARInstance3DBoxes, _ = _get_mmdet3d_box_types()
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            #gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)
        
        return anns_results
        
        
        
    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t
            
            
            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
        
        
        
        
        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        
        return input_dict


@OPENOCC_DATASET.register_module()
class OccStressNuScenesSceneDatasetLidarTraverse:
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset,
        protocol_path,
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
        future_aligned=False,
        max_samples=None,
    ):
        del nusc, times, input_dataset, output_dataset
        with open(protocol_path, 'rb') as f:
            self.protocol_samples = pickle.load(f)
        if max_samples is not None:
            self.protocol_samples = self.protocol_samples[:int(max_samples)]
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.token2info = {}
        for infos in self.nusc_infos.values():
            for info in infos:
                self.token2info[info['token']] = info

        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.test_mode = test_mode
        self.use_valid_flag = use_valid_flag
        self.future_aligned = bool(future_aligned)
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        _, Box3DMode = _get_mmdet3d_box_types()
        self.box_mode_3d = Box3DMode.LIDAR

    def __len__(self):
        return len(self.protocol_samples)

    def __getitem__(self, index):
        record = self.protocol_samples[index]
        sequence = self._build_sequence(record)
        input_occs = np.stack(
            [self._load_occ(entry['input_occ_path']) for entry in sequence],
            dtype=np.int64)
        output_occs = np.stack(
            [self._load_occ(entry['target_occ_path']) for entry in sequence],
            dtype=np.int64)

        frame_tokens = [entry['token'] for entry in sequence]
        metas = dict(
            scene_name=record['scene_name'],
            scene_token=record['scene_token'],
            sample_id=record['sample_id'],
            anchor_token=record['anchor_token'],
        )
        if self.future_aligned:
            metas.update(
                temporal_alignment='future_aligned_v1',
                input_time_offsets_sec=(-2.0, -1.5, -1.0, -0.5, 0.0),
                future_time_offsets_sec=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                future_start_index=5,
                future_length=6,
            )
        metas.update(self.get_meta_data_from_tokens(frame_tokens))
        if self.test_mode:
            metas.update(self.get_meta_info_from_token(record['target']['token']))
        metas.update(self.get_image_info_from_token(record['target']['token']))
        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def _build_sequence(self, record):
        sequence = []
        for frame in record['history']:
            sequence.append(dict(
                token=frame['token'],
                input_occ_path=frame['occ_path'],
                target_occ_path=self._clean_occ_path(record['scene_name'], frame['token']),
            ))
        sequence.append(dict(
            token=record['current_input']['token'],
            input_occ_path=record['current_input']['occ_path'],
            target_occ_path=self._clean_occ_path(record['scene_name'], record['current_input']['token']),
        ))
        if not self.future_aligned:
            sequence.append(dict(
                token=record['target']['token'],
                input_occ_path=record['target']['occ_path'],
                target_occ_path=record['target']['occ_path'],
            ))
        for frame in record['future_targets']:
            sequence.append(dict(
                token=frame['token'],
                input_occ_path=frame['occ_path'],
                target_occ_path=frame['occ_path'],
            ))

        expected = self.return_len + self.offset
        if len(sequence) < expected:
            raise ValueError(
                f'Protocol sample {record["sample_id"]} only has {len(sequence)} frames, expected {expected}.'
            )
        return sequence[:expected]

    def _clean_occ_path(self, scene_name, token):
        return os.path.join(self.data_path, f'gts/{scene_name}/{token}/labels.npz')

    def _load_occ(self, occ_path):
        import sys
        root = Path(os.environ.get('OCCSTRESS_CODE_ROOT', Path(__file__).resolve().parents[4]))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tools.adapters.occstress_paths import resolve_occstress_path
        occ_path = resolve_occstress_path(occ_path, code_root=root)
        label = np.load(occ_path)
        return label['semantics']

    def get_meta_data_from_tokens(self, tokens):
        xys = []
        gt_modes = []
        for token in tokens:
            info = self.token2info[token]
            xys.append(info['gt_ego_fut_trajs'][0])
            gt_modes.append(info['pose_mode'])
        return {'rel_poses': np.asarray(xys), 'gt_mode': np.asarray(gt_modes)}

    def get_meta_info_from_token(self, token):
        info = self.token2info[token]
        fut_valid_flag = info['valid_flag']
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw],
                axis=-1).astype(np.float32)

        LiDARInstance3DBoxes, _ = _get_mmdet3d_box_types()
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,
        )

    def get_image_info_from_token(self, token):
        info = self.token2info[token]
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for _, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t

            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))
        return input_dict


CARLA_MIRROR_Y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)
WAYMO_TO_OCC3D = {
    0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
    8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17,
}


@OPENOCC_DATASET.register_module()
class OccStressCARLASceneDataset:
    """Protocol-backed H4+current/F6 CARLA dataset for OccWorld."""

    def __init__(
            self,
            data_path,
            return_len,
            offset,
            imageset,
            protocol_path,
            max_samples=None,
            **kwargs):
        del data_path, kwargs
        if return_len != 11 or offset != 0:
            raise ValueError(
                'OccWorld OccStress-CARLA requires return_len=11 and offset=0.')
        with open(protocol_path, 'rb') as stream:
            records = pickle.load(stream)
        from occstress.datasets.metadata import load_metadata
        base = load_metadata(imageset, trusted_pickle=True)
        if max_samples is not None:
            records = records[:int(max_samples)]
        self.protocol_samples = records
        self.token2info = {
            info['token']: info
            for scene_infos in base['infos'].values()
            for info in scene_infos
        }
        self.return_len = return_len
        self.offset = offset
        self.dataset_name = base.get('metadata', {}).get(
            'dataset', 'UniOcc-CARLA')

    def __len__(self):
        return len(self.protocol_samples)

    def _resolve_occ_path(self, frame):
        from occstress.datasets.paths import release_occ_path
        released = release_occ_path(frame['occ_path'])
        if released is not None:
            return released
        path = Path(frame['occ_path'])
        if path.suffix != '.npz':
            path = path / 'labels.npz'
        if self.dataset_name in ('Occ3D-Waymo', 'OccStress-Waymo'):
            source = frame.get('occ_source', frame.get('source'))
            if source is None:
                source = 'corrupted' if '/occ/manual/' in str(path) else 'clean'
            scene_name = self.token2info[frame['token']]['scene_name']
            clean_root = os.environ.get('OCCSTRESS_WAYMO_CLEAN_OCC_CACHE')
            if source == 'clean' and clean_root:
                candidate = (
                    Path(clean_root) / str(scene_name).zfill(3) /
                    frame['token'] / 'labels.npz')
                if candidate.is_file():
                    return candidate

            asset_root = os.environ.get('OCCSTRESS_WAYMO_ASSET_CACHE')
            normalized = str(path).replace('\\', '/')
            if source != 'clean' and asset_root:
                for marker, relative_root in (
                    ('/occ/manual/', 'occ/manual'),
                    ('/occ/upstream/', 'occ/upstream'),
                    (
                        '/effocc-waymo-upstream/',
                        'occ/upstream/pointcloud_fusion/effocc',
                    ),
                ):
                    if marker not in normalized:
                        continue
                    candidate = (
                        Path(asset_root) / relative_root /
                        normalized.split(marker, 1)[1])
                    if candidate.is_file():
                        return candidate
            if path.is_file():
                return path
        elif path.is_file():
            return path
        raise FileNotFoundError(f'cannot resolve occupancy path: {path}')

    def _load_occ(self, frame):
        path = self._resolve_occ_path(frame)
        with np.load(path, allow_pickle=False) as labels:
            if 'semantics' in labels:
                semantics = np.asarray(labels['semantics'], dtype=np.int64)
            elif self.dataset_name in ('Occ3D-Waymo', 'OccStress-Waymo') and 'voxel_label' in labels:
                raw = np.asarray(labels['voxel_label'], dtype=np.int64)
                unknown = set(np.unique(raw).tolist()) - set(WAYMO_TO_OCC3D)
                if unknown:
                    raise ValueError(
                        f'{path} contains unmapped Waymo classes: '
                        f'{sorted(unknown)}')
                semantics = np.empty(raw.shape, dtype=np.int64)
                for source, target in WAYMO_TO_OCC3D.items():
                    semantics[raw == source] = target
            else:
                raise KeyError(
                    f'{path} has neither semantics nor supported voxel_label')
        if semantics.shape != (200, 200, 16):
            raise ValueError(
                f'{path} has occupancy shape {semantics.shape}, '
                'expected (200, 200, 16)')
        if semantics.min() < 0 or semantics.max() > 17:
            raise ValueError(f'{path} contains an invalid semantic class')
        return semantics

    def _input_sequence(self, record):
        return (
            record['history'] + [record['current_input']] +
            record['future_targets'])

    def _target_sequence(self, record):
        if record.get('traffic_mirror'):
            history = record['history']
        else:
            history = [
                {
                    'token': frame['token'],
                    'occ_path': self.token2info[frame['token']]['occ_path'],
                }
                for frame in record['history']
            ]
        return history + [record['target']] + record['future_targets']

    def _poses(self, record, tokens):
        clean = np.stack([
            np.asarray(self.token2info[token]['pose_mat'], dtype=np.float64)
            for token in tokens
        ])
        if record.get('traffic_mirror'):
            clean = np.stack([
                CARLA_MIRROR_Y @ pose @ CARLA_MIRROR_Y
                for pose in clean
            ])
        poses = clean.copy()

        misalignment = record.get('misalignment') or {}
        if misalignment:
            anchor_pose = clean[4]
            deltas = np.asarray(
                misalignment['delta_rt'], dtype=np.float64)
            affected = np.asarray(
                misalignment['affected_mask'], dtype=np.uint8)
            for index in range(4):
                if not affected[index]:
                    continue
                anchor_to_history = np.linalg.inv(clean[index]) @ anchor_pose
                corrupted = deltas[index] @ anchor_to_history
                poses[index] = anchor_pose @ np.linalg.inv(corrupted)
            if misalignment.get('current_active'):
                current_delta = np.asarray(
                    misalignment['current_delta_rt'], dtype=np.float64)
                poses[4] = anchor_pose @ np.linalg.inv(current_delta)
        return poses

    def _metadata(self, record, tokens):
        poses = self._poses(record, tokens)
        relative = np.linalg.inv(poses[:-1]) @ poses[1:]
        relative = np.concatenate([relative, relative[-1:]], axis=0)
        rel_poses = relative[:, :2, 3].astype(np.float32)
        modes = np.stack([
            np.asarray(self.token2info[token]['pose_mode'], dtype=np.float32)
            for token in tokens
        ])
        if record.get('traffic_mirror'):
            modes[:, [0, 1]] = modes[:, [1, 0]]
        return {
            'rel_poses': rel_poses,
            'gt_mode': modes,
            'scene_name': record['scene_name'],
            'scene_token': record['scene_token'],
            'sample_id': record['sample_id'],
            'anchor_token': record['anchor_token'],
            'future_start_index': 5,
            'future_length': 6,
        }

    def __getitem__(self, index):
        record = self.protocol_samples[index]
        inputs = self._input_sequence(record)
        targets = self._target_sequence(record)
        if len(inputs) != 11 or len(targets) != 11:
            raise ValueError(
                f'{record["sample_id"]}: expected 11 frames, '
                f'got input={len(inputs)}, target={len(targets)}')
        input_occs = np.stack([self._load_occ(frame) for frame in inputs])
        target_occs = np.stack([self._load_occ(frame) for frame in targets])
        tokens = [frame['token'] for frame in inputs]
        return input_occs, target_occs, self._metadata(record, tokens)


@OPENOCC_DATASET.register_module()
class OccStressWaymoSceneDataset(OccStressCARLASceneDataset):
    """Waymo specialization of the protocol-backed occupancy-only loader."""

    pass
