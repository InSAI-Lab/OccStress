# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import pickle
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from configs.scene_tokenizer.ii_scene_tokenizer_waymo_4f import dataset_type
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

from torchvision.transforms.functional import rotate

@PIPELINES.register_module()
class LoadStreamLatentToken(object):
    def __init__(self,
                 data_path=None,
                 to_long=False,
                 dataset_type='occ3d',
                 ):
        self.data_path = data_path
        self.to_long = to_long
        self.dataset_type = dataset_type

    def __call__(self, results):
        # current frame
        sample_idx = results['sample_idx']
        scene_name = results['scene_name']
        if self.dataset_type == 'waymo':
            scene_name = str(scene_name).zfill(3)
            occ_index = results['occ_path'].split('/')[-1].split('.')[0]
            latent_token_path = os.path.join(self.data_path, scene_name, f'{occ_index}.npz')
        else:
            latent_token_path = os.path.join(self.data_path, str(scene_name), f'{sample_idx}.npz')
        latent_token = np.load(latent_token_path)
        token = latent_token['token'][None]

        # future frame
        future_tokens = []
        future_frame = len(results['future_occ_path'])
        for frame_idx in range(future_frame):
            future_sample_idx = results['future_occ_path'][frame_idx].split('/')[-1]
            if self.dataset_type == 'waymo':
                scene_name = str(scene_name).zfill(3)
                future_occ_index = results['future_occ_path'][frame_idx].split('/')[-1].split('.')[0]
                latent_token_path = os.path.join(self.data_path, scene_name, f'{future_occ_index}.npz')
            else:
                latent_token_path = os.path.join(self.data_path, str(scene_name), f'{future_sample_idx}.npz')
            latent_token = np.load(latent_token_path)
            future_token = latent_token['token']
            future_tokens.append(future_token)
        future_tokens = np.stack(future_tokens)

        input_token = np.concatenate([token, future_tokens], axis=0)

        results['latent'] = input_token
        return results


@PIPELINES.register_module()
class LoadStreamLatentHistoryToken(LoadStreamLatentToken):
    """Load current/future stage2 tokens plus explicit previous-frame tokens.

    The original stage2 loader only exposes ``[current, future...]`` and the
    world model internally repeats the current token as memory. This loader
    keeps the baseline token root unchanged while also returning the preceding
    history tokens so stage2-only history fusion variants can consume them.
    """

    def __init__(self,
                 data_path=None,
                 to_long=False,
                 dataset_type='occ3d',
                 history_frame_number=4,
                 history_key='history_latent',
                 load_flow_summary=False,
                 flow_root=None,
                 flow_summary_key='history_flow_summary',
                 flow_key='flow_backward',
                 valid_key='flow_valid_backward',
                 dynamic_key='dynamic_mask'):
        super().__init__(data_path=data_path, to_long=to_long, dataset_type=dataset_type)
        self.history_frame_number = int(history_frame_number)
        self.history_key = history_key
        self.load_flow_summary = bool(load_flow_summary)
        self.flow_root = flow_root
        self.flow_summary_key = flow_summary_key
        self.flow_key = flow_key
        self.valid_key = valid_key
        self.dynamic_key = dynamic_key

    def _token_path(self, scene_name, sample_idx, occ_path=None):
        if self.dataset_type == 'waymo':
            scene_name = str(scene_name).zfill(3)
            occ_index = os.path.basename(str(occ_path)).split('.')[0] if occ_path is not None else str(sample_idx)
            return os.path.join(self.data_path, scene_name, f'{occ_index}.npz')
        return os.path.join(self.data_path, str(scene_name), f'{sample_idx}.npz')

    def _load_token(self, scene_name, sample_idx, occ_path=None):
        latent_token_path = self._token_path(scene_name, sample_idx, occ_path=occ_path)
        latent_token = np.load(latent_token_path)
        return latent_token['token']

    def _sample_id_from_occ_path(self, occ_path):
        sample_id = os.path.basename(str(occ_path).rstrip('/'))
        if self.dataset_type == 'waymo':
            sample_id = sample_id.split('.')[0]
        return sample_id

    def _history_occ_paths(self, results):
        paths = list(results.get('previous_occ_path', []))
        if len(paths) == 0:
            paths = [results['occ_path']] * self.history_frame_number
        if len(paths) < self.history_frame_number:
            pad_value = paths[0] if paths else results['occ_path']
            paths = [pad_value] * (self.history_frame_number - len(paths)) + paths
        return paths[-self.history_frame_number:]

    def _flow_step_summary(self, scene_name, prev_token, curr_token):
        if self.flow_root is None or prev_token == curr_token:
            return np.zeros(4, dtype=np.float32)
        flow_path = os.path.join(self.flow_root, str(scene_name), f'{curr_token}.npz')
        if not os.path.exists(flow_path):
            return np.zeros(4, dtype=np.float32)
        flow_data = np.load(flow_path)
        if self.valid_key not in flow_data.files or self.dynamic_key not in flow_data.files:
            return np.zeros(4, dtype=np.float32)
        valid = flow_data[self.valid_key].astype(np.bool_)
        dynamic = flow_data[self.dynamic_key].astype(np.bool_)
        valid_ratio = float(valid.mean())
        dynamic_ratio = float(dynamic.mean())
        valid_dynamic_ratio = float((valid & dynamic).mean())
        mean_mag = 0.0
        if self.flow_key in flow_data.files and valid.any():
            flow = flow_data[self.flow_key].astype(np.float32)
            mean_mag = float(np.linalg.norm(flow[..., :2][valid], axis=-1).mean())
        return np.array([valid_ratio, dynamic_ratio, valid_dynamic_ratio, mean_mag], dtype=np.float32)

    def _history_flow_summary(self, scene_name, history_tokens, current_token):
        seq_tokens = list(history_tokens) + [str(current_token)]
        summaries = []
        num_history = len(history_tokens)
        for hist_idx in range(num_history):
            step_summaries = [
                self._flow_step_summary(scene_name, seq_tokens[step_idx], seq_tokens[step_idx + 1])
                for step_idx in range(hist_idx, num_history)
            ]
            step_summary = np.stack(step_summaries, axis=0).mean(axis=0) if step_summaries else np.zeros(4, dtype=np.float32)
            age = float(num_history - hist_idx) / max(float(num_history), 1.0)
            summaries.append(np.concatenate([np.array([age], dtype=np.float32), step_summary], axis=0))
        return np.stack(summaries, axis=0).astype(np.float32)

    def __call__(self, results):
        sample_idx = results['sample_idx']
        scene_name = results['scene_name']

        token = self._load_token(scene_name, sample_idx, occ_path=results.get('occ_path'))[None]

        future_tokens = []
        future_frame = len(results['future_occ_path'])
        for frame_idx in range(future_frame):
            future_occ_path = results['future_occ_path'][frame_idx]
            future_sample_idx = self._sample_id_from_occ_path(future_occ_path)
            future_tokens.append(self._load_token(scene_name, future_sample_idx, occ_path=future_occ_path))
        future_tokens = np.stack(future_tokens)
        results['latent'] = np.concatenate([token, future_tokens], axis=0)

        history_paths = self._history_occ_paths(results)
        history_sample_ids = [self._sample_id_from_occ_path(path) for path in history_paths]
        history_tokens = [
            self._load_token(scene_name, history_sample_id, occ_path=history_path)
            for history_sample_id, history_path in zip(history_sample_ids, history_paths)
        ]
        results[self.history_key] = np.stack(history_tokens, axis=0)

        if self.load_flow_summary:
            results[self.flow_summary_key] = self._history_flow_summary(scene_name, history_sample_ids, sample_idx)

        return results

@PIPELINES.register_module()
class LoadLatentToken(object):
    def __init__(self,
                 data_path=None,
                 to_long=False,
                 ):
        self.data_path = data_path
        self.to_long = to_long

    def __call__(self, results):
        sample_idx = results['sample_idx']
        scene_name = results['scene_name']
        latent_token_path = os.path.join(self.data_path, scene_name, f'{sample_idx}.npz')
        latent_token = np.load(latent_token_path)
        token = latent_token['token']
        gt_mode = latent_token['gt_mode']
        rel_poses = latent_token['rel_poses']

        results['latent'] = token
        results['gt_mode'] = gt_mode
        results['rel_poses'] = rel_poses
        return results

@PIPELINES.register_module()
class LoadStreamOcc3D(object):
    def __init__(self,
                 to_long=True,
                 to_float=False,
                 dataset_type='occ3d',
                 corruption_type='origin',
                 corruption_path=None,
                 semantic_corruption_src=None,
                 semantic_corruption_dst=None,
                 ):
        self.to_long = to_long
        self.to_float = to_float
        self.dataset_type = dataset_type
        self.corruption_type = corruption_type
        self.corruption_path = corruption_path
        self.semantic_corruption_src = semantic_corruption_src
        self.semantic_corruption_dst = semantic_corruption_dst
        self.use_explicit_semantic_corruption = (
            self.semantic_corruption_src is not None and self.semantic_corruption_dst is not None
        )
        if self.corruption_type in ['discontinue', 'fragmentary', 'semantic', 'semantic_curr_only'] and not self.use_explicit_semantic_corruption:
            assert self.corruption_path is not None, 'corruption_path should be provided when corruption_type is not origin.'
            with open(self.corruption_path, 'rb') as f:
                self.corruption_data = pickle.load(f)
        else:
            # origin do not need corruption files
            self.corruption_data = None
        if (self.semantic_corruption_src is None) ^ (self.semantic_corruption_dst is None):
            raise ValueError('semantic_corruption_src and semantic_corruption_dst must be set together.')
        self.angles = None
        # waymo dataset cls map to occ3d
        self.waymo_map = {
            0: 0,  # TYPE_GENERALOBJECT
            1: 4,  # TYPE_VEHICLE
            2: 7,  # TYPE_PEDESTRIAN
            3: 15,  # TYPE_SIGN
            4: 2,  # TYPE_CYCLIST
            5: 15,  # TYPE_TRAFFIC_LIGHT
            6: 15,  # TYPE_POLE
            7: 8,  # TYPE_CONSTRUCTION_CONE
            8: 2,  # TYPE_BICYCLE
            9: 6,  # TYPE_MOTORCYCLE
            10: 15,  # TYPE_BUILDING
            11: 16,  # TYPE_VEGETATION
            12: 16,  # TYPE_TREE_TRUNK
            13: 11,  # TYPE_ROAD
            14: 13,  # TYPE_WALKABLE
            23: 17,  # TYPE_FREE
        }

    def apply_semantic_corruption(self, occ, semantic_random_pair):
        for original_label, new_label in semantic_random_pair.items():
            occ[occ == original_label] = new_label
        return occ

    def calculate_from_angle(self, start, end, shape):
        """Calculate the mask for the given angle range."""
        if self.angles is None:
            center_x = shape[1] / 2
            center_y = shape[0] / 2
            Y, X = np.ogrid[:shape[0], :shape[1]]
            angles = np.arctan2(Y - center_y, X - center_x)
            angles = (angles + 2 * np.pi) % (2 * np.pi)
        else:
            angles = self.angles

        start = start % (2 * np.pi)
        end = end % (2 * np.pi)

        if start < end:
            mask = (angles >= start) & (angles <= end)
        else:
            mask = (angles >= start) | (angles <= end)
        mask = mask.astype(np.uint8)
        mask = np.repeat(mask[:, :, np.newaxis], shape[2], axis=2)
        return mask

    def apply_corruption(self, occ, val):
        if self.corruption_type == 'fragmentary':
            for (start, end) in val:
                mask = self.calculate_from_angle(start, end, occ.shape)
                occ[mask == 1] = 17
        elif self.corruption_type in ['semantic', 'semantic_curr_only']:
            if self.semantic_corruption_src is not None:
                val = {int(self.semantic_corruption_src): int(self.semantic_corruption_dst)}
            occ = self.apply_semantic_corruption(occ, val)
        return occ

    def __call__(self, results):
        curr_occ_path = results['occ_path']
        previous_occ_path = results['previous_occ_path']
        future_occ_path = results['future_occ_path']
        # load current frame
        if self.dataset_type == 'occ3d':
            occ_gt_label = os.path.join(curr_occ_path, "labels.npz")
            occ_labels = np.load(occ_gt_label)
            curr_semantics = occ_labels['semantics']
        elif self.dataset_type == 'waymo':
            occ_labels = np.load(curr_occ_path)
            curr_semantics = occ_labels['voxel_label']
            # map the waymo cls to occ3d cls
            map_semantics = copy.deepcopy(curr_semantics)
            for key in self.waymo_map.keys():
                map_semantics[curr_semantics == key] = self.waymo_map[key]
            curr_semantics = map_semantics
        elif self.dataset_type == 'stcocc':
            curr_occ_path = curr_occ_path.replace('gts', 'stc-results')
            occ_gt_label = os.path.join(curr_occ_path, "labels.npz")
            occ_labels = np.load(occ_gt_label)
            curr_semantics = occ_labels['semantics']
        else:
            raise NotImplementedError
        curr_semantics_clean = copy.deepcopy(curr_semantics)
        
        scene_name = results['scene_name']
        sample_idx = results['sample_idx']
        # postprocess corruption
        if self.corruption_type == 'origin':
            pass
        elif self.corruption_type == 'reverse':
            pass
        elif self.corruption_type == 'discontinue':
            pass
        elif self.corruption_type == 'fragmentary':
            curr_semantics = self.apply_corruption(curr_semantics, self.corruption_data['dict']['fragmentary'][scene_name][sample_idx])
        elif self.corruption_type in ['semantic', 'semantic_curr_only']:
            semantic_val = None if self.use_explicit_semantic_corruption else self.corruption_data['dict']['semantic'][scene_name][sample_idx]
            curr_semantics = self.apply_corruption(curr_semantics, semantic_val)
        else:
            raise NotImplementedError

        # load previous frame
        previous_semantics = []
        previous_semantics_clean = []
        for path in previous_occ_path:
            if self.dataset_type == 'waymo':
                previous_occ_gt_label = path
                previous_occ_label = np.load(previous_occ_gt_label)
                previous_semantic = previous_occ_label['voxel_label']
                # map the waymo cls to occ3d cls
                map_semantics = copy.deepcopy(previous_semantic)
                for key in self.waymo_map.keys():
                    map_semantics[previous_semantic == key] = self.waymo_map[key]
                previous_semantic = map_semantics
            elif self.dataset_type == 'stcocc':
                path = path.replace('gts', 'stc-results')
                previous_occ_gt_label = os.path.join(path, "labels.npz")
                previous_occ_label = np.load(previous_occ_gt_label)
                previous_semantic = previous_occ_label['semantics']
            elif self.dataset_type == 'occ3d':
                previous_occ_gt_label = os.path.join(path, "labels.npz")
                previous_occ_label = np.load(previous_occ_gt_label)
                previous_semantic = previous_occ_label['semantics']
            else:
                raise NotImplementedError
            previous_semantics_clean.append(copy.deepcopy(previous_semantic))
            # postprocess corruption
            if self.corruption_type == 'origin':
                pass
            elif self.corruption_type == 'reverse':
                pass
            elif self.corruption_type == 'discontinue':
                pass
            elif self.corruption_type == 'fragmentary':
                previous_semantic = self.apply_corruption(previous_semantic, self.corruption_data['dict']['fragmentary'][scene_name][sample_idx])
            elif self.corruption_type == 'semantic':
                semantic_val = None if self.use_explicit_semantic_corruption else self.corruption_data['dict']['semantic'][scene_name][sample_idx]
                previous_semantic = self.apply_corruption(previous_semantic, semantic_val)
            elif self.corruption_type == 'semantic_curr_only':
                pass
            else:
                raise NotImplementedError
            previous_semantics.append(previous_semantic)

        # load future frame
        future_semantics = []
        future_semantics_clean = []
        for path in future_occ_path:
            if self.dataset_type == 'waymo':
                future_occ_gt_label = path
                future_occ_label = np.load(future_occ_gt_label)
                future_semantic = future_occ_label['voxel_label']
                # map the waymo cls to occ3d cls
                map_semantics = copy.deepcopy(future_semantic)
                for key in self.waymo_map.keys():
                    map_semantics[future_semantic == key] = self.waymo_map[key]
                future_semantic = map_semantics
            elif self.dataset_type == 'stcocc':
                path = path.replace('gts', 'stc-results')
                future_occ_gt_label = os.path.join(path, "labels.npz")
                future_occ_label = np.load(future_occ_gt_label)
                future_semantic = future_occ_label['semantics']
            elif self.dataset_type == 'occ3d':
                future_occ_gt_label = os.path.join(path, "labels.npz")
                future_occ_label = np.load(future_occ_gt_label)
                future_semantic = future_occ_label['semantics']

            future_semantics_clean.append(copy.deepcopy(future_semantic))
            future_semantics.append(future_semantic)

        occ_semantics = previous_semantics + [curr_semantics] + future_semantics
        occ_semantics_clean = previous_semantics_clean + [curr_semantics_clean] + future_semantics_clean
        occ_semantics = np.array(occ_semantics)
        occ_semantics_clean = np.array(occ_semantics_clean)
        if self.corruption_type == 'reverse':
            occ_semantics = occ_semantics[:, :, ::-1, :].copy()
            occ_semantics_clean = occ_semantics_clean[:, :, ::-1, :].copy()

        if self.to_long:
            occ_semantics = occ_semantics.astype(np.int64)
            occ_semantics_clean = occ_semantics_clean.astype(np.int64)
        if self.to_float:
            occ_semantics = occ_semantics.astype(np.float32)
            occ_semantics_clean = occ_semantics_clean.astype(np.float32)
        results['voxel_semantics'] = occ_semantics
        results['voxel_semantics_clean'] = occ_semantics_clean

        return results


@PIPELINES.register_module()
class RandomMaskStreamOcc3D(object):
    """Apply lightweight online masking to the tokenizer input only.

    This transform expects ``LoadStreamOcc3D`` to have already populated both
    ``voxel_semantics`` and ``voxel_semantics_clean``. It keeps the clean copy
    untouched and only masks ``voxel_semantics``.
    """

    def __init__(self,
                 masked_pass_prob=0.3,
                 free_class_idx=17,
                 current_frame_only=True,
                 block_mask_prob=1.0,
                 block_mask_area_range=(0.10, 0.20),
                 block_mask_num_range=(1, 2),
                 sector_mask_prob=0.0,
                 sector_width_range=(0.35, 0.70)):
        self.masked_pass_prob = masked_pass_prob
        self.free_class_idx = free_class_idx
        self.current_frame_only = current_frame_only
        self.block_mask_prob = block_mask_prob
        self.block_mask_area_range = block_mask_area_range
        self.block_mask_num_range = block_mask_num_range
        self.sector_mask_prob = sector_mask_prob
        self.sector_width_range = sector_width_range

    def __call__(self, results):
        if 'voxel_semantics' not in results or 'voxel_semantics_clean' not in results:
            return results
        if np.random.rand() >= self.masked_pass_prob:
            return results

        voxel_semantics = np.array(results['voxel_semantics'], copy=True)
        frame_indices = self._get_target_frame_indices(results, voxel_semantics.shape[0])

        for frame_idx in frame_indices:
            if np.random.rand() < self.block_mask_prob:
                voxel_semantics[frame_idx] = self._apply_block_mask(voxel_semantics[frame_idx])
            if np.random.rand() < self.sector_mask_prob:
                voxel_semantics[frame_idx] = self._apply_sector_mask(voxel_semantics[frame_idx])

        results['voxel_semantics'] = voxel_semantics
        results['mask_applied'] = True
        return results

    def _get_target_frame_indices(self, results, num_frames):
        if self.current_frame_only:
            current_idx = len(results.get('previous_occ_path', []))
            current_idx = min(max(current_idx, 0), num_frames - 1)
            return [current_idx]
        return list(range(num_frames))

    def _apply_block_mask(self, occ):
        masked = np.array(occ, copy=True)
        height, width, _ = masked.shape
        total_area = height * width
        num_blocks = np.random.randint(
            self.block_mask_num_range[0],
            self.block_mask_num_range[1] + 1,
        )

        for _ in range(num_blocks):
            area_frac = np.random.uniform(*self.block_mask_area_range) / max(num_blocks, 1)
            target_area = max(1, int(total_area * area_frac))
            aspect = np.random.uniform(0.5, 2.0)
            block_h = max(1, min(height, int(np.sqrt(target_area / aspect))))
            block_w = max(1, min(width, int(np.sqrt(target_area * aspect))))
            start_h = np.random.randint(0, max(height - block_h + 1, 1))
            start_w = np.random.randint(0, max(width - block_w + 1, 1))
            masked[start_h:start_h + block_h, start_w:start_w + block_w, :] = self.free_class_idx

        return masked

    def _apply_sector_mask(self, occ):
        masked = np.array(occ, copy=True)
        angle_start = np.random.uniform(0, 2 * np.pi)
        angle_width = np.random.uniform(*self.sector_width_range)
        angle_end = angle_start + angle_width
        sector_mask = self._calculate_sector_mask(masked.shape, angle_start, angle_end)
        masked[sector_mask == 1] = self.free_class_idx
        return masked

    def _calculate_sector_mask(self, shape, start, end):
        center_x = shape[1] / 2
        center_y = shape[0] / 2
        y_grid, x_grid = np.ogrid[:shape[0], :shape[1]]
        angles = np.arctan2(y_grid - center_y, x_grid - center_x)
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        start = start % (2 * np.pi)
        end = end % (2 * np.pi)
        if start < end:
            mask = (angles >= start) & (angles <= end)
        else:
            mask = (angles >= start) | (angles <= end)
        mask = mask.astype(np.uint8)
        return np.repeat(mask[:, :, np.newaxis], shape[2], axis=2)
    
    def apply_semantic_corruption(self, occ, semantic_random_pair):
        for original_label, new_label in semantic_random_pair.items():
            occ[occ == original_label] = new_label
        return occ

    def calculate_from_angle(self, start, end, shape):
        """Calculate the mask for the given angle range."""
        if self.angles is None:
            center_x = shape[1] / 2
            center_y = shape[0] / 2
            Y, X = np.ogrid[:shape[0], :shape[1]]
            angles = np.arctan2(Y - center_y, X - center_x)  # 计算每个点的角度
            angles = (angles + 2 * np.pi) % (2 * np.pi)  # 将角度转换为 [0, 2π] 范围内
        else:
            angles = self.angles

        start = start % (2 * np.pi)
        end = end % (2 * np.pi)

        if start < end:
            mask = (angles >= start) & (angles <= end)
        else:
            mask = (angles >= start) | (angles <= end)
        mask = mask.astype(np.uint8)
        mask = np.repeat(mask[:, :, np.newaxis], shape[2], axis=2)
        return mask

    def apply_corruption(self, occ, val):
        if self.corruption_type == 'fragmentary':
            for (start, end) in val:
                mask = self.calculate_from_angle(start, end, occ.shape)
                occ[mask == 1] = 17
        elif self.corruption_type in ['semantic', 'semantic_curr_only']:
            if self.semantic_corruption_src is not None:
                val = {int(self.semantic_corruption_src): int(self.semantic_corruption_dst)}
            occ = self.apply_semantic_corruption(occ, val)
        return occ


@PIPELINES.register_module()
class RandomDisagreementStreamOcc3D(object):
    """Create clean-only synthetic current/history disagreement online.

    This transform never reads external corruption benchmark assets. It starts
    from clean occupancy labels and perturbs only the tokenizer input while
    keeping the clean copy intact for reconstruction targets.
    """

    def __init__(self,
                 perturb_prob=0.3,
                 free_class_idx=17,
                 current_frame_only=True,
                 sample_current_vs_history=False,
                 current_frame_prob=0.5,
                 block_mask_prob=0.7,
                 block_mask_area_range=(0.04, 0.10),
                 block_mask_num_range=(1, 1),
                 semantic_swap_prob=0.3,
                 semantic_swap_area_range=(0.02, 0.06),
                 semantic_classes=None):
        self.perturb_prob = perturb_prob
        self.free_class_idx = free_class_idx
        self.current_frame_only = current_frame_only
        self.sample_current_vs_history = sample_current_vs_history
        self.current_frame_prob = current_frame_prob
        self.block_mask_prob = block_mask_prob
        self.block_mask_area_range = block_mask_area_range
        self.block_mask_num_range = block_mask_num_range
        self.semantic_swap_prob = semantic_swap_prob
        self.semantic_swap_area_range = semantic_swap_area_range
        self.semantic_classes = semantic_classes or list(range(17))

    def __call__(self, results):
        if 'voxel_semantics' not in results or 'voxel_semantics_clean' not in results:
            return results

        voxel_semantics = np.array(results['voxel_semantics'], copy=True)
        frame_reliability_map = np.ones(voxel_semantics.shape[:3], dtype=np.float32)

        if np.random.rand() >= self.perturb_prob:
            results['voxel_semantics'] = voxel_semantics
            results['frame_reliability_map'] = frame_reliability_map
            results['synthetic_disagreement_applied'] = False
            return results

        frame_indices = self._get_target_frame_indices(results, voxel_semantics.shape[0])
        frame_idx = int(np.random.choice(frame_indices))

        total_prob = max(self.block_mask_prob + self.semantic_swap_prob, 1e-6)
        block_prob = self.block_mask_prob / total_prob

        if np.random.rand() < block_prob:
            voxel_semantics[frame_idx], frame_reliability_map[frame_idx] = self._apply_block_mask(
                voxel_semantics[frame_idx], frame_reliability_map[frame_idx])
        else:
            voxel_semantics[frame_idx], frame_reliability_map[frame_idx] = self._apply_semantic_swap(
                voxel_semantics[frame_idx], frame_reliability_map[frame_idx])

        results['voxel_semantics'] = voxel_semantics
        results['frame_reliability_map'] = frame_reliability_map
        results['synthetic_disagreement_applied'] = True
        return results

    def _get_target_frame_indices(self, results, num_frames):
        if self.current_frame_only:
            current_idx = len(results.get('previous_occ_path', []))
            current_idx = min(max(current_idx, 0), num_frames - 1)
            return [current_idx]
        current_idx = len(results.get('previous_occ_path', []))
        current_idx = min(max(current_idx, 0), num_frames - 1)
        if self.sample_current_vs_history:
            history_indices = [idx for idx in range(num_frames) if idx != current_idx]
            if not history_indices or np.random.rand() < self.current_frame_prob:
                return [current_idx]
            return [int(np.random.choice(history_indices))]
        return list(range(num_frames))

    def _sample_box(self, height, width, area_range, num_parts=1):
        total_area = height * width
        area_frac = np.random.uniform(*area_range) / max(num_parts, 1)
        target_area = max(1, int(total_area * area_frac))
        aspect = np.random.uniform(0.5, 2.0)
        block_h = max(1, min(height, int(np.sqrt(target_area / aspect))))
        block_w = max(1, min(width, int(np.sqrt(target_area * aspect))))
        start_h = np.random.randint(0, max(height - block_h + 1, 1))
        start_w = np.random.randint(0, max(width - block_w + 1, 1))
        return start_h, start_w, block_h, block_w

    def _apply_block_mask(self, occ, reliability_map):
        masked = np.array(occ, copy=True)
        reliability_map = np.array(reliability_map, copy=True)
        height, width, _ = masked.shape
        num_blocks = np.random.randint(
            self.block_mask_num_range[0],
            self.block_mask_num_range[1] + 1,
        )

        for _ in range(num_blocks):
            start_h, start_w, block_h, block_w = self._sample_box(
                height, width, self.block_mask_area_range, num_parts=num_blocks)
            masked[start_h:start_h + block_h, start_w:start_w + block_w, :] = self.free_class_idx
            reliability_map[start_h:start_h + block_h, start_w:start_w + block_w] = 0.0

        return masked, reliability_map

    def _apply_semantic_swap(self, occ, reliability_map):
        swapped = np.array(occ, copy=True)
        reliability_map = np.array(reliability_map, copy=True)
        height, width, _ = swapped.shape
        start_h, start_w, block_h, block_w = self._sample_box(
            height, width, self.semantic_swap_area_range, num_parts=1)

        region = swapped[start_h:start_h + block_h, start_w:start_w + block_w, :]
        valid_mask = np.logical_and(region != self.free_class_idx, region != 255)
        if valid_mask.any():
            dst_label = int(np.random.choice(self.semantic_classes))
            region[valid_mask] = dst_label
            swapped[start_h:start_h + block_h, start_w:start_w + block_w, :] = region
            reliability_map[start_h:start_h + block_h, start_w:start_w + block_w] = 0.0
        return swapped, reliability_map

@PIPELINES.register_module()
class BEVAugStream(object):
    def __init__(self, bda_aug_conf, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy, tran_bda):
        # get rotation matrix
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([
            [rot_cos, -rot_sin, 0],
            [rot_sin, rot_cos, 0],
            [0, 0, 1]])
        scale_mat = torch.Tensor([
            [scale_ratio, 0, 0],
            [0, scale_ratio, 0],
            [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
        )

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])

        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def voxel_transform(self, results, flip_dx, flip_dy, rotate_bda=None):
        if flip_dx:
            results['voxel_semantics'] = results['voxel_semantics'][:, ::-1,...].copy()

        if flip_dy:
            results['voxel_semantics'] = results['voxel_semantics'][:, :, ::-1,...].copy()

        return results

    def __call__(self, results):
        # sample bda augmentation
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()

        # get bda matrix
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)

        # do voxel transformation
        results = self.voxel_transform(results, flip_dx=flip_dx, flip_dy=flip_dy)
        results['bda_mat'] = bda_mat

        return results

@PIPELINES.register_module()
class LoadOccGTFromFileCVPR2023(object):
    def __init__(self,
                 scale_1_2=False,
                 scale_1_4=False,
                 scale_1_8=False,
                 load_mask=False,
                 load_flow=False,
                 flow_gt_path=None,
                 ignore_invisible=False,
                 to_long=False,
                 ):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.ignore_invisible = ignore_invisible
        self.load_mask = load_mask
        self.load_flow = load_flow
        self.flow_gt_path = flow_gt_path
        self.to_long = to_long

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")

        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        if self.load_mask:
            voxel_mask = occ_labels['mask_camera']
            results['voxel_mask_camera'] = voxel_mask.astype(bool)
            if self.ignore_invisible:
                semantics[voxel_mask==0] = 255
        results['voxel_semantics'] = semantics

        if self.load_flow:
            W, H, Z = semantics.shape[0], semantics.shape[1], semantics.shape[2]
            scene_token = occ_gt_path.split('/')[-1]
            sparse_flow_path = os.path.join(self.flow_gt_path, scene_token+'.bin')
            sparse_flow_idx_path = os.path.join(self.flow_gt_path, scene_token+'_idx.bin')
            occ_flow = np.zeros((W*H*Z, 2), dtype=np.float16)
            sparse_flow = np.fromfile(sparse_flow_path, dtype=np.float16).reshape(-1, 3)[:, :2]
            sparse_idx = np.fromfile(sparse_flow_idx_path, dtype=np.int32).reshape(-1)
            occ_flow[sparse_idx] = sparse_flow
            occ_flow = occ_flow.reshape(W, H, Z, 2)
            if self.ignore_invisible:
                occ_flow[voxel_mask==0] = 255
            results['voxel_flow'] = occ_flow

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_2['mask_camera']
                if self.ignore_invisible:
                    semantics_1_2[voxel_mask==0] = 255
                results['voxel_mask_camera_1_2'] = voxel_mask
            results['voxel_semantics_1_2'] = semantics_1_2
        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_4['mask_camera']
                if self.ignore_invisible:
                    semantics_1_4[voxel_mask==0] = 255
                results['voxel_mask_camera_1_4'] = voxel_mask
            results['voxel_semantics_1_4'] = semantics_1_4

        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_8['mask_camera']
                if self.ignore_invisible:
                    semantics_1_8[voxel_mask==0] = 255
                results['voxel_mask_camera_1_8'] = voxel_mask
            results['voxel_semantics_1_8'] = semantics_1_8

        if self.to_long:
            results['voxel_semantics'] = results['voxel_semantics'].astype(np.int64)
            if self.scale_1_2:
                results['voxel_semantics_1_2'] = results['voxel_semantics_1_2'].astype(np.int64)
            if self.scale_1_4:
                results['voxel_semantics_1_4'] = results['voxel_semantics_1_4'].astype(np.int64)
            if self.scale_1_8:
                results['voxel_semantics_1_8'] = results['voxel_semantics_1_8'].astype(np.int64)

        return results

@PIPELINES.register_module()
class LoadOccGTFromFileOpenOcc(object):
    def __init__(self, scale_1_2=False, scale_1_4=False, scale_1_8=False, load_ray_mask=False):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.load_ray_mask = load_ray_mask

    def __call__(self, results):
        gts_occ_gt_path = results['occ_gt_path']

        occ_ray_mask_path = gts_occ_gt_path.replace('gts', 'openocc_v2_ray_mask')
        occ_ray_mask = os.path.join(occ_ray_mask_path, 'labels.npz')
        occ_ray_mask_1_2 = os.path.join(occ_ray_mask_path, 'labels_1_2.npz')
        occ_ray_mask_1_4 = os.path.join(occ_ray_mask_path, 'labels_1_4.npz')
        occ_ray_mask_1_8 = os.path.join(occ_ray_mask_path, 'labels_1_8.npz')

        occ_gt_path = gts_occ_gt_path.replace('gts', 'openocc_v2')
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")
        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        flow = occ_labels['flow']

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']
            flow_1_2 = occ_labels_1_2['flow']
            results['voxel_semantics_1_2'] = semantics_1_2
            results['voxel_flow_1_2'] = flow_1_2
            if self.load_ray_mask:
                ray_mask_1_2 = np.load(occ_ray_mask_1_2)
                ray_mask_1_2 = ray_mask_1_2['ray_mask2']
                results['ray_mask_1_2'] = ray_mask_1_2
        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']
            flow_1_4 = occ_labels_1_4['flow']
            results['voxel_semantics_1_4'] = semantics_1_4
            results['voxel_flow_1_4'] = flow_1_4
            if self.load_ray_mask:
                ray_mask_1_4 = np.load(occ_ray_mask_1_4)
                ray_mask_1_4 = ray_mask_1_4['ray_mask2']
                results['ray_mask_1_4'] = ray_mask_1_4
        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']
            flow_1_8 = occ_labels_1_8['flow']
            results['voxel_semantics_1_8'] = semantics_1_8
            results['voxel_flow_1_8'] = flow_1_8
            if self.load_ray_mask:
                ray_mask_1_8 = np.load(occ_ray_mask_1_8)
                ray_mask_1_8 = ray_mask_1_8['ray_mask2']
                results['ray_mask_1_8'] = ray_mask_1_8

        if self.load_ray_mask:
            ray_mask = np.load(occ_ray_mask)
            ray_mask = ray_mask['ray_mask2']
            results['ray_mask'] = ray_mask

        results['voxel_semantics'] = semantics
        results['voxel_flows'] = flow

        return results


@PIPELINES.register_module()
class LoadAnnotations(object):

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        return results
