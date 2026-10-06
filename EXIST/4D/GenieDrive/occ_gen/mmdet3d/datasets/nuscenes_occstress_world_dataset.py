# OccStress adapter/portability modifications; see docs/source-imports.json.
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from pathlib import Path

import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.geometry_utils import transform_matrix
from terminaltables import AsciiTable
from tqdm import tqdm

from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.utils import get_root_logger

from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .nuscenes_world_dataset import NuScenesWorldDataset
from .occ_metrics import Metric_mIoU
from .utils import nuscenes_get_rt_matrix


ROBUSTOCC_ROOT = Path(
    os.environ.get('OCCSTRESS_CODE_ROOT', Path(__file__).resolve().parents[6]))
OCCSTRESS_ROOT = Path(
    os.environ.get('OCCSTRESS_ROOT', os.environ.get(
        'OCCSTRESS_DATA_ROOT', ROBUSTOCC_ROOT / 'data' / 'OccStress')))
MIRROR_Y_4X4 = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
MIRROR_Y_3X3 = np.diag([1.0, -1.0, 1.0]).astype(np.float32)


def _resolve_path(path):
    from occstress.datasets.paths import resolve_occstress_path
    return str(resolve_occstress_path(path, code_root=ROBUSTOCC_ROOT, dataset='nuscenes'))


def _resolve_occ_npz(path):
    path = Path(_resolve_path(path))
    if path.suffix == '.npz':
        return str(path)
    return str(path / 'labels.npz')


def _identity_rt():
    return np.eye(4, dtype=np.float32)


def _is_traffic_mirror(record):
    corruption = record.get('corruption', {}) if record is not None else {}
    return corruption.get('type') == 'traffic'


def _mirror_y_rt(rt):
    rt = np.asarray(rt, dtype=np.float32)
    return (MIRROR_Y_4X4 @ rt @ MIRROR_Y_4X4).astype(np.float32)


def _mirror_y_translation(translation):
    translation = np.asarray(translation, dtype=np.float32).copy()
    translation[..., 1] *= -1.0
    return translation


def _mirror_y_quaternion(quaternion):
    rotation = pyquaternion.Quaternion(quaternion).rotation_matrix.astype(np.float64)
    mirror = MIRROR_Y_3X3.astype(np.float64)
    mirrored = mirror @ rotation @ mirror
    u, _, vh = np.linalg.svd(mirrored)
    mirrored = u @ vh
    if np.linalg.det(mirrored) < 0:
        u[:, -1] *= -1
        mirrored = u @ vh
    return np.array(pyquaternion.Quaternion(matrix=mirrored).elements, dtype=np.float32)


def _quaternion_from_rt(rt):
    rotation = np.asarray(rt, dtype=np.float64)[:3, :3]
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return np.array(
        pyquaternion.Quaternion(matrix=rotation).elements, dtype=np.float32)


def _mirror_y_can_bus(can_bus):
    can_bus = np.asarray(can_bus, dtype=np.float32).copy()
    if can_bus.shape[0] >= 2:
        can_bus[1] *= -1.0
    if can_bus.shape[0] >= 7:
        can_bus[3:7] = _mirror_y_quaternion(can_bus[3:7])
    if can_bus.shape[0] >= 9:
        can_bus[8] *= -1.0
    if can_bus.shape[0] >= 13:
        can_bus[10] *= -1.0
        can_bus[12] *= -1.0
    if can_bus.shape[0] >= 15:
        can_bus[14] *= -1.0
    if can_bus.shape[0] >= 17:
        can_bus[-2] *= -1.0
        can_bus[-1] *= -1.0
    return can_bus


def _mirror_y_ego_lcf_feat(ego_lcf_feat):
    ego_lcf_feat = np.asarray(ego_lcf_feat, dtype=np.float32).copy()
    feat_dim = ego_lcf_feat.shape[-1]
    if feat_dim >= 2:
        ego_lcf_feat[..., 1] *= -1.0
    if feat_dim == 3:
        ego_lcf_feat[..., 2] *= -1.0
    elif feat_dim >= 9:
        ego_lcf_feat[..., 3] *= -1.0
        ego_lcf_feat[..., 4] *= -1.0
        ego_lcf_feat[..., 8] *= -1.0
    return ego_lcf_feat


def _mirror_y_ego_trajs(trajs):
    trajs = np.asarray(trajs, dtype=np.float32).copy()
    if trajs.shape[-1] >= 2:
        trajs[..., 1] *= -1.0
    return trajs


def _mirror_y_ego_cmd(cmd):
    cmd = np.asarray(cmd, dtype=np.float32).copy()
    if cmd.shape[-1] >= 2:
        cmd[..., [0, 1]] = cmd[..., [1, 0]]
    return cmd


def _mirror_y_box_array(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if boxes.size == 0:
        return boxes
    boxes[:, 1] *= -1.0
    if boxes.shape[1] > 6:
        boxes[:, 6] *= -1.0
    if boxes.shape[1] > 8:
        boxes[:, 8] *= -1.0
    return boxes


def _mirror_y_agent_attr(gt_fut_trajs, gt_fut_goal, gt_lcf_feat, gt_fut_yaw):
    gt_fut_trajs = np.asarray(gt_fut_trajs, dtype=np.float32).copy()
    gt_lcf_feat = np.asarray(gt_lcf_feat, dtype=np.float32).copy()
    gt_fut_yaw = np.asarray(gt_fut_yaw, dtype=np.float32).copy()
    gt_fut_goal = np.asarray(gt_fut_goal, dtype=np.float32).copy()

    gt_fut_trajs[:, 1::2] *= -1.0
    gt_lcf_feat[:, 1] *= -1.0
    gt_lcf_feat[:, 2] *= -1.0
    gt_lcf_feat[:, 4] *= -1.0
    gt_fut_yaw *= -1.0

    moving = (gt_fut_goal >= 0) & (gt_fut_goal < 8)
    gt_fut_goal[moving] = 7 - gt_fut_goal[moving]
    return gt_fut_trajs, gt_fut_goal, gt_lcf_feat, gt_fut_yaw


def _load_misalignment_cache(record):
    corruption = record.get('corruption', {})
    if corruption.get('type') != 'misalignment':
        return None
    severity = corruption.get('severity')
    if severity is None:
        return None
    cache_path = OCCSTRESS_ROOT / 'cache' / 'misalignment' / severity / record[
        'scene_name'] / f"{record['sample_id']}.npz"
    if not cache_path.exists():
        return None
    return np.load(cache_path, allow_pickle=True)


def _build_anchor_to_history_rts(record, token_to_info):
    anchor_info = token_to_info[record['anchor_token']]
    history_rts = []
    for hist_token in record['history_tokens']:
        hist_info = token_to_info[hist_token]
        history_rts.append(
            nuscenes_get_rt_matrix(anchor_info, hist_info, 'ego', 'ego').astype(np.float32))

    cache = _load_misalignment_cache(record)
    if cache is not None:
        misaligned = cache['rt_misaligned'].astype(np.float32)
        clean = cache['rt_clean'].astype(np.float32)
    elif (record.get('misalignment') or {}).get('delta_rt') is not None:
        clean = np.asarray(history_rts, dtype=np.float32)
        delta = np.asarray(
            record['misalignment']['delta_rt'], dtype=np.float32)
        if delta.shape != clean.shape:
            raise ValueError(
                f"{record['sample_id']} has delta_rt shape {delta.shape}, "
                f'expected {clean.shape}')
        misaligned = delta @ clean
    else:
        clean = None
        misaligned = None

    if misaligned is None:
        resolved = history_rts
    else:
        resolved = []
        for idx, hist in enumerate(record['history']):
            if hist.get('rt_source') == 'misalignment':
                resolved.append(misaligned[idx])
            else:
                resolved.append(clean[idx] if idx < len(clean) else history_rts[idx])

    if _is_traffic_mirror(record):
        resolved = [_mirror_y_rt(rt) for rt in resolved]
    return resolved


def _build_stepwise_rts(anchor_to_frames):
    sequence_rts = []
    for idx, curr_anchor_to_frame in enumerate(anchor_to_frames):
        if idx == 0:
            sequence_rts.append(_identity_rt())
            continue
        prev_anchor_to_frame = anchor_to_frames[idx - 1]
        curr_to_prev = prev_anchor_to_frame @ np.linalg.inv(curr_anchor_to_frame)
        sequence_rts.append(curr_to_prev.astype(np.float32))
    return sequence_rts


@DATASETS.register_module()
class OccStressNuScenesWorldDataset(NuScenesWorldDataset):

    def __init__(self,
                 protocol_path,
                 native_history_frames=3,
                 max_samples=None,
                 *args,
                 **kwargs):
        self.protocol_path = _resolve_path(protocol_path)
        self.native_history_frames = native_history_frames
        self.max_samples = max_samples
        self.expected_future_frames = kwargs.get('load_future_frame_number', 0)
        self.base_ann_file = kwargs.get('ann_file')
        self.protocol_records = None
        self.base_data_infos = None
        self.base_metadata = None
        self.token_to_info = None
        self.token_to_index = None
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        base = mmcv.load(ann_file, file_format='pkl')
        base_infos = list(sorted(base['infos'], key=lambda e: e['timestamp']))
        base_infos = base_infos[::self.load_interval]
        self.base_data_infos = base_infos
        self.base_metadata = base['metadata']
        self.metadata = base['metadata']
        self.version = self.metadata['version']
        self.token_to_info = {info['token']: info for info in self.base_data_infos}
        self.token_to_index = {info['token']: idx for idx, info in enumerate(self.base_data_infos)}

        records = mmcv.load(self.protocol_path, file_format='pkl')
        records = list(sorted(records, key=lambda e: e['anchor_timestamp']))
        if self.max_samples is not None:
            records = records[:self.max_samples]
        for record in records:
            if record['history_length'] < self.native_history_frames:
                raise ValueError(
                    f"{record['sample_id']} has only {record['history_length']} history frames")
            if record['future_length'] != self.expected_future_frames:
                raise ValueError(
                    f"{record['sample_id']} has {record['future_length']} future frames, "
                    f"expected {self.expected_future_frames}")
        self.protocol_records = records
        self.flag = np.arange(len(records), dtype=np.int64)
        return records

    def _get_base_info(self, token):
        return self.token_to_info[token]

    def _build_future_info(self, record):
        mirror_y = _is_traffic_mirror(record)
        anchor_info = self._get_base_info(record['anchor_token'])
        future_occ_path = []
        future_occ_index = []
        curr_to_future_ego_rt = []
        curr_ego_to_global_rt = []
        ego_to_global_rotation = []
        ego_to_global_translation = []
        pose_mat = []
        ego_to_lidar = []

        anchor_ego_to_global = transform_matrix(
            anchor_info['ego2global_translation'],
            pyquaternion.Quaternion(anchor_info['ego2global_rotation']))
        anchor_ego_to_lidar = transform_matrix(
            anchor_info['lidar2ego_translation'],
            pyquaternion.Quaternion(anchor_info['lidar2ego_rotation']),
            inverse=True)
        curr_ego_to_global_rt.append(_mirror_y_rt(anchor_ego_to_global) if mirror_y else anchor_ego_to_global)
        ego_to_global_rotation.append(
            _mirror_y_quaternion(anchor_info['ego2global_rotation']) if mirror_y else anchor_info['ego2global_rotation'])
        ego_to_global_translation.append(
            _mirror_y_translation(anchor_info['ego2global_translation']) if mirror_y else anchor_info['ego2global_translation'])
        pose_mat.append(_mirror_y_rt(anchor_info['pose_mat']) if mirror_y else anchor_info['pose_mat'])
        ego_to_lidar.append(_mirror_y_rt(anchor_ego_to_lidar) if mirror_y else anchor_ego_to_lidar)

        prev_info = anchor_info
        for future in record['future_targets']:
            future_info = self._get_base_info(future['token'])
            future_occ_path.append(_resolve_occ_npz(future['occ_path']))
            future_occ_index.append(self.token_to_index[future['token']])
            step_rt = nuscenes_get_rt_matrix(prev_info, future_info, 'ego', 'ego').astype(np.float32)
            future_ego_to_global = transform_matrix(
                future_info['ego2global_translation'],
                pyquaternion.Quaternion(future_info['ego2global_rotation']))
            future_ego_to_lidar = transform_matrix(
                future_info['lidar2ego_translation'],
                pyquaternion.Quaternion(future_info['lidar2ego_rotation']),
                inverse=True)
            curr_to_future_ego_rt.append(_mirror_y_rt(step_rt) if mirror_y else step_rt)
            curr_ego_to_global_rt.append(_mirror_y_rt(future_ego_to_global) if mirror_y else future_ego_to_global)
            ego_to_global_rotation.append(
                _mirror_y_quaternion(future_info['ego2global_rotation']) if mirror_y else future_info['ego2global_rotation'])
            ego_to_global_translation.append(
                _mirror_y_translation(future_info['ego2global_translation']) if mirror_y else future_info['ego2global_translation'])
            pose_mat.append(_mirror_y_rt(future_info['pose_mat']) if mirror_y else future_info['pose_mat'])
            ego_to_lidar.append(_mirror_y_rt(future_ego_to_lidar) if mirror_y else future_ego_to_lidar)
            prev_info = future_info

        return dict(
            future_occ_path=future_occ_path,
            future_occ_index=future_occ_index,
            curr_to_future_ego_rt=np.array(curr_to_future_ego_rt, dtype=np.float32),
            curr_ego_to_global=np.array(curr_ego_to_global_rt, dtype=np.float32),
            ego_to_global_rotation=np.array(ego_to_global_rotation, dtype=np.float32),
            ego_to_global_translation=np.array(ego_to_global_translation, dtype=np.float32),
            pose_mat=np.array(pose_mat, dtype=np.float32),
            valid_frame=np.ones(record['future_length'], dtype=np.bool_),
            ego_to_lidar=np.array(ego_to_lidar, dtype=np.float32),
        )

    def _build_previous_info(self, record):
        selected_history = record['history'][-self.native_history_frames:]
        selected_tokens = record['history_tokens'][-self.native_history_frames:]
        previous_occ_path = [_resolve_occ_npz(hist['occ_path']) for hist in selected_history]
        anchor_to_history = _build_anchor_to_history_rts(record, self.token_to_info)
        current_anchor_rt = _identity_rt()
        misalignment = record.get('misalignment') or {}
        if misalignment.get('current_active'):
            current_anchor_rt = np.array(
                misalignment.get('current_delta_rt', _identity_rt().tolist()),
                dtype=np.float32)
        anchor_to_frames = anchor_to_history + [current_anchor_rt]
        stepwise_rts = _build_stepwise_rts(anchor_to_frames)
        previous_curr_to_prev_ego_rt = stepwise_rts[-(self.native_history_frames + 1):-1]
        previous_occ_index = [self.token_to_index[token] for token in selected_tokens]
        return dict(
            previous_occ_path=previous_occ_path,
            previous_curr_to_prev_ego_rt=previous_curr_to_prev_ego_rt,
            previous_occ_index=previous_occ_index,
            curr_to_prev_ego_rt=stepwise_rts[-1],
        )

    def _build_protocol_observe_info(self, record):
        """Build the state that the official sequential evaluator would cache."""
        anchor_info = self._get_base_info(record['anchor_token'])
        anchor_to_history = _build_anchor_to_history_rts(record, self.token_to_info)
        current_anchor_rt = _identity_rt()
        misalignment = record.get('misalignment') or {}
        if misalignment.get('current_active'):
            current_anchor_rt = np.array(
                misalignment.get('current_delta_rt', _identity_rt().tolist()),
                dtype=np.float32)

        anchor_global = transform_matrix(
            anchor_info['ego2global_translation'],
            pyquaternion.Quaternion(anchor_info['ego2global_rotation']))
        if _is_traffic_mirror(record):
            anchor_global = _mirror_y_rt(anchor_global)

        global_poses = [
            anchor_global @ np.linalg.inv(anchor_to_frame)
            for anchor_to_frame in anchor_to_history + [current_anchor_rt]
        ]
        rotations = np.array(
            [_quaternion_from_rt(pose) for pose in global_poses],
            dtype=np.float32)
        translations = np.array(
            [pose[:3, 3] for pose in global_poses], dtype=np.float32)

        ego_lcf = []
        for token in record['history_tokens']:
            feat = self._get_base_info(token)['gt_ego_lcf_feat']
            compact = np.array([feat[0], feat[1], feat[4]], dtype=np.float32)
            if _is_traffic_mirror(record):
                compact = _mirror_y_ego_lcf_feat(compact)
            ego_lcf.append(compact)

        observe_count = self.native_history_frames + 1
        return dict(
            protocol_observe_rotations=rotations[-(observe_count + 1):],
            protocol_observe_translations=translations[-(observe_count + 1):],
            protocol_observe_ego_lcf_feat=np.array(
                ego_lcf[-observe_count:], dtype=np.float32),
            protocol_mode=True,
            observed_offsets=(-1.5, -1.0, -0.5, 0.0),
        )

    def _build_ego_trajs_info(self, record):
        mirror_y = _is_traffic_mirror(record)
        infos = [self._get_base_info(record['anchor_token'])]
        for future in record['future_targets'][:-1]:
            infos.append(self._get_base_info(future['token']))

        gt_ego_fut_trajs = []
        gt_ego_fut_cmd = []
        gt_ego_lcf_feat = []
        for info in infos:
            trajs = info['gt_ego_fut_trajs']
            gt_ego_fut_trajs.append(_mirror_y_ego_trajs(trajs[0]) if mirror_y else trajs[0])
            gt_ego_fut_cmd.append(_mirror_y_ego_cmd(info['gt_ego_fut_cmd']) if mirror_y else info['gt_ego_fut_cmd'])
            ego_feat = info['gt_ego_lcf_feat']
            ego_lcf_feat = np.array([ego_feat[0], ego_feat[1], ego_feat[4]], dtype=np.float32)
            gt_ego_lcf_feat.append(_mirror_y_ego_lcf_feat(ego_lcf_feat) if mirror_y else ego_lcf_feat)

        return dict(
            gt_ego_fut_trajs=np.array(gt_ego_fut_trajs, dtype=np.float32),
            gt_ego_fut_cmd=np.array(gt_ego_fut_cmd, dtype=np.float32),
            gt_ego_lcf_feat=np.array(gt_ego_lcf_feat, dtype=np.float32),
        )

    def _build_box_info(self, token, record=None):
        mirror_y = _is_traffic_mirror(record)
        info = self._get_base_info(token)
        mask = info['valid_flag']
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_velocity = info['gt_velocity'][mask]
        nan_mask = np.isnan(gt_velocity[:, 0])
        gt_velocity[nan_mask] = [0.0, 0.0]
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        if mirror_y:
            gt_bboxes_3d = _mirror_y_box_array(gt_bboxes_3d)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
        gt_fut_masks = info['gt_agent_fut_masks'][mask]
        gt_fut_goal = info['gt_agent_fut_goal'][mask]
        gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
        gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
        if mirror_y:
            gt_fut_trajs, gt_fut_goal, gt_lcf_feat, gt_fut_yaw = _mirror_y_agent_attr(
                gt_fut_trajs, gt_fut_goal, gt_lcf_feat, gt_fut_yaw)
        attr_labels = np.concatenate(
            [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
        ).astype(np.float32)
        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_names_3d=gt_names_3d,
            gt_attr_labels=attr_labels,
            fut_valid_flag=mask,
        )

    def get_data_info(self, index):
        record = self.data_infos[index]
        mirror_y = _is_traffic_mirror(record)
        anchor_info = self._get_base_info(record['anchor_token'])
        current_input = record.get('current_input', record['target'])
        selected_history_tokens = record['history_tokens'][-self.native_history_frames:]
        scene_name = record['scene_name']
        input_dict = dict(
            index=index,
            occ_path=_resolve_occ_npz(current_input['occ_path']),
            sample_idx=record['anchor_token'],
            pts_filename=anchor_info['lidar_path'],
            sweeps=anchor_info['sweeps'],
            timestamp=anchor_info['timestamp'] / 1e6,
            can_bus=_mirror_y_can_bus(anchor_info['can_bus']) if mirror_y else anchor_info['can_bus'],
            prev=anchor_info['prev'],
            scene_name=scene_name,
            sample_weight=1.0,
            ego_from_sensor=_mirror_y_rt(anchor_info['ego_from_sensor']) if mirror_y else anchor_info['ego_from_sensor'],
            protocol_sample_id=record['sample_id'],
            occstress_protocol=record,
            occstress_corruption=record['corruption'],
            occstress_misalignment=record.get('misalignment'),
            start_of_sequence=True,
            sequence_group_idx=index,
            history_tokens=selected_history_tokens,
            protocol_history_tokens=record['history_tokens'],
            future_tokens=record['future_tokens'],
            future_length=record['future_length'],
            anchor_token=record['anchor_token'],
            occ_index=selected_history_tokens + [record['anchor_token']] + record['future_tokens'],
        )
        input_dict.update(self._build_previous_info(record))
        input_dict.update(self._build_future_info(record))
        input_dict.update(self._build_ego_trajs_info(record))
        input_dict.update(self._build_box_info(record['anchor_token'], record))
        input_dict.update(self._build_protocol_observe_info(record))
        return input_dict

    def evaluate_miou(self, results, logger=None):
        pred_sems, data_index, times = [], [], []
        num_classes = 17 if self.dataset_name == 'openocc' else 18
        self.miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=False,
            logger=logger)

        processed_set = set()
        for result in results:
            for i, idx in enumerate(result['index']):
                if idx in processed_set:
                    continue
                processed_set.add(idx)
                data_index.append(idx)
                pred_sems.append(result['semantics'][i])
                times.append(result['time'] if 'time' in result else 0)

        if times:
            print(f'Average time: {sum(times) / len(times)}')

        for idx in tqdm(data_index):
            record = self.data_infos[idx]
            gt_semantics = np.load(_resolve_occ_npz(record['target']['occ_path']), allow_pickle=True)['semantics']
            pr_semantics = pred_sems[data_index.index(idx)]
            self.miou_metric.add_batch(pr_semantics, gt_semantics, None, None)
            self.miou_metric.add_iou_batch(pr_semantics, gt_semantics, None, None)

        _, miou, _, _, _ = self.miou_metric.count_miou()
        iou = self.miou_metric.count_iou()
        return dict(semantics_miou=miou, binary_iou=iou)


@DATASETS.register_module()
class OccStressNuScenesTokenizerDataset(Custom3DDataset):

    def __init__(self,
                 protocol_path,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 test_mode=False,
                 filter_empty_gt=False,
                 dataset_name='occ3d',
                 eval_metric='miou',
                 **kwargs):
        self.protocol_path = _resolve_path(protocol_path)
        self.base_ann_file = ann_file
        self.load_interval = load_interval
        self.base_data_infos = None
        self.token_to_info = None
        self.token_to_index = None
        self.anchor_indices = set()
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        kwargs.pop('load_future_frame_number', None)
        kwargs.pop('load_previous_frame_number', None)
        kwargs.pop('load_previous_data', None)
        kwargs.pop('use_sequence_group_flag', None)
        kwargs.pop('sequences_split_num', None)
        kwargs.setdefault('box_type_3d', 'LiDAR')
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            **kwargs)
        self.box_mode_3d = Box3DMode.LIDAR

    def load_annotations(self, ann_file):
        base = mmcv.load(ann_file, file_format='pkl')
        self.base_data_infos = list(sorted(base['infos'], key=lambda e: e['timestamp']))
        self.base_data_infos = self.base_data_infos[::self.load_interval]
        self.metadata = base['metadata']
        self.version = self.metadata['version']
        self.token_to_info = {info['token']: info for info in self.base_data_infos}
        self.token_to_index = {info['token']: idx for idx, info in enumerate(self.base_data_infos)}

        records = mmcv.load(self.protocol_path, file_format='pkl')
        records = list(sorted(records, key=lambda e: e['anchor_timestamp']))
        expanded = []
        global_index = 0
        for clip_index, record in enumerate(records):
            mirror_y = _is_traffic_mirror(record)
            anchor_to_history = _build_anchor_to_history_rts(record, self.token_to_info)
            current_anchor_rt = _identity_rt()
            misalignment = record.get('misalignment') or {}
            if misalignment.get('current_active'):
                current_anchor_rt = np.array(
                    misalignment.get('current_delta_rt', _identity_rt().tolist()),
                    dtype=np.float32)
            anchor_to_frames = anchor_to_history + [current_anchor_rt]
            stepwise_rts = _build_stepwise_rts(anchor_to_frames)
            current_input = record.get('current_input', record['target'])
            frame_tokens = record['history_tokens'] + [record['anchor_token']]
            frame_paths = [hist['occ_path'] for hist in record['history']] + [current_input['occ_path']]
            frame_sources = [hist['occ_source'] for hist in record['history']] + [current_input['source']]
            frame_save_flags = [False] * len(record['history']) + [True]
            for frame_idx, (token, occ_path, curr_to_prev_ego_rt, save_flag, occ_source) in enumerate(
                    zip(frame_tokens, frame_paths, stepwise_rts, frame_save_flags, frame_sources)):
                info = self.token_to_info[token]
                expanded.append(dict(
                    index=global_index,
                    clip_index=clip_index,
                    protocol_sample_id=record['sample_id'],
                    sample_idx=token,
                    anchor_token=record['anchor_token'],
                    scene_name=record['scene_name'],
                    occ_path=_resolve_occ_npz(occ_path),
                    timestamp=info['timestamp'] / 1e6,
                    pts_filename=info['lidar_path'],
                    sweeps=info['sweeps'],
                    can_bus=_mirror_y_can_bus(info['can_bus']) if mirror_y else info['can_bus'],
                    prev=info['prev'],
                    start_of_sequence=frame_idx == 0,
                    curr_to_prev_ego_rt=curr_to_prev_ego_rt.astype(np.float32),
                    sequence_group_idx=clip_index,
                    occstress_save_token=save_flag,
                    occstress_protocol=record,
                    occstress_corruption=record['corruption'],
                    occstress_occ_source=occ_source,
                    gt_occ_path=_resolve_occ_npz(record['target']['occ_path']) if save_flag else _resolve_occ_npz(occ_path),
                    ego_from_sensor=_mirror_y_rt(info['ego_from_sensor']) if mirror_y else info['ego_from_sensor'],
                ))
                if save_flag:
                    self.anchor_indices.add(global_index)
                global_index += 1
        self.flag = np.array([item['sequence_group_idx'] for item in expanded], dtype=np.int64)
        return expanded

    def get_data_info(self, index):
        return copy.deepcopy(self.data_infos[index])

    def evaluate(self, results, logger=None, runner=None, show_dir=None, **eval_kwargs):
        pred_sems, data_index, times = [], [], []
        num_classes = 17 if self.dataset_name == 'openocc' else 18
        metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=False,
            logger=logger)

        processed_set = set()
        for result in results:
            for i, idx in enumerate(result['index']):
                if idx not in self.anchor_indices or idx in processed_set:
                    continue
                processed_set.add(idx)
                data_index.append(idx)
                pred_sems.append(result['semantics'][i])
                times.append(result['time'] if 'time' in result else 0)

        if times:
            print(f'Average time: {sum(times) / len(times)}')

        for idx in tqdm(data_index):
            info = self.data_infos[idx]
            gt_occ_path = info.get('gt_occ_path', info['occ_path'])
            gt_semantics = np.load(gt_occ_path, allow_pickle=True)['semantics']
            pr_semantics = pred_sems[data_index.index(idx)]
            metric.add_batch(pr_semantics, gt_semantics, None, None)
            metric.add_iou_batch(pr_semantics, gt_semantics, None, None)

        _, miou, _, _, _ = metric.count_miou()
        iou = metric.count_iou()
        return dict(semantics_miou=miou, binary_iou=iou)
