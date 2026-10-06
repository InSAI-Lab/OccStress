# OccStress adapter/portability modifications; see docs/source-imports.json.
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import numpy as np

from ..builder import PIPELINES


def _resolve_occ_npz(path):
    if path.endswith('.npz'):
        return path
    return os.path.join(path, 'labels.npz')


@PIPELINES.register_module()
class OccStressLoadStreamOcc3D(object):
    def __init__(self, to_long=True, to_float=False, dataset_type='occ3d'):
        self.to_long = to_long
        self.to_float = to_float
        self.dataset_type = dataset_type
        self.waymo_map = {
            0: 0,
            1: 4,
            2: 7,
            3: 15,
            4: 2,
            5: 15,
            6: 15,
            7: 8,
            8: 2,
            9: 6,
            10: 15,
            11: 16,
            12: 16,
            13: 11,
            14: 13,
            23: 17,
        }

    def _load_semantics(self, path):
        if self.dataset_type == 'occ3d':
            occ_labels = np.load(_resolve_occ_npz(path))
            semantics = occ_labels['semantics']
        elif self.dataset_type == 'stcocc':
            path = path.replace('gts', 'stc-results')
            occ_labels = np.load(_resolve_occ_npz(path))
            semantics = occ_labels['semantics']
        elif self.dataset_type in {'waymo', 'waymo_occstress'}:
            occ_labels = np.load(path)
            if 'semantics' in occ_labels:
                semantics = occ_labels['semantics']
            else:
                semantics = occ_labels['voxel_label']
                map_semantics = copy.deepcopy(semantics)
                for key, value in self.waymo_map.items():
                    map_semantics[semantics == key] = value
                semantics = map_semantics
        else:
            raise NotImplementedError
        return semantics

    def __call__(self, results):
        curr_occ_path = results['occ_path']
        previous_occ_path = results.get('previous_occ_path', [])
        future_occ_path = results.get('future_occ_path', [])

        curr_semantics = self._load_semantics(curr_occ_path)
        previous_semantics = [self._load_semantics(path) for path in previous_occ_path]
        future_semantics = [self._load_semantics(path) for path in future_occ_path]
        occ_semantics = np.array(previous_semantics + [curr_semantics] + future_semantics)

        if self.to_long:
            occ_semantics = occ_semantics.astype(np.int64)
        if self.to_float:
            occ_semantics = occ_semantics.astype(np.float32)
        results['voxel_semantics'] = occ_semantics
        return results


@PIPELINES.register_module()
class OccStressLoadStreamLatentToken(object):
    def __init__(self, current_data_path, future_data_path=None, dataset_type='occ3d'):
        self.current_data_path = current_data_path
        self.future_data_path = future_data_path or current_data_path
        self.current_data_paths = self._as_list(current_data_path)
        self.future_data_paths = self._as_list(self.future_data_path)
        self.dataset_type = dataset_type

    @staticmethod
    def _as_list(paths):
        if isinstance(paths, (list, tuple)):
            return list(paths)
        return [paths]

    def _latent_path(self, root, scene_name, sample_name):
        return os.path.join(root, str(scene_name), f'{sample_name}.npz')

    def _load_first_existing(self, roots, scene_name, sample_name):
        for root in roots:
            path = self._latent_path(root, scene_name, sample_name)
            if os.path.exists(path):
                return path, np.load(path)
        return None, None

    def __call__(self, results):
        scene_name = results['scene_name']
        if self.dataset_type == 'waymo_occstress':
            scene_name = str(scene_name).zfill(3)
        current_name_candidates = []
        if self.dataset_type == 'waymo_occstress' and results.get('current_latent_name'):
            current_name_candidates.append(results['current_latent_name'])
        protocol_sample_id = results.get('protocol_sample_id')
        if protocol_sample_id:
            current_name_candidates.append(protocol_sample_id)
        current_name_candidates.append(results['sample_idx'])

        latent_token = None
        current_path = None
        for current_name in current_name_candidates:
            current_path, latent_token = self._load_first_existing(
                self.current_data_paths, scene_name, current_name)
            if latent_token is not None:
                break
        if latent_token is None:
            raise FileNotFoundError(
                f'Could not resolve current latent token for scene={scene_name}. '
                f'Tried names={current_name_candidates} under roots={self.current_data_paths}'
            )
        token = latent_token['token'][None]

        future_tokens = []
        explicit_future_tokens = results.get('future_tokens')
        future_occ_path = results.get('future_occ_path', [])
        future_names = explicit_future_tokens or [
            os.path.splitext(os.path.basename(path))[0] for path in future_occ_path
        ]
        if self.dataset_type == 'waymo_occstress':
            future_names = results.get('future_latent_names') or future_names
        for future_name in future_names:
            latent_token_path, latent_token = self._load_first_existing(
                self.future_data_paths, scene_name, future_name)
            if latent_token is None:
                raise FileNotFoundError(
                    f'Could not resolve future latent token for scene={scene_name}, '
                    f'name={future_name}, roots={self.future_data_paths}'
                )
            future_tokens.append(latent_token['token'])

        if future_tokens:
            future_tokens = np.stack(future_tokens)
            input_token = np.concatenate([token, future_tokens], axis=0)
        else:
            input_token = token

        results['latent'] = input_token
        return results


@PIPELINES.register_module()
class OccStressLoadStreamLatentHistoryToken(OccStressLoadStreamLatentToken):
    """Load OccStress current token plus clean future and explicit clean history tokens.

    OccStress protocols export a corrupted anchor token named by ``protocol_sample_id``.
    Future targets and explicit history slots are read from clean stage2 token
    roots, so stage2-only history-gate experiments can evaluate whether clean
    aligned history helps when the anchor token is corrupted by OccStress protocols.
    """

    def __init__(self,
                 current_data_path,
                 future_data_path=None,
                 history_data_path=None,
                 dataset_type='occ3d',
                 history_frame_number=4,
                 history_key='history_latent',
                 load_flow_summary=False,
                 flow_root=None,
                 flow_summary_key='history_flow_summary',
                 flow_key='flow_backward',
                 valid_key='flow_valid_backward',
                 dynamic_key='dynamic_mask'):
        super().__init__(
            current_data_path=current_data_path,
            future_data_path=future_data_path,
            dataset_type=dataset_type)
        self.history_data_path = history_data_path or future_data_path or current_data_path
        self.history_data_paths = self._as_list(self.history_data_path)
        self.history_frame_number = int(history_frame_number)
        self.history_key = history_key
        self.load_flow_summary = bool(load_flow_summary)
        self.flow_root = flow_root
        self.flow_summary_key = flow_summary_key
        self.flow_key = flow_key
        self.valid_key = valid_key
        self.dynamic_key = dynamic_key

    def _history_names(self, results):
        history_names = list(results.get('history_tokens') or [])
        if not history_names:
            history_names = [
                os.path.splitext(os.path.basename(path))[0]
                for path in results.get('previous_occ_path', [])
            ]
        if len(history_names) == 0:
            history_names = [results.get('anchor_token', results['sample_idx'])] * self.history_frame_number
        if len(history_names) < self.history_frame_number:
            pad_value = history_names[0]
            history_names = [pad_value] * (self.history_frame_number - len(history_names)) + history_names
        return history_names[-self.history_frame_number:]

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
            step_summary = (
                np.stack(step_summaries, axis=0).mean(axis=0)
                if step_summaries else np.zeros(4, dtype=np.float32)
            )
            age = float(num_history - hist_idx) / max(float(num_history), 1.0)
            summaries.append(np.concatenate([np.array([age], dtype=np.float32), step_summary], axis=0))
        return np.stack(summaries, axis=0).astype(np.float32)

    def __call__(self, results):
        super().__call__(results)

        scene_name = results['scene_name']
        history_names = self._history_names(results)
        history_tokens = []
        for history_name in history_names:
            _, latent_token = self._load_first_existing(
                self.history_data_paths, scene_name, history_name)
            if latent_token is None:
                raise FileNotFoundError(
                    f'Could not resolve history latent token for scene={scene_name}, '
                    f'name={history_name}, roots={self.history_data_paths}'
                )
            history_tokens.append(latent_token['token'])
        results[self.history_key] = np.stack(history_tokens, axis=0)

        if self.load_flow_summary:
            current_token = results.get('anchor_token', results['sample_idx'])
            results[self.flow_summary_key] = self._history_flow_summary(
                scene_name, history_names, current_token)

        return results
