# OccStress adapter/portability modifications; see docs/source-imports.json.
import os
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from .dataset import get_meta_data


MIRROR_Y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)
OBSERVED_OFFSETS = (-1.5, -1.0, -0.5, 0.0)
FUTURE_OFFSETS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
WAYMO_TO_OCC3D = {
    0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
    8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17,
}


def _pose(info):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Quaternion(
        info['ego2global_rotation']).rotation_matrix
    pose[:3, 3] = np.asarray(
        info['ego2global_translation'], dtype=np.float64)
    return pose


class OccStressProtocolDataset:
    """Build DOME H4/F6 samples directly from OccStress anchor records."""

    def __init__(
            self,
            protocol_path,
            base_info,
            occstress_root,
            nuscenes_root=None,
            max_samples=None,
            num_shards=1,
            shard_index=0):
        from occstress.datasets.metadata import load_metadata
        from occstress.datasets.paths import resolve_occstress_path
        self.protocol_path = resolve_occstress_path(protocol_path, occstress_root=occstress_root)
        self.base_info_path = resolve_occstress_path(base_info, occstress_root=occstress_root)
        self.occstress_root = Path(occstress_root).resolve()
        self.nuscenes_root = Path(
            nuscenes_root or self.occstress_root.parent / 'nuscenes').resolve()

        base = load_metadata(self.base_info_path, trusted_pickle=True)
        self.dataset_name = base.get('metadata', {}).get(
            'dataset', 'Occ3D-nuScenes')
        scene_infos = base['infos']
        if not isinstance(scene_infos, dict):
            raise TypeError(
                'DOME base info must contain a scene-name to frame-list mapping')
        self.token_to_info = {
            info['token']: info
            for frames in scene_infos.values()
            for info in frames
        }

        with self.protocol_path.open('rb') as handle:
            records = pickle.load(handle)
        records = sorted(records, key=lambda item: (
            item.get('anchor_timestamp', 0),
            item.get('scene_name', ''),
            item.get('anchor_token', ''),
        ))
        if max_samples is not None:
            records = records[:max_samples]
        if num_shards <= 0:
            raise ValueError('num_shards must be positive')
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError('shard_index must be in [0, num_shards)')
        self.full_record_count = len(records)
        self.num_shards = num_shards
        self.shard_index = shard_index
        records = records[shard_index::num_shards]
        self.records = records
        self._validate_records()

    def _validate_records(self):
        for record in self.records:
            if record['history_length'] < 4:
                raise ValueError(
                    f"{record['sample_id']} has fewer than four history frames")
            if record['future_length'] != 6:
                raise ValueError(
                    f"{record['sample_id']} has "
                    f"{record['future_length']} future frames, expected six")
            tokens = (
                record['history_tokens'][-3:] +
                [record['anchor_token']] +
                record.get('future_tokens', [
                    item['token'] for item in record['future_targets']
                ]))
            missing = [token for token in tokens if token not in self.token_to_info]
            if missing:
                raise KeyError(
                    f"{record['sample_id']} has tokens absent from base info: "
                    f'{missing}')

    def __len__(self):
        return len(self.records)

    def _resolve_occ_path(self, item, scene_name):
        from occstress.datasets.paths import release_occ_path
        released = release_occ_path(item['occ_path'], occstress_root=self.occstress_root)
        if released is not None:
            return released
        raw_path = item['occ_path']
        path = Path(raw_path)
        if path.suffix != '.npz':
            path = path / 'labels.npz'
        if self.dataset_name in ('Occ3D-Waymo', 'OccStress-Waymo'):
            source = item.get('occ_source', item.get('source'))
            if source is None:
                source = 'corrupted' if '/occ/manual/' in str(path) else 'clean'
            clean_root = os.environ.get('OCCSTRESS_WAYMO_CLEAN_OCC_CACHE')
            if source == 'clean' and clean_root:
                candidate = (
                    Path(clean_root) / str(scene_name).zfill(3) /
                    item['token'] / 'labels.npz')
                if candidate.is_file():
                    return candidate

            normalized = str(path).replace('\\', '/')
            asset_root = os.environ.get('OCCSTRESS_WAYMO_ASSET_CACHE')
            if source != 'clean' and asset_root:
                remaps = (
                    ('/occ/manual/', 'occ/manual'),
                    ('/occ/upstream/', 'occ/upstream'),
                    (
                        '/effocc-waymo-upstream/',
                        'occ/upstream/pointcloud_fusion/effocc',
                    ),
                )
                for marker, relative_root in remaps:
                    if marker not in normalized:
                        continue
                    candidate = (
                        Path(asset_root) / relative_root /
                        normalized.split(marker, 1)[1])
                    if candidate.is_file():
                        return candidate

        if path.is_file():
            return path

        normalized = str(path).replace('\\', '/')
        remaps = (
            ('/data/OccStress/', self.occstress_root),
            ('/data/nuscenes/', self.nuscenes_root),
        )
        for marker, root in remaps:
            if marker in normalized:
                candidate = root / normalized.split(marker, 1)[1]
                if candidate.is_file():
                    return candidate

        if not path.is_absolute():
            candidates = (
                self.occstress_root / path,
                self.nuscenes_root / path,
                self.occstress_root.parent.parent / path,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f'cannot resolve occupancy path: {raw_path}')

    def _load_occ(self, item, scene_name):
        path = self._resolve_occ_path(item, scene_name)
        with np.load(path, allow_pickle=False) as label:
            if 'semantics' in label:
                semantics = np.asarray(label['semantics'], dtype=np.int64)
            elif self.dataset_name in ('Occ3D-Waymo', 'OccStress-Waymo') and 'voxel_label' in label:
                raw = np.asarray(label['voxel_label'], dtype=np.int64)
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

    def _observed_poses(self, record):
        anchor_pose = _pose(self.token_to_info[record['anchor_token']])
        history_poses = [
            _pose(self.token_to_info[token])
            for token in record['history_tokens']
        ]
        clean_anchor_to_history = [
            np.linalg.inv(history_pose) @ anchor_pose
            for history_pose in history_poses
        ]

        misalignment = record.get('misalignment') or {}
        deltas = misalignment.get('delta_rt')
        if deltas is not None:
            deltas = np.asarray(deltas, dtype=np.float64)
            if deltas.shape != (len(history_poses), 4, 4):
                raise ValueError(
                    f"{record['sample_id']} has invalid delta_rt shape "
                    f'{deltas.shape}')

        resolved_history = []
        for index, (history, clean_rt) in enumerate(zip(
                record['history'], clean_anchor_to_history)):
            anchor_to_history = clean_rt
            if history.get('rt_source') == 'misalignment':
                if deltas is None:
                    raise ValueError(
                        f"{record['sample_id']} requires missing delta_rt")
                anchor_to_history = deltas[index] @ clean_rt
            resolved_history.append(
                anchor_pose @ np.linalg.inv(anchor_to_history))

        current_delta = np.eye(4, dtype=np.float64)
        if misalignment.get('current_active'):
            current_delta = np.asarray(
                misalignment.get('current_delta_rt'), dtype=np.float64)
            if current_delta.shape != (4, 4):
                raise ValueError(
                    f"{record['sample_id']} has invalid current_delta_rt")
        current_pose = anchor_pose @ np.linalg.inv(current_delta)
        return resolved_history[-3:] + [current_pose]

    def _sequence_poses(self, record):
        poses = self._observed_poses(record)
        poses.extend([
            _pose(self.token_to_info[target['token']])
            for target in record['future_targets']
        ])
        if record.get('corruption', {}).get('type') == 'traffic':
            poses = [MIRROR_Y @ pose @ MIRROR_Y for pose in poses]
        return np.asarray(poses, dtype=np.float64)

    def __getitem__(self, index):
        record = self.records[index]
        observed_items = (
            record['history'][-3:] +
            [record.get('current_input', record['target'])])
        sequence_items = observed_items + record['future_targets']
        occupancy = np.stack(
            [
                self._load_occ(item, record['scene_name'])
                for item in sequence_items
            ], axis=0)
        poses = self._sequence_poses(record)
        if occupancy.shape != (10, 200, 200, 16):
            raise ValueError(
                f"{record['sample_id']} has sequence shape {occupancy.shape}")
        if poses.shape != (10, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(
                f"{record['sample_id']} has invalid pose sequence")

        metadata = get_meta_data(poses)
        metadata.update({
            'sample_id': record['sample_id'],
            'anchor_token': record['anchor_token'],
            'scene_name': record['scene_name'],
            'corruption': record['corruption'],
            'observed_offsets_seconds': OBSERVED_OFFSETS,
            'future_offsets_seconds': FUTURE_OFFSETS,
        })
        for key, value in metadata.items():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ValueError(
                    f"{record['sample_id']}.{key} contains NaN/Inf")
        return occupancy, occupancy.copy(), metadata
