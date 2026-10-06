# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Protocol-backed OccStress-Waymo dataset."""

# Keep this implementation identical to II-World's adapter. Both projects
# vendor mmdet3d and cannot share a runtime import without destabilizing their
# registries.
import os
from pathlib import Path

import mmcv
import numpy as np
from pyquaternion import Quaternion

from .builder import DATASETS
from .waymo_world_dataset import WaymoWorldDataset


def _occ_npz(path):
    path = Path(path)
    return str(path if path.suffix == ".npz" else path / "labels.npz")


def _frame_occ_path(frame, scene_name):
    from occstress.datasets.paths import release_occ_path
    released = release_occ_path(frame['occ_path'], dataset=os.environ.get('OCCSTRESS_DATASET'))
    if released is not None:
        return str(released)
    source = frame.get("occ_source", frame.get("source"))
    path = str(frame["occ_path"])
    if source is None:
        source = "corrupted" if "/occ/manual/" in path else "clean"
    clean_root = os.environ.get("OCCSTRESS_WAYMO_CLEAN_OCC_CACHE")
    if source == "clean" and clean_root:
        cached = (
            Path(clean_root) / str(scene_name).zfill(3) /
            frame["token"] / "labels.npz"
        )
        if cached.exists():
            return str(cached)
    asset_root = os.environ.get("OCCSTRESS_WAYMO_ASSET_CACHE")
    if source != "clean" and asset_root:
        for marker, relative_root in (
            ("/occ/manual/", "occ/manual"),
            ("/occ/upstream/", "occ/upstream"),
        ):
            if marker not in path:
                continue
            relative = path.split(marker, 1)[1]
            cached = Path(asset_root) / relative_root / relative
            if cached.exists():
                return str(cached)
        variant = frame.get("prediction_variant")
        if variant:
            for subtrack in (
                "camera_only", "pointcloud_fusion", "camera_fusion"
            ):
                cached = (
                    Path(asset_root) / "occ" / "upstream" / subtrack /
                    str(source) / str(variant) / str(scene_name).zfill(3) /
                    frame["token"] / "labels.npz"
                )
                if cached.exists():
                    return str(cached)
    return _occ_npz(path)


def _load_semantics(path):
    with np.load(_occ_npz(path)) as labels:
        if "semantics" in labels:
            return labels["semantics"]
        raw = labels["voxel_label"]
    mapping = {0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
               8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17}
    output = np.empty(raw.shape, dtype=np.uint8)
    for source, target in mapping.items():
        output[raw == source] = target
    return output


def _mirror_rt(rt):
    mirror = np.diag([1.0, -1.0, 1.0, 1.0])
    return mirror @ np.asarray(rt) @ mirror


def _scene_shards(value):
    if not value:
        return None
    return {
        str(scene).strip().zfill(3)
        for scene in str(value).split(",")
        if str(scene).strip()
    }


def _quaternion_from_rotation(rotation):
    """Project float32 pose rotations back onto SO(3)."""
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    orthogonal = u @ vh
    if np.linalg.det(orthogonal) < 0:
        u[:, -1] *= -1
        orthogonal = u @ vh
    return Quaternion(matrix=orthogonal).elements


def _build_stepwise_rts(anchor_to_frames):
    stepwise = []
    for index, current in enumerate(anchor_to_frames):
        if index == 0:
            stepwise.append(np.eye(4, dtype=np.float32))
            continue
        previous = anchor_to_frames[index - 1]
        stepwise.append(
            (previous @ np.linalg.inv(current)).astype(np.float32))
    return stepwise


@DATASETS.register_module()
class OccStressWaymoWorldDataset(WaymoWorldDataset):
    def __init__(self, protocol_path, base_info, native_history_frames=3,
                 max_samples=None, scene_shard=None, *args, **kwargs):
        from occstress.datasets.paths import resolve_occstress_path
        self.protocol_path = str(resolve_occstress_path(protocol_path))
        self.base_info_path = str(resolve_occstress_path(base_info))
        self.native_history_frames = native_history_frames
        self.max_samples = max_samples
        self.scene_shards = _scene_shards(scene_shard)
        self.token_to_info = {}
        kwargs["ann_file"] = self.base_info_path
        kwargs["pose_file"] = None
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        from occstress.datasets.metadata import load_metadata
        base = load_metadata(ann_file, trusted_pickle=True)
        infos = [info for values in base["infos"].values() for info in values]
        self.token_to_info = {info["token"]: info for info in infos}
        records = mmcv.load(self.protocol_path, file_format="pkl")
        if self.scene_shards is not None:
            records = [
                record for record in records
                if str(record["scene_name"]).zfill(3) in self.scene_shards
            ]
        if self.max_samples:
            records = records[:self.max_samples]
        self.flag = np.arange(len(records), dtype=np.int64)
        return records

    def _pose(self, token, mirror=False):
        pose = np.asarray(self.token_to_info[token]["pose_mat"], dtype=np.float32)
        return _mirror_rt(pose).astype(np.float32) if mirror else pose

    def get_data_info(self, index):
        record = self.data_infos[index]
        mirror = record.get("traffic_mirror", False)
        history = record["history"][-self.native_history_frames:]
        current, future = record["current_input"], record["future_targets"]
        poses = [self._pose(frame["token"], mirror) for frame in history]
        anchor_pose = self._pose(current["token"], mirror)
        future_poses = [self._pose(frame["token"], mirror) for frame in future]
        anchor_to_history = [
            (np.linalg.inv(pose) @ anchor_pose).astype(np.float32)
            for pose in poses
        ]
        misalignment = record.get("misalignment") or {}
        if misalignment:
            deltas = np.asarray(
                misalignment["delta_rt"], dtype=np.float32)[
                    -self.native_history_frames:]
            affected = np.asarray(
                misalignment["affected_mask"], dtype=np.uint8)[
                    -self.native_history_frames:]
            anchor_to_history = [
                deltas[index] @ rt if affected[index] else rt
                for index, rt in enumerate(anchor_to_history)
            ]
        current_anchor_rt = np.eye(4, dtype=np.float32)
        if misalignment.get("current_active"):
            current_anchor_rt = np.asarray(
                misalignment["current_delta_rt"], dtype=np.float32)
        previous_rts = _build_stepwise_rts(
            anchor_to_history + [current_anchor_rt])

        future_rts, previous = [], anchor_pose
        for pose in future_poses:
            future_rts.append(np.linalg.inv(pose) @ previous)
            previous = pose
        all_future = [anchor_pose] + future_poses
        translations = np.array([pose[:3, 3] for pose in all_future])
        rotations = np.array([
            _quaternion_from_rotation(pose[:3, :3]) for pose in all_future
        ], dtype=np.float32)
        controls = [self.token_to_info[current["token"]]] + [
            self.token_to_info[frame["token"]] for frame in future[:-1]]
        commands = np.array([info["gt_ego_fut_cmd"] for info in controls], dtype=np.float32)
        trajectories = np.array(
            [info["gt_ego_fut_trajs"][0] for info in controls], dtype=np.float32)
        if mirror:
            commands[:, [0, 1]] = commands[:, [1, 0]]
            trajectories[:, 1] *= -1
        occ_index = (
            [frame["token"] for frame in history] +
            [current["token"]] +
            [frame["token"] for frame in future]
        )
        all_history_poses = [
            self._pose(frame["token"], mirror) for frame in record["history"]
        ]
        all_anchor_to_history = [
            (np.linalg.inv(pose) @ anchor_pose).astype(np.float32)
            for pose in all_history_poses
        ]
        if misalignment:
            deltas = np.asarray(misalignment["delta_rt"], dtype=np.float32)
            affected = np.asarray(
                misalignment["affected_mask"], dtype=np.uint8)
            all_anchor_to_history = [
                deltas[offset] @ rt if affected[offset] else rt
                for offset, rt in enumerate(all_anchor_to_history)
            ]
        observe_global_poses = [
            anchor_pose @ np.linalg.inv(anchor_to_frame)
            for anchor_to_frame in all_anchor_to_history + [current_anchor_rt]
        ]
        observe_count = self.native_history_frames + 1
        observe_global_poses = observe_global_poses[-(observe_count + 1):]

        return {
            "index": index,
            "occ_path": _frame_occ_path(current, record["scene_name"]),
            "occ_index": occ_index,
            "sample_idx": current["token"],
            "timestamp": self.token_to_info[current["token"]]["timestamp"] / 1e6,
            "prev": None, "scene_name": record["scene_name"],
            "previous_occ_path": [
                _frame_occ_path(frame, record["scene_name"])
                for frame in history
            ],
            "previous_occ_index": list(range(len(history))),
            "previous_curr_to_prev_ego_rt": np.array(previous_rts[:-1]),
            "curr_to_prev_ego_rt": np.asarray(previous_rts[-1]),
            "future_occ_path": [
                _frame_occ_path(frame, record["scene_name"])
                for frame in future
            ],
            "future_occ_index": list(range(len(future))),
            "curr_to_future_ego_rt": np.array(future_rts, dtype=np.float32),
            "curr_ego_to_global": np.array(all_future, dtype=np.float32),
            "ego_to_global_rotation": rotations,
            "ego_to_global_translation": translations,
            "valid_frame": np.ones(len(future), dtype=np.bool_),
            "start_of_sequence": True, "sequence_group_idx": index,
            "gt_ego_fut_trajs": trajectories, "gt_ego_fut_cmd": commands,
            "gt_ego_lcf_feat": np.zeros((len(controls), 3), dtype=np.float32),
            "protocol_observe_rotations": np.array([
                _quaternion_from_rotation(pose[:3, :3])
                for pose in observe_global_poses
            ], dtype=np.float32),
            "protocol_observe_translations": np.array([
                pose[:3, 3] for pose in observe_global_poses
            ], dtype=np.float32),
            "protocol_observe_ego_lcf_feat": np.zeros(
                (observe_count, 3), dtype=np.float32),
            "protocol_mode": True,
            "observed_offsets": (-1.5, -1.0, -0.5, 0.0),
            "protocol_sample_id": record["sample_id"], "occstress_protocol": record,
        }

    def evaluate_forecasting_miou(self, results, logger=None):
        from .occ_metrics import Metric_mIoU
        metrics = [
            Metric_mIoU(
                num_classes=18, use_lidar_mask=False, use_image_mask=False,
                logger=logger)
            for _ in range(6)
        ]
        predictions = {}
        for result in results:
            for offset, index in enumerate(result["index"]):
                key = (
                    "pred_futu_semantics"
                    if "pred_futu_semantics" in result else "semantics")
                predictions[int(index)] = result[key][offset]
        for index, prediction in sorted(predictions.items()):
            target = np.stack([
                _load_semantics(_frame_occ_path(
                    frame, self.data_infos[index]["scene_name"]))
                for frame in self.data_infos[index]["future_targets"]])
            if prediction.shape[0] != 6 or target.shape[0] != 6:
                raise ValueError(
                    f"expected F6, got pred={prediction.shape}, target={target.shape}")
            for horizon, metric in enumerate(metrics):
                metric.add_batch(
                    prediction[horizon], target[horizon], None, None)
                metric.add_iou_batch(
                    prediction[horizon], target[horizon], None, None)
        output = {}
        for horizon, metric in enumerate(metrics, start=1):
            _, miou, _, _, _ = metric.count_miou()
            output[f"semantics_miou_time_{horizon * 0.5:.1f}s"] = miou
            output[f"binary_iou_time_{horizon * 0.5:.1f}s"] = metric.count_iou()
        return output

    def evaluate_miou(self, results, logger=None):
        return self.evaluate_forecasting_miou(results, logger=logger)

    def start_streaming_evaluation(self, protocol=None):
        del protocol
        from .occ_metrics import Metric_mIoU
        self._stream_metrics = [
            Metric_mIoU(
                num_classes=18, use_lidar_mask=False, use_image_mask=False)
            for _ in range(6)
        ]
        self._stream_seen = set()
        self.streaming_evaluated_records = 0

    def add_streaming_results(self, results):
        if not hasattr(self, "_stream_metrics"):
            raise RuntimeError("start_streaming_evaluation must be called first")
        for result in results:
            key = (
                "pred_futu_semantics"
                if "pred_futu_semantics" in result else "semantics")
            for offset, raw_index in enumerate(result["index"]):
                index = int(raw_index)
                if index in self._stream_seen:
                    continue
                prediction = result[key][offset]
                target = np.stack([
                    _load_semantics(_frame_occ_path(
                        frame, self.data_infos[index]["scene_name"]))
                    for frame in self.data_infos[index]["future_targets"]
                ])
                if prediction.shape[0] != 6 or target.shape[0] != 6:
                    raise ValueError(
                        f"expected F6, got pred={prediction.shape}, "
                        f"target={target.shape}")
                for horizon, metric in enumerate(self._stream_metrics):
                    pred_frame = prediction[horizon].copy()
                    target_frame = target[horizon].copy()
                    metric.add_batch(pred_frame, target_frame, None, None)
                    metric.add_iou_batch(
                        pred_frame, target_frame, None, None)
                self._stream_seen.add(index)
                self.streaming_evaluated_records += 1

    def finish_streaming_evaluation(self):
        if not hasattr(self, "_stream_metrics"):
            raise RuntimeError("streaming evaluation was not started")
        output = {}
        for horizon, metric in enumerate(self._stream_metrics, start=1):
            _, miou, _, _, _ = metric.count_miou()
            output[f"semantics_miou_time_{horizon * 0.5:.1f}s"] = miou
            output[f"binary_iou_time_{horizon * 0.5:.1f}s"] = metric.count_iou()
        output["_aggregate_counts"] = {
            "semantic_confusion": [
                metric.hist.astype(np.int64).tolist()
                for metric in self._stream_metrics
            ],
            "binary_confusion": [
                metric.iou_hist.astype(np.int64).tolist()
                for metric in self._stream_metrics
            ],
        }
        return output
