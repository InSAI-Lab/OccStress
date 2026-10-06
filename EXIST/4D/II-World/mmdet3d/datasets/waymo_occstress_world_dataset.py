# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Protocol-backed OccStress-Waymo dataset."""

import copy
import os
from pathlib import Path

import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet3d.core.bbox.structures import Box3DMode

from .builder import DATASETS
from .custom_3d import Custom3DDataset
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
    if source is None:
        source = (
            "corrupted"
            if "/occ/manual/" in str(frame["occ_path"]) else "clean"
        )
    clean_root = os.environ.get("OCCSTRESS_WAYMO_CLEAN_OCC_CACHE")
    if source == "clean" and clean_root:
        cached = (
            Path(clean_root) / str(scene_name).zfill(3) /
            frame["token"] / "labels.npz"
        )
        if cached.exists():
            return str(cached)
    asset_root = os.environ.get("OCCSTRESS_WAYMO_ASSET_CACHE")
    marker = "/occ/manual/"
    path = str(frame["occ_path"])
    if source != "clean" and asset_root and marker in path:
        relative = path.split(marker, 1)[1]
        cached = Path(asset_root) / "occ" / "manual" / relative
        if cached.exists():
            return str(cached)
    dataset = os.environ.get("OCCSTRESS_DATASET", "waymo")
    root = os.environ.get(
        "OCCSTRESS_CARLA_ROOT" if dataset == "carla" else "OCCSTRESS_WAYMO_ROOT")
    if root:
        for marker in ("/occ/", "/clean_occ/"):
            if marker in path:
                path = str(Path(root) / (marker.strip("/") + "/" + path.split(marker, 1)[1]))
                break
    return _occ_npz(path)


def _load_semantics(path):
    with np.load(_occ_npz(path)) as labels:
        if "semantics" in labels:
            return labels["semantics"]
        raw = labels["voxel_label"]
    mapping = {
        0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
        8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17,
    }
    unknown = set(np.unique(raw).tolist()) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped Waymo labels: {sorted(unknown)}")
    output = np.empty(raw.shape, dtype=np.uint8)
    for source, target in mapping.items():
        output[raw == source] = target
    return output


def _mirror_rt(rt):
    mirror = np.diag([1.0, -1.0, 1.0, 1.0])
    return mirror @ np.asarray(rt) @ mirror


def _rotation_quaternion(rotation):
    """Project float32 pose rotations back onto SO(3) for pyquaternion."""
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    orthogonal = u @ vh
    if np.linalg.det(orthogonal) < 0:
        u[:, -1] *= -1
        orthogonal = u @ vh
    return Quaternion(matrix=orthogonal).elements


def _scene_shards(value):
    if not value:
        return None
    return {
        str(scene).strip().zfill(3)
        for scene in str(value).split(",")
        if str(scene).strip()
    }


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


def _build_official_waymo_stepwise_rts(poses):
    """Match WaymoWorldDataset.get_data_info exactly."""
    stepwise = []
    for index, current in enumerate(poses):
        if index == 0:
            stepwise.append(np.eye(4, dtype=np.float32))
            continue
        previous = poses[index - 1]
        stepwise.append(
            (current @ np.linalg.inv(previous)).astype(np.float32))
    return stepwise


@DATASETS.register_module()
class OccStressWaymoWorldDataset(WaymoWorldDataset):
    def __init__(self, protocol_path, base_info, native_history_frames=4,
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
        infos = []
        for scene_infos in base["infos"].values():
            infos.extend(scene_infos)
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
        current = record["current_input"]
        future = record["future_targets"]
        history_tokens = [frame["token"] for frame in history]
        poses = [self._pose(token, mirror) for token in history_tokens]
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
        stepwise_previous = _build_stepwise_rts(
            anchor_to_history + [current_anchor_rt])

        future_stepwise = []
        previous = anchor_pose
        for pose in future_poses:
            future_stepwise.append(np.linalg.inv(pose) @ previous)
            previous = pose
        all_future_poses = [anchor_pose] + future_poses
        translations = np.array([pose[:3, 3] for pose in all_future_poses])
        rotations = np.array([
            _rotation_quaternion(pose[:3, :3])
            for pose in all_future_poses
        ], dtype=np.float32)

        control_infos = [self.token_to_info[current["token"]]] + [
            self.token_to_info[frame["token"]] for frame in future[:-1]
        ]
        commands = np.array(
            [info["gt_ego_fut_cmd"] for info in control_infos], dtype=np.float32)
        trajectories = np.array(
            [info["gt_ego_fut_trajs"][0] for info in control_infos],
            dtype=np.float32)
        if mirror:
            commands[:, [0, 1]] = commands[:, [1, 0]]
            trajectories[:, 1] *= -1

        return {
            "index": index,
            "occ_index": [index],
            "occ_path": _frame_occ_path(current, record["scene_name"]),
            "sample_idx": current["token"],
            "timestamp": self.token_to_info[current["token"]]["timestamp"] / 1e6,
            "prev": None,
            "scene_name": record["scene_name"],
            "previous_occ_path": [
                _frame_occ_path(frame, record["scene_name"])
                for frame in history
            ],
            "previous_occ_index": list(range(len(history))),
            "previous_curr_to_prev_ego_rt": np.array(stepwise_previous[:-1]),
            "curr_to_prev_ego_rt": np.asarray(stepwise_previous[-1]),
            "future_occ_path": [
                _frame_occ_path(frame, record["scene_name"])
                for frame in future
            ],
            "future_occ_index": list(range(len(future))),
            "curr_to_future_ego_rt": np.array(future_stepwise, dtype=np.float32),
            "curr_ego_to_global": np.array(all_future_poses, dtype=np.float32),
            "ego_to_global_rotation": rotations,
            "ego_to_global_translation": translations,
            "valid_frame": np.ones(len(future), dtype=np.bool_),
            "start_of_sequence": True,
            "sequence_group_idx": index,
            "gt_ego_fut_trajs": trajectories,
            "gt_ego_fut_cmd": commands,
            "gt_ego_lcf_feat": np.zeros((len(control_infos), 3), dtype=np.float32),
            "protocol_sample_id": record["sample_id"],
            "history_tokens": history_tokens,
            "future_tokens": [frame["token"] for frame in future],
            "future_latent_names": [
                f"{int(self.token_to_info[frame['token']]['frame_idx']):03d}_04"
                for frame in future
            ],
            "current_latent_name": (
                f"{int(self.token_to_info[current['token']]['frame_idx']):03d}_04"),
            "future_length": len(future),
            "occstress_protocol": record,
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
            future = self.data_infos[index]["future_targets"]
            target = np.stack([
                _load_semantics(_frame_occ_path(
                    frame, self.data_infos[index]["scene_name"]))
                for frame in future
            ])
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

    def start_streaming_evaluation(self):
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


@DATASETS.register_module()
class WaymoWorldReferenceDataset(WaymoWorldDataset):
    """Run the unmodified official Waymo loader on a small scene subset."""

    def __init__(self, scene_shard=None, max_frames_per_scene=None,
                 *args, **kwargs):
        self.reference_scenes = _scene_shards(scene_shard)
        self.max_frames_per_scene = max_frames_per_scene
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        infos = super().load_annotations(ann_file)
        if self.reference_scenes is not None:
            infos = [
                info for info in infos
                if str(info["scene_idx"]).zfill(3) in self.reference_scenes
            ]
        if self.max_frames_per_scene:
            counts = {}
            selected = []
            for info in infos:
                scene = str(info["scene_idx"]).zfill(3)
                count = counts.get(scene, 0)
                if count >= self.max_frames_per_scene:
                    continue
                selected.append(info)
                counts[scene] = count + 1
            infos = selected
        if not infos:
            raise ValueError("official Waymo reference subset is empty")
        previous_scene = None
        for info in infos:
            scene = str(info["scene_idx"]).zfill(3)
            if scene != previous_scene:
                info["prev"] = None
                previous_scene = scene
        return infos


@DATASETS.register_module()
class OccStressWaymoTokenizerDataset(Custom3DDataset):
    """Expand each protocol anchor into H4+current tokenizer states."""

    def __init__(self, protocol_path, base_info, ann_file=None, pipeline=None,
                 data_root=None, classes=None, test_mode=True,
                 filter_empty_gt=False, max_samples=None, scene_shard=None,
                 **kwargs):
        from occstress.datasets.paths import resolve_occstress_path
        self.protocol_path = str(resolve_occstress_path(protocol_path))
        self.base_info_path = str(resolve_occstress_path(base_info))
        self.max_samples = max_samples
        self.scene_shards = _scene_shards(scene_shard)
        self.token_to_info = {}
        kwargs.pop("pose_file", None)
        kwargs.pop("load_future_frame_number", None)
        kwargs.pop("load_previous_frame_number", None)
        kwargs.pop("load_previous_data", None)
        kwargs.pop("use_sequence_group_flag", None)
        kwargs.pop("sequences_split_num", None)
        kwargs.pop("dataset_name", None)
        kwargs.pop("eval_metric", None)
        kwargs.pop("load_interval", None)
        kwargs.setdefault("box_type_3d", "LiDAR")
        super().__init__(
            data_root=data_root,
            ann_file=self.base_info_path,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            **kwargs,
        )
        self.box_mode_3d = Box3DMode.LIDAR

    def load_annotations(self, ann_file):
        from occstress.datasets.metadata import load_metadata
        base = load_metadata(ann_file, trusted_pickle=True)
        infos = [
            info for scene_infos in base["infos"].values()
            for info in scene_infos
        ]
        self.token_to_info = {info["token"]: info for info in infos}
        records = mmcv.load(self.protocol_path, file_format="pkl")
        if self.scene_shards is not None:
            records = [
                record for record in records
                if str(record["scene_name"]).zfill(3) in self.scene_shards
            ]
        if self.max_samples:
            records = records[:self.max_samples]

        expanded = []
        for clip_index, record in enumerate(records):
            mirror = record.get("traffic_mirror", False)
            history = record["history"]
            frames = history + [record["current_input"]]
            poses = []
            for frame in frames:
                pose = np.asarray(
                    self.token_to_info[frame["token"]]["pose_mat"],
                    dtype=np.float32)
                if mirror:
                    pose = _mirror_rt(pose).astype(np.float32)
                poses.append(pose)
            stepwise = _build_official_waymo_stepwise_rts(poses)
            misalignment = record.get("misalignment") or {}
            if misalignment:
                deltas = np.asarray(
                    misalignment["delta_rt"], dtype=np.float32)
                affected = np.asarray(
                    misalignment["affected_mask"], dtype=np.uint8)
                for frame_index, active in enumerate(affected):
                    if active:
                        stepwise[frame_index] = (
                            deltas[frame_index] @ stepwise[frame_index])
            if misalignment.get("current_active"):
                stepwise[-1] = (
                    np.asarray(
                        misalignment["current_delta_rt"], dtype=np.float32)
                    @ stepwise[-1]
                )
            anchor_frame_idx = int(
                self.token_to_info[
                    record["anchor_token"]]["frame_idx"])
            current_latent_name = f"{anchor_frame_idx:03d}_04"
            for frame_index, (frame, rt) in enumerate(zip(frames, stepwise)):
                info = self.token_to_info[frame["token"]]
                expanded.append({
                    "index": len(expanded),
                    "clip_index": clip_index,
                    "protocol_sample_id": record["sample_id"],
                    "sample_idx": frame["token"],
                    "anchor_token": record["anchor_token"],
                    "current_latent_name": current_latent_name,
                    "scene_name": record["scene_name"],
                    "occ_path": _frame_occ_path(
                        frame, record["scene_name"]),
                    "timestamp": info["timestamp"] / 1e6,
                    "prev": None,
                    "start_of_sequence": frame_index == 0,
                    "curr_to_prev_ego_rt": rt,
                    "sequence_group_idx": clip_index,
                    "occstress_save_token": frame_index == 4,
                    "occstress_protocol": record,
                    "occstress_corruption": record["corruption"],
                    "gt_occ_path": _frame_occ_path(
                        record["target"] if frame_index == 4 else frame,
                        record["scene_name"]),
                })
        self.flag = np.asarray(
            [item["sequence_group_idx"] for item in expanded],
            dtype=np.int64,
        )
        return expanded

    def get_data_info(self, index):
        return copy.deepcopy(self.data_infos[index])

    def evaluate(self, results, logger=None, **kwargs):
        from .occ_metrics import Metric_mIoU
        metric = Metric_mIoU(
            num_classes=18, use_lidar_mask=False, use_image_mask=False,
            logger=logger)
        for result in results:
            for offset, index in enumerate(result["index"]):
                info = self.data_infos[int(index)]
                if not info["occstress_save_token"]:
                    continue
                target = _load_semantics(info["gt_occ_path"])
                prediction = result["semantics"][offset]
                metric.add_batch(prediction, target, None, None)
                metric.add_iou_batch(prediction, target, None, None)
        _, miou, _, _, _ = metric.count_miou()
        return {
            "semantics_miou": miou,
            "binary_iou": metric.count_iou(),
        }


@DATASETS.register_module()
class OccStressWaymoFutureTokenizerDataset(OccStressWaymoTokenizerDataset):
    """Tokenize the unique F6 targets needed by one protocol scene shard."""

    def load_annotations(self, ann_file):
        from occstress.datasets.metadata import load_metadata
        base = load_metadata(ann_file, trusted_pickle=True)
        infos = [
            info for scene_infos in base["infos"].values()
            for info in scene_infos
        ]
        self.token_to_info = {info["token"]: info for info in infos}
        records = mmcv.load(self.protocol_path, file_format="pkl")
        if self.scene_shards is not None:
            records = [
                record for record in records
                if str(record["scene_name"]).zfill(3) in self.scene_shards
            ]
        if self.max_samples:
            records = records[:self.max_samples]

        unique = {}
        for record in records:
            for frame in record["future_targets"]:
                unique[(frame["token"], frame["occ_path"])] = (
                    record, frame)

        expanded = []
        for record, frame in unique.values():
            info = self.token_to_info[frame["token"]]
            save_name = f"{int(info['frame_idx']):03d}_04"
            expanded.append({
                "index": len(expanded),
                "clip_index": len(expanded),
                "protocol_sample_id": save_name,
                "sample_idx": frame["token"],
                "anchor_token": record["anchor_token"],
                "scene_name": record["scene_name"],
                "occ_path": _frame_occ_path(frame, record["scene_name"]),
                "timestamp": info["timestamp"] / 1e6,
                "prev": None,
                "start_of_sequence": True,
                "curr_to_prev_ego_rt": np.eye(4, dtype=np.float32),
                "sequence_group_idx": len(expanded),
                "occstress_save_token": True,
                "occstress_protocol": record,
                "occstress_corruption": record["corruption"],
                "gt_occ_path": _frame_occ_path(
                    frame, record["scene_name"]),
            })
        expanded.sort(key=lambda item: item["timestamp"])
        for index, item in enumerate(expanded):
            item["index"] = index
            item["clip_index"] = index
            item["sequence_group_idx"] = index
        self.flag = np.arange(len(expanded), dtype=np.int64)
        return expanded
