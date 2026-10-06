# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Occ3D-Waymo zero-shot camera adapter for SparseWorld-TC."""

from __future__ import annotations

import functools
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mmdet3d.datasets.pipelines.formating import (
    Collect4D,
    DefaultFormatBundle3D,
)
from mmdet3d.datasets.pipelines.test_time_aug import MultiScaleFlipAug3D


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[2]
for directory in (
    PROJECT_ROOT / "scripts/waymo",
    PROJECT_ROOT / "runtime/CVT-Occ/python-packages",
    Path(os.environ["SPARSEWORLD_PYTHON_PACKAGES"])
    if "SPARSEWORLD_PYTHON_PACKAGES" in os.environ else None,
):
    if directory is None:
        continue
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from effocc_waymo_camera_corruptions import (  # noqa: E402
    SEVERITY as CAMERA_SEVERITIES,
    corrupt_views,
)
from waymo_occstress_common import (  # noqa: E402
    OCC3D_CLASSES,
    map_waymo_semantics,
)


CAMERA_NAMES = (
    "FRONT",
    "FRONT_LEFT",
    "SIDE_LEFT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
)
# The KITTI converter stores SIDE_LEFT and FRONT_RIGHT in image_3 and image_2.
KITTI_CAMERA_INDICES = (0, 1, 3, 2, 4)
CORRUPTION_NAMES = {
    "brightness": "Brightness",
    "camera_crash": "CameraCrash",
    "color_quant": "ColorQuant",
    "fog": "Fog",
    "frame_lost": "FrameLost",
    "low_light": "LowLight",
    "motion_blur": "MotionBlur",
    "snow": "Snow",
}
FRAME_PROTOCOLS = {"current", "history_k1", "all_frame"}
FINAL_HEIGHT = 256
FINAL_WIDTH = 704


@functools.lru_cache(maxsize=32)
def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


@functools.lru_cache(maxsize=8)
def _load_corrupted_views(
    paths: tuple[str, ...],
    corruption: str,
    severity: str,
    sample_key: str,
) -> tuple[np.ndarray, ...]:
    views = [_load_rgb(path).copy() for path in paths]
    return tuple(corrupt_views(views, corruption, severity, sample_key))


def _yaw(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _lookup(mapping: dict, key: int) -> object:
    if key in mapping:
        return mapping[key]
    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]
    raise KeyError(key)


class SparseWorldWaymoDataset(torch.utils.data.Dataset):
    """Build the fixed 5,978-anchor OccStress-Waymo H4/F6 camera track."""

    CLASSES = OCC3D_CLASSES

    def __init__(
        self,
        base_info: str | Path,
        pose_file: str | Path,
        camera_root: str | Path,
        waymo_root: str | Path,
        protocol: dict,
        history_length: int = 4,
        future_length: int = 6,
    ) -> None:
        from occstress.datasets.metadata import load_metadata
        from occstress.datasets.paths import resolve_occstress_path
        self.base_info_path = resolve_occstress_path(base_info, dataset='waymo')
        self.pose_file_path = resolve_occstress_path(pose_file, dataset='waymo')
        self.camera_root = resolve_occstress_path(camera_root, dataset='waymo')
        self.waymo_root = resolve_occstress_path(waymo_root, dataset='waymo')
        payload = load_metadata(self.base_info_path, trusted_pickle=True)
        if payload.get("metadata", {}).get("dataset") not in ("Occ3D-Waymo", "OccStress-Waymo"):
            raise ValueError(f"Unexpected Waymo metadata: {self.base_info_path}")
        with self.pose_file_path.open("rb") as stream:
            self.pose_all = pickle.load(stream)

        self.protocol = dict(protocol)
        self.history_length = history_length
        self.future_length = future_length
        self._validate_protocol()
        self.scenes: dict[str, list[dict]] = {}
        self.anchors: list[tuple[str, int]] = []
        for scene_name in sorted(payload["infos"]):
            frames = sorted(
                payload["infos"][scene_name], key=lambda item: item["frame_idx"]
            )
            frame_indices = [int(frame["frame_idx"]) for frame in frames]
            if any(
                later - earlier != 5
                for earlier, later in zip(frame_indices, frame_indices[1:])
            ):
                raise ValueError(f"Non-2Hz frames in Waymo scene {scene_name}")
            self.scenes[str(scene_name)] = frames
            for frame_index in range(history_length, len(frames) - future_length):
                self.anchors.append((str(scene_name), frame_index))
        if len(self.anchors) != 5978:
            raise ValueError(
                f"Expected 5,978 strict H4/F6 anchors, got {len(self.anchors)}"
            )
        self.flag = np.zeros(len(self.anchors), dtype=np.uint8)
        self.camera_layout = self._detect_camera_layout()
        self.tta_transform = MultiScaleFlipAug3D(
            img_scale=(FINAL_WIDTH, FINAL_HEIGHT),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                DefaultFormatBundle3D(
                    class_names=list(OCC3D_CLASSES), with_label=False
                ),
                Collect4D(
                    keys=["img", "temporal_semantics", "temporal_ego_states"],
                    meta_keys=(
                        "filename", "ori_shape", "img_shape", "pad_shape",
                        "lidar2img", "ego2lidar", "sample_idx", "scene_name",
                        "frame_idx",
                    ),
                ),
            ],
        )

    def _validate_protocol(self) -> None:
        corruption = self.protocol.get("corruption", "clean")
        if corruption == "clean":
            return
        if corruption not in CORRUPTION_NAMES:
            raise ValueError(f"Unsupported Waymo corruption: {corruption}")
        source_name = CORRUPTION_NAMES[corruption]
        if source_name not in CAMERA_SEVERITIES:
            raise ValueError(f"Corruption implementation missing: {corruption}")
        severity = self.protocol.get("severity")
        if severity not in CAMERA_SEVERITIES[source_name]:
            raise ValueError(f"Unsupported severity: {severity}")
        if self.protocol.get("frame_protocol") not in FRAME_PROTOCOLS:
            raise ValueError(
                f"Unsupported temporal protocol: "
                f"{self.protocol.get('frame_protocol')}"
            )

    def _detect_camera_layout(self) -> str:
        first_scene = min(self.scenes)
        first_frame = self.scenes[first_scene][0]
        canonical = (
            self.camera_root / first_scene
            / f"{int(first_frame['frame_idx']):03d}" / "FRONT.jpg"
        )
        if canonical.is_file():
            return "waymo_occstress"
        kitti = (
            self.camera_root / "training/image_0"
            / f"{int(first_frame['sample_idx']):07d}.png"
        )
        if kitti.is_file():
            return "kitti"
        raise FileNotFoundError(
            f"Could not detect Waymo camera layout below {self.camera_root}"
        )

    def __len__(self) -> int:
        return len(self.anchors)

    def _corruption_active(self, history_rank: int) -> bool:
        if self.protocol.get("corruption", "clean") == "clean":
            return False
        frame_protocol = self.protocol["frame_protocol"]
        if frame_protocol == "current":
            return history_rank == 0
        if frame_protocol == "history_k1":
            return history_rank in (0, 1)
        if frame_protocol == "all_frame":
            return history_rank > 0
        raise AssertionError(frame_protocol)

    def _camera_path(
        self, scene_name: str, frame: dict, camera_rank: int
    ) -> Path:
        if self.camera_layout == "waymo_occstress":
            return (
                self.camera_root / scene_name / f"{int(frame['frame_idx']):03d}"
                / f"{CAMERA_NAMES[camera_rank]}.jpg"
            )
        image_index = KITTI_CAMERA_INDICES[camera_rank]
        return (
            self.camera_root / f"training/image_{image_index}"
            / f"{int(frame['sample_idx']):07d}.png"
        )

    def _camera_metadata(
        self, scene_name: str, frame: dict, camera_rank: int
    ) -> dict:
        scene = _lookup(self.pose_all, int(scene_name))
        pose_frame = _lookup(scene, int(frame["frame_idx"]))
        return _lookup(pose_frame, camera_rank)

    @staticmethod
    def _lidar2img(
        current_pose: np.ndarray,
        source_pose: np.ndarray,
        camera: dict,
    ) -> np.ndarray:
        sensor2ego = np.asarray(camera["sensor2ego"], dtype=np.float64)
        camera_to_global = source_pose @ sensor2ego
        camera_to_current = np.linalg.inv(current_pose) @ camera_to_global
        intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
        return (intrinsic @ np.linalg.inv(camera_to_current)).astype(np.float32)

    @staticmethod
    def _resize_crop(
        image: np.ndarray, lidar2img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        resize = max(FINAL_HEIGHT / height, FINAL_WIDTH / width)
        resized_width = int(width * resize)
        resized_height = int(height * resize)
        crop_x = max(0, resized_width - FINAL_WIDTH) // 2
        crop_y = resized_height - FINAL_HEIGHT
        if crop_y < 0:
            raise ValueError(
                f"Image {image.shape} cannot cover {FINAL_HEIGHT}x{FINAL_WIDTH}"
            )
        transformed = Image.fromarray(image).resize(
            (resized_width, resized_height)
        ).crop(
            (crop_x, crop_y, crop_x + FINAL_WIDTH, crop_y + FINAL_HEIGHT)
        )
        ida = np.eye(4, dtype=np.float32)
        ida[0, 0] = resize
        ida[1, 1] = resize
        ida[0, 2] = -crop_x
        ida[1, 2] = -crop_y
        output = np.asarray(transformed, dtype=np.uint8)
        if output.shape != (FINAL_HEIGHT, FINAL_WIDTH, 3):
            raise ValueError(f"Bad transformed image shape: {output.shape}")
        return output, ida @ lidar2img

    @staticmethod
    def _stp3_relative(current_pose: np.ndarray, other_pose: np.ndarray) -> np.ndarray:
        relative = np.linalg.inv(current_pose) @ other_pose
        # Waymo ego coordinates are x-forward/y-left; ST-P3 controls are
        # x-right/y-forward.
        return np.array(
            [-relative[1, 3], relative[0, 3], _wrap_angle(_yaw(relative))],
            dtype=np.float32,
        )

    def _ego_state(self, frames: list[dict], frame_index: int) -> torch.Tensor:
        current_pose = np.asarray(frames[frame_index]["pose_mat"], dtype=np.float64)
        history = [
            self._stp3_relative(
                current_pose,
                np.asarray(frames[frame_index - lag]["pose_mat"], dtype=np.float64),
            )
            for lag in range(1, self.history_length + 1)
        ]
        step_seconds = 0.5
        velocity = -history[0] / step_seconds
        previous_velocity = (history[0] - history[1]) / step_seconds
        acceleration = (velocity - previous_velocity) / step_seconds
        command = np.asarray(frames[frame_index]["pose_mode"], dtype=np.float32)
        state = np.concatenate(
            [acceleration, command, velocity, np.concatenate(history)]
        ).astype(np.float32)
        if state.shape != (21,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid SparseWorld Waymo ego state: {state}")
        return torch.from_numpy(state[None, :])

    def _load_gt(self, scene_name: str, frame: dict) -> dict[str, np.ndarray]:
        from occstress.datasets.paths import release_occ_path
        path = release_occ_path(frame.get('occ_path', ''), dataset='waymo') or (
            self.waymo_root / "clean_occ" / scene_name
            / f"{int(frame['frame_idx']):03d}_04.npz"
        )
        with np.load(path, allow_pickle=False) as payload:
            semantics = map_waymo_semantics(payload["voxel_label"])
            mask_lidar = np.asarray(payload["origin_voxel_state"], dtype=bool)
            mask_camera = np.asarray(payload["final_voxel_state"], dtype=bool)
        if semantics.shape != (200, 200, 16):
            raise ValueError(f"Bad Waymo occupancy shape: {semantics.shape}")
        if semantics.min() < 0 or semantics.max() > 17:
            raise ValueError("Mapped Waymo labels fall outside the Occ3D ontology")
        return {
            "voxel_semantics": semantics,
            "mask_lidar": mask_lidar,
            "mask_camera": mask_camera,
        }

    def __getitem__(self, index: int) -> dict:
        scene_name, frame_index = self.anchors[index]
        frames = self.scenes[scene_name]
        current = frames[frame_index]
        current_pose = np.asarray(current["pose_mat"], dtype=np.float64)

        images = []
        filenames = []
        lidar2img = []
        original_shapes = []
        for history_rank in range(self.history_length + 1):
            source = frames[frame_index - history_rank]
            paths = [
                self._camera_path(scene_name, source, camera_rank)
                for camera_rank in range(len(CAMERA_NAMES))
            ]
            if self._corruption_active(history_rank):
                views = _load_corrupted_views(
                    tuple(str(path) for path in paths),
                    CORRUPTION_NAMES[self.protocol["corruption"]],
                    self.protocol["severity"],
                    source["token"],
                )
            else:
                views = [_load_rgb(str(path)).copy() for path in paths]
            source_pose = np.asarray(source["pose_mat"], dtype=np.float64)
            for camera_rank, (path, image) in enumerate(zip(paths, views)):
                original_shapes.append(image.shape)
                projection = self._lidar2img(
                    current_pose,
                    source_pose,
                    self._camera_metadata(scene_name, source, camera_rank),
                )
                image, projection = self._resize_crop(image, projection)
                # SparseWorld's normalization follows mmcv's BGR convention.
                images.append(np.ascontiguousarray(image[..., ::-1]))
                filenames.append(str(path))
                lidar2img.append(projection)

        temporal_semantics = {
            horizon: self._load_gt(scene_name, frames[frame_index + horizon])
            for horizon in range(1, self.future_length + 1)
        }
        transformed_shapes = [image.shape for image in images]
        results = {
            "img": images,
            "filename": filenames,
            "ori_shape": original_shapes,
            "img_shape": transformed_shapes,
            "pad_shape": transformed_shapes,
            "lidar2img": lidar2img,
            "ego2lidar": np.eye(4, dtype=np.float32),
            "sample_idx": current["token"],
            "scene_name": scene_name,
            "frame_idx": int(current["frame_idx"]),
            "temporal_semantics": temporal_semantics,
            "temporal_ego_states": {0: self._ego_state(frames, frame_index)},
        }
        return self.tta_transform(results)
