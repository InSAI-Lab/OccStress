# OccStress adapter/portability modifications; see docs/source-imports.json.
"""UniOcc-CARLA camera-only input adapter for SparseWorld-TC."""

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
from mmdet3d.datasets.pipelines.loading import RandomTransformImage
from mmdet3d.datasets.pipelines.test_time_aug import MultiScaleFlipAug3D


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[2]
for directory in (
    PROJECT_ROOT / "scripts/carla",
    PROJECT_ROOT / "scripts/waymo",
    PROJECT_ROOT / "runtime/CVT-Occ/python-packages",
    Path(os.environ["SPARSEWORLD_PYTHON_PACKAGES"])
    if "SPARSEWORLD_PYTHON_PACKAGES" in os.environ else None,
):
    if directory is None:
        continue
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from effocc_carla_upstream_adapter import (  # noqa: E402
    CAMERA_CORRUPTIONS,
    CAMERA_SEVERITIES,
    corrupt_carla_views,
)


CAMERA_NAMES = ("CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK")
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
OPENCV_TO_CARLA_CAMERA = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
CARLA_TO_MODEL = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
CLASS_NAMES = (
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
)


@functools.lru_cache(maxsize=128)
def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _yaw(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


class SparseWorldCarlaDataset(torch.utils.data.Dataset):
    """Build strict H4/F6 CARLA anchors without nuScenes-only side files."""

    CLASSES = CLASS_NAMES

    def __init__(
        self,
        base_info: str | Path,
        protocol: dict,
        history_length: int = 4,
        future_length: int = 6,
    ) -> None:
        from occstress.datasets.metadata import load_metadata
        from occstress.datasets.paths import resolve_occstress_path
        self.base_info_path = resolve_occstress_path(base_info, dataset='carla')
        payload = load_metadata(self.base_info_path, trusted_pickle=True)
        if payload.get("metadata", {}).get("dataset") not in ("UniOcc-CARLA", "OccStress-CARLA"):
            raise ValueError(f"Unexpected CARLA metadata: {self.base_info_path}")

        self.protocol = dict(protocol)
        self.history_length = history_length
        self.future_length = future_length
        self._validate_protocol()
        self.scenes = {}
        self.anchors = []
        for scene_name in sorted(payload["infos"]):
            frames = sorted(
                payload["infos"][scene_name], key=lambda item: item["frame_idx"]
            )
            if [frame["frame_idx"] for frame in frames] != list(range(len(frames))):
                raise ValueError(f"Non-contiguous frames in {scene_name}")
            self.scenes[scene_name] = frames
            for frame_index in range(history_length, len(frames) - future_length):
                self.anchors.append((scene_name, frame_index))
        if len(self.anchors) != 330:
            raise ValueError(f"Expected 330 strict H4/F6 anchors, got {len(self.anchors)}")
        self.flag = np.zeros(len(self.anchors), dtype=np.uint8)

        self.image_transform = RandomTransformImage(
            ida_aug_conf={
                "resize_lim": (0.32, 0.88),
                "final_dim": (256, 704),
                # The established EFFOcc CARLA adapter uses 0.25 for square
                # 800x800 views. It keeps the horizon near the same output
                # row as SparseWorld's nuScenes preprocessing instead of
                # retaining only the bottom of the image.
                "bot_pct_lim": (0.25, 0.25),
                "rot_lim": (0.0, 0.0),
                "H": 800,
                "W": 800,
                "rand_flip": False,
            },
            training=False,
        )
        self.tta_transform = MultiScaleFlipAug3D(
            img_scale=(800, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                DefaultFormatBundle3D(
                    class_names=list(CLASS_NAMES), with_label=False
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

    @staticmethod
    def _camera_path(frame, camera):
        from occstress.datasets.paths import resolve_occstress_path
        return str(resolve_occstress_path(frame['cams'][camera]['data_path'], dataset='carla'))

    def _validate_protocol(self) -> None:
        corruption = self.protocol.get("corruption", "clean")
        if corruption == "clean":
            return
        if corruption not in CORRUPTION_NAMES:
            raise ValueError(f"Unsupported CARLA corruption: {corruption}")
        if CORRUPTION_NAMES[corruption] not in CAMERA_CORRUPTIONS:
            raise ValueError(f"Corruption implementation missing: {corruption}")
        if self.protocol.get("severity") not in CAMERA_SEVERITIES:
            raise ValueError(f"Unsupported severity: {self.protocol.get('severity')}")
        if self.protocol.get("frame_protocol") not in FRAME_PROTOCOLS:
            raise ValueError(
                f"Unsupported temporal protocol: {self.protocol.get('frame_protocol')}"
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

    @staticmethod
    def _camera_to_ego(camera: dict) -> np.ndarray:
        raw = np.asarray(camera["extrinsics"], dtype=np.float64)
        if raw.shape != (4, 4):
            raise ValueError(f"Bad CARLA camera extrinsic shape: {raw.shape}")
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = (
            CARLA_TO_MODEL @ raw[:3, :3] @ OPENCV_TO_CARLA_CAMERA
        )
        output[:3, 3] = CARLA_TO_MODEL @ raw[:3, 3]
        return output

    @classmethod
    def _lidar2img(
        cls,
        current_pose: np.ndarray,
        source_frame: dict,
        camera_name: str,
    ) -> np.ndarray:
        camera = source_frame["cams"][camera_name]
        camera_to_global = (
            np.asarray(source_frame["pose_mat"], dtype=np.float64)
            @ cls._camera_to_ego(camera)
        )
        camera_to_current = np.linalg.inv(current_pose) @ camera_to_global
        intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
        projection = np.eye(4, dtype=np.float64)
        projection[:3, :3] = intrinsic
        return (projection @ np.linalg.inv(camera_to_current)).astype(np.float32)

    @staticmethod
    def _stp3_relative(current_pose: np.ndarray, other_pose: np.ndarray) -> np.ndarray:
        relative = np.linalg.inv(current_pose) @ other_pose
        # ST-P3 controls use x-right/y-forward, while OccStress-CARLA is
        # x-forward/y-left.
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
            raise ValueError(f"Invalid SparseWorld ego state: {state}")
        return torch.from_numpy(state[None, :])

    @staticmethod
    def _load_gt(frame: dict) -> dict[str, np.ndarray]:
        from occstress.datasets.paths import resolve_occstress_path
        with np.load(resolve_occstress_path(frame["occ_path"], dataset='carla'), allow_pickle=False) as payload:
            semantics = np.asarray(payload["semantics"], dtype=np.uint8)
            visibility = np.asarray(
                payload["infov"] if "infov" in payload.files
                else np.ones_like(semantics),
                dtype=bool,
            )
            output = {
                "voxel_semantics": semantics,
                # SparseWorld-TC does not consume these masks at inference;
                # expose CARLA's released camera visibility for pipeline
                # compatibility without inventing a LiDAR mask.
                "mask_lidar": visibility,
                "mask_camera": visibility,
            }
        if output["voxel_semantics"].shape != (200, 200, 16):
            raise ValueError(
                f"Bad CARLA occupancy shape: {output['voxel_semantics'].shape}"
            )
        labels = np.unique(output["voxel_semantics"])
        if labels.min() < 0 or labels.max() > 17:
            raise ValueError(f"CARLA labels outside Occ3D ontology: {labels}")
        return output

    def __getitem__(self, index: int) -> dict:
        scene_name, frame_index = self.anchors[index]
        frames = self.scenes[scene_name]
        current = frames[frame_index]
        current_pose = np.asarray(current["pose_mat"], dtype=np.float64)

        images = []
        filenames = []
        lidar2img = []
        for history_rank in range(self.history_length + 1):
            source = frames[frame_index - history_rank]
            views = [
                _load_rgb(self._camera_path(source, camera)).copy()
                for camera in CAMERA_NAMES
            ]
            if self._corruption_active(history_rank):
                views = corrupt_carla_views(
                    views,
                    CORRUPTION_NAMES[self.protocol["corruption"]],
                    self.protocol["severity"],
                    source["token"],
                )
            for camera_name, image in zip(CAMERA_NAMES, views):
                if image.shape != (800, 800, 3):
                    raise ValueError(
                        f"Expected 800x800 CARLA RGB, got {image.shape}"
                    )
                # SparseWorld's GPU normalization expects mmcv-style BGR.
                images.append(np.ascontiguousarray(image[..., ::-1]))
                filenames.append(self._camera_path(source, camera_name))
                lidar2img.append(
                    self._lidar2img(current_pose, source, camera_name)
                )

        temporal_semantics = {
            horizon: self._load_gt(frames[frame_index + horizon])
            for horizon in range(1, self.future_length + 1)
        }
        results = {
            "img": images,
            "filename": filenames,
            "lidar2img": lidar2img,
            "ego2lidar": np.eye(4, dtype=np.float32),
            "sample_idx": current["token"],
            "scene_name": scene_name,
            "frame_idx": frame_index,
            "temporal_semantics": temporal_semantics,
            "temporal_ego_states": {0: self._ego_state(frames, frame_index)},
        }
        results = self.image_transform(results)
        return self.tta_transform(results)
