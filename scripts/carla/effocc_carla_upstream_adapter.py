#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""CARLA adapters for RoboBEV camera and Robo3D LiDAR corruptions."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from mmdet.datasets.builder import PIPELINES
from PIL import Image, ImageOps

from mmdet3d.datasets.pipelines.loading import PrepareImageInputs
from waymo_sdgocc_corruptions import (
    ADAPTER_VERSION as ROBO3D_ADAPTER_VERSION,
    CORRUPTIONS as LIDAR_CORRUPTIONS,
    PARAMETERS as LIDAR_PARAMETERS,
    ROBO3D_COMMIT,
    SEVERITIES as LIDAR_SEVERITIES,
    FramePointCloud,
    apply_corruption,
)


ADAPTER_VERSION = "effocc-carla-upstream-v1"
ROBOBEV_REPOSITORY = "https://github.com/Daniel-xsy/RoboBEV"
ROBOBEV_COMMIT = "3a32edaba9434dc27791bd25a1168951d091bd89"
CAMERA_CORRUPTIONS = (
    "CameraCrash",
    "FrameLost",
    "MotionBlur",
    "ColorQuant",
    "Brightness",
    "LowLight",
    "Fog",
    "Snow",
)
CAMERA_SEVERITIES = ("easy", "mid", "hard")
CAMERA_PARAMETERS = {
    "CameraCrash": {"easy": 2, "mid": 4, "hard": 5},
    "FrameLost": {"easy": 2, "mid": 4, "hard": 5},
    "MotionBlur": {"easy": 2, "mid": 4, "hard": 5},
    "ColorQuant": {"easy": 1, "mid": 2, "hard": 3},
    "Brightness": {"easy": 2, "mid": 4, "hard": 5},
    "LowLight": {"easy": 2, "mid": 3, "hard": 4},
    "Fog": {"easy": 2, "mid": 4, "hard": 5},
    "Snow": {"easy": 1, "mid": 2, "hard": 3},
}
IMAGECORRUPTIONS_NAMES = {
    "MotionBlur": "motion_blur",
    "Brightness": "brightness",
    "Fog": "fog",
    "Snow": "snow",
}
CAMERA_NAMES = ("CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK")
CARLA_TO_MODEL = np.diag((1.0, -1.0, 1.0, 1.0)).astype(np.float64)


def stable_seed(*parts: object) -> int:
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


@contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _low_light(
    image: np.ndarray,
    severity: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    brightness = (0.60, 0.50, 0.40, 0.30, 0.20)[severity]
    image_float = image.astype(np.float64) / 255.0
    minimum = float(image_float.min())
    maximum = float(image_float.max())
    if maximum > minimum:
        scaled = ((image_float - minimum) / (maximum - minimum)) ** 2
    else:
        scaled = np.zeros_like(image_float)
    scaled *= brightness
    poisson_scale = 10 * (60, 25, 12, 5, 3)[severity]
    noisy = np.clip(
        rng.poisson(scaled * poisson_scale) / poisson_scale, 0.0, 1.0
    )
    gaussian_scale = 0.1 * (0.08, 0.12, 0.18, 0.26, 0.38)[severity]
    noisy = np.clip(
        noisy + rng.normal(size=noisy.shape, scale=gaussian_scale),
        0.0,
        1.0,
    )
    return np.uint8(noisy * 255.0)


def _corrupt_view(
    image: np.ndarray,
    corruption: str,
    amount: int,
    seed: int,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if corruption == "ColorQuant":
        return np.asarray(
            ImageOps.posterize(Image.fromarray(image), 5 - amount)
        )
    if corruption == "LowLight":
        return _low_light(image, amount, np.random.RandomState(seed))

    from imagecorruptions import corrupt

    with numpy_seed(seed):
        return np.asarray(
            corrupt(
                image,
                corruption_name=IMAGECORRUPTIONS_NAMES[corruption],
                severity=amount,
            ),
            dtype=np.uint8,
        )


def corrupt_carla_views(
    images: Iterable[np.ndarray],
    corruption: str,
    severity: str,
    token: str,
) -> list[np.ndarray]:
    if corruption not in CAMERA_CORRUPTIONS:
        raise ValueError(f"unsupported camera corruption: {corruption}")
    if severity not in CAMERA_SEVERITIES:
        raise ValueError(f"unsupported camera severity: {severity}")
    views = [np.asarray(image, dtype=np.uint8).copy() for image in images]
    if len(views) != len(CAMERA_NAMES):
        raise ValueError(f"expected four CARLA views, got {len(views)}")
    if any(view.ndim != 3 or view.shape[2] != 3 for view in views):
        raise ValueError("CARLA camera corruption expects RGB images")

    amount = CAMERA_PARAMETERS[corruption][severity]
    if corruption == "CameraCrash":
        # RoboBEV draws with replacement and fixes the selected cameras for
        # the entire protocol. Only the camera index space changes from 6 to 4.
        rng = np.random.RandomState(
            stable_seed(ADAPTER_VERSION, "protocol-fixed", corruption, severity)
        )
        for camera_index in rng.choice(len(views), size=amount, replace=True):
            views[int(camera_index)].fill(0)
        return views

    frame_seed = stable_seed(ADAPTER_VERSION, token, corruption, severity)
    rng = np.random.RandomState(frame_seed)
    if corruption == "FrameLost":
        for view in views:
            if rng.rand() < amount / 6.0:
                view.fill(0)
        return views

    return [
        _corrupt_view(
            view,
            corruption,
            amount,
            (frame_seed ^ ((index + 1) * 0x9E3779B1)) & 0xFFFFFFFF,
        )
        for index, view in enumerate(views)
    ]


@PIPELINES.register_module()
class PrepareCorruptedCarlaImageInputs(PrepareImageInputs):
    """Apply RoboBEV at source resolution before EFFOcc preprocessing."""

    def __init__(self, corruption: str, severity: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if corruption not in CAMERA_CORRUPTIONS:
            raise ValueError(corruption)
        if severity not in CAMERA_SEVERITIES:
            raise ValueError(severity)
        self.corruption = corruption
        self.severity = severity

    def get_inputs(self, results: dict[str, Any], flip=None, scale=None):
        if tuple(self.data_config["cams"]) != CAMERA_NAMES:
            raise ValueError("unexpected CARLA camera order")
        paths = [results["curr"]["cams"][name]["data_path"] for name in CAMERA_NAMES]
        real_open = Image.open
        sources = []
        for path in paths:
            with real_open(path) as image:
                sources.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        outputs = corrupt_carla_views(
            sources,
            self.corruption,
            self.severity,
            str(results["sample_idx"]),
        )
        replacements = {
            str(Path(path).resolve()): Image.fromarray(output, mode="RGB")
            for path, output in zip(paths, outputs)
        }

        def open_corrupted(filename, *args, **kwargs):
            replacement = replacements.get(str(Path(filename).resolve()))
            if replacement is not None:
                return replacement.copy()
            return real_open(filename, *args, **kwargs)

        Image.open = open_corrupted
        try:
            return super().get_inputs(results, flip=flip, scale=scale)
        finally:
            Image.open = real_open
            for replacement in replacements.values():
                replacement.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(corruption={self.corruption!r}, "
            f"severity={self.severity!r})"
        )


def _carla_boxes(raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(raw_path, allow_pickle=True) as payload:
        annotations = payload["annotations"].tolist()
    boxes = []
    for annotation in annotations:
        transform = np.asarray(annotation["agent_to_ego"], dtype=np.float64)
        transform = CARLA_TO_MODEL @ transform @ CARLA_TO_MODEL
        size = np.asarray(annotation["size"], dtype=np.float64)
        heading = np.arctan2(transform[1, 0], transform[0, 0])
        boxes.append(
            np.concatenate((transform[:3, 3], size, [heading])).astype(
                np.float32
            )
        )
    if not boxes:
        return np.empty((0, 7), dtype=np.float32), np.empty(0, dtype=np.int8)
    # UniOcc annotations contain traffic agents. Mark all as dynamic so the
    # released incomplete-echo box rule can be applied without a lost PLY tag.
    return np.stack(boxes), np.ones(len(boxes), dtype=np.int8)


def _frame_point_cloud(
    clean: np.ndarray,
    raw_path: Path,
) -> FramePointCloud:
    xyz = clean[:, :3].astype(np.float32, copy=False)
    elevation = np.round(
        np.degrees(np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))),
        1,
    )
    _, beam_id = np.unique(elevation, return_inverse=True)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    column_id = np.floor((azimuth + np.pi) / (2.0 * np.pi) * 2048.0)
    column_id = np.mod(column_id.astype(np.int32), 2048)
    boxes, box_type = _carla_boxes(raw_path)
    frame = FramePointCloud(
        points=clean[:, :4].astype(np.float32, copy=True),
        laser_id=np.zeros(len(clean), dtype=np.int8),
        beam_id=beam_id.astype(np.int16, copy=False),
        column_id=column_id,
        return_id=np.zeros(len(clean), dtype=np.int8),
        boxes=boxes,
        box_type=box_type,
        lidar_origins={0: np.zeros(3, dtype=np.float64)},
    )
    frame.validate()
    return frame


@PIPELINES.register_module()
class CorruptEFFOccCarlaPoints:
    """Apply the released Robo3D equations to CARLA's surround LiDAR."""

    def __init__(
        self,
        corruption: str,
        severity: str,
        raw_root: str,
        robo3d_root: str,
    ) -> None:
        if corruption not in LIDAR_CORRUPTIONS:
            raise ValueError(corruption)
        if severity not in LIDAR_SEVERITIES:
            raise ValueError(severity)
        self.corruption = corruption
        self.severity = severity
        self.raw_root = Path(raw_root).resolve()
        self.robo3d_root = Path(robo3d_root).resolve()

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        clean = results["points"].tensor.detach().cpu().numpy()
        if clean.ndim != 2 or clean.shape[1] != 5:
            raise ValueError(f"expected CARLA points with shape (N, 5), got {clean.shape}")
        point_path = Path(results["pts_filename"])
        scene = point_path.parent.name
        frame_index = int(point_path.stem)
        raw_path = self.raw_root / scene / f"{frame_index}.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        token = str(results["sample_idx"])
        source = _frame_point_cloud(clean, raw_path)
        corrupted = apply_corruption(
            source,
            self.corruption,
            self.severity,
            token=token,
            robo3d_root=self.robo3d_root,
        )
        output = clean[corrupted.source_indices].copy()
        output[:, :3] = corrupted.points[:, :3]
        output[:, 3] = np.clip(corrupted.points[:, 3] / 255.0, 0.0, 1.0)
        results["points"] = results["points"].new_point(output)
        results["effocc_upstream"] = {
            "adapter_version": ADAPTER_VERSION,
            "corruption": self.corruption,
            "severity": self.severity,
            "statistics": corrupted.statistics,
            "token": token,
        }
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(corruption={self.corruption!r}, "
            f"severity={self.severity!r})"
        )


def implementation_manifest(track: str) -> dict[str, Any]:
    if track == "camera":
        return {
            "adapter_version": ADAPTER_VERSION,
            "camera_count": len(CAMERA_NAMES),
            "camera_names": list(CAMERA_NAMES),
            "corruptions": list(CAMERA_CORRUPTIONS),
            "parameters": CAMERA_PARAMETERS,
            "reference_commit": ROBOBEV_COMMIT,
            "reference_repository": ROBOBEV_REPOSITORY,
            "severity_order": list(CAMERA_SEVERITIES),
            "source_resolution_corruption": True,
            "adaptations": {
                "CameraCrash": (
                    "RoboBEV severity draws with replacement over the four "
                    "available CARLA cameras; fixed for the protocol"
                ),
                "FrameLost": "RoboBEV severity/6 per-view probability",
            },
        }
    if track == "lidar":
        return {
            "adapter_version": ADAPTER_VERSION,
            "corruptions": list(LIDAR_CORRUPTIONS),
            "parameters": LIDAR_PARAMETERS,
            "reference_commit": ROBO3D_COMMIT,
            "severity_order": list(LIDAR_SEVERITIES),
            "underlying_adapter_version": ROBO3D_ADAPTER_VERSION,
            "carla_adaptations": {
                "sensor": "single 360-degree CARLA semantic LiDAR",
                "beam_id": "0.1-degree quantized elevation (64 observed rows)",
                "column_id": "2048-bin azimuth index",
                "return_id": "single return",
                "incomplete_echo": "all UniOcc traffic-agent oriented boxes",
                "intensity": "CARLA I in [0,1] mapped to Robo3D [0,255]",
            },
        }
    raise ValueError(track)
