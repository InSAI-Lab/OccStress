#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""EFFOcc adapter for deterministic Robo3D-style Waymo corruptions."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from mmdet.datasets.builder import PIPELINES

from waymo_sdgocc_corruptions import (
    ADAPTER_VERSION as ROBO3D_ADAPTER_VERSION,
    FramePointCloud,
    apply_corruption,
)


ADAPTER_VERSION = "effocc-waymo-robo3d-v2"


class SceneCache:
    """Small per-DataLoader-worker cache for compressed scene packs."""

    def __init__(
        self,
        index_root: Path,
        sensor_root: Path,
        alignment_root: Path,
        max_scenes: int = 2,
    ) -> None:
        self.index_root = Path(index_root).resolve()
        self.sensor_root = Path(sensor_root).resolve()
        self.alignment_root = Path(alignment_root).resolve()
        self.max_scenes = max_scenes
        self._scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, scene: str) -> dict[str, Any]:
        scene = str(scene).zfill(3)
        cached = self._scenes.pop(scene, None)
        if cached is not None:
            self._scenes[scene] = cached
            return cached

        index_path = self.index_root / f"{scene}.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        data_path = self.sensor_root / index["data_path"]
        with np.load(data_path) as source:
            arrays = {name: source[name] for name in source.files}
        alignment_path = self.alignment_root / f"{scene}.npz"
        with np.load(alignment_path) as source:
            alignment = {name: source[name] for name in source.files}
        frame_indices = np.asarray(
            [frame["frame_index"] for frame in index["frames"]],
            dtype=np.int16,
        )
        if not np.array_equal(alignment["frame_indices"], frame_indices):
            raise ValueError(f"{alignment_path}: frame indices do not match")
        if not np.array_equal(
            alignment["source_counts"],
            np.diff(arrays["frame_offsets"]),
        ):
            raise ValueError(f"{alignment_path}: source point counts do not match")
        by_frame = {
            int(frame["frame_index"]): offset
            for offset, frame in enumerate(index["frames"])
        }
        origins = {
            int(record["laser_id"]): np.asarray(
                record["extrinsic"], dtype=np.float64
            )[:3, 3]
            for record in index["lidar_calibrations"]
        }
        cached = {
            "arrays": arrays,
            "alignment": alignment,
            "by_frame": by_frame,
            "index": index,
            "origins": origins,
        }
        self._scenes[scene] = cached
        while len(self._scenes) > self.max_scenes:
            self._scenes.popitem(last=False)
        return cached


def frame_from_scene(
    cached: dict[str, Any],
    frame_index: int,
) -> tuple[str, FramePointCloud]:
    try:
        offset = cached["by_frame"][int(frame_index)]
    except KeyError as exc:
        raise KeyError(
            f"frame {frame_index} is not part of the canonical 2 Hz stream"
        ) from exc
    arrays = cached["arrays"]
    alignment = cached["alignment"]
    point_start, point_end = arrays["frame_offsets"][offset : offset + 2]
    box_start, box_end = arrays["box_offsets"][offset : offset + 2]
    point_slice = slice(int(point_start), int(point_end))
    box_slice = slice(int(box_start), int(box_end))
    metadata = cached["index"]["frames"][offset]
    byte_start, byte_end = alignment["byte_offsets"][offset : offset + 2]
    source_count = int(point_end - point_start)
    keep = np.unpackbits(
        alignment["packed_keep"][int(byte_start) : int(byte_end)],
        count=source_count,
        bitorder="little",
    ).astype(bool, copy=False)
    expected_kept = int(alignment["kept_counts"][offset])
    if int(keep.sum()) != expected_kept:
        raise ValueError(
            f"{metadata['token']}: corrupt EFFOcc alignment mask "
            f"({int(keep.sum())} != {expected_kept})"
        )
    frame = FramePointCloud(
        points=arrays["points"][point_slice][keep],
        laser_id=arrays["laser_id"][point_slice][keep],
        beam_id=arrays["beam_id"][point_slice][keep],
        column_id=arrays["column_id"][point_slice][keep],
        return_id=arrays["return_id"][point_slice][keep],
        boxes=arrays["boxes"][box_slice],
        box_type=arrays["box_type"][box_slice],
        lidar_origins=cached["origins"],
    )
    frame.validate()
    return str(metadata["token"]), frame


@PIPELINES.register_module()
class CorruptEFFOccWaymoPoints:
    """Corrupt EFFOcc points before BEV augmentation and depth projection."""

    def __init__(
        self,
        corruption: str,
        severity: str,
        index_root: str,
        sensor_root: str,
        alignment_root: str,
        robo3d_root: str,
        cache_scenes: int = 2,
        verify_alignment: bool = True,
    ) -> None:
        if corruption == "clean":
            raise ValueError("clean EFFOcc inference must omit this transform")
        self.corruption = corruption
        self.severity = severity
        self.robo3d_root = Path(robo3d_root).resolve()
        self.verify_alignment = verify_alignment
        self.cache = SceneCache(
            Path(index_root),
            Path(sensor_root),
            Path(alignment_root),
            max_scenes=cache_scenes,
        )

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        sample_idx = int(results["sample_idx"])
        scene = sample_idx % 1_000_000 // 1_000
        frame_index = sample_idx % 1_000
        token, source = frame_from_scene(
            self.cache.get(f"{scene:03d}"),
            frame_index,
        )
        expected_token = f"waymo-val-{scene:03d}-{frame_index:03d}"
        if token != expected_token:
            raise ValueError(f"token mismatch: {token} != {expected_token}")

        clean = results["points"].tensor.detach().cpu().numpy()
        if clean.shape != (len(source.points), 5):
            raise ValueError(
                f"{token}: EFFOcc/Waymo point shapes differ: "
                f"{clean.shape} vs {source.points.shape}"
            )
        if self.verify_alignment and not np.array_equal(
            clean[:, :3], source.points[:, :3]
        ):
            error = float(
                np.max(np.abs(clean[:, :3] - source.points[:, :3]))
            )
            raise ValueError(
                f"{token}: EFFOcc/Waymo point order differs; max error={error}"
            )

        corrupted = apply_corruption(
            source,
            self.corruption,
            self.severity,
            token=token,
            robo3d_root=self.robo3d_root,
        )
        output = clean[corrupted.source_indices].copy()
        output[:, :3] = corrupted.points[:, :3]
        # Robo3D operates on an SDGOcc-compatible [0, 255] intensity scale.
        output[:, 3] = np.tanh(corrupted.points[:, 3] / 255.0)
        results["points"] = results["points"].new_point(output)
        results["effocc_upstream"] = {
            "adapter_version": ADAPTER_VERSION,
            "corruption": self.corruption,
            "robo3d_adapter_version": ROBO3D_ADAPTER_VERSION,
            "severity": self.severity,
            "statistics": corrupted.statistics,
            "token": token,
        }
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"corruption={self.corruption!r}, severity={self.severity!r})"
        )
