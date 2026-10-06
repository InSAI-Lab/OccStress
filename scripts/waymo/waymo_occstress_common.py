#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Shared schema and transforms for the OccStress-Waymo manual track."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


WAYMO_TO_OCC3D = {
    0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
    8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17,
}
OCC3D_CLASSES = (
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
)
SEVERITIES = ("easy", "mid", "hard")
TEMPORAL_PATTERNS = {
    "current_only": (0,),
    "history_only": (-4, -3, -2, -1),
    "recent_burst": (-1, 0),
}
CONTENT_FAMILIES = ("semantic", "hole", "dropout")
MATRIX_FAMILY = "misalignment"
TRAFFIC_FAMILY = "traffic"
OBSERVED_OFFSETS_SECONDS = (-2.0, -1.5, -1.0, -0.5, 0.0)
FUTURE_OFFSETS_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
MISALIGNMENT_CONFIGS = {
    "easy": {
        "mode": "constant_bias", "dx_range": 0.8, "dy_range": 0.8,
        "yaw_deg_range": 4.0,
    },
    "mid": {
        "mode": "drift", "dx_range": 2.4, "dy_range": 2.4,
        "yaw_deg_range": 12.0,
    },
    "hard": {
        "mode": "drift", "dx_range": 4.5, "dy_range": 4.5,
        "yaw_deg_range": 20.0,
    },
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def map_waymo_semantics(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    unknown = sorted(set(np.unique(raw).tolist()) - set(WAYMO_TO_OCC3D))
    if unknown:
        raise ValueError(f"unmapped Waymo labels: {unknown}")
    mapped = np.empty(raw.shape, dtype=np.uint8)
    for source, target in WAYMO_TO_OCC3D.items():
        mapped[raw == source] = target
    return mapped


def mirror_y_semantics(semantics: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(semantics, axis=1))


def mirror_y_rt(rt: np.ndarray) -> np.ndarray:
    mirror = np.diag([1.0, -1.0, 1.0, 1.0])
    return mirror @ np.asarray(rt, dtype=np.float64) @ mirror


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return a normalized wxyz quaternion without external dependencies."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s,
                          (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s,
                          (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def relative_xy(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.linalg.inv(origin) @ target
    return delta[:2, 3].astype(np.float32)


def command_from_trajectory(cumulative_xy: np.ndarray) -> np.ndarray:
    # Match the official nuScenes converter: x is lateral in its local frame.
    lateral = float(cumulative_xy[-1, 0])
    if lateral >= 2.0:
        return np.array([1, 0, 0], dtype=np.float32)
    if lateral <= -2.0:
        return np.array([0, 1, 0], dtype=np.float32)
    return np.array([0, 0, 1], dtype=np.float32)


def make_planar_delta(dx_m: float, dy_m: float, yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    delta = np.eye(4, dtype=np.float32)
    delta[0, 0], delta[0, 1] = c, -s
    delta[1, 0], delta[1, 1] = s, c
    delta[0, 3], delta[1, 3] = dx_m, dy_m
    return delta


def build_misalignment_series(
    token: str, severity: str, affected_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match the OccStress-nuScenes magnitude policy for an arbitrary active mask."""
    config = MISALIGNMENT_CONFIGS[severity]
    affected_mask = np.asarray(affected_mask, dtype=np.uint8)
    length = len(affected_mask)
    rng = np.random.default_rng(
        stable_seed("misalignment", severity, token, length))
    max_dx = float(rng.uniform(-config["dx_range"], config["dx_range"]))
    max_dy = float(rng.uniform(-config["dy_range"], config["dy_range"]))
    max_yaw = float(
        rng.uniform(-config["yaw_deg_range"], config["yaw_deg_range"]))

    dx = np.zeros(length, dtype=np.float32)
    dy = np.zeros(length, dtype=np.float32)
    yaw = np.zeros(length, dtype=np.float32)
    deltas = np.repeat(np.eye(4, dtype=np.float32)[None], length, axis=0)
    active = np.flatnonzero(affected_mask)
    if len(active) == 0:
        return deltas, dx, dy, yaw

    for rank, index in enumerate(active, start=1):
        scale = 1.0
        if config["mode"] == "drift":
            scale = (rank / len(active)) ** 1.25
        dx[index] = max_dx * scale
        dy[index] = max_dy * scale
        yaw[index] = max_yaw * scale
        deltas[index] = make_planar_delta(dx[index], dy[index], yaw[index])
    return deltas, dx, dy, yaw
