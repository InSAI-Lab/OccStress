#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Deterministic Robo3D-style point-cloud corruptions for OccStress-Waymo.

The public Robo3D repository does not contain its WOD-C generator.  This
module ports the released nuScenes-C definitions while adapting beam-based
operations to Waymo's five LiDARs and replacing semantic point masks with
Waymo boxes or a fitted ground plane.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROBO3D_REPOSITORY = "https://github.com/worldbench/Robo3D"
ROBO3D_COMMIT = "481a3b8634b2d291b84736cab1b546ed266efafa"
ADAPTER_VERSION = "waymo-robo3d-v1"

CORRUPTIONS = (
    "beam_missing",
    "cross_sensor",
    "crosstalk",
    "fog",
    "incomplete_echo",
    "motion_blur",
    "snow",
    "wet_ground",
)
SEVERITIES = ("light", "moderate", "heavy")

PARAMETERS: dict[str, dict[str, dict[str, float]]] = {
    "beam_missing": {
        "light": {"beam_drop_fraction": 0.25},
        "moderate": {"beam_drop_fraction": 0.50},
        "heavy": {"beam_drop_fraction": 0.75},
    },
    "cross_sensor": {
        "light": {
            "vertical_keep_fraction": 0.75,
            "horizontal_keep_fraction": 0.50,
        },
        "moderate": {
            "vertical_keep_fraction": 0.50,
            "horizontal_keep_fraction": 0.50,
        },
        "heavy": {
            "vertical_keep_fraction": 0.25,
            "horizontal_keep_fraction": 0.50,
        },
    },
    "crosstalk": {
        "light": {"point_fraction": 0.03, "noise_std": 3.0},
        "moderate": {"point_fraction": 0.07, "noise_std": 3.0},
        "heavy": {"point_fraction": 0.12, "noise_std": 3.0},
    },
    "fog": {
        "light": {"beta": 0.008},
        "moderate": {"beta": 0.05},
        "heavy": {"beta": 0.2},
    },
    "incomplete_echo": {
        "light": {"drop_fraction": 0.75},
        "moderate": {"drop_fraction": 0.85},
        "heavy": {"drop_fraction": 0.95},
    },
    "motion_blur": {
        "light": {"translation_std_m": 0.2},
        "moderate": {"translation_std_m": 0.3},
        "heavy": {"translation_std_m": 0.4},
    },
    "snow": {
        "light": {"snowfall_rate_mm_h": 0.5, "terminal_velocity_m_s": 2.0},
        "moderate": {
            "snowfall_rate_mm_h": 1.0,
            "terminal_velocity_m_s": 1.6,
        },
        "heavy": {
            "snowfall_rate_mm_h": 2.5,
            "terminal_velocity_m_s": 1.6,
        },
    },
    "wet_ground": {
        "light": {"water_height_m": 0.0002, "noise_floor": 0.2},
        "moderate": {"water_height_m": 0.0010, "noise_floor": 0.3},
        "heavy": {"water_height_m": 0.0012, "noise_floor": 0.7},
    },
}

_FOG_ALPHA_VALUES = np.asarray(
    (0.0, 0.005, 0.01, 0.02, 0.03, 0.06), dtype=np.float64
)
_FOG_TABLE_DIRS = {
    "light": "integral_lookup_tables_seg_light_0.008beta",
    "moderate": "integral_lookup_tables_seg_moderate_0.05beta",
    "heavy": "integral_lookup_tables_seg_heavy_0.2beta",
}
_DYNAMIC_BOX_TYPES = frozenset((1, 4))  # vehicle and cyclist


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def implementation_manifest() -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "corruptions": list(CORRUPTIONS),
        "parameters": PARAMETERS,
        "reference_commit": ROBO3D_COMMIT,
        "reference_repository": ROBO3D_REPOSITORY,
        "severity_order": list(SEVERITIES),
        "waymo_adaptations": {
            "beam_missing": (
                "Robo3D 25/50/75 percent beam loss, independently scaled "
                "to each Waymo LiDAR's observed beam rows"
            ),
            "cross_sensor": (
                "uniform vertical beam thinning followed by alternating "
                "range-image column thinning"
            ),
            "crosstalk": "released Robo3D point ratios and Gaussian scale",
            "fog": "released Robo3D fog lookup tables and equations",
            "incomplete_echo": (
                "released drop ratios applied to points inside Waymo "
                "vehicle/cyclist boxes"
            ),
            "motion_blur": (
                "released translation and per-point jitter definition"
            ),
            "snow": (
                "Robo3D snowfall-rate severities with deterministic "
                "beam-interception simulation; public particle patterns "
                "are sensor-specific and are not reused"
            ),
            "wet_ground": (
                "released water/noise severities and Fresnel equations "
                "applied to a robustly fitted Waymo ground plane"
            ),
        },
    }


@dataclass(frozen=True)
class FramePointCloud:
    points: np.ndarray
    laser_id: np.ndarray
    beam_id: np.ndarray
    column_id: np.ndarray
    return_id: np.ndarray
    boxes: np.ndarray
    box_type: np.ndarray
    lidar_origins: Mapping[int, np.ndarray]

    def validate(self) -> None:
        points = np.asarray(self.points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(f"points must have shape (N, 4), got {points.shape}")
        count = len(points)
        for name in ("laser_id", "beam_id", "column_id", "return_id"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,):
                raise ValueError(
                    f"{name} must have shape ({count},), got {value.shape}"
                )
        if np.asarray(self.boxes).ndim != 2 or self.boxes.shape[1] != 7:
            raise ValueError(f"boxes must have shape (M, 7), got {self.boxes.shape}")
        if np.asarray(self.box_type).shape != (len(self.boxes),):
            raise ValueError("box_type length does not match boxes")
        if not np.isfinite(points).all() or not np.isfinite(self.boxes).all():
            raise ValueError("point cloud or boxes contain non-finite values")


@dataclass(frozen=True)
class CorruptionResult:
    points: np.ndarray
    source_indices: np.ndarray
    statistics: dict[str, Any]

    def validate(self, source_count: int) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 4:
            raise ValueError(f"invalid corrupted point shape {self.points.shape}")
        if self.source_indices.shape != (len(self.points),):
            raise ValueError("source index count does not match corrupted points")
        if len(self.source_indices):
            if self.source_indices.min() < 0 or self.source_indices.max() >= source_count:
                raise ValueError("corrupted point source index is out of bounds")
        if not np.isfinite(self.points).all():
            raise ValueError("corruption produced non-finite points")


def normalize_waymo_points(points: np.ndarray) -> np.ndarray:
    """Map Waymo's normalized intensity to SDGOcc's nuScenes 0-255 scale."""
    output = np.asarray(points, dtype=np.float32).copy()
    if output.ndim != 2 or output.shape[1] != 4:
        raise ValueError(f"expected (N, 4) points, got {output.shape}")
    output[:, 3] = np.clip(output[:, 3], 0.0, 1.0) * 255.0
    return output


def _result(
    points: np.ndarray,
    source_indices: np.ndarray,
    source_count: int,
    corruption: str,
    severity: str,
    **statistics: Any,
) -> CorruptionResult:
    result = CorruptionResult(
        np.asarray(points, dtype=np.float32),
        np.asarray(source_indices, dtype=np.int64),
        {
            "adapter_version": ADAPTER_VERSION,
            "corruption": corruption,
            "input_points": int(source_count),
            "output_points": int(len(points)),
            "retained_fraction": (
                float(len(points) / source_count) if source_count else 0.0
            ),
            "severity": severity,
            **statistics,
        },
    )
    result.validate(source_count)
    return result


def _clean(frame: FramePointCloud, severity: str) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    return _result(
        points,
        np.arange(len(points)),
        len(points),
        "clean",
        severity,
    )


def _beam_missing(
    frame: FramePointCloud, severity: str, rng: np.random.Generator
) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    fraction = PARAMETERS["beam_missing"][severity]["beam_drop_fraction"]
    keep = np.ones(len(points), dtype=bool)
    dropped: dict[str, list[int]] = {}
    for laser in sorted(np.unique(frame.laser_id).tolist()):
        sensor = frame.laser_id == laser
        beams = np.unique(frame.beam_id[sensor])
        drop_count = min(len(beams) - 1, int(round(len(beams) * fraction)))
        selected = (
            rng.choice(beams, size=drop_count, replace=False)
            if drop_count
            else np.empty(0, dtype=beams.dtype)
        )
        keep &= ~(sensor & np.isin(frame.beam_id, selected))
        dropped[str(int(laser))] = [int(value) for value in selected]
    indices = np.flatnonzero(keep)
    return _result(
        points[indices],
        indices,
        len(points),
        "beam_missing",
        severity,
        dropped_beams=dropped,
    )


def _uniform_beam_subset(beams: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(round(len(beams) * fraction)))
    positions = np.floor(np.arange(count) * len(beams) / count).astype(int)
    return beams[positions]


def _cross_sensor(frame: FramePointCloud, severity: str) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    params = PARAMETERS["cross_sensor"][severity]
    keep = np.zeros(len(points), dtype=bool)
    selected_beams: dict[str, list[int]] = {}
    for laser in sorted(np.unique(frame.laser_id).tolist()):
        sensor = frame.laser_id == laser
        beams = np.unique(frame.beam_id[sensor])
        selected = _uniform_beam_subset(
            beams, params["vertical_keep_fraction"]
        )
        beam_keep = np.isin(frame.beam_id, selected)
        columns = frame.column_id[sensor]
        column_phase = int(columns.min(initial=0)) % 2
        column_keep = (frame.column_id % 2) == column_phase
        keep |= sensor & beam_keep & column_keep
        selected_beams[str(int(laser))] = [int(value) for value in selected]
    indices = np.flatnonzero(keep)
    return _result(
        points[indices],
        indices,
        len(points),
        "cross_sensor",
        severity,
        selected_beams=selected_beams,
    )


def _crosstalk(
    frame: FramePointCloud, severity: str, rng: np.random.Generator
) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    params = PARAMETERS["crosstalk"][severity]
    count = int(params["point_fraction"] * len(points))
    selected = rng.choice(len(points), size=count, replace=False)
    points[selected] += rng.normal(
        0.0, params["noise_std"], size=(count, 4)
    ).astype(np.float32)
    points[:, 3] = np.clip(points[:, 3], 0.0, 255.0)
    return _result(
        points,
        np.arange(len(points)),
        len(points),
        "crosstalk",
        severity,
        changed_points=int(count),
        changed_fraction=float(count / len(points)) if len(points) else 0.0,
    )


def _robo3d_root(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("ROBO3D_ROOT"):
        candidates.append(Path(os.environ["ROBO3D_ROOT"]))
    candidates.extend(
        (
            Path(__file__).resolve().parents[2] / "EXIST/3D/Robo3D",
            Path("/tmp/Robo3D-official"),
        )
    )
    for candidate in candidates:
        if (candidate / "create/nuscenes_c/fog").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Robo3D fog tables are missing; set ROBO3D_ROOT to the official "
        f"repository at commit {ROBO3D_COMMIT}"
    )


@lru_cache(maxsize=18)
def _fog_table_cached(
    severity: str, alpha: float, robo3d_root: str
) -> dict[float, tuple[float, float]]:
    root = Path(robo3d_root)
    path = (
        root
        / "create/nuscenes_c/fog"
        / _FOG_TABLE_DIRS[severity]
        / "original"
        / (
            "integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_"
            f"alpha_{alpha}.pickle"
        )
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fog_table(
    severity: str, alpha: float, robo3d_root: Path | None
) -> dict[float, tuple[float, float]]:
    root = _robo3d_root(robo3d_root)
    return _fog_table_cached(severity, alpha, str(root))


def _fog(
    frame: FramePointCloud,
    severity: str,
    rng: np.random.Generator,
    robo3d_root: Path | None,
) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    beta = PARAMETERS["fog"][severity]["beta"]
    alpha = float(rng.choice(_FOG_ALPHA_VALUES))
    table = _fog_table(severity, alpha, robo3d_root)

    distance = np.linalg.norm(points[:, :3], axis=1)
    original_intensity = points[:, 3].copy()
    points[:, 3] = np.rint(
        np.exp(-2.0 * alpha * distance) * points[:, 3]
    )

    distance_bins = np.clip(
        np.rint(distance * 10.0).astype(np.int64), 0, 2000
    )
    lookup = np.asarray(
        [table[round(index / 10.0, 1)] for index in range(2001)],
        dtype=np.float64,
    )
    table_values = lookup[distance_bins]
    fog_distance = table_values[:, 0]
    fog_response = (
        table_values[:, 1]
        * original_intensity
        * np.square(distance)
        * beta
        / (1.0e-6 / np.pi)
    )
    replace = (fog_response > points[:, 3]) & (distance > 1.0e-6)
    replacement_distance = fog_distance.copy()
    valid = replace & (fog_distance > 1.0e-6)
    replacement_distance[valid] += (
        10.0 * rng.beta(2.0, 20.0, size=int(valid.sum()))
    )
    scale = np.ones(len(points), dtype=np.float64)
    scale[replace] = replacement_distance[replace] / distance[replace]
    points[replace, :3] *= scale[replace, None].astype(np.float32)
    points[replace, 3] = fog_response[replace].astype(np.float32)
    return _result(
        points,
        np.arange(len(points)),
        len(points),
        "fog",
        severity,
        alpha=alpha,
        beta=beta,
        changed_points=int(replace.sum()),
        changed_fraction=float(replace.mean()) if len(points) else 0.0,
    )


def points_in_oriented_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    center = box[:3]
    length, width, height, heading = box[3:]
    relative = points[:, :3] - center
    cosine = np.cos(heading)
    sine = np.sin(heading)
    local_x = cosine * relative[:, 0] + sine * relative[:, 1]
    local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
    return (
        (np.abs(local_x) <= length / 2.0)
        & (np.abs(local_y) <= width / 2.0)
        & (np.abs(relative[:, 2]) <= height / 2.0)
    )


def _dynamic_box_mask(frame: FramePointCloud) -> np.ndarray:
    mask = np.zeros(len(frame.points), dtype=bool)
    for box, box_type in zip(frame.boxes, frame.box_type):
        if int(box_type) not in _DYNAMIC_BOX_TYPES:
            continue
        inside = points_in_oriented_box(frame.points, box)
        if int(inside.sum()) > 10:
            mask |= inside
    return mask


def _incomplete_echo(
    frame: FramePointCloud, severity: str, rng: np.random.Generator
) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    candidates = np.flatnonzero(_dynamic_box_mask(frame))
    drop_count = int(
        len(candidates)
        * PARAMETERS["incomplete_echo"][severity]["drop_fraction"]
    )
    dropped = (
        rng.choice(candidates, size=drop_count, replace=False)
        if drop_count
        else np.empty(0, dtype=np.int64)
    )
    keep = np.ones(len(points), dtype=bool)
    keep[dropped] = False
    indices = np.flatnonzero(keep)
    return _result(
        points[indices],
        indices,
        len(points),
        "incomplete_echo",
        severity,
        candidate_points=int(len(candidates)),
        dropped_points=int(drop_count),
    )


def _motion_blur(
    frame: FramePointCloud, severity: str, rng: np.random.Generator
) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    sigma = PARAMETERS["motion_blur"][severity]["translation_std_m"]
    translation = rng.normal(0.0, sigma, size=3)
    points[:, :3] += translation.astype(np.float32)
    jitter_scale = np.asarray((sigma * 0.1, sigma * 0.1, sigma * 0.05))
    jitter = rng.normal(0.0, jitter_scale, size=(len(points), 3))
    jitter = np.clip(jitter, -3.0 * sigma, 3.0 * sigma)
    points[:, :3] += jitter.astype(np.float32)
    return _result(
        points,
        np.arange(len(points)),
        len(points),
        "motion_blur",
        severity,
        translation_m=translation.tolist(),
        jitter_std_m=jitter_scale.tolist(),
    )


def _snow(
    frame: FramePointCloud, severity: str, rng: np.random.Generator
) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    params = PARAMETERS["snow"][severity]
    rate = params["snowfall_rate_mm_h"]
    velocity = params["terminal_velocity_m_s"]
    distance = np.linalg.norm(points[:, :3], axis=1)

    density = rate / velocity
    hit_probability = np.clip(
        density * (0.025 + 0.0015 * np.minimum(distance, 100.0)),
        0.0,
        0.45,
    )
    hit = rng.random(len(points)) < hit_probability
    attenuation = np.exp(-0.0025 * density * distance)
    points[:, 3] *= attenuation.astype(np.float32)

    hit_count = int(hit.sum())
    if hit_count:
        original_distance = distance[hit]
        flake_distance = original_distance * rng.beta(
            2.0, 5.0, size=hit_count
        )
        valid = original_distance > 1.0e-6
        scale = np.ones(hit_count, dtype=np.float64)
        scale[valid] = flake_distance[valid] / original_distance[valid]
        points[hit, :3] *= scale[:, None].astype(np.float32)
        points[hit, 3] = np.maximum(
            points[hit, 3],
            rng.uniform(0.35, 0.9, size=hit_count) * 255.0,
        )
    return _result(
        points,
        np.arange(len(points)),
        len(points),
        "snow",
        severity,
        changed_points=hit_count,
        changed_fraction=float(hit.mean()) if len(points) else 0.0,
        snowfall_rate_mm_h=rate,
        terminal_velocity_m_s=velocity,
    )


def _fit_ground_plane(frame: FramePointCloud) -> tuple[np.ndarray, float, np.ndarray]:
    xyz = np.asarray(frame.points[:, :3], dtype=np.float64)
    radial = np.linalg.norm(xyz[:, :2], axis=1)
    candidates = (
        (radial > 3.0)
        & (radial < 70.0)
        & (xyz[:, 2] > -0.45)
        & (xyz[:, 2] < 0.45)
    )
    dynamic = _dynamic_box_mask(frame)
    candidates &= ~dynamic
    selected = np.flatnonzero(candidates)
    if len(selected) < 1000:
        normal = np.asarray((0.0, 0.0, 1.0))
        offset = -float(np.median(xyz[:, 2]))
    else:
        active = selected
        coefficients = np.asarray((0.0, 0.0, 0.0))
        for _ in range(3):
            design = np.column_stack(
                (xyz[active, 0], xyz[active, 1], np.ones(len(active)))
            )
            coefficients = np.linalg.lstsq(
                design, xyz[active, 2], rcond=None
            )[0]
            residual = (
                xyz[selected, 2]
                - coefficients[0] * xyz[selected, 0]
                - coefficients[1] * xyz[selected, 1]
                - coefficients[2]
            )
            median = np.median(residual)
            mad = np.median(np.abs(residual - median))
            threshold = max(0.05, min(0.18, 3.5 * 1.4826 * mad))
            active = selected[np.abs(residual - median) < threshold]
        normal = np.asarray(
            (-coefficients[0], -coefficients[1], 1.0), dtype=np.float64
        )
        normal /= np.linalg.norm(normal)
        offset = -float(coefficients[2]) / np.linalg.norm(
            (-coefficients[0], -coefficients[1], 1.0)
        )
    signed = xyz @ normal + offset
    ground = (
        (signed > -0.12)
        & (signed < 0.22)
        & (radial > 2.0)
        & ~dynamic
    )
    return normal, offset, ground


def _fresnel_transmission(
    incident_angle: np.ndarray,
    reflectivity: np.ndarray,
    air_index: float = 1.0003,
    water_index: float = 1.33,
) -> tuple[np.ndarray, np.ndarray]:
    outgoing = np.arcsin(
        np.clip(np.sin(incident_angle) * air_index / water_index, -1.0, 1.0)
    )
    power_fraction = (
        np.cos(incident_angle)
        * air_index
        / water_index
        / np.maximum(np.cos(outgoing), 1.0e-6)
    )
    rs_air = (
        (air_index * np.cos(incident_angle) - water_index * np.cos(outgoing))
        / (air_index * np.cos(incident_angle) + water_index * np.cos(outgoing))
    )
    ts_air = (
        2.0
        * air_index
        * np.cos(incident_angle)
        / (air_index * np.cos(incident_angle) + water_index * np.cos(outgoing))
    )
    rp_air = (
        (water_index * np.cos(incident_angle) - air_index * np.cos(outgoing))
        / (water_index * np.cos(incident_angle) + air_index * np.cos(outgoing))
    )
    tp_air = (
        2.0
        * air_index
        * np.cos(incident_angle)
        / (water_index * np.cos(incident_angle) + air_index * np.cos(outgoing))
    )
    rs_air = np.square(rs_air)
    ts_air = np.square(ts_air) / power_fraction
    rp_air = np.square(rp_air)
    tp_air = np.square(tp_air) / power_fraction

    reverse_fraction = (
        np.cos(outgoing)
        * water_index
        / air_index
        / np.maximum(np.cos(incident_angle), 1.0e-6)
    )
    rs_water = np.square(
        (
            water_index * np.cos(outgoing)
            - air_index * np.cos(incident_angle)
        )
        / (
            water_index * np.cos(outgoing)
            + air_index * np.cos(incident_angle)
        )
    )
    ts_water = np.square(
        2.0
        * water_index
        * np.cos(outgoing)
        / (
            water_index * np.cos(outgoing)
            + air_index * np.cos(incident_angle)
        )
    ) / reverse_fraction
    rp_water = np.square(
        (
            air_index * np.cos(outgoing)
            - water_index * np.cos(incident_angle)
        )
        / (
            air_index * np.cos(outgoing)
            + water_index * np.cos(incident_angle)
        )
    )
    tp_water = np.square(
        2.0
        * water_index
        * np.cos(outgoing)
        / (
            air_index * np.cos(outgoing)
            + water_index * np.cos(incident_angle)
        )
    ) / reverse_fraction
    transmission_s = (
        ts_air * reflectivity * ts_water
        / np.maximum(1.0 - reflectivity * rs_water, 1.0e-6)
    )
    transmission_p = (
        tp_air * reflectivity * tp_water
        / np.maximum(1.0 - reflectivity * rp_water, 1.0e-6)
    )
    return transmission_s, transmission_p


def _wet_ground(frame: FramePointCloud, severity: str) -> CorruptionResult:
    points = normalize_waymo_points(frame.points)
    params = PARAMETERS["wet_ground"][severity]
    normal, offset, ground = _fit_ground_plane(frame)
    ground_indices = np.flatnonzero(ground)
    if len(ground_indices) < 1000:
        return _result(
            points,
            np.arange(len(points)),
            len(points),
            "wet_ground",
            severity,
            dropped_points=0,
            ground_points=int(len(ground_indices)),
            plane_normal=normal.tolist(),
            plane_offset=float(offset),
        )

    xyz = points[ground_indices, :3].astype(np.float64)
    sensor_origins = np.stack(
        [
            np.asarray(frame.lidar_origins[int(laser)], dtype=np.float64)
            for laser in frame.laser_id[ground_indices]
        ]
    )
    rays = xyz - sensor_origins
    distance = np.linalg.norm(rays, axis=1)
    cosine = np.clip(
        -(rays @ normal) / np.maximum(distance, 1.0e-6), 1.0e-3, 1.0
    )
    incident = np.arccos(cosine)
    intensity = points[ground_indices, 3].astype(np.float64) / 255.0

    slope, intercept = np.polyfit(distance, intensity / cosine, 1)
    emitted = np.maximum(1.0e-5, 15.0 * (slope * distance + intercept))
    reflectivity = np.clip(intensity / cosine / emitted, 0.05, 1.0)
    ts, tp = _fresnel_transmission(incident, reflectivity)
    wet_transmission = np.maximum(ts, tp)
    blend = np.clip(params["water_height_m"] / 0.0012, 0.0, 1.0)
    wet_reflectivity = (
        (1.0 - blend) * reflectivity
        + blend * wet_transmission / np.maximum(incident, 1.0e-3)
    )
    new_intensity = np.clip(
        emitted * cosine * wet_reflectivity, 0.0, intensity
    )
    adaptive_threshold = (
        params["noise_floor"]
        * np.maximum(0.003, np.percentile(intensity, 5))
        * cosine
    )
    keep_ground = new_intensity > adaptive_threshold
    keep = np.ones(len(points), dtype=bool)
    keep[ground_indices[~keep_ground]] = False
    points[ground_indices[keep_ground], 3] = (
        new_intensity[keep_ground] * 255.0
    ).astype(np.float32)
    indices = np.flatnonzero(keep)
    return _result(
        points[indices],
        indices,
        len(points),
        "wet_ground",
        severity,
        dropped_points=int((~keep_ground).sum()),
        ground_points=int(len(ground_indices)),
        plane_normal=normal.tolist(),
        plane_offset=float(offset),
        water_height_m=params["water_height_m"],
    )


def apply_corruption(
    frame: FramePointCloud,
    corruption: str,
    severity: str = "light",
    *,
    token: str,
    robo3d_root: Path | None = None,
) -> CorruptionResult:
    frame.validate()
    if corruption == "clean":
        result = _clean(frame, "clean")
    else:
        if corruption not in CORRUPTIONS:
            raise ValueError(
                f"unknown corruption {corruption!r}; expected {CORRUPTIONS}"
            )
        if severity not in SEVERITIES:
            raise ValueError(
                f"unknown severity {severity!r}; expected {SEVERITIES}"
            )
        rng = np.random.default_rng(
            stable_seed(ADAPTER_VERSION, token, corruption, severity)
        )
        if corruption == "beam_missing":
            result = _beam_missing(frame, severity, rng)
        elif corruption == "cross_sensor":
            result = _cross_sensor(frame, severity)
        elif corruption == "crosstalk":
            result = _crosstalk(frame, severity, rng)
        elif corruption == "fog":
            result = _fog(frame, severity, rng, robo3d_root)
        elif corruption == "incomplete_echo":
            result = _incomplete_echo(frame, severity, rng)
        elif corruption == "motion_blur":
            result = _motion_blur(frame, severity, rng)
        elif corruption == "snow":
            result = _snow(frame, severity, rng)
        elif corruption == "wet_ground":
            result = _wet_ground(frame, severity)
        else:  # pragma: no cover
            raise AssertionError(corruption)
    result.validate(len(frame.points))
    return result


def main() -> None:
    print(json.dumps(implementation_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
