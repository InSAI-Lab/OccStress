#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Create a right-handed, model-ready view of OccStress-CARLA manual protocols."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np


MIRROR_Y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-occupancy-files", type=int, default=3960)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_matrix(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {value.shape}")
    return MIRROR_Y @ value @ MIRROR_Y


def rotation_to_quaternion(rotation: np.ndarray) -> list[float]:
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(np.float32).tolist()


def convert_frame(
    frame: dict, source_root: Path, output_root: Path
) -> dict:
    output = copy.deepcopy(frame)
    pose = mirror_matrix(frame["pose_mat"])
    output["pose_mat"] = pose.astype(np.float32).tolist()
    output["ego2global"] = pose.astype(np.float32).tolist()
    output["ego2global_translation"] = pose[:3, 3].astype(np.float32).tolist()
    output["ego2global_rotation"] = rotation_to_quaternion(pose[:3, :3])

    trajectories = np.asarray(frame["gt_ego_fut_trajs"], dtype=np.float32).copy()
    trajectories[..., 1] *= -1.0
    output["gt_ego_fut_trajs"] = trajectories.tolist()
    command = np.asarray(frame["gt_ego_fut_cmd"], dtype=np.float32).copy()
    command[[0, 1]] = command[[1, 0]]
    output["gt_ego_fut_cmd"] = command.tolist()
    output["pose_mode"] = command.tolist()
    output["occ_path"] = rewrite_occ_path(
        frame["occ_path"], source_root, output_root)
    return output


def rewrite_occ_path(path: str, source_root: Path, output_root: Path) -> str:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    value = Path(path).expanduser()
    if ".." in value.parts:
        raise ValueError(f"occupancy path contains parent traversal: {value}")
    candidate = value if value.is_absolute() else source_root / value
    try:
        relative = candidate.resolve().relative_to(source_root)
    except ValueError:
        normalized = str(value).replace("\\", "/")
        marker = f"/{source_root.name}/"
        if marker not in normalized:
            raise ValueError(f"occupancy path is outside source root {source_root}: {value}")
        relative = Path(normalized.split(marker, 1)[1])
    destination = (output_root / relative).resolve()
    destination.relative_to(output_root)
    return str(destination)


def convert_record(record: dict, source_root: Path, output_root: Path) -> dict:
    output = copy.deepcopy(record)
    for key in ("history", "future_targets"):
        for frame in output[key]:
            frame["occ_path"] = rewrite_occ_path(
                frame["occ_path"], source_root, output_root)
    for key in ("current_input", "target"):
        output[key]["occ_path"] = rewrite_occ_path(
            output[key]["occ_path"], source_root, output_root)

    traffic = output.get("traffic_transform")
    if traffic:
        # Already mirrored diagnostic controls still need the coordinate conversion.
        trajectory = np.asarray(traffic["current_gt_ego_fut_trajs"], dtype=np.float32).copy()
        trajectory[..., 1] *= -1.0
        traffic["current_gt_ego_fut_trajs"] = trajectory.tolist()
        command = np.asarray(traffic["current_gt_ego_fut_cmd"], dtype=np.float32)
        traffic["current_gt_ego_fut_cmd"] = command[[1, 0, 2]].tolist()

    misalignment = output.get("misalignment")
    if misalignment:
        misalignment["delta_rt"] = [
            mirror_matrix(matrix).astype(np.float32).tolist()
            for matrix in misalignment["delta_rt"]
        ]
        misalignment["current_delta_rt"] = mirror_matrix(
            misalignment["current_delta_rt"]
        ).astype(np.float32).tolist()
        misalignment["dy_m"] = (-np.asarray(
            misalignment["dy_m"], dtype=np.float32)).tolist()
        misalignment["yaw_deg"] = (-np.asarray(
            misalignment["yaw_deg"], dtype=np.float32)).tolist()
        misalignment["current_dy_m"] = -float(
            misalignment["current_dy_m"])
        misalignment["current_yaw_deg"] = -float(
            misalignment["current_yaw_deg"])

    output["coordinate_adapter"] = {
        "source": "CARLA left-handed x-forward/y-right/z-up",
        "target": "Occ3D right-handed x-forward/y-left/z-up",
        "occupancy_transform": "flip axis 1",
        "pose_transform": "M @ pose @ M",
        "mirror_matrix": MIRROR_Y.tolist(),
    }
    return output


def convert_npz(source: Path, destination: Path, overwrite: bool) -> int:
    if destination.is_file() and not overwrite:
        with np.load(destination, allow_pickle=False) as payload:
            semantics = payload["semantics"]
            infov = payload["infov"]
        if semantics.shape == (200, 200, 16) and infov.shape == semantics.shape:
            return destination.stat().st_size
    with np.load(source, allow_pickle=False) as payload:
        semantics = np.ascontiguousarray(np.flip(payload["semantics"], axis=1))
        infov = np.ascontiguousarray(np.flip(payload["infov"], axis=1))
    if semantics.shape != (200, 200, 16):
        raise ValueError(f"{source}: invalid shape {semantics.shape}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, semantics=semantics, infov=infov)
    os.replace(temporary, destination)
    return destination.stat().st_size


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("source and output roots must differ")

    source_base = source_root / "meta/manual/carla_base_info.pkl"
    with source_base.open("rb") as stream:
        base_info = pickle.load(stream)
    converted_base = copy.deepcopy(base_info)
    converted_base["metadata"].update({
        "source_coordinate_convention": base_info["metadata"][
            "coordinate_convention"],
        "coordinate_convention": (
            "Occ3D right-handed ego-local x-forward/y-left/z-up"),
        "model_view": True,
    })
    converted_base["infos"] = {
        scene: [
            convert_frame(frame, source_root, output_root)
            for frame in frames
        ]
        for scene, frames in base_info["infos"].items()
    }

    npz_sources = sorted((source_root / "occ").rglob("*.npz"))
    if len(npz_sources) != args.expected_occupancy_files:
        raise ValueError(
            f"expected {args.expected_occupancy_files} occupancy files, "
            f"found {len(npz_sources)}")
    converted_bytes = 0
    for source in npz_sources:
        destination = output_root / source.relative_to(source_root)
        converted_bytes += convert_npz(source, destination, args.overwrite)

    protocol_sources = sorted((source_root / "protocols/manual").rglob("*.pkl"))
    record_count = 0
    for source in protocol_sources:
        with source.open("rb") as stream:
            records = pickle.load(stream)
        converted = [
            convert_record(record, source_root, output_root)
            for record in records
        ]
        record_count += len(converted)
        atomic_pickle(output_root / source.relative_to(source_root), converted)

    metadata_root = output_root / "meta/manual"
    atomic_pickle(metadata_root / "carla_base_info.pkl", converted_base)
    for name in ("class_mapping.json", "dataset.json"):
        with (source_root / "meta/manual" / name).open() as stream:
            payload = json.load(stream)
        payload["source_coordinate_convention"] = payload.get(
            "coordinate_convention")
        payload["coordinate_convention"] = (
            "Occ3D right-handed ego-local x-forward/y-left/z-up")
        payload["model_view"] = True
        if name == "dataset.json":
            payload["control_policy"] = (
                "GT future ego poses; command order right/left/straight; "
                "right for local y<=-2m, left for y>=2m")
        atomic_json(metadata_root / name, payload)

    manifest = {
        "status": "success",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_base_info_sha256": sha256(source_base),
        "base_info": str((metadata_root / "carla_base_info.pkl").resolve()),
        "base_info_sha256": sha256(metadata_root / "carla_base_info.pkl"),
        "coordinate_convention": (
            "Occ3D right-handed ego-local x-forward/y-left/z-up"),
        "occupancy_files": len(npz_sources),
        "occupancy_bytes": converted_bytes,
        "protocols": len(protocol_sources),
        "protocol_records": record_count,
        "expected_anchors_per_protocol": 330,
        "present_occupied_classes": [4, 11, 14, 15, 16],
        "free_class": 17,
    }
    atomic_json(metadata_root / "model_view_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
