#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Build canonical UniOcc-CARLA metadata and 38 OccStress manual protocols."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WAYMO_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "waymo"
sys.path.insert(0, str(WAYMO_SCRIPT_DIR))

from waymo_occstress_common import (  # noqa: E402
    CONTENT_FAMILIES,
    FUTURE_OFFSETS_SECONDS,
    MATRIX_FAMILY,
    OBSERVED_OFFSETS_SECONDS,
    OCC3D_CLASSES,
    SEVERITIES,
    TEMPORAL_PATTERNS,
    TRAFFIC_FAMILY,
    atomic_json,
    build_misalignment_series,
    relative_xy,
    rotation_matrix_to_quaternion,
    sha256sum,
)


UNIOCC_TO_OCC3D = {
    0: 0,
    1: 4,
    2: 2,
    3: 6,
    4: 7,
    5: 8,
    6: 16,
    7: 11,
    8: 14,
    9: 15,
    10: 17,
}
CAMERA_NAMES = ("CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK")
EXPECTED_SCENES = {
    "scene_Town05": 120,
    "scene_Town05_Opt": 120,
    "scene_Town07": 120,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=6)
    parser.add_argument("--expected-frames", type=int, default=360)
    parser.add_argument("--expected-anchors", type=int, default=330)
    parser.add_argument("--overwrite-clean", action="store_true")
    parser.add_argument("--skip-protocols", action="store_true")
    return parser.parse_args()


def atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def map_uniocc_semantics(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    unknown = sorted(set(np.unique(raw).tolist()) - set(UNIOCC_TO_OCC3D))
    if unknown:
        raise ValueError(f"unmapped UniOcc labels: {unknown}")
    mapped = np.empty(raw.shape, dtype=np.uint8)
    for source, target in UNIOCC_TO_OCC3D.items():
        mapped[raw == source] = target
    return mapped


def canonical_clean_path(root: Path, scene_name: str, token: str) -> Path:
    return root / "occ" / "clean" / scene_name / token / "labels.npz"


def write_clean_occ(
    source: Path, destination: Path, overwrite: bool
) -> tuple[list[int], int]:
    if destination.exists() and not overwrite:
        with np.load(destination) as payload:
            semantics = payload["semantics"]
            infov = payload["infov"]
        if semantics.shape != (200, 200, 16) or infov.shape != semantics.shape:
            raise ValueError(f"invalid existing canonical occupancy: {destination}")
        return np.unique(semantics).astype(int).tolist(), destination.stat().st_size

    with np.load(source, allow_pickle=True) as payload:
        semantics = map_uniocc_semantics(payload["occ_label"])
        infov = np.asarray(payload["occ_mask_camera"], dtype=bool)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, semantics=semantics, infov=infov)
    os.replace(temporary, destination)
    return np.unique(semantics).astype(int).tolist(), destination.stat().st_size


def carla_command(cumulative_xy: np.ndarray) -> np.ndarray:
    """CARLA uses x-forward/y-right in its left-handed ego-local frame."""
    lateral = float(cumulative_xy[-1, 1])
    if lateral >= 2.0:
        return np.array([0, 1, 0], dtype=np.float32)
    if lateral <= -2.0:
        return np.array([1, 0, 0], dtype=np.float32)
    return np.array([0, 0, 1], dtype=np.float32)


def enrich_controls(scene_frames: list[dict], future_length: int) -> None:
    for index, frame in enumerate(scene_frames):
        available = scene_frames[index + 1:index + 1 + future_length]
        cumulative = np.zeros((future_length, 2), dtype=np.float32)
        masks = np.zeros(future_length, dtype=np.float32)
        for step, target in enumerate(available):
            cumulative[step] = relative_xy(
                frame["ego2global"], target["ego2global"])
            masks[step] = 1
        per_step = cumulative.copy()
        if future_length > 1:
            per_step[1:] -= cumulative[:-1]
        command = (
            carla_command(cumulative)
            if len(available) == future_length
            else np.array([0, 0, 1], dtype=np.float32)
        )
        frame["gt_ego_fut_trajs"] = per_step
        frame["gt_ego_fut_masks"] = masks
        frame["gt_ego_fut_cmd"] = command
        frame["pose_mode"] = command
        frame["gt_ego_lcf_feat"] = np.zeros(9, dtype=np.float32)
        frame["ego2global_translation"] = frame["ego2global"][:3, 3].astype(
            np.float32)
        frame["ego2global_rotation"] = rotation_matrix_to_quaternion(
            frame["ego2global"]).astype(np.float32)
        frame["pose_mat"] = frame["ego2global"].astype(np.float32)


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def build_frames(args: argparse.Namespace) -> tuple[list[dict], set[int], int]:
    scene_info_path = args.data_root / "scene_infos.pkl"
    with scene_info_path.open("rb") as stream:
        scene_infos = pickle.load(stream)
    observed = {
        scene["scene_name"]: len(scene["occ_in_scene_paths"])
        for scene in scene_infos
    }
    if observed != EXPECTED_SCENES:
        raise ValueError(f"unexpected scene inventory: {observed}")

    frames = []
    labels_present: set[int] = set()
    clean_bytes = 0
    for scene in sorted(scene_infos, key=lambda item: item["scene_name"]):
        scene_name = scene["scene_name"]
        for frame_index, relative_occ in enumerate(scene["occ_in_scene_paths"]):
            expected = f"{scene_name}/{frame_index}.npz"
            if relative_occ != expected:
                raise ValueError(f"unexpected frame ordering: {relative_occ}")
            source_occ = (args.data_root / relative_occ).resolve()
            token = f"carla-val-{scene_name.removeprefix('scene_')}-{frame_index:03d}"
            clean_occ = canonical_clean_path(
                args.output_root, scene_name, token).resolve()
            present, size = write_clean_occ(
                source_occ, clean_occ, args.overwrite_clean)
            labels_present.update(present)
            clean_bytes += size

            with np.load(source_occ, allow_pickle=True) as payload:
                pose = np.asarray(
                    payload["ego_to_world_transformation"], dtype=np.float64)
                cameras = payload["cameras"]
                annotations = payload["annotations"]
                if pose.shape != (4, 4) or len(cameras) != 4:
                    raise ValueError(f"invalid source metadata: {source_occ}")

                camera_records = {}
                camera_paths = []
                for camera_index, camera_name in enumerate(CAMERA_NAMES):
                    image = (
                        args.data_root
                        / scene_name
                        / f"{frame_index}_camera_{camera_index}.png"
                    ).resolve()
                    if not image.is_file():
                        raise FileNotFoundError(image)
                    camera = cameras[camera_index]
                    if camera["cam_name"] != camera_name:
                        raise ValueError(f"camera order mismatch: {source_occ}")
                    camera_paths.append(str(image))
                    camera_records[camera_name] = {
                        "data_path": str(image),
                        "cam_intrinsic": np.asarray(
                            camera["intrinsics"], dtype=np.float32),
                        "extrinsics": np.asarray(
                            camera["extrinsics"], dtype=np.float32),
                    }

                lidar_path = (
                    args.data_root
                    / scene_name
                    / f"{frame_index}_surround_lidar.ply"
                ).resolve()
                annotation_categories = sorted({
                    int(annotation["category_id"]) for annotation in annotations
                })

            frames.append({
                "token": token,
                "scene_name": scene_name,
                "scene_token": f"carla-val-{scene_name.removeprefix('scene_')}",
                "frame_idx": frame_index,
                "sample_idx": frame_index,
                "timestamp": frame_index * 500_000,
                "occ_path": str(clean_occ),
                "raw_occ_path": str(source_occ),
                "camera_paths": camera_paths,
                "cams": camera_records,
                "lidar_path": str(lidar_path),
                "annotation_count": len(annotations),
                "annotation_categories": annotation_categories,
                "ego2global": pose,
            })
    return frames, labels_present, clean_bytes


def frame_ref(frame: dict, path: Path | str | None = None, source="clean") -> dict:
    return {
        "token": frame["token"],
        "occ_path": str(path or frame["occ_path"]),
        "occ_source": source,
    }


def corrupted_path(
    root: Path, family: str, severity: str, frame: dict
) -> Path:
    return (
        root / "occ" / "manual" / family / severity
        / frame["scene_name"] / frame["token"] / "labels.npz"
    )


def traffic_path(root: Path, frame: dict) -> Path:
    return (
        root / "occ" / "manual" / "traffic"
        / frame["scene_name"] / frame["token"] / "labels.npz"
    )


def protocol_record(
    window: list[dict],
    root: Path,
    family="clean",
    severity=None,
    pattern="clean",
) -> dict:
    history, current, future = window[:4], window[4], window[5:]
    active = set(TEMPORAL_PATTERNS.get(pattern, ()))

    def input_ref(frame: dict, offset: int) -> dict:
        if family in CONTENT_FAMILIES and offset in active:
            return frame_ref(
                frame,
                corrupted_path(root, family, severity, frame),
                source="corrupted",
            )
        if family == TRAFFIC_FAMILY:
            return frame_ref(frame, traffic_path(root, frame), source="corrupted")
        return frame_ref(frame)

    history_refs = [
        input_ref(frame, offset)
        for frame, offset in zip(history, (-4, -3, -2, -1))
    ]
    current_ref = input_ref(current, 0)
    target = (
        frame_ref(current, traffic_path(root, current), source="corrupted")
        if family == TRAFFIC_FAMILY else frame_ref(current)
    )
    future_refs = [
        (
            frame_ref(frame, traffic_path(root, frame), source="corrupted")
            if family == TRAFFIC_FAMILY else frame_ref(frame)
        )
        for frame in future
    ]
    record = {
        "schema_version": 1,
        "dataset": "UniOcc-CARLA",
        "sample_id": (
            f"{current['token']}__{family}__{severity or 'none'}__{pattern}"
        ),
        "anchor_token": current["token"],
        "scene_name": current["scene_name"],
        "scene_token": current["scene_token"],
        "history_length": 4,
        "future_length": 6,
        "history_tokens": [frame["token"] for frame in history],
        "history": history_refs,
        "current_input": current_ref,
        "target": target,
        "future_targets": future_refs,
        "frame_protocol": pattern,
        "corruption": {"type": family, "severity": severity},
        "corruption_family": family,
        "severity": severity,
        "observed_offsets_seconds": list(OBSERVED_OFFSETS_SECONDS),
        "future_offsets_seconds": list(FUTURE_OFFSETS_SECONDS),
        "traffic_mirror": family == TRAFFIC_FAMILY,
    }
    if family == TRAFFIC_FAMILY:
        trajectory = np.asarray(
            current["gt_ego_fut_trajs"], dtype=np.float32).copy()
        trajectory[:, 1] *= -1
        command = np.asarray(
            current["gt_ego_fut_cmd"], dtype=np.float32)[[1, 0, 2]]
        record["traffic_transform"] = {
            "mirror_axis": "ego_local_y",
            "mirror_matrix": np.diag([1.0, -1.0, 1.0, 1.0]).tolist(),
            "relative_pose_policy": "M @ relative_pose @ M",
            "trajectory_component_scale": [1.0, -1.0],
            "command_permutation": [1, 0, 2],
            "current_gt_ego_fut_trajs": trajectory.tolist(),
            "current_gt_ego_fut_masks": to_builtin(
                current["gt_ego_fut_masks"]),
            "current_gt_ego_fut_cmd": command.tolist(),
        }
    if family == MATRIX_FAMILY:
        affected_mask = np.array(
            [offset in active for offset in (-4, -3, -2, -1)],
            dtype=np.uint8,
        )
        deltas, dx_m, dy_m, yaw_deg = build_misalignment_series(
            current["token"], severity, affected_mask)
        current_active = 0 in active
        current_deltas, current_dx, current_dy, current_yaw = (
            build_misalignment_series(
                current["token"] + "__current",
                severity,
                np.array([current_active], dtype=np.uint8),
            )
        )
        for index, ref in enumerate(history_refs):
            ref["rt_source"] = (
                "misalignment" if affected_mask[index] else "clean")
        current_ref["rt_source"] = (
            "misalignment" if current_active else "clean")
        record["misalignment"] = {
            "severity": severity,
            "affected_mask": affected_mask.tolist(),
            "current_active": current_active,
            "delta_rt": deltas.tolist(),
            "current_delta_rt": current_deltas[0].tolist(),
            "dx_m": dx_m.tolist(),
            "dy_m": dy_m.tolist(),
            "yaw_deg": yaw_deg.tolist(),
            "current_dx_m": float(current_dx[0]),
            "current_dy_m": float(current_dy[0]),
            "current_yaw_deg": float(current_yaw[0]),
        }
    return record


def protocol_path(
    root: Path, family: str, severity: str | None, pattern: str
) -> Path:
    base = root / "protocols" / "manual"
    if family == "clean":
        return base / "clean" / "H4_F6_val_backbone.pkl"
    if family == TRAFFIC_FAMILY:
        return base / family / "H4_F6_val_backbone.pkl"
    return base / family / severity / f"{pattern}_H4_F6_val_backbone.pkl"


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.history_length != 4 or args.future_length != 6:
        raise ValueError("OccStress-CARLA currently fixes the OccStress H4/F6 contract")

    frames, labels_present, clean_bytes = build_frames(args)
    if len(frames) != args.expected_frames:
        raise RuntimeError(
            f"expected {args.expected_frames} frames, got {len(frames)}")

    by_scene = defaultdict(list)
    for frame in frames:
        by_scene[frame["scene_name"]].append(frame)
    for scene_frames in by_scene.values():
        scene_frames.sort(key=lambda frame: frame["timestamp"])
        enrich_controls(scene_frames, args.future_length)

    width = args.history_length + 1 + args.future_length
    windows = []
    for scene_frames in by_scene.values():
        windows.extend(
            scene_frames[start:start + width]
            for start in range(len(scene_frames) - width + 1)
        )
    if len(windows) != args.expected_anchors:
        raise RuntimeError(
            f"expected {args.expected_anchors} anchors, got {len(windows)}")

    meta = args.output_root / "meta" / "manual"
    public_frames = [to_builtin(frame) for frame in frames]
    atomic_json(meta / "frames_H4_F6_val.json", public_frames)
    atomic_json(meta / "anchors_H4_F6_val.json", [
        {
            "anchor_token": window[4]["token"],
            "scene_token": window[4]["scene_token"],
            "history_tokens": [frame["token"] for frame in window[:4]],
            "current_token": window[4]["token"],
            "future_tokens": [frame["token"] for frame in window[5:]],
        }
        for window in windows
    ])
    base_info = {
        "metadata": {
            "dataset": "UniOcc-CARLA",
            "split": "Carla-2Hz-val",
            "coordinate_convention": (
                "CARLA left-handed ego-local x-forward/y-right/z-up"
            ),
        },
        "infos": {
            scene: [to_builtin(frame) for frame in scene_frames]
            for scene, scene_frames in sorted(by_scene.items())
        },
    }
    atomic_pickle(meta / "carla_base_info.pkl", base_info)

    scene_info = args.data_root / "scene_infos.pkl"
    dataset_meta = {
        "schema_version": 1,
        "dataset": "UniOcc-CARLA",
        "benchmark_name": "OccStress-CARLA",
        "source_repository": "tasl-lab/uniocc",
        "source_revision": "e81775b36e376f591a1145ca054b72fd541a67eb",
        "split": "Carla-2Hz-val",
        "domain": "simulated",
        "native_hz": 2,
        "evaluation_hz": 2,
        "frame_count": len(frames),
        "scene_count": len(by_scene),
        "anchor_count": len(windows),
        "history_length": args.history_length,
        "future_length": args.future_length,
        "window_definition": "4 history + current + 6 future",
        "observed_offsets_seconds": list(OBSERVED_OFFSETS_SECONDS),
        "future_offsets_seconds": list(FUTURE_OFFSETS_SECONDS),
        "grid_shape": [200, 200, 16],
        "voxel_size_m": [0.4, 0.4, 0.4],
        "point_cloud_range_m": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        "coordinate_convention": (
            "CARLA left-handed ego-local x-forward/y-right/z-up"
        ),
        "control_policy": (
            "GT future ego poses; right for local y>=2m, left for y<=-2m"
        ),
        "source_data_root": str(args.data_root),
        "source_scene_info": str(scene_info.resolve()),
        "source_scene_info_sha256": sha256sum(scene_info),
        "canonical_clean_bytes": clean_bytes,
        "labels_present_occ3d": sorted(labels_present),
        "scene_length_histogram": {
            str(length): count
            for length, count in sorted(Counter(
                len(value) for value in by_scene.values()).items())
        },
    }
    atomic_json(meta / "dataset.json", dataset_meta)
    atomic_json(meta / "class_mapping.json", {
        "source": "UniOcc common ontology (occ_label)",
        "target": "Occ3D 18-class",
        "mapping": {
            str(key): value for key, value in UNIOCC_TO_OCC3D.items()
        },
        "classes": list(OCC3D_CLASSES),
        "unsupported_target_classes_are_absent": True,
        "labels_present": sorted(labels_present),
    })

    protocol_count = 0
    if not args.skip_protocols:
        specs = [("clean", None, "clean")]
        specs.extend(
            (family, severity, pattern)
            for family in CONTENT_FAMILIES + (MATRIX_FAMILY,)
            for severity in SEVERITIES
            for pattern in TEMPORAL_PATTERNS
        )
        specs.append((TRAFFIC_FAMILY, None, "all_frame"))
        if len(specs) != 38:
            raise RuntimeError(f"expected 38 protocols, got {len(specs)}")
        for family, severity, pattern in specs:
            records = [
                protocol_record(
                    window, args.output_root, family, severity, pattern)
                for window in windows
            ]
            atomic_pickle(
                protocol_path(args.output_root, family, severity, pattern),
                records,
            )
        protocol_count = len(specs)

    atomic_json(meta / "generation_summary.json", {
        **dataset_meta,
        "protocol_count": protocol_count,
        "base_info": str((meta / "carla_base_info.pkl").resolve()),
    })
    print(json.dumps({
        "status": "success",
        "frames": len(frames),
        "scenes": len(by_scene),
        "anchors": len(windows),
        "protocols": protocol_count,
        "canonical_clean_bytes": clean_bytes,
        "labels_present_occ3d": sorted(labels_present),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
