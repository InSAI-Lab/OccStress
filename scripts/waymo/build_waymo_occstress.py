#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Build the canonical OccStress-Waymo index and 38 manual protocols."""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from waymo_occstress_common import (
    CONTENT_FAMILIES, FUTURE_OFFSETS_SECONDS, MATRIX_FAMILY,
    OBSERVED_OFFSETS_SECONDS, OCC3D_CLASSES, SEVERITIES, TEMPORAL_PATTERNS,
    TRAFFIC_FAMILY, WAYMO_TO_OCC3D, atomic_json, build_misalignment_series,
    command_from_trajectory, relative_xy, rotation_matrix_to_quaternion,
    sha256sum,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--occ-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=6)
    parser.add_argument("--expected-frames", type=int, default=7998)
    parser.add_argument("--expected-anchors", type=int, default=5978)
    parser.add_argument("--skip-protocols", action="store_true")
    return parser.parse_args()


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def sample_idx(info):
    if "sample_idx" in info:
        return int(info["sample_idx"])
    return int(info["image"]["image_idx"])


def pose_matrix(pose_info, scene, frame):
    scene_info = pose_info[scene] if scene in pose_info else pose_info[str(scene)]
    frame_info = scene_info[frame] if not isinstance(scene_info, dict) else (
        scene_info[frame] if frame in scene_info else scene_info[str(frame)]
    )
    if isinstance(frame_info, (list, tuple)):
        frame_info = frame_info[0]
    if "ego2global" not in frame_info:
        camera_zero = (
            frame_info[0] if 0 in frame_info else frame_info["0"])
        frame_info = camera_zero
    if "ego2global" not in frame_info:
        raise KeyError(
            f"missing ego2global for Waymo scene={scene} frame={frame}")
    return np.asarray(frame_info["ego2global"], dtype=np.float64)


def occ_path(root, scene, frame):
    return (Path(root) / f"{scene:03d}" / f"{frame:03d}_04.npz").resolve()


def build_frames(annotations, poses, args):
    ordered = sorted(annotations, key=lambda item: item["timestamp"])[::args.stride]
    frames = []
    for info in ordered:
        idx = sample_idx(info)
        scene = idx % 1_000_000 // 1000
        frame = idx % 1000
        pose = pose_matrix(poses, scene, frame)
        source = occ_path(args.occ_root, scene, frame)
        token = f"waymo-val-{scene:03d}-{frame:03d}"
        frames.append({
            "token": token,
            "scene_name": f"{scene:03d}",
            "scene_token": f"waymo-val-{scene:03d}",
            "scene_idx": scene,
            "frame_idx": frame,
            "sample_idx": idx,
            "timestamp": int(info["timestamp"]),
            "occ_path": str(source),
            "ego2global": pose,
        })
    return frames


def enrich_controls(scene_frames, future_length):
    for index, frame in enumerate(scene_frames):
        available = scene_frames[index + 1:index + 1 + future_length]
        cumulative = np.zeros((future_length, 2), dtype=np.float32)
        masks = np.zeros(future_length, dtype=np.float32)
        for step, target in enumerate(available):
            cumulative[step] = relative_xy(frame["ego2global"], target["ego2global"])
            masks[step] = 1
        per_step = cumulative.copy()
        if future_length > 1:
            per_step[1:] -= cumulative[:-1]
        command = command_from_trajectory(cumulative) if available else np.array(
            [0, 0, 1], dtype=np.float32)
        frame["gt_ego_fut_trajs"] = per_step
        frame["gt_ego_fut_masks"] = masks
        frame["gt_ego_fut_cmd"] = command
        frame["pose_mode"] = command
        frame["gt_ego_lcf_feat"] = np.zeros(9, dtype=np.float32)
        frame["ego2global_translation"] = frame["ego2global"][:3, 3].astype(np.float32)
        frame["ego2global_rotation"] = rotation_matrix_to_quaternion(
            frame["ego2global"]).astype(np.float32)
        frame["pose_mat"] = frame["ego2global"].astype(np.float32)


def public_frame(frame):
    result = dict(frame)
    for key, value in list(result.items()):
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
    return result


def frame_ref(frame, path=None):
    return {"token": frame["token"], "occ_path": str(path or frame["occ_path"])}


def corrupted_path(root, family, severity, frame):
    return Path(root) / "occ" / "manual" / family / severity / (
        frame["scene_name"]) / frame["token"] / "labels.npz"


def traffic_path(root, frame):
    return Path(root) / "occ" / "manual" / "traffic" / frame[
        "scene_name"] / frame["token"] / "labels.npz"


def protocol_record(window, root, family="clean", severity=None, pattern="clean"):
    history = window[:4]
    current = window[4]
    future = window[5:]
    active = set(TEMPORAL_PATTERNS.get(pattern, ()))

    def input_path(frame, offset):
        if family in CONTENT_FAMILIES and offset in active:
            return corrupted_path(root, family, severity, frame)
        if family == TRAFFIC_FAMILY:
            return traffic_path(root, frame)
        return frame["occ_path"]

    history_refs = [
        frame_ref(frame, input_path(frame, offset))
        for frame, offset in zip(history, (-4, -3, -2, -1))
    ]
    current_ref = frame_ref(current, input_path(current, 0))
    target_path = traffic_path(root, current) if family == TRAFFIC_FAMILY else current["occ_path"]
    future_refs = [
        frame_ref(frame, traffic_path(root, frame) if family == TRAFFIC_FAMILY else frame["occ_path"])
        for frame in future
    ]
    record = {
        "schema_version": 1,
        "dataset": "Occ3D-Waymo",
        "sample_id": f"{current['token']}__{family}__{severity or 'none'}__{pattern}",
        "anchor_token": current["token"],
        "scene_name": current["scene_name"],
        "scene_token": current["scene_token"],
        "history_length": 4,
        "future_length": 6,
        "history_tokens": [frame["token"] for frame in history],
        "history": history_refs,
        "current_input": current_ref,
        "target": frame_ref(current, target_path),
        "future_targets": future_refs,
        "frame_protocol": pattern,
        "corruption": {"type": family, "severity": severity},
        "corruption_family": family,
        "severity": severity,
        "observed_offsets_seconds": list(OBSERVED_OFFSETS_SECONDS),
        "future_offsets_seconds": list(FUTURE_OFFSETS_SECONDS),
        "traffic_mirror": family == TRAFFIC_FAMILY,
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
                current["token"] + "__current", severity,
                np.array([current_active], dtype=np.uint8))
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


def protocol_path(root, family, severity, pattern):
    base = Path(root) / "protocols" / "manual"
    if family == "clean":
        return base / "clean" / "H4_F6_val_backbone.pkl"
    if family == TRAFFIC_FAMILY:
        return base / family / "H4_F6_val_backbone.pkl"
    return base / family / severity / f"{pattern}_H4_F6_val_backbone.pkl"


def atomic_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main():
    args = parse_args()
    annotations = load_pickle(args.ann_file)
    poses = load_pickle(args.pose_file)
    frames = build_frames(annotations, poses, args)
    if args.expected_frames and len(frames) != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} frames, got {len(frames)}")

    by_scene = defaultdict(list)
    for frame in frames:
        by_scene[frame["scene_name"]].append(frame)
    for scene_frames in by_scene.values():
        scene_frames.sort(key=lambda frame: frame["timestamp"])
        enrich_controls(scene_frames, args.future_length)

    windows = []
    for scene_frames in by_scene.values():
        width = args.history_length + 1 + args.future_length
        windows.extend(
            scene_frames[start:start + width]
            for start in range(len(scene_frames) - width + 1)
        )
    if args.expected_anchors and len(windows) != args.expected_anchors:
        raise RuntimeError(f"expected {args.expected_anchors} anchors, got {len(windows)}")

    meta = args.output_root / "meta" / "manual"
    meta.mkdir(parents=True, exist_ok=True)
    atomic_json(meta / "frames_H4_F6_val.json", [public_frame(frame) for frame in frames])
    atomic_json(meta / "anchors_H4_F6_val.json", [
        {
            "anchor_token": window[4]["token"],
            "scene_token": window[4]["scene_token"],
            "tokens": [frame["token"] for frame in window],
        }
        for window in windows
    ])
    base_info = {"metadata": {"dataset": "Occ3D-Waymo"}, "infos": {
        scene: [public_frame(frame) for frame in scene_frames]
        for scene, scene_frames in sorted(by_scene.items())
    }}
    atomic_pickle(meta / "waymo_base_info.pkl", base_info)

    dataset_meta = {
        "schema_version": 1,
        "dataset": "Occ3D-Waymo",
        "zero_shot_source_dataset": "Occ3D-nuScenes",
        "native_hz": 10,
        "evaluation_hz": 2,
        "stride": args.stride,
        "frame_count": len(frames),
        "scene_count": len(by_scene),
        "anchor_count": len(windows),
        "history_length": args.history_length,
        "future_length": args.future_length,
        "observed_offsets_seconds": list(OBSERVED_OFFSETS_SECONDS),
        "future_offsets_seconds": list(FUTURE_OFFSETS_SECONDS),
        "grid_shape": [200, 200, 16],
        "source_ann_file": str(args.ann_file.resolve()),
        "source_ann_sha256": sha256sum(args.ann_file),
        "source_pose_file": str(args.pose_file.resolve()),
        "source_pose_sha256": sha256sum(args.pose_file),
        "occ_root": str(args.occ_root.resolve()),
        "scene_length_histogram": dict(sorted(Counter(
            len(value) for value in by_scene.values()).items())),
    }
    atomic_json(meta / "dataset.json", dataset_meta)
    atomic_json(meta / "class_mapping.json", {
        "source": "Occ3D-Waymo voxel_label",
        "target": "Occ3D 18-class",
        "mapping": {str(key): value for key, value in WAYMO_TO_OCC3D.items()},
        "classes": list(OCC3D_CLASSES),
    })

    if not args.skip_protocols:
        specs = [("clean", None, "clean")]
        specs.extend(
            (family, severity, pattern)
            for family in CONTENT_FAMILIES + (MATRIX_FAMILY,)
            for severity in SEVERITIES
            for pattern in TEMPORAL_PATTERNS
        )
        specs.append((TRAFFIC_FAMILY, None, "all_frame"))
        for family, severity, pattern in specs:
            records = [
                protocol_record(window, args.output_root, family, severity, pattern)
                for window in windows
            ]
            atomic_pickle(protocol_path(
                args.output_root, family, severity, pattern), records)
        if len(specs) != 38:
            raise RuntimeError(f"expected 38 protocols, got {len(specs)}")

    atomic_json(meta / "generation_summary.json", {
        **dataset_meta,
        "protocol_count": 0 if args.skip_protocols else 38,
        "base_info": str((meta / "waymo_base_info.pkl").resolve()),
    })
    print(f"frames={len(frames)} scenes={len(by_scene)} anchors={len(windows)}")


if __name__ == "__main__":
    main()
