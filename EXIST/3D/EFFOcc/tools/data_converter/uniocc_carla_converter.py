#!/usr/bin/env python3
"""Convert UniOcc-CARLA into an EFFOcc-friendly node-local training view."""

from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


CAMERA_NAMES = ("CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK")
# CARLA camera extrinsics use x-forward/y-right/z-up sensor axes, while
# EFFOcc back-projects pixels in OpenCV x-right/y-down/z-forward coordinates.
OPENCV_TO_CARLA_CAMERA = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
CARLA_TO_MODEL = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
CARLA_TO_MODEL_HOMOGENEOUS = np.diag([1.0, -1.0, 1.0, 1.0]).astype(
    np.float64
)
REQUIRED_NPZ_KEYS = {
    "occ_label",
    "occ_mask_lidar",
    "occ_mask_camera",
    "ego_to_world_transformation",
    "cameras",
}
INTENSITY_FIELDS = ("intensity", "I", "reflectance", "scalar_Intensity")
TIME_FIELDS = ("time", "timestamp", "t")
PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--compress-labels", action="store_true")
    return parser.parse_args()


def rotation_to_quaternion(rotation):
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif index == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(np.float32).tolist()


def atomic_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_npz(path, compressed, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    writer = np.savez_compressed if compressed else np.savez
    writer(temporary, **arrays)
    os.replace(temporary, path)


def first_field(names, candidates):
    return next((name for name in candidates if name in names), None)


def read_ply_vertices(path):
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"{path}: invalid PLY magic")
        encoding = None
        vertex_count = None
        active_element = None
        properties = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            fields = line.decode("ascii").strip().split()
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[0] == "format":
                encoding = fields[1]
            elif fields[0] == "element":
                active_element = fields[1]
                if active_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and active_element == "vertex":
                if fields[1] == "list":
                    raise ValueError(f"{path}: list-valued vertex property unsupported")
                properties.append((fields[2], fields[1]))
            elif fields[0] == "end_header":
                break

        if vertex_count is None or encoding is None:
            raise ValueError(f"{path}: missing vertex count or format")
        unknown = [kind for _, kind in properties if kind not in PLY_DTYPES]
        if unknown:
            raise ValueError(f"{path}: unsupported PLY property types {unknown}")
        names = [name for name, _ in properties]
        if encoding == "ascii":
            values = np.loadtxt(stream, max_rows=vertex_count, ndmin=2)
            if values.shape != (vertex_count, len(properties)):
                raise ValueError(f"{path}: malformed ASCII vertex table {values.shape}")
            return {name: values[:, index] for index, name in enumerate(names)}
        if encoding not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"{path}: unsupported PLY encoding {encoding}")
        endian = ">" if encoding == "binary_big_endian" else "<"
        dtype_fields = []
        for name, kind in properties:
            dtype = np.dtype(PLY_DTYPES[kind])
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder(endian)
            dtype_fields.append((name, dtype))
        vertices = np.fromfile(stream, dtype=np.dtype(dtype_fields), count=vertex_count)
        if len(vertices) != vertex_count:
            raise ValueError(f"{path}: truncated binary vertex table")
        return {name: vertices[name] for name in names}


def convert_ply(source, destination):
    if destination.is_file() and destination.stat().st_size > 0:
        return
    vertices = read_ply_vertices(source)
    names = set(vertices)
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{source}: PLY does not contain x/y/z")
    count = len(vertices["x"])
    points = np.zeros((count, 5), dtype=np.float32)
    for column, name in enumerate(("x", "y", "z")):
        points[:, column] = np.asarray(vertices[name], dtype=np.float32)
    points[:, 1] *= -1.0
    intensity = first_field(names, INTENSITY_FIELDS)
    timestamp = first_field(names, TIME_FIELDS)
    if intensity:
        points[:, 3] = np.asarray(vertices[intensity], dtype=np.float32)
    if timestamp:
        points[:, 4] = np.asarray(vertices[timestamp], dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    points.tofile(temporary)
    os.replace(temporary, destination)


def convert_one(task):
    data_root, output_root, split, scene_name, frame_index, relative_occ, compress = task
    data_root = Path(data_root)
    output_root = Path(output_root)
    source_occ = data_root / relative_occ
    with np.load(source_occ, allow_pickle=True) as payload:
        missing = REQUIRED_NPZ_KEYS - set(payload.files)
        if missing:
            raise ValueError(f"{source_occ}: missing {sorted(missing)}")
        semantics = np.asarray(payload["occ_label"], dtype=np.uint8)[:, ::-1, :].copy()
        mask_lidar = np.asarray(payload["occ_mask_lidar"], dtype=bool)[:, ::-1, :].copy()
        mask_camera = np.asarray(payload["occ_mask_camera"], dtype=bool)[:, ::-1, :].copy()
        raw_pose = np.asarray(
            payload["ego_to_world_transformation"], dtype=np.float64
        )
        pose = CARLA_TO_MODEL_HOMOGENEOUS @ raw_pose @ CARLA_TO_MODEL_HOMOGENEOUS
        cameras = payload["cameras"]

        if semantics.shape != (200, 200, 16):
            raise ValueError(f"{source_occ}: bad occupancy shape {semantics.shape}")
        labels = set(np.unique(semantics).astype(int).tolist())
        if not labels.issubset(range(11)):
            raise ValueError(f"{source_occ}: unsupported labels {sorted(labels)}")
        if len(cameras) != 4 or pose.shape != (4, 4):
            raise ValueError(f"{source_occ}: invalid sensor metadata")

        token = f"carla-{split}-{scene_name.removeprefix('scene_')}-{frame_index:05d}"
        label_path = output_root / "labels" / scene_name / f"{frame_index}.npz"
        if not label_path.is_file() or label_path.stat().st_size == 0:
            atomic_npz(
                label_path,
                compress,
                semantics=semantics,
                mask_lidar=mask_lidar,
                mask_camera=mask_camera,
            )

        camera_records = {}
        pose_quaternion = rotation_to_quaternion(pose[:3, :3])
        for camera_index, camera_name in enumerate(CAMERA_NAMES):
            camera = cameras[camera_index]
            if camera["cam_name"] != camera_name:
                raise ValueError(f"{source_occ}: camera order mismatch")
            relative_image = camera.get(
                "filename",
                f"{scene_name}/{frame_index}_camera_{camera_index}.png",
            )
            image_path = data_root / relative_image
            if not image_path.is_file():
                # The official 2 Hz CARLA archive renumbers downsampled images
                # contiguously, but some camera metadata retains the 10 Hz name.
                fallback_image = source_occ.parent / (
                    f"{source_occ.stem}_camera_{camera_index}.png"
                )
                if not fallback_image.is_file():
                    raise FileNotFoundError(
                        f"neither {image_path} nor {fallback_image} exists"
                    )
                image_path = fallback_image
            intrinsic = camera.get("intrinsics", camera.get("instrinsic"))
            if intrinsic is None:
                raise ValueError(f"{source_occ}: missing camera intrinsics")
            extrinsic = np.asarray(camera["extrinsics"], dtype=np.float64)
            sensor_to_ego_rotation = (
                CARLA_TO_MODEL
                @ extrinsic[:3, :3]
                @ OPENCV_TO_CARLA_CAMERA
            )
            if not np.allclose(
                sensor_to_ego_rotation.T @ sensor_to_ego_rotation,
                np.eye(3),
                atol=1e-5,
            ) or not np.isclose(
                np.linalg.det(sensor_to_ego_rotation), 1.0, atol=1e-5
            ):
                raise ValueError(f"{source_occ}: invalid camera rotation")
            camera_records[camera_name] = {
                "data_path": str(image_path),
                "cam_intrinsic": np.asarray(intrinsic, dtype=np.float32),
                "sensor2ego_rotation": rotation_to_quaternion(
                    sensor_to_ego_rotation
                ),
                "sensor2ego_translation": (
                    CARLA_TO_MODEL @ extrinsic[:3, 3]
                ).astype(np.float32).tolist(),
                "ego2global_rotation": pose_quaternion,
                "ego2global_translation": pose[:3, 3].astype(np.float32).tolist(),
            }

    source_lidar = data_root / scene_name / f"{frame_index}_surround_lidar.ply"
    point_path = output_root / "points" / scene_name / f"{frame_index}.bin"
    convert_ply(source_lidar, point_path)

    return {
        "sample_idx": token,
        "sample_token": token,
        "scene_token": f"carla-{split}-{scene_name.removeprefix('scene_')}",
        "timestamp": frame_index,
        "pts_filename": str(point_path),
        "occ_gt_path": str(label_path),
        "ego2global": pose.astype(np.float32),
        "curr": {
            "cams": camera_records,
            "lidar_path": str(point_path),
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "lidar2ego_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": pose_quaternion,
            "ego2global_translation": pose[:3, 3].astype(np.float32).tolist(),
        },
    }


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    with (data_root / "scene_infos.pkl").open("rb") as stream:
        scenes = pickle.load(stream)

    tasks = []
    for scene in sorted(scenes, key=lambda value: value["scene_name"]):
        scene_name = scene["scene_name"]
        for frame_index, relative_occ in enumerate(scene["occ_in_scene_paths"]):
            tasks.append((
                str(data_root),
                str(output_root),
                args.split,
                scene_name,
                frame_index,
                relative_occ,
                args.compress_labels,
            ))
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]

    infos = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, info in enumerate(
            executor.map(convert_one, tasks, chunksize=8), start=1
        ):
            infos.append(info)
            if completed % 500 == 0 or completed == len(tasks):
                print(
                    f"Converted {completed}/{len(tasks)} {args.split} frames",
                    flush=True,
                )

    payload = {
        "metadata": {
            "dataset": "UniOcc-CARLA",
            "split": args.split,
            "coordinate_convention": "right-handed x-forward/y-left/z-up",
            "source_coordinate_convention": "CARLA x-forward/y-right/z-up",
            "class_ontology": "UniOcc native 0..10",
            "frame_count": len(infos),
        },
        "infos": infos,
    }
    info_path = output_root / f"uniocc_carla_{args.split}_infos.pkl"
    atomic_pickle(info_path, payload)
    print(f"Wrote {len(infos)} records to {info_path}", flush=True)


if __name__ == "__main__":
    main()
