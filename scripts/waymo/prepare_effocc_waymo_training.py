#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Convert Waymo TFRecords into the 10 Hz files used by EFFOcc."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WAYMO_PROTO_ROOT = WORKSPACE_ROOT / ".runtime" / "waymo-proto"
if WAYMO_PROTO_ROOT.is_dir():
    sys.path.insert(0, str(WAYMO_PROTO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="training",
        help="Select the Waymo split and corresponding OpenMMLab sample prefix.",
    )
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-end", type=int)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--image-storage",
        choices=("source", "png"),
        default="source",
        help="Keep source JPEG bytes or reproduce the official PNG rewrite.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Export only EFFOcc LiDAR bins; images may be linked separately.",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def tfrecord_payloads(path: Path):
    with path.open("rb") as stream:
        record_index = 0
        while True:
            length_raw = stream.read(8)
            if not length_raw:
                return
            if len(length_raw) != 8:
                raise RuntimeError(
                    f"{path}: truncated record length at {record_index}"
                )
            length = struct.unpack("<Q", length_raw)[0]
            if len(stream.read(4)) != 4:
                raise RuntimeError(
                    f"{path}: truncated length CRC at {record_index}"
                )
            payload = stream.read(length)
            if len(payload) != length:
                raise RuntimeError(
                    f"{path}: truncated payload at {record_index}"
                )
            if len(stream.read(4)) != 4:
                raise RuntimeError(
                    f"{path}: truncated payload CRC at {record_index}"
                )
            yield payload
            record_index += 1


def load_expected_frames(
    metadata: Path, sample_prefix: int
) -> dict[int, list[int]]:
    with metadata.open("rb") as stream:
        records = pickle.load(stream)
    by_scene: dict[int, list[int]] = {}
    for record in records:
        sample_index = int(record["image"]["image_idx"])
        if sample_index // 1_000_000 != sample_prefix:
            raise RuntimeError(
                f"Unexpected sample prefix for index {sample_index}: "
                f"expected {sample_prefix}"
            )
        scene_index = sample_index % 1_000_000 // 1000
        frame_index = sample_index % 1000
        by_scene.setdefault(scene_index, []).append(frame_index)
    del records
    gc.collect()
    for scene_index, frame_indices in by_scene.items():
        if frame_indices != sorted(set(frame_indices)):
            raise RuntimeError(
                f"Scene {scene_index:03d} has duplicate/unsorted metadata frames"
            )
    return by_scene


def initialize_worker() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def frame_to_effocc_points(frame):
    import numpy as np
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils

    range_images = {}
    camera_projections = {}
    top_pose = dataset_pb2.MatrixFloat()
    for laser in frame.lasers:
        returns = (laser.ri_return1, laser.ri_return2)
        for return_index, laser_return in enumerate(returns):
            if not laser_return.range_image_compressed:
                continue
            range_image = dataset_pb2.MatrixFloat()
            range_image.ParseFromString(
                zlib.decompress(laser_return.range_image_compressed)
            )
            range_images.setdefault(laser.name, []).append(range_image)

            camera_projection = dataset_pb2.MatrixInt32()
            camera_projection.ParseFromString(
                zlib.decompress(laser_return.camera_projection_compressed)
            )
            camera_projections.setdefault(laser.name, []).append(
                camera_projection
            )
            if (
                laser.name == dataset_pb2.LaserName.TOP
                and return_index == 0
            ):
                top_pose.ParseFromString(
                    zlib.decompress(laser_return.range_image_pose_compressed)
                )

    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda item: item.name
    )
    point_parts = []
    for return_index in (0, 1):
        cartesian = frame_utils.convert_range_image_to_cartesian(
            frame,
            range_images,
            top_pose,
            ri_index=return_index,
            keep_polar_features=True,
        )
        for calibration in calibrations:
            laser_id = int(calibration.name)
            range_image = range_images[laser_id][return_index]
            shape = tuple(int(value) for value in range_image.shape.dims)
            values = np.asarray(range_image.data, dtype=np.float32).reshape(shape)
            valid = (values[..., 0] > 0) & (values[..., 3] != 1.0)
            rows, columns = np.nonzero(valid)
            selected = np.asarray(cartesian[laser_id].numpy())[valid]
            points = np.empty((len(selected), 6), dtype=np.float32)
            points[:, :3] = selected[:, 3:6]
            points[:, 3] = selected[:, 1]
            points[:, 4] = selected[:, 2]
            if laser_id == int(dataset_pb2.LaserName.TOP):
                points[:, 5] = (
                    (return_index * shape[0] + rows) * shape[1] + columns
                )
            else:
                points[:, 5] = -1
            point_parts.append(points)
    points = np.concatenate(point_parts)
    if not np.isfinite(points).all():
        raise RuntimeError("LiDAR conversion produced non-finite values")
    return points


def scene_is_complete(
    done_path: Path,
    source: Path,
    expected_frames: list[int],
    output_root: Path,
    image_storage: str,
    skip_images: bool,
    sample_prefix: int,
) -> bool:
    if not done_path.is_file():
        return False
    try:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not (
        payload.get("status") == "success"
        and payload.get("source_bytes") == source.stat().st_size
        and payload.get("frame_indices") == expected_frames
        and payload.get("image_storage") == image_storage
        and bool(payload.get("skip_images", False)) == skip_images
        and payload.get("sample_prefix") == sample_prefix
    ):
        return False
    scene_index = int(payload["scene_index"])
    first_frame = expected_frames[0]
    last_frame = expected_frames[-1]
    for frame_index in (first_frame, last_frame):
        stem = f"{sample_prefix}{scene_index:03d}{frame_index:03d}"
        if not (output_root / "velodyne" / f"{stem}.bin").is_file():
            return False
        if not skip_images:
            for camera_index in range(5):
                if not (
                    output_root / f"image_{camera_index}" / f"{stem}.png"
                ).is_file():
                    return False
    return True


def write_image(path: Path, encoded: bytes, storage: str) -> None:
    if storage == "source":
        atomic_bytes(path, encoded)
        return
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode Waymo image for {path}")
    success, png = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Could not encode PNG for {path}")
    atomic_bytes(path, png.tobytes())


def extract_scene(
    scene_index: int,
    source_text: str,
    expected_frames: list[int],
    output_root_text: str,
    state_root_text: str,
    image_storage: str,
    skip_images: bool,
    sample_prefix: int,
    overwrite: bool,
) -> dict:
    import numpy as np
    from waymo_open_dataset import dataset_pb2

    source = Path(source_text)
    output_root = Path(output_root_text)
    state_root = Path(state_root_text)
    done_path = state_root / f"{scene_index:03d}.done.json"
    if not overwrite and scene_is_complete(
        done_path,
        source,
        expected_frames,
        output_root,
        image_storage,
        skip_images,
        sample_prefix,
    ):
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        payload["status"] = "skipped"
        return payload

    expected_set = set(expected_frames)
    actual_frames = []
    image_bytes = 0
    lidar_bytes = 0
    started = time.monotonic()
    for frame_index, payload in enumerate(tfrecord_payloads(source)):
        if frame_index not in expected_set:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(payload)
        stem = f"{sample_prefix}{scene_index:03d}{frame_index:03d}"
        if not skip_images:
            images = {int(image.name): image for image in frame.images}
            if set(images) != {1, 2, 3, 4, 5}:
                raise RuntimeError(
                    f"{source} frame {frame_index}: cameras={sorted(images)}"
                )
            for camera_id, image in images.items():
                image_path = (
                    output_root
                    / f"image_{camera_id - 1}"
                    / f"{stem}.png"
                )
                encoded = bytes(image.image)
                write_image(image_path, encoded, image_storage)
                image_bytes += image_path.stat().st_size

        points = frame_to_effocc_points(frame)
        lidar_path = output_root / "velodyne" / f"{stem}.bin"
        temporary = lidar_path.with_name(lidar_path.name + ".partial")
        lidar_path.parent.mkdir(parents=True, exist_ok=True)
        points.astype(np.float32, copy=False).tofile(temporary)
        os.replace(temporary, lidar_path)
        lidar_bytes += lidar_path.stat().st_size
        actual_frames.append(frame_index)

    if actual_frames != expected_frames:
        raise RuntimeError(
            f"{source}: expected frames {expected_frames[:3]}...{expected_frames[-3:]}, "
            f"got {actual_frames[:3]}...{actual_frames[-3:]}"
        )
    result = {
        "elapsed_seconds": time.monotonic() - started,
        "frame_indices": actual_frames,
        "frames": len(actual_frames),
        "image_bytes": image_bytes,
        "image_storage": image_storage,
        "lidar_bytes": lidar_bytes,
        "scene_index": scene_index,
        "sample_prefix": sample_prefix,
        "skip_images": skip_images,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "status": "success",
    }
    atomic_json(done_path, result)
    return result


def main() -> None:
    args = parse_args()
    sample_prefix = 0 if args.split == "training" else 1
    expected_scenes = 798 if args.split == "training" else 202
    sources = sorted(args.raw_root.glob("*.tfrecord"))
    expected_by_scene = load_expected_frames(args.metadata, sample_prefix)
    if len(sources) != expected_scenes or len(expected_by_scene) != expected_scenes:
        raise RuntimeError(
            f"Expected {expected_scenes} raw/metadata scenes, got "
            f"{len(sources)}/{len(expected_by_scene)}"
        )
    directories = [args.output_root / "velodyne", args.state_root]
    if not args.skip_images:
        directories.extend(
            args.output_root / f"image_{index}" for index in range(5)
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    end = len(sources) if args.scene_end is None else args.scene_end
    selected = list(range(args.scene_start, min(end, len(sources))))
    if args.max_scenes is not None:
        selected = selected[: args.max_scenes]
    tasks = [
        (
            scene_index,
            str(sources[scene_index]),
            expected_by_scene[scene_index],
            str(args.output_root),
            str(args.state_root),
            args.image_storage,
            args.skip_images,
            sample_prefix,
            args.overwrite,
        )
        for scene_index in selected
    ]
    if not tasks:
        raise RuntimeError("No Waymo scenes selected")

    started = time.monotonic()
    completed = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker
    ) as executor:
        futures = {executor.submit(extract_scene, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                        "scene": f"{result['scene_index']:03d}",
                        "scene_seconds": round(result["elapsed_seconds"], 1),
                        "status": result["status"],
                        "total": len(tasks),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = {
        "elapsed_seconds": time.monotonic() - started,
        "image_storage": args.image_storage,
        "metadata": str(args.metadata),
        "output_root": str(args.output_root),
        "raw_root": str(args.raw_root),
        "scene_indices": selected,
        "scenes": len(completed),
        "split": args.split,
        "sample_prefix": sample_prefix,
        "skip_images": args.skip_images,
        "status": "success",
        "total_frames": sum(result["frames"] for result in completed),
        "workers": args.workers,
    }
    atomic_json(args.state_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
