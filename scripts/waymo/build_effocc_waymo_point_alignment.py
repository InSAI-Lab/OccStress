#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Build compact masks aligning raw Waymo points with EFFOcc input points."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


SCENES = 202
FRAMES = 7_998


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--effocc-data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_infos(annotation: Path) -> list[dict[str, Any]]:
    with annotation.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != FRAMES:
        raise ValueError(f"expected {FRAMES} annotation records")
    return payload


def scene_records(
    infos: list[dict[str, Any]],
) -> dict[int, dict[int, str]]:
    records: dict[int, dict[int, str]] = {}
    for info in infos:
        sample_idx = int(info["image"]["image_idx"])
        scene = sample_idx % 1_000_000 // 1_000
        frame = sample_idx % 1_000
        path = str(info["point_cloud"]["velodyne_path"])
        if frame in records.setdefault(scene, {}):
            raise ValueError(f"duplicate scene/frame {scene:03d}/{frame:03d}")
        records[scene][frame] = path
    if set(records) != set(range(SCENES)):
        raise ValueError("annotation does not contain scenes 000 through 201")
    return records


def xyz_rows(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return contiguous.view(np.dtype((np.void, 12))).ravel()


def exact_ordered_keep(
    raw_xyz: np.ndarray,
    clean_xyz: np.ndarray,
) -> np.ndarray:
    raw_rows = xyz_rows(raw_xyz)
    clean_rows = xyz_rows(clean_xyz)
    keep = np.isin(raw_rows, clean_rows)
    selected = raw_xyz[keep]
    if selected.shape == clean_xyz.shape and np.array_equal(
        selected, clean_xyz
    ):
        return keep

    # Exact duplicate XYZ rows can straddle the NLZ boundary. Resolve those
    # rare frames as an ordered subsequence so each clean point is used once.
    keep = np.zeros(len(raw_rows), dtype=bool)
    source_index = 0
    for clean_row in clean_rows:
        while (
            source_index < len(raw_rows)
            and raw_rows[source_index] != clean_row
        ):
            source_index += 1
        if source_index == len(raw_rows):
            raise ValueError("EFFOcc points are not an ordered raw-point subset")
        keep[source_index] = True
        source_index += 1
    return keep


def build_scene(
    scene: int,
    records: dict[int, str],
    effocc_data_root: str,
    index_root: str,
    sensor_root: str,
    output_root: str,
    overwrite: bool,
) -> dict[str, Any]:
    output = Path(output_root) / f"{scene:03d}.npz"
    done = Path(output_root) / f"{scene:03d}.done.json"
    if output.is_file() and done.is_file() and not overwrite:
        payload = json.loads(done.read_text(encoding="utf-8"))
        if payload.get("status") == "success":
            return {**payload, "status": "skipped"}

    index_path = Path(index_root) / f"{scene:03d}.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    with np.load(Path(sensor_root) / index["data_path"]) as source:
        points = source["points"]
        frame_offsets = source["frame_offsets"]

        frame_indices = []
        source_counts = []
        kept_counts = []
        byte_offsets = [0]
        packed_parts = []
        filtered = 0
        for offset, metadata in enumerate(index["frames"]):
            frame = int(metadata["frame_index"])
            point_start, point_end = frame_offsets[offset : offset + 2]
            raw_xyz = points[int(point_start) : int(point_end), :3]
            point_path = Path(effocc_data_root) / records[frame]
            clean = np.fromfile(point_path, dtype=np.float32)
            if clean.size % 6:
                raise ValueError(f"{point_path}: malformed six-feature point file")
            clean_xyz = clean.reshape(-1, 6)[:, :3]

            keep = exact_ordered_keep(raw_xyz, clean_xyz)
            selected = raw_xyz[keep]
            if selected.shape != clean_xyz.shape or not np.array_equal(
                selected, clean_xyz
            ):
                raise ValueError(
                    f"waymo-val-{scene:03d}-{frame:03d}: EFFOcc points are "
                    "not an exact ordered subset of the raw Waymo points"
                )
            packed = np.packbits(keep, bitorder="little")
            packed_parts.append(packed)
            byte_offsets.append(byte_offsets[-1] + len(packed))
            frame_indices.append(frame)
            source_counts.append(len(raw_xyz))
            kept_counts.append(len(clean_xyz))
            filtered += len(raw_xyz) - len(clean_xyz)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            frame_indices=np.asarray(frame_indices, dtype=np.int16),
            source_counts=np.asarray(source_counts, dtype=np.int32),
            kept_counts=np.asarray(kept_counts, dtype=np.int32),
            byte_offsets=np.asarray(byte_offsets, dtype=np.int64),
            packed_keep=np.concatenate(packed_parts),
        )
    os.replace(temporary, output)
    payload = {
        "alignment": "raw-waymo-to-effocc-nlz-filtered",
        "filtered_points": int(filtered),
        "frames": len(frame_indices),
        "kept_points": int(sum(kept_counts)),
        "scene": f"{scene:03d}",
        "source_points": int(sum(source_counts)),
        "status": "success",
    }
    atomic_json(done, payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    records = scene_records(load_infos(args.annotation.resolve()))
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                build_scene,
                scene,
                records[scene],
                str(args.effocc_data_root.resolve()),
                str(args.index_root.resolve()),
                str(args.sensor_root.resolve()),
                str(args.output_root.resolve()),
                args.overwrite,
            ): scene
            for scene in range(SCENES)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if completed % 10 == 0 or completed == SCENES:
                elapsed = time.monotonic() - start
                print(
                    f"aligned {completed}/{SCENES} scenes "
                    f"({completed / elapsed:.2f} scenes/s)",
                    flush=True,
                )

    if sum(int(record["frames"]) for record in results) != FRAMES:
        raise ValueError("alignment output frame count mismatch")
    payload = {
        "alignment": "raw-waymo-to-effocc-nlz-filtered",
        "filtered_points": sum(
            int(record["filtered_points"]) for record in results
        ),
        "frames": FRAMES,
        "kept_points": sum(int(record["kept_points"]) for record in results),
        "scenes": SCENES,
        "source_points": sum(
            int(record["source_points"]) for record in results
        ),
        "status": "success",
    }
    atomic_json(args.output_root / "manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
