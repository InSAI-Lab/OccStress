#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Generate deterministic frame-level manual corruption assets for OccStress-CARLA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

WAYMO_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "waymo"
sys.path.insert(0, str(WAYMO_SCRIPT_DIR))

from generate_waymo_occstress_assets import (  # noqa: E402
    FREE,
    dropout_corruption,
    hole_corruption,
    semantic_corruption,
)
from waymo_occstress_common import (  # noqa: E402
    OCC3D_CLASSES,
    SEVERITIES,
    atomic_json,
    mirror_y_semantics,
    stable_seed,
)


ALL_TASKS = tuple(
    [f"{family}:{severity}" for family in ("semantic", "hole", "dropout")
     for severity in SEVERITIES]
    + ["traffic"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--task",
        default="all",
        help="all, traffic, or FAMILY:SEVERITY",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scene")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(root: Path, task: str, frame: dict) -> tuple[Path, Path]:
    if task == "traffic":
        asset_base = root / "occ" / "manual" / "traffic"
        event_base = root / "events" / "manual" / "traffic"
    else:
        family, severity = task.split(":")
        asset_base = root / "occ" / "manual" / family / severity
        event_base = root / "events" / "manual" / family / severity
    return (
        asset_base / frame["scene_name"] / frame["token"] / "labels.npz",
        event_base / frame["scene_name"] / f"{frame['token']}.json",
    )


def load_clean(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        semantics = np.asarray(payload["semantics"], dtype=np.uint8)
        infov = np.asarray(payload["infov"], dtype=bool)
    if semantics.shape != (200, 200, 16) or infov.shape != semantics.shape:
        raise ValueError(f"invalid canonical clean occupancy: {path}")
    return semantics, infov


def sparse_surface_hole(
    semantics: np.ndarray, severity: str, token: str
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fallback for CARLA objects represented as surfaces without 3D interiors."""
    configs = {
        "easy": {"regions": 1, "half_size": (2, 2, 1)},
        "mid": {"regions": 2, "half_size": (4, 4, 2)},
        "hard": {"regions": 3, "half_size": (6, 6, 3)},
    }
    config = configs[severity]
    output = semantics.copy()
    changed = np.zeros(output.shape, dtype=bool)
    vehicle_classes = (3, 4, 5, 9, 10)
    vehicle_mask = np.isin(semantics, vehicle_classes)
    candidate_mask = vehicle_mask if vehicle_mask.any() else semantics != FREE
    candidates = np.argwhere(candidate_mask)
    rng = np.random.default_rng(stable_seed("carla-surface-hole", token))
    order = rng.permutation(len(candidates))
    half_size = np.asarray(config["half_size"], dtype=np.int64)
    shape = np.asarray(output.shape, dtype=np.int64)
    events = []

    for candidate_index in order:
        center = candidates[int(candidate_index)]
        low = np.maximum(center - half_size, 0)
        high = np.minimum(center + half_size + 1, shape)
        view = output[
            low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        removable_mask = (
            np.isin(view, vehicle_classes)
            if vehicle_mask.any() else view != FREE
        )
        removable = np.argwhere(removable_mask) + low
        if not len(removable):
            continue
        output[
            removable[:, 0], removable[:, 1], removable[:, 2]
        ] = FREE
        changed[
            removable[:, 0], removable[:, 1], removable[:, 2]
        ] = True
        events.append({
            "branch": "carla_sparse_surface_fallback",
            "target": "vehicle_surface" if vehicle_mask.any() else "occupied_surface",
            "center_voxel": center.tolist(),
            "half_size_voxels": half_size.tolist(),
            "removed_voxels": int(len(removable)),
        })
        if len(events) >= config["regions"]:
            break
    if not events:
        raise ValueError(f"{token}: no occupied voxel available for hole fallback")
    return output, changed, events


def run_corruption(
    semantics: np.ndarray, task: str, token: str
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if task == "traffic":
        output = mirror_y_semantics(semantics)
        changed = output != semantics
        events = [{
            "branch": "mirror_y",
            "axis": 1,
            "changed_voxels": int(changed.sum()),
        }]
        return output, changed, events
    family, severity = task.split(":")
    if family == "semantic":
        return semantic_corruption(semantics, severity, token)
    if family == "hole":
        output, changed, events = hole_corruption(
            semantics, severity, token)
        if changed.any():
            return output, changed, events
        return sparse_surface_hole(semantics, severity, token)
    if family == "dropout":
        return dropout_corruption(semantics, severity, token)
    raise ValueError(f"unsupported task: {task}")


def process_task(
    frame: dict,
    root: Path,
    task: str,
    semantics: np.ndarray,
    infov: np.ndarray,
    overwrite: bool,
) -> dict:
    output_path, event_path = output_paths(root, task, frame)
    if output_path.exists() and event_path.exists() and not overwrite:
        event = json.loads(event_path.read_text(encoding="utf-8"))
        return {
            "status": "skipped",
            "task": task,
            "token": frame["token"],
            "changed_voxels": int(event["changed_voxels"]),
            "occupied_voxels": int(event["occupied_voxels"]),
            "normalization_voxels": int(event["normalization_voxels"]),
            "output_bytes": output_path.stat().st_size,
        }

    output, changed, events = run_corruption(
        semantics, task, frame["token"])
    occupied_voxels = int((semantics != FREE).sum())
    normalization_voxels = int(np.count_nonzero(
        (semantics != FREE) | (output != FREE)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(temporary, semantics=output, infov=infov)
    os.replace(temporary, output_path)

    values, counts = np.unique(output, return_counts=True)
    atomic_json(event_path, {
        "schema_version": 1,
        "dataset": "UniOcc-CARLA",
        "benchmark_name": "OccStress-CARLA",
        "task": task,
        "token": frame["token"],
        "source_path": frame["occ_path"],
        "output_path": str(output_path),
        "changed_voxels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "occupied_voxels": occupied_voxels,
        "changed_fraction_occupied": float(
            changed.sum() / max(1, occupied_voxels)),
        "normalization_voxels": normalization_voxels,
        "changed_fraction_support": float(
            changed.sum() / max(1, normalization_voxels)),
        "adaptation_policy": (
            "6-neighbor voxel components replace nuScenes instance boxes; "
            "severity ranges are identical to the OccStress-Waymo manual adapter; "
            "CARLA surface-only objects use a deterministic local hole fallback"
        ),
        "events": events,
        "class_histogram": {
            OCC3D_CLASSES[int(value)]: int(count)
            for value, count in zip(values, counts)
        },
    })
    return {
        "status": "success",
        "task": task,
        "token": frame["token"],
        "changed_voxels": int(changed.sum()),
        "occupied_voxels": occupied_voxels,
        "normalization_voxels": normalization_voxels,
        "output_bytes": output_path.stat().st_size,
    }


def process_frame(
    frame: dict, root: Path, tasks: tuple[str, ...], overwrite: bool
) -> list[dict]:
    semantics, infov = load_clean(frame["occ_path"])
    return [
        process_task(
            frame, root, task, semantics, infov, overwrite)
        for task in tasks
    ]


def validate_task(task: str) -> None:
    if task == "traffic":
        return
    family, severity = task.split(":")
    if family not in {"semantic", "hole", "dropout"} or severity not in SEVERITIES:
        raise ValueError(f"invalid task: {task}")


def task_summary(task: str, results: list[dict], requested: int) -> dict:
    selected = [result for result in results if result["task"] == task]
    changed = [result["changed_voxels"] for result in selected]
    changed_total = sum(changed)
    occupied_total = sum(result["occupied_voxels"] for result in selected)
    support_total = sum(
        result["normalization_voxels"] for result in selected)
    return {
        "schema_version": 1,
        "dataset": "UniOcc-CARLA",
        "benchmark_name": "OccStress-CARLA",
        "task": task,
        "requested_frames": requested,
        "success": sum(
            result["status"] == "success" for result in selected),
        "skipped": sum(
            result["status"] == "skipped" for result in selected),
        "complete": len(selected),
        "changed_voxels": changed_total,
        "changed_voxels_min": min(changed) if changed else 0,
        "changed_voxels_mean": float(np.mean(changed)) if changed else 0.0,
        "changed_voxels_max": max(changed) if changed else 0,
        "zero_changed_frames": sum(value == 0 for value in changed),
        "changed_fraction_occupied": float(
            changed_total / max(1, occupied_total)),
        "changed_fraction_support": float(
            changed_total / max(1, support_total)),
        "output_bytes": sum(result["output_bytes"] for result in selected),
    }


def main() -> None:
    args = parse_args()
    tasks = ALL_TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        validate_task(task)
    frames = json.loads(args.frames_json.read_text(encoding="utf-8"))
    frames = [
        {
            "token": frame["token"],
            "scene_name": frame["scene_name"],
            "occ_path": frame["occ_path"],
        }
        for frame in frames
        if args.scene is None or frame["scene_name"] == args.scene
    ]
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        raise RuntimeError("no frames selected")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_frame,
                frame,
                args.output_root.resolve(),
                tasks,
                args.overwrite,
            )
            for frame in frames
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    summaries = {}
    summary_root = args.output_root / "meta" / "manual" / "assets"
    for task in tasks:
        summary = task_summary(task, results, len(frames))
        summaries[task] = summary
        atomic_json(
            summary_root / f"{task.replace(':', '_')}.json", summary)
    print(json.dumps({
        "status": "success",
        "frames": len(frames),
        "workers": args.workers,
        "summaries": summaries,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
