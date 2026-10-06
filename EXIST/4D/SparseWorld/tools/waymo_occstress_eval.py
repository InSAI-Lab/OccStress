#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Evaluate SparseWorld-TC zero-shot on OccStress-Waymo camera protocols."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[2]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from carla_occstress_eval import (  # noqa: E402
    FUTURE_SECONDS,
    acquire_output_lock,
    add_prediction,
    atomic_json_dump,
    git_value,
    load_framework,
    metrics_from_hist,
    resolve_path,
    sha256_file,
    sha256_files,
    unwrap_array,
)


EXPECTED_SAMPLES = 5978


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-info", required=True)
    parser.add_argument("--pose-file", required=True)
    parser.add_argument("--camera-root", required=True)
    parser.add_argument("--waymo-root", required=True)
    parser.add_argument("--manifest", default="occstress/protocols.json")
    parser.add_argument("--protocol-index", type=int)
    parser.add_argument("--protocol-id", default="clean")
    parser.add_argument("--corruption", default="clean")
    parser.add_argument("--severity")
    parser.add_argument("--frame-protocol")
    parser.add_argument("--output-json")
    parser.add_argument("--output-root", default="occstress/waymo/results")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_protocol(args):
    if args.protocol_index is None:
        return {
            "id": args.protocol_id,
            "corruption": args.corruption,
            "severity": args.severity,
            "frame_protocol": args.frame_protocol,
        }
    with resolve_path(args.manifest).open() as stream:
        protocols = json.load(stream)["protocols"]
    if not 0 <= args.protocol_index < len(protocols):
        raise IndexError(args.protocol_index)
    return protocols[args.protocol_index]


def successful_result_exists(path, protocol_id, max_samples):
    if not path.is_file():
        return False
    try:
        with path.open() as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    expected = min(max_samples, EXPECTED_SAMPLES) if max_samples else EXPECTED_SAMPLES
    return (
        payload.get("status") == "success"
        and payload.get("protocol", {}).get("id") == protocol_id
        and payload.get("sample_count") == expected
    )


def hash_from_environment(path: Path, variable: str) -> str:
    value = os.environ.get(variable)
    return value if value else sha256_file(path)


def build_runtime(args, protocol, framework):
    from occstress_adapters.waymo_dataset import SparseWorldWaymoDataset

    config_path = resolve_path(args.config)
    cfg = framework["Config"].fromfile(str(config_path))
    cfg = framework["compat_cfg"](cfg)
    cfg = framework["patch_config"](cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.num_views = 5
    cfg.model.pts_bbox_head.transformer.num_views = 5
    cfg.data.workers_per_gpu = args.workers
    framework["setup_multi_processes"](cfg)
    framework["set_random_seed"](args.seed, deterministic=True)

    base_info = resolve_path(args.base_info)
    pose_file = resolve_path(args.pose_file)
    camera_root = resolve_path(args.camera_root)
    waymo_root = resolve_path(args.waymo_root)
    dataset = SparseWorldWaymoDataset(
        base_info=base_info,
        pose_file=pose_file,
        camera_root=camera_root,
        waymo_root=waymo_root,
        protocol=protocol,
    )
    loader = framework["build_dataloader"](
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )
    model = framework["build_model"](
        cfg.model, test_cfg=cfg.get("test_cfg")
    )
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = framework["load_checkpoint"](
        model, str(checkpoint_path), map_location="cpu"
    )
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = framework["MMDataParallel"](model.cuda(), device_ids=[0])
    model.eval()
    return (
        dataset, loader, model, checkpoint_path, base_info, pose_file,
        camera_root, waymo_root, config_path,
    )


def evaluate(args, protocol, output_path):
    framework = load_framework()
    torch = framework["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("SparseWorld-TC Waymo evaluation requires CUDA")
    (
        dataset, loader, model, checkpoint, base_info, pose_file,
        camera_root, waymo_root, config,
    ) = build_runtime(args, protocol, framework)
    expected_samples = len(dataset)
    if expected_samples != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Waymo dataset has {expected_samples}, expected {EXPECTED_SAMPLES}"
        )
    sample_limit = (
        expected_samples
        if args.max_samples is None
        else min(args.max_samples, expected_samples)
    )
    semantic_hist = np.zeros((6, 18, 18), dtype=np.int64)
    occupancy_hist = np.zeros((6, 2, 2), dtype=np.int64)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    processed = 0

    for data in loader:
        if processed >= sample_limit:
            break
        temporal_gt = data["temporal_semantics"][0]
        with torch.inference_mode():
            result = model(return_loss=False, rescale=True, **data)
        for horizon in range(6):
            prediction = unwrap_array(result[f"semantic_occ_{horizon + 1}s"][0])
            target = unwrap_array(
                temporal_gt[horizon + 1]["voxel_semantics"]
            )
            add_prediction(
                semantic_hist, occupancy_hist, horizon, prediction, target
            )
        processed += 1
        if processed == 1 or processed % 100 == 0:
            elapsed = time.time() - started
            print(
                f"[{protocol['id']}] {processed}/{sample_limit} "
                f"({processed / elapsed:.2f} samples/s)",
                flush=True,
            )
    if processed != sample_limit:
        raise RuntimeError(f"Processed {processed}, expected {sample_limit}")

    horizons = [
        metrics_from_hist(semantic_hist[index], occupancy_hist[index])
        for index in range(6)
    ]
    elapsed = time.time() - started
    adapter_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "occstress/waymo_dataset.py",
        REPO_ROOT / "mmdet3d/models/sparsedetectors/opus.py",
        PROJECT_ROOT / "scripts/waymo/effocc_waymo_camera_corruptions.py",
        PROJECT_ROOT / "scripts/waymo/waymo_occstress_common.py",
        resolve_path(args.manifest),
    ]
    payload = {
        "status": "success",
        "method": "SparseWorld-TC",
        "dataset": "OccStress-Waymo",
        "track": "camera-only",
        "evaluation_regime": "Occ3D-nuScenes checkpoint zero-shot",
        "protocol": protocol,
        "sample_count": processed,
        "dataset_sample_count": expected_samples,
        "input_times_seconds": [-2.0, -1.5, -1.0, -0.5, 0.0],
        "input_order": "current, then nearest-to-oldest history",
        "camera_names": [
            "FRONT", "FRONT_LEFT", "SIDE_LEFT", "FRONT_RIGHT", "SIDE_RIGHT"
        ],
        "camera_layout": dataset.camera_layout,
        "image_transform": (
            "per-view aspect-preserving resize to width 704, then bottom crop "
            "to 256 pixels with matching per-view projection update"
        ),
        "control_strategy": (
            "GT command plus pose-derived ST-P3-format acceleration, velocity, "
            "and four-step history"
        ),
        "class_mapping": "Occ3D-Waymo 23-label source to Occ3D 18-class",
        "future_times_seconds": FUTURE_SECONDS,
        "metrics": {
            "horizons": [
                {"seconds": seconds, **metrics}
                for seconds, metrics in zip(FUTURE_SECONDS, horizons)
            ],
            "mean_six_frames": {
                key: float(np.mean([item[key] for item in horizons]))
                for key in ("miou", "iou")
            },
        },
        "semantic_confusion": semantic_hist.tolist(),
        "occupancy_confusion": occupancy_hist.tolist(),
        "runtime": {
            "elapsed_seconds": elapsed,
            "samples_per_second": processed / elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "hostname": socket.gethostname(),
        },
        "provenance": {
            "code_commit": git_value("rev-parse", "HEAD"),
            "code_dirty": bool(git_value("status", "--porcelain")),
            "adapter_sha256": sha256_files(adapter_paths),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hash_from_environment(
                checkpoint, "SPARSEWORLD_CHECKPOINT_SHA256"
            ),
            "base_info": str(base_info),
            "base_info_sha256": hash_from_environment(
                base_info, "SPARSEWORLD_WAYMO_BASE_INFO_SHA256"
            ),
            "pose_file": str(pose_file),
            "pose_file_sha256": hash_from_environment(
                pose_file, "SPARSEWORLD_WAYMO_POSE_SHA256"
            ),
            "camera_root": str(camera_root),
            "waymo_root": str(waymo_root),
            "config_path": str(config),
            "num_views_override": 5,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    atomic_json_dump(payload, output_path)
    print(json.dumps({
        "status": payload["status"],
        "protocol": protocol["id"],
        "sample_count": processed,
        "mean_six_frames": payload["metrics"]["mean_six_frames"],
        "output_json": str(output_path),
    }, indent=2), flush=True)


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    protocol = resolve_protocol(args)
    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else resolve_path(args.output_root) / f"{protocol['id']}.json"
    )
    output_lock = acquire_output_lock(output_path)
    if output_lock is None:
        print(f"SKIP protocol already running: {protocol['id']}")
        return
    try:
        if not args.force and successful_result_exists(
            output_path, protocol["id"], args.max_samples
        ):
            print(f"SKIP existing successful result: {output_path}")
            return
        evaluate(args, protocol, output_path)
    finally:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_UN)
        output_lock.close()


if __name__ == "__main__":
    main()
