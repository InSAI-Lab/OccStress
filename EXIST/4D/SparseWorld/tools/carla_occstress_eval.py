#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Evaluate SparseWorld-TC zero-shot on UniOcc-CARLA camera protocols."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CLASS_NAMES = [
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
]
FUTURE_SECONDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-info", required=True)
    parser.add_argument("--manifest", default="occstress/protocols.json")
    parser.add_argument("--protocol-index", type=int)
    parser.add_argument("--protocol-id", default="clean")
    parser.add_argument("--corruption", default="clean")
    parser.add_argument("--severity")
    parser.add_argument("--frame-protocol")
    parser.add_argument("--output-json")
    parser.add_argument("--output-root", default="occstress/carla/results")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    from occstress.datasets.paths import resolve_occstress_path
    return resolve_occstress_path(value, code_root=REPO_ROOT)


def resolve_protocol(args):
    if args.protocol_index is None:
        protocol = {
            "id": args.protocol_id,
            "corruption": args.corruption,
            "severity": args.severity,
            "frame_protocol": args.frame_protocol,
        }
    else:
        with resolve_path(args.manifest).open() as stream:
            protocols = json.load(stream)["protocols"]
        if not 0 <= args.protocol_index < len(protocols):
            raise IndexError(args.protocol_index)
        protocol = protocols[args.protocol_index]
    return protocol


def atomic_json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def successful_result_exists(path, protocol_id, max_samples):
    if not path.is_file():
        return False
    try:
        with path.open() as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "success":
        return False
    if payload.get("protocol", {}).get("id") != protocol_id:
        return False
    expected = max_samples if max_samples is not None else 330
    return payload.get("sample_count") == expected


def acquire_output_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    stream = lock_path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def unwrap_array(value):
    while value.__class__.__name__ == "DataContainer":
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        import torch
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    value = np.asarray(value)
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    return value


def confusion(prediction, target, classes):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != (200, 200, 16) or target.shape != prediction.shape:
        raise ValueError(
            f"Bad prediction/target shapes: {prediction.shape}/{target.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or Inf")
    if prediction.min() < 0 or prediction.max() >= classes:
        raise ValueError(
            f"Prediction outside [0,{classes - 1}]: "
            f"{prediction.min()}..{prediction.max()}"
        )
    valid = (target >= 0) & (target < classes)
    encoded = classes * target[valid].astype(np.int64)
    encoded += prediction[valid].astype(np.int64)
    return np.bincount(
        encoded, minlength=classes * classes
    ).reshape(classes, classes)


def metrics_from_hist(semantic_hist, occupancy_hist):
    denominator = (
        semantic_hist.sum(1) + semantic_hist.sum(0) - np.diag(semantic_hist)
    )
    semantic_iou = np.divide(
        np.diag(semantic_hist),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator != 0,
    )
    # CARLA only exposes a subset of Occ3D classes. Present-class mIoU is
    # defined by GT support, not by classes spuriously predicted by a model.
    present = semantic_hist.sum(1) > 0
    present[17] = False
    occupancy_denominator = (
        occupancy_hist.sum(1) + occupancy_hist.sum(0)
        - np.diag(occupancy_hist)
    )
    occupancy_iou = np.divide(
        np.diag(occupancy_hist),
        occupancy_denominator,
        out=np.full(occupancy_denominator.shape, np.nan, dtype=np.float64),
        where=occupancy_denominator != 0,
    )
    return {
        "miou": float(np.mean(semantic_iou[present]) * 100),
        "iou": float(occupancy_iou[1] * 100),
        "present_class_ids": np.flatnonzero(present).tolist(),
        "present_class_names": [
            CLASS_NAMES[index] for index in np.flatnonzero(present)
        ],
        "per_class_iou": {
            name: None if np.isnan(value) else float(value * 100)
            for name, value in zip(CLASS_NAMES, semantic_iou)
        },
    }


def add_prediction(hist_semantic, hist_occupancy, horizon, prediction, target):
    hist_semantic[horizon] += confusion(prediction, target, 18)
    hist_occupancy[horizon] += confusion(
        (prediction != 17).astype(np.uint8),
        (target != 17).astype(np.uint8),
        2,
    )


def load_framework():
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.parallel import _functions as mmcv_parallel_functions
    from mmcv.runner import load_checkpoint
    import mmdet
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    torch_get_stream = torch.nn.parallel._functions._get_stream

    def get_stream_compat(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return torch_get_stream(device)

    mmcv_parallel_functions._get_stream = get_stream_compat
    if mmdet.__version__ > "2.23.0":
        from mmdet.utils import compat_cfg, setup_multi_processes
    else:
        from mmdet3d.utils import compat_cfg, setup_multi_processes
    return locals()


def build_runtime(args, protocol, framework):
    from occstress_adapters.carla_dataset import SparseWorldCarlaDataset

    config_path = resolve_path(args.config)
    cfg = framework["Config"].fromfile(str(config_path))
    cfg = framework["compat_cfg"](cfg)
    cfg = framework["patch_config"](cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.num_views = 4
    cfg.model.pts_bbox_head.transformer.num_views = 4
    cfg.data.workers_per_gpu = args.workers
    framework["setup_multi_processes"](cfg)
    framework["set_random_seed"](args.seed, deterministic=True)

    base_info = resolve_path(args.base_info)
    dataset = SparseWorldCarlaDataset(base_info, protocol)
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
    return dataset, loader, model, checkpoint_path, base_info, config_path


def evaluate(args, protocol, output_path):
    framework = load_framework()
    torch = framework["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("SparseWorld-TC CARLA evaluation requires CUDA")
    dataset, loader, model, checkpoint, base_info, config = build_runtime(
        args, protocol, framework
    )
    expected_samples = len(dataset)
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
        if processed == 1 or processed % 25 == 0:
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
        REPO_ROOT / "occstress/carla_dataset.py",
        REPO_ROOT / "mmdet3d/models/sparsedetectors/opus.py",
        PROJECT_ROOT / "scripts/carla/effocc_carla_upstream_adapter.py",
        resolve_path(args.manifest),
    ]
    payload = {
        "status": "success",
        "method": "SparseWorld-TC",
        "dataset": "UniOcc-CARLA",
        "track": "camera-only",
        "evaluation_regime": "Occ3D-nuScenes checkpoint zero-shot",
        "protocol": protocol,
        "sample_count": processed,
        "dataset_sample_count": expected_samples,
        "input_times_seconds": [-2.0, -1.5, -1.0, -0.5, 0.0],
        "input_order": "current, then nearest-to-oldest history",
        "camera_names": ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK"],
        "control_strategy": (
            "GT command plus pose-derived ST-P3-format acceleration, velocity, "
            "and four-step history"
        ),
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
            "checkpoint_sha256": sha256_file(checkpoint),
            "base_info": str(base_info),
            "base_info_sha256": sha256_file(base_info),
            "config_path": str(config),
            "num_views_override": 4,
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
        # Recheck after acquiring the lock because another worker may have
        # completed while this process was waiting to start.
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
