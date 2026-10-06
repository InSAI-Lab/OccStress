#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Export EFFOcc predictions for camera-corrupted, clean-LiDAR OccStress-Waymo."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader

from export_effocc_waymo_upstream import (
    EXPECTED_FRAMES,
    EXPECTED_SCENES,
    VOXEL_SHAPE,
    WAYMO_TO_OCCSTRESS,
    OrderedIndexSampler,
    atomic_json,
    atomic_npz,
    git_commit,
    git_is_dirty,
    metrics_from_hist,
    normalize_predictions,
    sample_scene_frame,
    sha256_file,
    update_confusion,
)
from effocc_waymo_camera_corruptions import SEVERITY


ADAPTER_VERSION = "effocc-waymo-robobev-camera-v1"
SOURCE_MODALITY = "camera_lidar"
CORRUPTED_MODALITY = "camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corruption", choices=("Clean", *sorted(SEVERITY)))
    parser.add_argument(
        "severity",
        nargs="?",
        choices=("clean", "easy", "mid", "hard"),
        default="clean",
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--occ-gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-end", type=int, default=EXPECTED_SCENES)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setting_root(root: Path, corruption: str, severity: str) -> Path:
    if corruption == "Clean":
        return root / "clean"
    return root / corruption / severity


def scene_done(
    root: Path,
    scene: int,
    expected_frames: int,
    checkpoint_sha256: str,
    corruption: str,
    severity: str,
) -> bool:
    path = root / f"{scene:03d}" / ".done.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "success"
        and payload.get("adapter_version") == ADAPTER_VERSION
        and payload.get("checkpoint_sha256") == checkpoint_sha256
        and payload.get("corruption") == corruption
        and payload.get("severity") == severity
        and int(payload.get("frames", -1)) == expected_frames
    )


def configure_dataset(config: Config, args: argparse.Namespace) -> None:
    clean = args.corruption == "Clean"
    if clean != (args.severity == "clean"):
        raise ValueError("Clean requires severity=clean and corruptions require easy/mid/hard")

    config.model.pretrained = None
    config.model.train_cfg = None
    config.data.test.ann_file = str(args.annotation.resolve())
    config.data.test.data_root = str(args.data_root.resolve()) + "/"
    config.data.test.pose_file = str(args.pose_file.resolve())
    config.data.test.occ_gt_data_root = str(args.occ_gt_root.resolve()) + "/"
    config.data.test.test_mode = True

    pipeline = list(config.data.test.pipeline)
    indexes = [
        index
        for index, transform in enumerate(pipeline)
        if transform["type"] == "PrepareImageInputsWaymo"
    ]
    if len(indexes) != 1:
        raise ValueError(
            "expected exactly one PrepareImageInputsWaymo transform, "
            f"found {len(indexes)}"
        )
    prepare = dict(pipeline[indexes[0]])
    if not clean:
        prepare.update(
            image_corruption=args.corruption,
            image_corruption_severity=args.severity,
        )
    pipeline[indexes[0]] = prepare
    config.data.test.pipeline = pipeline


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if not 0 <= args.scene_start < args.scene_end <= EXPECTED_SCENES:
        raise ValueError("expected 0 <= scene-start < scene-end <= 202")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid batch size or worker count")

    set_random_seed(args.seed, deterministic=True)
    config = Config.fromfile(str(args.config.resolve()))
    configure_dataset(config, args)
    dataset = build_dataset(config.data.test)
    if len(dataset) != EXPECTED_FRAMES:
        raise ValueError(
            f"expected {EXPECTED_FRAMES} canonical frames, got {len(dataset)}"
        )

    scene_indices: dict[int, list[int]] = {}
    for index, info in enumerate(dataset.data_infos):
        scene, _ = sample_scene_frame(info)
        if args.scene_start <= scene < args.scene_end:
            scene_indices.setdefault(scene, []).append(index)
    if not scene_indices:
        raise ValueError("scene selection produced no samples")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    output_setting_root = setting_root(
        args.output_root, args.corruption, args.severity
    )
    pending = []
    skipped_scenes = []
    for scene, indices in sorted(scene_indices.items()):
        if (
            not args.overwrite
            and scene_done(
                output_setting_root,
                scene,
                len(indices),
                checkpoint_sha256,
                args.corruption,
                args.severity,
            )
        ):
            skipped_scenes.append(scene)
        else:
            pending.extend(indices)
    if args.max_samples is not None:
        pending = pending[: args.max_samples]
    if not pending:
        payload = {
            "adapter_version": ADAPTER_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "corrupted_modality": CORRUPTED_MODALITY,
            "corruption": args.corruption,
            "severity": args.severity,
            "skipped_scenes": skipped_scenes,
            "source_modality": SOURCE_MODALITY,
            "status": "nothing-to-do",
        }
        atomic_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=OrderedIndexSampler(pending),
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=args.batch_size),
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    checkpoint = load_checkpoint(
        model,
        str(args.checkpoint.resolve()),
        map_location="cpu",
        strict=False,
    )
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()

    expected_counts = Counter(
        sample_scene_frame(dataset.data_infos[index])[0] for index in pending
    )
    completed_counts: Counter[int] = Counter()
    completed_scenes = []
    confusion = np.zeros((16, 16), dtype=np.int64)
    start_time = time.monotonic()
    completed = 0
    for data in data_loader:
        data = scatter(data, [0])[0]
        image_metas = data["img_metas"][0]
        with torch.inference_mode():
            result = model(return_loss=False, rescale=True, **data)
        predictions = normalize_predictions(result, len(image_metas))
        for metadata, native in zip(image_metas, predictions):
            sample_idx = int(metadata["sample_idx"])
            scene = sample_idx % 1_000_000 // 1_000
            frame = sample_idx % 1_000
            token = f"waymo-val-{scene:03d}-{frame:03d}"
            atomic_npz(
                output_setting_root
                / f"{scene:03d}"
                / token
                / "labels.npz",
                WAYMO_TO_OCCSTRESS[native],
            )

            info = dataset.data_infos[pending[completed]]
            target, _, mask_camera, mask_infov = dataset.get_occ_gt_from_info(info)
            update_confusion(
                confusion,
                native,
                target,
                np.logical_and(mask_camera, mask_infov),
            )
            completed += 1
            completed_counts[scene] += 1
            if completed_counts[scene] == expected_counts[scene]:
                atomic_json(
                    output_setting_root / f"{scene:03d}" / ".done.json",
                    {
                        "adapter_version": ADAPTER_VERSION,
                        "checkpoint_sha256": checkpoint_sha256,
                        "corrupted_modality": CORRUPTED_MODALITY,
                        "corruption": args.corruption,
                        "frames": completed_counts[scene],
                        "scene": f"{scene:03d}",
                        "severity": args.severity,
                        "source_modality": SOURCE_MODALITY,
                        "status": "success",
                        "voxel_shape": list(VOXEL_SHAPE),
                    },
                )
                completed_scenes.append(scene)
        if completed % args.log_interval < len(predictions) or completed == len(pending):
            elapsed = time.monotonic() - start_time
            print(
                f"exported {completed}/{len(pending)} frames "
                f"({completed / elapsed:.3f} frames/s)",
                flush=True,
            )

    elapsed = time.monotonic() - start_time
    code_root = args.config.resolve().parents[2]
    corruption_impl = Path(__file__).resolve().with_name(
        "effocc_waymo_camera_corruptions.py"
    )
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "batch_size": args.batch_size,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": git_commit(code_root),
        "code_dirty": git_is_dirty(code_root),
        "completed_scenes": completed_scenes,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "corrupted_modality": CORRUPTED_MODALITY,
        "corruption": args.corruption,
        "corruption_implementation": str(corruption_impl),
        "corruption_implementation_sha256": sha256_file(corruption_impl),
        "elapsed_seconds": elapsed,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "frame_count": len(pending),
        "gpu": torch.cuda.get_device_name(0),
        "mapping": WAYMO_TO_OCCSTRESS.tolist(),
        "metrics": metrics_from_hist(confusion),
        "output_root": str(output_setting_root.resolve()),
        "peak_gpu_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "robobev_severity": SEVERITY,
        "sample_count": len(pending),
        "scene_end": args.scene_end,
        "scene_ids": sorted(expected_counts),
        "scene_start": args.scene_start,
        "severity": args.severity,
        "skipped_scenes": skipped_scenes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_modality": SOURCE_MODALITY,
        "status": "success",
        "workers": args.workers,
    }
    atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
