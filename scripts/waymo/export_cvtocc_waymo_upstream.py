#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Export rolling CVT-Occ predictions for the OccStress-Waymo upstream track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader, Sampler

from eval_cvtocc_waymo_protocol import (
    CLASS_NAMES,
    enable_mmcv_torch28_scatter_compat,
    corrupt_image_tensor,
)
from waymo_corruptions import SEVERITY


EXPECTED_SCENES = 202
EXPECTED_FRAMES = 7_998
VOXEL_SHAPE = (200, 200, 16)
ADAPTER_VERSION = "cvtocc-waymo-2hz-v1"
CVTOCC_TO_OCCSTRESS = np.asarray(
    [0, 4, 7, 15, 2, 15, 15, 8, 2, 6, 15, 16, 16, 11, 13, 17],
    dtype=np.uint8,
)


class OrderedIndexSampler(Sampler):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corruption", choices=["Clean", *sorted(SEVERITY)])
    parser.add_argument(
        "severity",
        nargs="?",
        choices=("easy", "mid", "hard"),
        default="easy",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-end", type=int, default=EXPECTED_SCENES)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, semantics: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, semantics=semantics)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def setting_root(output_root: Path, corruption: str, severity: str) -> Path:
    if corruption == "Clean":
        return output_root / "clean"
    return output_root / corruption / severity


def sample_scene_frame(info: dict) -> tuple[int, int]:
    sample_idx = int(info["image"]["image_idx"])
    return sample_idx % 1_000_000 // 1_000, sample_idx % 1_000


def scene_done(
    root: Path,
    scene: int,
    expected_frames: int,
    checkpoint_sha256: str,
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
        and int(payload.get("frames", -1)) == expected_frames
        and payload.get("checkpoint_sha256") == checkpoint_sha256
    )


def metrics_from_hist(hist: np.ndarray) -> dict[str, Any]:
    true_positive = np.diag(hist)
    denominator = hist.sum(1) + hist.sum(0) - true_positive
    iou = np.divide(
        true_positive,
        denominator,
        out=np.full(16, np.nan, dtype=np.float64),
        where=denominator != 0,
    )
    without_motorcycle = np.concatenate((iou[:9], iou[10:-1]))
    return {
        "class_iou": {
            name: None if np.isnan(value) else float(value * 100)
            for name, value in zip(CLASS_NAMES, iou)
        },
        "miou_non_free": float(np.nanmean(iou[:-1]) * 100),
        "miou_non_free_without_motorcycle": float(
            np.nanmean(without_motorcycle) * 100
        ),
    }


def update_confusion(
    confusion: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> None:
    valid = mask.astype(bool)
    target = target[valid].astype(np.int64, copy=False)
    prediction = prediction[valid].astype(np.int64, copy=False)
    counts = np.bincount(
        16 * target + prediction,
        minlength=16 * 16,
    ).reshape(16, 16)
    confusion += counts


def main() -> int:
    args = parse_args()
    if args.workers < 0 or args.corruption_workers < 0:
        raise ValueError("worker counts must be non-negative")
    if not 0 <= args.scene_start < args.scene_end <= EXPECTED_SCENES:
        raise ValueError("expected 0 <= scene-start < scene-end <= 202")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expose exactly one GPU with CUDA_VISIBLE_DEVICES"
        )

    enable_mmcv_torch28_scatter_compat()
    set_random_seed(args.seed, deterministic=True)
    config = Config.fromfile(str(args.config))
    config.model.pretrained = None
    config.model.train_cfg = None
    config.model.save_results = True
    config.data.test.test_mode = True

    import projects.mmdet3d_plugin  # noqa: F401

    dataset = build_dataset(config.data.test)
    scene_indices: dict[int, list[int]] = {}
    for index, info in enumerate(dataset.data_infos_full):
        scene, _ = sample_scene_frame(info)
        if args.scene_start <= scene < args.scene_end:
            scene_indices.setdefault(scene, []).append(index)
    if not scene_indices:
        raise ValueError("scene selection produced no samples")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    output_setting_root = setting_root(
        args.output_root,
        args.corruption,
        args.severity,
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
            "corruption": args.corruption,
            "severity": args.severity,
            "skipped_scenes": skipped_scenes,
            "status": "nothing-to-do",
        }
        atomic_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=OrderedIndexSampler(pending),
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=1),
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    checkpoint = load_checkpoint(
        model,
        str(args.checkpoint),
        map_location="cpu",
        strict=False,
    )
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()

    executor = (
        ProcessPoolExecutor(
            max_workers=args.corruption_workers,
            mp_context=get_context("spawn"),
        )
        if args.corruption_workers
        else None
    )
    confusion = np.zeros((16, 16), dtype=np.int64)
    expected_counts = Counter(
        sample_scene_frame(dataset.data_infos_full[index])[0]
        for index in pending
    )
    completed_counts: Counter[int] = Counter()
    completed_scenes = []
    start_time = time.monotonic()
    try:
        for completed, data in enumerate(data_loader, start=1):
            data = scatter(data, [0])[0]
            image = data["img"][0]
            image_metas = data["img_metas"][0]
            voxel_semantics = data["voxel_semantics"][0]
            valid_mask = data["valid_mask"][0]
            sample_idx = int(image_metas[0]["sample_idx"])
            scene = sample_idx % 1_000_000 // 1_000
            frame = sample_idx % 1_000
            token = f"waymo-val-{scene:03d}-{frame:03d}"
            if args.corruption != "Clean":
                data["img"][0] = corrupt_image_tensor(
                    image,
                    image_metas,
                    corruption=args.corruption,
                    severity=args.severity,
                    sample_key=str(sample_idx),
                    executor=executor,
                )

            with torch.inference_mode():
                result = model(return_loss=False, rescale=True, **data)
            if (
                not isinstance(result, dict)
                or "voxel_semantics_preds" not in result
            ):
                raise TypeError(f"unexpected model result: {type(result)}")
            native = np.asarray(result["voxel_semantics_preds"]).squeeze()
            if native.shape != VOXEL_SHAPE:
                raise ValueError(
                    f"{token}: prediction shape {native.shape} != {VOXEL_SHAPE}"
                )
            if native.min(initial=0) < 0 or native.max(initial=0) > 15:
                raise ValueError(f"{token}: native labels outside [0, 15]")
            mapped = CVTOCC_TO_OCCSTRESS[native.astype(np.uint8, copy=False)]
            atomic_npz(
                output_setting_root
                / f"{scene:03d}"
                / token
                / "labels.npz",
                mapped,
            )
            update_confusion(
                confusion,
                native,
                voxel_semantics.detach().cpu().numpy().squeeze(),
                valid_mask.detach().cpu().numpy().squeeze(),
            )
            completed_counts[scene] += 1
            if completed_counts[scene] == expected_counts[scene]:
                scene_payload = {
                    "adapter_version": ADAPTER_VERSION,
                    "checkpoint_sha256": checkpoint_sha256,
                    "corruption": args.corruption,
                    "frames": completed_counts[scene],
                    "scene": f"{scene:03d}",
                    "severity": args.severity,
                    "status": "success",
                    "voxel_shape": list(VOXEL_SHAPE),
                }
                atomic_json(
                    output_setting_root / f"{scene:03d}" / ".done.json",
                    scene_payload,
                )
                completed_scenes.append(scene)
            if completed % args.log_interval == 0 or completed == len(pending):
                elapsed = time.monotonic() - start_time
                print(
                    f"exported {completed}/{len(pending)} frames "
                    f"({completed / elapsed:.3f} frames/s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed = time.monotonic() - start_time
    code_root = args.config.parents[3]
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": git_commit(code_root),
        "code_dirty": git_is_dirty(code_root),
        "completed_scenes": completed_scenes,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "confusion_matrix": confusion.tolist(),
        "corruption": args.corruption,
        "elapsed_seconds": elapsed,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "frame_count": len(pending),
        "gpu": torch.cuda.get_device_name(0),
        "mapping": CVTOCC_TO_OCCSTRESS.tolist(),
        "metrics": metrics_from_hist(confusion),
        "output_root": str(output_setting_root.resolve()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "sample_count": len(pending),
        "scene_end": args.scene_end,
        "scene_ids": sorted(expected_counts),
        "scene_start": args.scene_start,
        "severity": args.severity,
        "skipped_scenes": skipped_scenes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "ok",
    }
    atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
