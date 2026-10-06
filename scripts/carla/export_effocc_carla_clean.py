#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Export one resumable shard of clean EFFOcc CARLA predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader, Sampler


EXPECTED_FRAMES = 360
VOXEL_SHAPE = (200, 200, 16)
NATIVE_CLASSES = (
    "undefined",
    "car",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "vegetation",
    "road",
    "terrain",
    "building",
    "free",
)
OCC3D_CLASSES = (
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
)
UNIOCC_TO_OCC3D = np.asarray(
    [0, 4, 2, 6, 7, 8, 16, 11, 14, 15, 17], dtype=np.uint8
)
ADAPTER_VERSION = "effocc-carla-clean-v1"


class OrderedIndexSampler(Sampler):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", choices=("camera", "fusion"), required=True)
    parser.add_argument("--canonical-clean-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_state(path: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("EFFOCC_BASE_COMMIT")
        dirty = os.environ.get("EFFOCC_CODE_DIRTY")
        return commit, None if dirty is None else dirty == "1"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, semantics=semantics)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def valid_prediction(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != ["semantics"]:
                return False
            semantics = payload["semantics"]
            return (
                semantics.shape == VOXEL_SHAPE
                and semantics.dtype == np.uint8
                and set(np.unique(semantics).tolist()).issubset(
                    set(UNIOCC_TO_OCC3D.tolist())
                )
            )
    except (OSError, ValueError):
        return False


def scene_token(info: dict) -> tuple[str, str]:
    scene = Path(info["occ_gt_path"]).parent.name
    if not scene.startswith("scene_"):
        raise ValueError(f"cannot derive CARLA scene from {info['occ_gt_path']}")
    frame = int(info["timestamp"])
    token = f"carla-val-{scene.removeprefix('scene_')}-{frame:03d}"
    return scene, token


def output_path(root: Path, info: dict) -> Path:
    scene, token = scene_token(info)
    return root / scene / token / "labels.npz"


def load_target(
    info: dict, canonical_clean_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    scene, token = scene_token(info)
    with np.load(info["occ_gt_path"], allow_pickle=False) as payload:
        native = np.asarray(payload["semantics"], dtype=np.uint8)
        mask = np.asarray(payload["mask_camera"], dtype=bool)
    if native.shape != VOXEL_SHAPE or mask.shape != VOXEL_SHAPE:
        raise ValueError(f"invalid prepared GT shape for {scene}/{token}")
    if native.max(initial=0) >= len(UNIOCC_TO_OCC3D):
        raise ValueError(f"invalid native label in {info['occ_gt_path']}")
    mapped = UNIOCC_TO_OCC3D[native]

    canonical_path = canonical_clean_root / scene / token / "labels.npz"
    with np.load(canonical_path, allow_pickle=False) as payload:
        canonical = np.asarray(payload["semantics"], dtype=np.uint8)
        infov = np.asarray(payload["infov"], dtype=bool)
    if not np.array_equal(mapped, canonical):
        raise ValueError(f"canonical semantics mismatch for {scene}/{token}")
    if not np.array_equal(mask, infov):
        raise ValueError(f"canonical visibility mask mismatch for {scene}/{token}")
    return mapped, mask


def update_confusion(
    confusion: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> None:
    valid = mask & (target >= 0) & (target < len(OCC3D_CLASSES))
    encoded = len(OCC3D_CLASSES) * target[valid].astype(np.int64)
    encoded += prediction[valid].astype(np.int64)
    confusion += np.bincount(
        encoded, minlength=len(OCC3D_CLASSES) ** 2
    ).reshape(len(OCC3D_CLASSES), len(OCC3D_CLASSES))


def normalize_predictions(result: object, batch_size: int) -> list[np.ndarray]:
    if not isinstance(result, list) or len(result) != batch_size:
        raise TypeError(
            f"expected {batch_size} predictions, got {type(result)} "
            f"with length {len(result) if isinstance(result, list) else 'n/a'}"
        )
    predictions = []
    for prediction in result:
        native = np.asarray(prediction).squeeze()
        if native.shape != VOXEL_SHAPE:
            raise ValueError(f"unexpected prediction shape {native.shape}")
        if native.min(initial=0) < 0 or native.max(initial=0) >= len(NATIVE_CLASSES):
            raise ValueError("prediction labels are outside the native ontology")
        predictions.append(UNIOCC_TO_OCC3D[native.astype(np.uint8)])
    return predictions


def build_runtime(args: argparse.Namespace):
    config = Config.fromfile(str(args.config.resolve()))
    config.model.pretrained = None
    config.model.train_cfg = None
    config.data.test.test_mode = True
    dataset = build_dataset(config.data.test)
    if len(dataset) != EXPECTED_FRAMES:
        raise ValueError(f"expected {EXPECTED_FRAMES} frames, got {len(dataset)}")

    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    if config.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model, str(args.checkpoint.resolve()), map_location="cpu", strict=False
    )
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0]).eval()
    return model, dataset


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid batch size or worker count")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_random_seed(args.seed, deterministic=True)
    args.output_root = args.output_root.resolve()
    args.canonical_clean_root = args.canonical_clean_root.resolve()
    checkpoint_sha256 = sha256_file(args.checkpoint.resolve())
    model, dataset = build_runtime(args)
    torch.cuda.reset_peak_memory_stats()

    assigned = list(range(args.shard_index, len(dataset), args.num_shards))
    if args.max_samples is not None:
        assigned = assigned[:args.max_samples]
    pending = []
    skipped = 0
    confusion = np.zeros(
        (len(OCC3D_CLASSES), len(OCC3D_CLASSES)), dtype=np.int64
    )
    for index in assigned:
        info = dataset.data_infos[index]
        path = output_path(args.output_root, info)
        target, mask = load_target(info, args.canonical_clean_root)
        if not args.overwrite and valid_prediction(path):
            with np.load(path, allow_pickle=False) as payload:
                prediction = payload["semantics"]
            update_confusion(confusion, prediction, target, mask)
            skipped += 1
        else:
            pending.append(index)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=OrderedIndexSampler(pending),
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=args.batch_size),
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    started = time.monotonic()
    written = 0
    pending_offset = 0
    for data in loader:
        with torch.inference_mode():
            result = model(return_loss=False, rescale=True, **data)
        predictions = normalize_predictions(result, len(result))
        batch_indices = pending[
            pending_offset : pending_offset + len(predictions)
        ]
        if len(batch_indices) != len(predictions):
            raise RuntimeError("prediction/index alignment failed")
        for index, prediction in zip(batch_indices, predictions):
            info = dataset.data_infos[index]
            target, mask = load_target(info, args.canonical_clean_root)
            update_confusion(confusion, prediction, target, mask)
            atomic_npz(output_path(args.output_root, info), prediction)
            written += 1
        pending_offset += len(predictions)
        print(
            f"model={args.model} shard={args.shard_index}/{args.num_shards} "
            f"processed={pending_offset}/{len(pending)}",
            flush=True,
        )
    if pending_offset != len(pending):
        raise RuntimeError("not all pending frames were exported")

    code_commit, code_dirty = git_state(args.config.resolve().parents[2])
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "assigned_indices": assigned,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "canonical_clean_root": str(args.canonical_clean_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "confusion": confusion.tolist(),
        "elapsed_seconds": time.monotonic() - started,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "frame_count": len(assigned),
        "gpu": torch.cuda.get_device_name(0),
        "mapping": UNIOCC_TO_OCC3D.tolist(),
        "model": args.model,
        "num_shards": args.num_shards,
        "output_root": str(args.output_root),
        "peak_gpu_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "shard_index": args.shard_index,
        "skipped_frames": skipped,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "status": "success",
        "voxel_shape": list(VOXEL_SHAPE),
        "workers": args.workers,
        "written_frames": written,
    }
    atomic_json(args.manifest.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
