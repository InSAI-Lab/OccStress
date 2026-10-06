#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Export one rank's CARLA camera or LiDAR upstream corruption settings."""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader

from effocc_carla_upstream_adapter import (
    ADAPTER_VERSION,
    CAMERA_CORRUPTIONS,
    CAMERA_SEVERITIES,
    LIDAR_CORRUPTIONS,
    LIDAR_SEVERITIES,
    implementation_manifest,
)
from export_effocc_carla_clean import (
    EXPECTED_FRAMES,
    NATIVE_CLASSES,
    OCC3D_CLASSES,
    OrderedIndexSampler,
    atomic_json,
    atomic_npz,
    git_state,
    load_target,
    normalize_predictions,
    output_path,
    sha256_file,
    update_confusion,
    valid_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--track",
        choices=("camera", "lidar", "fusion_camera"),
        required=True,
    )
    parser.add_argument("--canonical-clean-root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--robo3d-root", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def corruption_modality(track: str) -> str:
    return "camera" if track in ("camera", "fusion_camera") else "lidar"


def model_type(track: str) -> str:
    return "camera" if track == "camera" else "fusion"


def all_settings(track: str) -> list[tuple[str, str]]:
    if corruption_modality(track) == "camera":
        return [
            (corruption, severity)
            for corruption in CAMERA_CORRUPTIONS
            for severity in CAMERA_SEVERITIES
        ]
    return [
        (corruption, severity)
        for corruption in LIDAR_CORRUPTIONS
        for severity in LIDAR_SEVERITIES
    ]


def configure_dataset(
    args: argparse.Namespace,
    corruption: str,
    severity: str,
):
    config = Config.fromfile(str(args.config.resolve()))
    config.data.test.test_mode = True
    pipeline = list(config.data.test.pipeline)
    if corruption_modality(args.track) == "camera":
        first = dict(pipeline[0])
        if first.get("type") != "PrepareImageInputs":
            raise ValueError(f"unexpected camera pipeline head: {first}")
        first.update(
            type="PrepareCorruptedCarlaImageInputs",
            corruption=corruption,
            severity=severity,
        )
        pipeline[0] = first
    else:
        if args.raw_root is None or args.robo3d_root is None:
            raise ValueError("LiDAR export requires --raw-root and --robo3d-root")
        load_index = next(
            index
            for index, transform in enumerate(pipeline)
            if transform["type"] == "LoadPointsFromFile"
        )
        pipeline.insert(
            load_index + 1,
            {
                "type": "CorruptEFFOccCarlaPoints",
                "corruption": corruption,
                "severity": severity,
                "raw_root": str(args.raw_root.resolve()),
                "robo3d_root": str(args.robo3d_root.resolve()),
            },
        )
    config.data.test.pipeline = pipeline
    dataset = build_dataset(config.data.test)
    if len(dataset) != EXPECTED_FRAMES:
        raise ValueError(f"expected {EXPECTED_FRAMES} frames, got {len(dataset)}")
    return dataset


def build_runtime(args: argparse.Namespace):
    config = Config.fromfile(str(args.config.resolve()))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    if config.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model, str(args.checkpoint.resolve()), map_location="cpu", strict=False
    )
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", NATIVE_CLASSES)
    return MMDataParallel(model, device_ids=[0]).eval()


def setting_root(base: Path, corruption: str, severity: str) -> Path:
    return base / corruption / severity


def export_setting(
    args: argparse.Namespace,
    model: MMDataParallel,
    corruption: str,
    severity: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    dataset = configure_dataset(args, corruption, severity)
    output_root = setting_root(args.output_base, corruption, severity)
    manifest_path = args.manifest_root / corruption / f"{severity}.json"
    selected = list(range(len(dataset)))
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    pending = []
    skipped = 0
    confusion = np.zeros(
        (len(OCC3D_CLASSES), len(OCC3D_CLASSES)), dtype=np.int64
    )
    for index in selected:
        info = dataset.data_infos[index]
        path = output_path(output_root, info)
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
        batch_indices = pending[pending_offset : pending_offset + len(predictions)]
        if len(batch_indices) != len(predictions):
            raise RuntimeError("prediction/index alignment failed")
        for index, prediction in zip(batch_indices, predictions):
            info = dataset.data_infos[index]
            target, mask = load_target(info, args.canonical_clean_root)
            update_confusion(confusion, prediction, target, mask)
            atomic_npz(output_path(output_root, info), prediction)
            written += 1
        pending_offset += len(predictions)
        print(
            f"track={args.track} setting={corruption}/{severity} "
            f"rank={args.shard_index}/{args.num_shards} "
            f"processed={pending_offset}/{len(pending)}",
            flush=True,
        )
    if pending_offset != len(pending):
        raise RuntimeError("not all pending frames were exported")

    code_commit, code_dirty = git_state(args.config.resolve().parents[2])
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "batch_size": args.batch_size,
        "canonical_clean_root": str(args.canonical_clean_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "confusion": confusion.tolist(),
        "corruption": corruption,
        "elapsed_seconds": time.monotonic() - started,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "expected_full_frames": EXPECTED_FRAMES,
        "frame_count": len(selected),
        "gpu": torch.cuda.get_device_name(0),
        "corruption_modality": corruption_modality(args.track),
        "implementation": implementation_manifest(
            corruption_modality(args.track)
        ),
        "max_samples": args.max_samples,
        "model": model_type(args.track),
        "num_shards": args.num_shards,
        "output_root": str(output_root),
        "peak_gpu_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "severity": severity,
        "shard_index": args.shard_index,
        "skipped_frames": skipped,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "status": "success",
        "track": args.track,
        "voxel_shape": [200, 200, 16],
        "workers": args.workers,
        "written_frames": written,
    }
    atomic_json(manifest_path.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    del loader, dataset
    return payload


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to each exporter rank")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid batch size or worker count")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if corruption_modality(args.track) == "camera" and args.raw_root is not None:
        raise ValueError("camera export does not use --raw-root")

    set_random_seed(args.seed, deterministic=True)
    args.canonical_clean_root = args.canonical_clean_root.resolve()
    args.output_base = args.output_base.resolve()
    args.manifest_root = args.manifest_root.resolve()
    checkpoint_sha256 = sha256_file(args.checkpoint.resolve())
    model = build_runtime(args)
    torch.cuda.reset_peak_memory_stats()

    settings = all_settings(args.track)
    assigned = settings[args.shard_index :: args.num_shards]
    if not assigned:
        raise ValueError("this rank has no assigned settings")
    print(
        f"track={args.track} rank={args.shard_index}/{args.num_shards} "
        f"assigned={assigned}",
        flush=True,
    )
    for corruption, severity in assigned:
        export_setting(
            args,
            model,
            corruption,
            severity,
            checkpoint_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
