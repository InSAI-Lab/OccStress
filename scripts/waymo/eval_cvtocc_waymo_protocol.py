#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Strict-anchor RoboBEV-style OccStress evaluation for CVT-Occ on Waymo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader, Sampler

from waymo_corruptions import (
    SEVERITY,
    SLOW_IMAGE_CORRUPTIONS,
    WAYMO_CAMERA_NAMES,
    camera_crash_draw,
    corrupt_views_parallel,
    corrupt_waymo_frame_files_into_memmap,
    corrupt_waymo_files_parallel,
)


PATTERN_CORRUPT_OFFSETS = {
    "current_only": (0,),
    "history_only": (-30, -25, -20, -15, -10, -5),
    "recent_burst": (-5, 0),
}
HISTORY_OFFSETS = (30, 25, 20, 15, 10, 5)
CLASS_NAMES = (
    "GO",
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_SIGN",
    "TYPE_BICYCLIST",
    "TYPE_TRAFFIC_LIGHT",
    "TYPE_POLE",
    "TYPE_CONSTRUCTION_CONE",
    "TYPE_BICYCLE",
    "TYPE_MOTORCYCLE",
    "TYPE_BUILDING",
    "TYPE_VEGETATION",
    "TYPE_TREE_TRUNK",
    "TYPE_ROAD",
    "TYPE_WALKABLE",
    "TYPE_FREE",
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
    parser.add_argument("severity", choices=("easy", "mid", "hard"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-end", type=int, default=202)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-workers", type=int, default=0)
    parser.add_argument("--corruption-prefetch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--sequential-bev", action="store_true")
    parser.add_argument("--sequential-patterns", action="store_true")
    parser.add_argument("--verify-cache", action="store_true")
    parser.add_argument("--verify-raw-preprocess", action="store_true")
    return parser.parse_args()


def enable_mmcv_torch28_scatter_compat() -> None:
    import mmcv.parallel._functions as mmcv_functions
    import torch.nn.parallel._functions as torch_functions

    torch_get_stream = torch_functions._get_stream

    def get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return torch_get_stream(device)

    mmcv_functions._get_stream = get_stream


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def dataset_indices(dataset, scene_start: int, scene_end: int) -> list[int]:
    if not 0 <= scene_start < scene_end <= 202:
        raise ValueError("Expected 0 <= scene-start < scene-end <= 202")
    indices = []
    for index, info in enumerate(dataset.data_infos_full):
        sample_idx = int(info["image"]["image_idx"])
        scene_idx = sample_idx % 1_000_000 // 1_000
        if scene_start <= scene_idx < scene_end:
            indices.append(index)
    if not indices:
        raise ValueError("Scene selection produced no samples")
    return indices


def metrics_from_hist(hist: np.ndarray) -> dict:
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


def unpack_batch(data: dict) -> tuple[torch.Tensor, list[dict], torch.Tensor, torch.Tensor]:
    data = scatter(data, [0])[0]
    image = data["img"][0]
    image_metas = data["img_metas"][0]
    voxel_semantics = data["voxel_semantics"][0]
    valid_mask = data["valid_mask"][0]
    if image.shape[:2] != (1, 5):
        raise ValueError(f"unexpected image shape: {tuple(image.shape)}")
    if len(image_metas) != 1:
        raise ValueError(f"expected one image meta, got {len(image_metas)}")
    return image, image_metas, voxel_semantics, valid_mask


def load_raw_waymo_views(image_metas: list[dict]) -> list[np.ndarray]:
    filenames = image_metas[0].get("filename")
    if not isinstance(filenames, (list, tuple)) or len(filenames) != 5:
        raise ValueError("image metadata must contain five source filenames")
    raw_views = []
    for filename in filenames:
        raw = mmcv.imread(filename, flag="unchanged")
        if raw is None:
            raise OSError(f"failed to read source image: {filename}")
        if raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError(
                f"expected HxWx3 source image, got {raw.shape}: {filename}"
            )
        raw_views.append(raw)
    return raw_views


def preprocess_raw_waymo_views(
    views: list[np.ndarray],
    image: torch.Tensor,
) -> torch.Tensor:
    if len(views) != 5:
        raise ValueError(f"Waymo preprocessing expects 5 views, got {len(views)}")
    target_height, target_width = image.shape[-2:]
    padded_height = target_height * 2
    padded_width = target_width * 2
    padded_views = []
    for view in views:
        if view.shape[0] != padded_height:
            if view.shape[0] > padded_height or view.shape[1] > padded_width:
                raise ValueError(
                    "source image exceeds the CVT Waymo padding canvas: "
                    f"{view.shape} vs {(padded_height, padded_width)}"
                )
            padded = np.zeros((padded_height, padded_width, 3))
            padded[: view.shape[0], : view.shape[1]] = view
            view = padded
        padded_views.append(view)

    # Match MyLoadMultiViewImageFromFiles and RandomScaleImageMultiViewImage:
    # stack, cast to float32, then resize each BGR view.
    stacked_padded = np.stack(padded_views, axis=-1).astype(np.float32)
    resized = [
        mmcv.imresize(
            stacked_padded[..., camera_idx],
            (target_width, target_height),
            return_scale=False,
        )
        for camera_idx in range(5)
    ]
    stacked = np.stack(resized)
    return (
        torch.from_numpy(stacked)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .to(device=image.device, dtype=image.dtype)
    )


def corrupt_image_tensor(
    image: torch.Tensor,
    image_metas: list[dict],
    corruption: str,
    severity: str,
    sample_key: str,
    executor=None,
    raw_views: list[np.ndarray] | None = None,
    parallel_output: np.ndarray | None = None,
    parallel_output_path: str | None = None,
) -> torch.Tensor:
    if corruption == "Clean":
        return image.clone()
    if executor is not None and corruption in SLOW_IMAGE_CORRUPTIONS:
        filenames = image_metas[0].get("filename")
        if not isinstance(filenames, (list, tuple)) or len(filenames) != 5:
            raise ValueError("image metadata must contain five source filenames")
        target_height, target_width = image.shape[-2:]
        stacked = corrupt_waymo_files_parallel(
            filenames,
            corruption=corruption,
            severity=severity,
            sample_key=sample_key,
            target_height=target_height,
            target_width=target_width,
            executor=executor,
            output=parallel_output,
            output_path=parallel_output_path,
        )
        return (
            torch.from_numpy(stacked)
            .unsqueeze(0)
            .to(device=image.device, dtype=image.dtype)
        )

    if raw_views is None:
        raw_views = load_raw_waymo_views(image_metas)
    corrupted = corrupt_views_parallel(
        raw_views,
        corruption=corruption,
        severity=severity,
        sample_key=sample_key,
        executor=executor,
    )
    return preprocess_raw_waymo_views(corrupted, image)


def get_bev_features(model, image: torch.Tensor, image_metas: list[dict]):
    features = model.extract_feat(img=image)
    head = model.pts_bbox_head
    transformer = head.transformer
    batch_size = features[0].shape[0]
    dtype = features[0].dtype
    bev_queries = head.bev_embedding.weight.to(dtype)
    if head.volume_flag:
        bev_mask = torch.zeros(
            (batch_size, head.bev_z, head.bev_h, head.bev_w),
            device=bev_queries.device,
            dtype=dtype,
        )
    else:
        bev_mask = torch.zeros(
            (batch_size, head.bev_h, head.bev_w),
            device=bev_queries.device,
            dtype=dtype,
        )
    bev_pos = head.positional_encoding(bev_mask).to(dtype)
    outputs = transformer.get_bev_features(
        features,
        bev_queries,
        bev_pos=bev_pos,
        img_metas=image_metas,
        prev_bev=None,
    )
    return features, outputs


def split_paired_features(features, outputs: dict, index: int):
    split_features = [feature[index : index + 1] for feature in features]
    split_outputs = {
        "bev_embed": outputs["bev_embed"][index : index + 1],
        "feat_flatten": outputs["feat_flatten"][:, :, index : index + 1],
        "spatial_shapes": outputs["spatial_shapes"],
        "level_start_index": outputs["level_start_index"],
        "shift": (
            outputs["shift"][index : index + 1]
            if outputs["shift"].shape[0] > 1
            else outputs["shift"]
        ),
    }
    return split_features, split_outputs


def get_clean_and_corrupt_bev_features(
    model,
    clean_image: torch.Tensor,
    corrupt_image: torch.Tensor,
    image_metas: list[dict],
    low_memory: bool = False,
):
    if low_memory:
        clean_features, clean_outputs = get_bev_features(
            model,
            clean_image,
            image_metas,
        )
        corrupt_features, corrupt_outputs = get_bev_features(
            model,
            corrupt_image,
            image_metas,
        )
        return (
            clean_features,
            corrupt_features,
            clean_outputs,
            corrupt_outputs,
        )

    paired_image = torch.cat((clean_image, corrupt_image), dim=0)
    paired_metas = [image_metas[0], image_metas[0]]
    features, outputs = get_bev_features(model, paired_image, paired_metas)
    clean_features, clean_outputs = split_paired_features(features, outputs, 0)
    corrupt_features, corrupt_outputs = split_paired_features(features, outputs, 1)
    return clean_features, corrupt_features, clean_outputs, corrupt_outputs


def selected_history(
    pattern: str,
    clean_queue: deque,
    corrupt_queue: deque,
    meta_queue: deque,
    current_bev: torch.Tensor,
    current_meta: dict,
) -> tuple[list[torch.Tensor], list[dict[dict]]]:
    history_bevs = []
    history_metas = []
    for offset in HISTORY_OFFSETS:
        if len(meta_queue) < offset:
            continue
        use_corrupt = pattern == "history_only" or (
            pattern == "recent_burst" and offset == 5
        )
        queue = corrupt_queue if use_corrupt else clean_queue
        history_bevs.append(queue[-offset])
        history_metas.append(meta_queue[-offset])

    if not history_bevs:
        history_bevs = [torch.zeros_like(current_bev)]
        history_metas = [current_meta.copy()]

    first_key = len(HISTORY_OFFSETS) - len(history_metas)
    metadata = {
        first_key + index: meta
        for index, meta in enumerate(history_metas)
    }
    return history_bevs, [metadata]


def stack_cached_inputs(feature_sets, output_sets):
    features = [
        torch.cat([feature_set[level] for feature_set in feature_sets], dim=0)
        for level in range(len(feature_sets[0]))
    ]
    outputs = {
        "bev_embed": torch.cat(
            [output["bev_embed"] for output in output_sets], dim=0
        ),
        "feat_flatten": torch.cat(
            [output["feat_flatten"] for output in output_sets], dim=2
        ),
        "spatial_shapes": output_sets[0]["spatial_shapes"],
        "level_start_index": output_sets[0]["level_start_index"],
        "shift": torch.cat([output["shift"] for output in output_sets], dim=0),
    }
    return features, outputs


def evaluate_patterns(
    model,
    clean_features,
    corrupt_features,
    clean_outputs: dict,
    corrupt_outputs: dict,
    clean_queue: deque,
    corrupt_queue: deque,
    meta_queue: deque,
    image_metas: list[dict],
    voxel_semantics: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    patterns = list(PATTERN_CORRUPT_OFFSETS)
    feature_sets = []
    output_sets = []
    histories = []
    metadata = []
    for pattern in patterns:
        current_corrupt = pattern in {"current_only", "recent_burst"}
        features = corrupt_features if current_corrupt else clean_features
        outputs = corrupt_outputs if current_corrupt else clean_outputs
        previous_bevs, previous_metas = selected_history(
            pattern,
            clean_queue,
            corrupt_queue,
            meta_queue,
            outputs["bev_embed"],
            image_metas[0],
        )
        feature_sets.append(features)
        output_sets.append(outputs)
        histories.append(previous_bevs)
        metadata.append(previous_metas[0])

    history_length = len(histories[0])
    if any(len(history) != history_length for history in histories):
        raise RuntimeError("temporal patterns produced different history lengths")
    batched_history = [
        torch.cat(
            [histories[pattern_idx][time_idx] for pattern_idx in range(len(patterns))],
            dim=0,
        )
        for time_idx in range(history_length)
    ]
    batched_features, batched_outputs = stack_cached_inputs(
        feature_sets, output_sets
    )
    predictions = model.pts_bbox_head(
        batched_features,
        [image_metas[0] for _ in patterns],
        batched_history,
        metadata,
        only_bev=False,
        precomputed_bev_features=batched_outputs,
    )
    occupancy = model.pts_bbox_head.get_occ(predictions)
    count_matrices = {}
    for pattern_idx, pattern in enumerate(patterns):
        metrics = model.pts_bbox_head.eval_metrics(
            voxel_semantics,
            occupancy[pattern_idx : pattern_idx + 1],
            valid_mask,
        )
        count_matrices[pattern] = np.asarray(
            metrics["count_matrix"], dtype=np.int64
        )
    return count_matrices, predictions["bev_embed"], batched_outputs["bev_embed"]


def evaluate_patterns_sequential(
    model,
    clean_features,
    corrupt_features,
    clean_outputs: dict,
    corrupt_outputs: dict,
    clean_queue: deque,
    corrupt_queue: deque,
    meta_queue: deque,
    image_metas: list[dict],
    voxel_semantics: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    count_matrices = {}
    returned_bev = None
    expected_bev = None
    for pattern in PATTERN_CORRUPT_OFFSETS:
        current_corrupt = pattern in {"current_only", "recent_burst"}
        features = corrupt_features if current_corrupt else clean_features
        outputs = corrupt_outputs if current_corrupt else clean_outputs
        previous_bevs, previous_metas = selected_history(
            pattern,
            clean_queue,
            corrupt_queue,
            meta_queue,
            outputs["bev_embed"],
            image_metas[0],
        )
        predictions = model.pts_bbox_head(
            features,
            image_metas,
            previous_bevs,
            previous_metas,
            only_bev=False,
            precomputed_bev_features=outputs,
        )
        returned_bev = predictions["bev_embed"]
        expected_bev = outputs["bev_embed"]
        if returned_bev.data_ptr() != expected_bev.data_ptr():
            raise RuntimeError(
                "head did not preserve the cached current BEV batch"
            )
        occupancy = model.pts_bbox_head.get_occ(predictions)
        metrics = model.pts_bbox_head.eval_metrics(
            voxel_semantics,
            occupancy,
            valid_mask,
        )
        count_matrices[pattern] = np.asarray(
            metrics["count_matrix"],
            dtype=np.int64,
        )
    return count_matrices, returned_bev, expected_bev


def verify_cached_path(
    model,
    clean_features,
    clean_outputs: dict,
    image_metas: list[dict],
) -> float:
    zero_bev = torch.zeros_like(clean_outputs["bev_embed"])
    previous_metas = [{5: image_metas[0].copy()}]
    reference = model.pts_bbox_head(
        clean_features,
        image_metas,
        [zero_bev],
        previous_metas,
        only_bev=False,
    )
    cached = model.pts_bbox_head(
        clean_features,
        image_metas,
        [zero_bev],
        previous_metas,
        only_bev=False,
        precomputed_bev_features=clean_outputs,
    )
    return float(torch.max(torch.abs(reference["occ"] - cached["occ"])).item())


def prefetched_corruption_batches(
    data_loader,
    args: argparse.Namespace,
    executor: ProcessPoolExecutor,
    output: np.ndarray,
    output_path: str,
):
    data_iterator = iter(data_loader)
    pending = deque()

    def submit(data, output_index: int):
        image, image_metas, voxel_semantics, valid_mask = unpack_batch(data)
        filenames = image_metas[0].get("filename")
        if not isinstance(filenames, (list, tuple)) or len(filenames) != 5:
            raise ValueError("image metadata must contain five source filenames")
        sample_key = str(int(image_metas[0]["sample_idx"]))
        future = executor.submit(
            corrupt_waymo_frame_files_into_memmap,
            filenames,
            args.corruption,
            args.severity,
            sample_key,
            image.shape[-2],
            image.shape[-1],
            output_path,
            tuple(output.shape),
            output_index,
        )
        return (
            output_index,
            future,
            image,
            image_metas,
            voxel_semantics,
            valid_mask,
        )

    for output_index in range(args.corruption_prefetch):
        try:
            pending.append(submit(next(data_iterator), output_index))
        except StopIteration:
            break

    while pending:
        (
            output_index,
            future,
            image,
            image_metas,
            voxel_semantics,
            valid_mask,
        ) = pending.popleft()
        if future.result() != output_index:
            raise RuntimeError("corruption prefetch returned the wrong slot")
        corrupted_image = (
            torch.from_numpy(output[output_index])
            .unsqueeze(0)
            .to(device=image.device, dtype=image.dtype)
        )
        try:
            pending.append(submit(next(data_iterator), output_index))
        except StopIteration:
            pass
        yield (
            image,
            image_metas,
            voxel_semantics,
            valid_mask,
            corrupted_image,
        )


def immediate_corruption_batches(
    data_loader,
    args: argparse.Namespace,
    executor,
    output,
    output_path,
):
    for data in data_loader:
        image, image_metas, voxel_semantics, valid_mask = unpack_batch(data)
        corrupted_image = corrupt_image_tensor(
            image,
            image_metas,
            corruption=args.corruption,
            severity=args.severity,
            sample_key=str(int(image_metas[0]["sample_idx"])),
            executor=executor,
            parallel_output=output,
            parallel_output_path=output_path,
        )
        yield (
            image,
            image_metas,
            voxel_semantics,
            valid_mask,
            corrupted_image,
        )


def main() -> int:
    args = parse_args()
    sequential_bev = args.low_memory or args.sequential_bev
    sequential_patterns = args.low_memory or args.sequential_patterns
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.corruption_workers < 0:
        raise ValueError("--corruption-workers must be non-negative")
    if args.corruption_prefetch < 1:
        raise ValueError("--corruption-prefetch must be positive")
    if args.corruption_prefetch > 1 and (
        args.corruption_workers == 0
        or args.corruption not in SLOW_IMAGE_CORRUPTIONS
    ):
        raise ValueError(
            "corruption prefetch requires workers and a slow image corruption"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES before evaluation"
        )

    enable_mmcv_torch28_scatter_compat()
    set_random_seed(args.seed, deterministic=True)
    config = Config.fromfile(str(args.config))
    config.model.pretrained = None
    config.model.train_cfg = None
    config.data.test.test_mode = True

    import projects.mmdet3d_plugin  # noqa: F401

    dataset = build_dataset(config.data.test)
    indices = dataset_indices(dataset, args.scene_start, args.scene_end)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        indices = indices[: args.max_samples]
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=OrderedIndexSampler(indices),
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
    corruption_executor = (
        ProcessPoolExecutor(
            max_workers=args.corruption_workers,
            mp_context=get_context("spawn"),
        )
        if args.corruption_workers
        else None
    )
    corruption_output = None
    corruption_output_path = None
    if (
        corruption_executor is not None
        and args.corruption in SLOW_IMAGE_CORRUPTIONS
    ):
        corruption_output_path = str(
            Path("/dev/shm")
            / f"occstress-waymo-{os.getpid()}-corruption.dat"
        )
        corruption_output = np.memmap(
            corruption_output_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                (args.corruption_prefetch, 5, 3, 640, 960)
                if args.corruption_prefetch > 1
                else (5, 3, 640, 960)
            ),
        )

    histories = {
        "clean": deque(maxlen=30),
        "corrupt": deque(maxlen=30),
        "meta": deque(maxlen=30),
    }
    confusion = {
        pattern: np.zeros((16, 16), dtype=np.int64)
        for pattern in PATTERN_CORRUPT_OFFSETS
    }
    seen_scenes: set[int] = set()
    current_scene = None
    cache_equivalence_max_abs = None
    raw_preprocess_equivalence_max_abs = None
    start_time = time.monotonic()

    if args.corruption_prefetch > 1:
        batch_iterator = prefetched_corruption_batches(
            data_loader,
            args,
            corruption_executor,
            corruption_output,
            corruption_output_path,
        )
    else:
        batch_iterator = immediate_corruption_batches(
            data_loader,
            args,
            corruption_executor,
            corruption_output,
            corruption_output_path,
        )

    for completed, batch in enumerate(batch_iterator, start=1):
        (
            image,
            image_metas,
            voxel_semantics,
            valid_mask,
            corrupted_image,
        ) = batch
        if (
            args.verify_raw_preprocess
            and raw_preprocess_equivalence_max_abs is None
        ):
            reconstructed = preprocess_raw_waymo_views(
                load_raw_waymo_views(image_metas),
                image,
            )
            raw_preprocess_equivalence_max_abs = float(
                torch.max(torch.abs(image - reconstructed)).item()
            )
            if raw_preprocess_equivalence_max_abs > 0:
                raise RuntimeError(
                    "raw Waymo preprocessing differs from the official loader: "
                    f"{raw_preprocess_equivalence_max_abs}"
                )
        sample_idx = int(image_metas[0]["sample_idx"])
        scene_idx = sample_idx % 1_000_000 // 1_000
        if scene_idx != current_scene:
            for queue in histories.values():
                queue.clear()
            current_scene = scene_idx
        seen_scenes.add(scene_idx)

        with torch.inference_mode():
            (
                clean_features,
                corrupt_features,
                clean_outputs,
                corrupt_outputs,
            ) = get_clean_and_corrupt_bev_features(
                model,
                image,
                corrupted_image,
                image_metas,
                low_memory=sequential_bev,
            )
            if args.verify_cache and cache_equivalence_max_abs is None:
                cache_equivalence_max_abs = verify_cached_path(
                    model,
                    clean_features,
                    clean_outputs,
                    image_metas,
                )
                if cache_equivalence_max_abs > 1e-6:
                    raise RuntimeError(
                        "cached BEV path differs from direct path: "
                        f"{cache_equivalence_max_abs}"
                    )

            evaluate = (
                evaluate_patterns_sequential
                if sequential_patterns
                else evaluate_patterns
            )
            count_matrices, returned_bev, expected_bev = evaluate(
                model,
                clean_features,
                corrupt_features,
                clean_outputs,
                corrupt_outputs,
                histories["clean"],
                histories["corrupt"],
                histories["meta"],
                image_metas,
                voxel_semantics,
                valid_mask,
            )
            if returned_bev.data_ptr() != expected_bev.data_ptr():
                raise RuntimeError("head did not preserve the cached current BEV batch")
            for pattern, count_matrix in count_matrices.items():
                confusion[pattern] += count_matrix

        histories["clean"].append(clean_outputs["bev_embed"].detach())
        histories["corrupt"].append(corrupt_outputs["bev_embed"].detach())
        histories["meta"].append(image_metas[0].copy())

        if completed % args.log_interval == 0 or completed == len(indices):
            elapsed = time.monotonic() - start_time
            print(
                f"evaluated {completed}/{len(indices)} anchors "
                f"({completed / elapsed:.3f} anchors/s)",
                flush=True,
            )

    if corruption_executor is not None:
        corruption_executor.shutdown()
    if corruption_output is not None:
        del corruption_output
        Path(corruption_output_path).unlink(missing_ok=True)
    elapsed = time.monotonic() - start_time
    checkpoint_sha256 = sha256sum(args.checkpoint)
    common = {
        "anchor_count": len(indices),
        "cache_equivalence_max_abs": cache_equivalence_max_abs,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "code_commit": git_commit(args.config.parents[3]),
        "code_dirty": git_is_dirty(args.config.parents[3]),
        "config": str(args.config.resolve()),
        "config_sha256": sha256sum(args.config),
        "corruption": args.corruption,
        "corruption_prefetch": args.corruption_prefetch,
        "corruption_workers": args.corruption_workers,
        "corruption_source_sha256": sha256sum(
            Path(__file__).with_name("waymo_corruptions.py")
        ),
        "corruption_stage": "raw_bgr_before_waymo_padding_and_resize",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": elapsed,
        "evaluator_sha256": sha256sum(Path(__file__).resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "history_offsets_seconds": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5],
        "input_offsets_seconds": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
        "low_memory": args.low_memory,
        "peak_gpu_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
        ),
        "peak_gpu_memory_reserved_mib": (
            torch.cuda.max_memory_reserved() / (1024**2)
        ),
        "raw_preprocess_equivalence_max_abs": (
            raw_preprocess_equivalence_max_abs
        ),
        "sequential_bev": sequential_bev,
        "sequential_patterns": sequential_patterns,
        "samples_per_second": len(indices) / elapsed,
        "scene_end": args.scene_end,
        "scene_ids": sorted(seen_scenes),
        "scene_start": args.scene_start,
        "seed": args.seed,
        "severity": args.severity,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "ok",
    }
    if args.corruption == "CameraCrash":
        draw = camera_crash_draw(args.severity)
        unique = sorted(set(draw))
        common.update(
            {
                "camera_draw_indices": draw,
                "camera_unique_indices": unique,
                "camera_unique_names": [
                    WAYMO_CAMERA_NAMES[index] for index in unique
                ],
            }
        )
    for pattern, matrix in confusion.items():
        payload = {
            **common,
            "confusion_matrix": matrix.tolist(),
            "corrupt_offsets_frames": list(PATTERN_CORRUPT_OFFSETS[pattern]),
            "metrics": metrics_from_hist(matrix),
            "protocol_id": (
                f"robobev__{args.corruption}__{args.severity}__{pattern}"
            ),
            "temporal_pattern": pattern,
        }
        output = args.output_dir / (
            f"{payload['protocol_id']}__scenes"
            f"{args.scene_start:03d}-{args.scene_end:03d}.json"
        )
        atomic_json(output, payload)
        print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
