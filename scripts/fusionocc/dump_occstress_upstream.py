#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import json
import os
import tempfile
import time
from functools import partial
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torch.utils.data import DataLoader, Sampler

import mmdet
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

if mmdet.__version__ > "2.23.0":
    from mmdet.utils import compat_cfg, setup_multi_processes
else:
    from mmdet3d.utils import compat_cfg, setup_multi_processes


EXPECTED_SHAPE = (200, 200, 16)


class IndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream FusionOcc predictions into canonical OccStress-nuScenes labels.npz files."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--existing-root",
        type=Path,
        help="Optional persistent root checked before inference when output-root is node-local staging.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=["none", "camera", "lidar"],
        default="none",
        help="Optionally set predictions outside the selected GT visibility mask to free.",
    )
    return parser.parse_args()


def atomic_npz(path, semantics):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, semantics=semantics)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def valid_prediction(path):
    if not path.is_file():
        return False
    try:
        with np.load(path) as payload:
            return (
                payload.files == ["semantics"]
                and payload["semantics"].shape == EXPECTED_SHAPE
                and payload["semantics"].dtype == np.uint8
            )
    except Exception:
        return False


def resolve_gt_path(repo_root, info):
    path = Path(info["occ_path"])
    if not path.is_absolute():
        path = repo_root / path
    if path.suffix != ".npz":
        path = path / "labels.npz"
    return path.resolve()


def scene_and_token(info):
    token = str(info["token"])
    occ_path = Path(info["occ_path"])
    scene = occ_path.parent.name
    if not scene.startswith("scene-"):
        scene = str(info.get("scene_name", scene))
    if not scene.startswith("scene-"):
        raise ValueError(f"Cannot resolve scene name for token {token}: {info['occ_path']}")
    return scene, token


def shard_indices(dataset, num_shards, shard_index):
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    total = len(dataset)
    indices = np.arange(shard_index, total, num_shards, dtype=np.int64)
    return total, indices


def build_runtime(args):
    cfg = Config.fromfile(str(args.config))
    cfg.model.img_view_transformer.is_train = False
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = str(args.ann_file.resolve())

    dataset = build_dataset(cfg.data.test)
    source_count, indices = shard_indices(
        dataset, args.num_shards, args.shard_index
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=IndexSampler(indices),
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=1),
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16")
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model, dataset, data_loader, source_count, indices


def main():
    args = parse_args()
    started = time.time()
    repo_root = Path.cwd().resolve()
    args.output_root = args.output_root.resolve()
    args.manifest = args.manifest.resolve()

    model, dataset, loader, source_count, source_indices = build_runtime(args)
    written = 0
    skipped = 0
    processed = 0

    for local_index, data in enumerate(loader):
        source_index = int(source_indices[local_index])
        info = dataset.data_infos[source_index]
        scene, token = scene_and_token(info)
        output_path = args.output_root / scene / token / "labels.npz"
        existing_path = (
            args.existing_root.resolve() / scene / token / "labels.npz"
            if args.existing_root
            else output_path
        )
        if not args.overwrite and valid_prediction(existing_path):
            skipped += 1
        else:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            if len(result) != 1:
                raise RuntimeError(f"Expected one prediction, got {len(result)}")
            semantics = np.asarray(result[0])
            if semantics.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Unexpected prediction shape for {scene}/{token}: {semantics.shape}"
                )
            semantics = semantics.astype(np.uint8, copy=False)
            if semantics.min() < 0 or semantics.max() > 17:
                raise ValueError(
                    f"Invalid label range for {scene}/{token}: "
                    f"[{semantics.min()}, {semantics.max()}]"
                )
            if args.mask_mode != "none":
                with np.load(resolve_gt_path(repo_root, info)) as gt:
                    mask = gt[f"mask_{args.mask_mode}"].astype(bool)
                semantics = semantics.copy()
                semantics[~mask] = 17
            atomic_npz(output_path, semantics)
            written += 1
        processed += 1
        if processed % args.progress_interval == 0 or processed == len(source_indices):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"shard={args.shard_index}/{args.num_shards} "
                f"processed={processed}/{len(source_indices)} written={written} "
                f"skipped={skipped} rate={processed / elapsed:.2f} frame/s",
                flush=True,
            )

    payload = {
        "status": "success",
        "source_model": "fusionocc",
        "source_annotation": str(args.ann_file.resolve()),
        "output_root": str(args.output_root),
        "existing_root": str(args.existing_root.resolve()) if args.existing_root else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "mask_mode": args.mask_mode,
        "source_frame_count": source_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "source_index_first": int(source_indices[0]) if len(source_indices) else None,
        "source_index_last": int(source_indices[-1]) if len(source_indices) else None,
        "expected_shard_frames": len(source_indices),
        "processed_frames": processed,
        "written_frames": written,
        "skipped_frames": skipped,
        "elapsed_seconds": time.time() - started,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_json(args.manifest, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
