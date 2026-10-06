#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np


CLASS_NAMES = (
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate one canonical FusionOcc OccStress-nuScenes export."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--modality", choices=["clean", "image", "pointcloud"], required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument("--expected-count", type=int, default=6019)
    parser.add_argument("--done-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def resolve_gt_path(repo_root, info):
    path = Path(info["occ_path"])
    if not path.is_absolute():
        path = repo_root / path
    if path.suffix != ".npz":
        path = path / "labels.npz"
    return path.resolve()


def scene_name(info):
    path = Path(info["occ_path"])
    scene = path.parent.name
    return str(info.get("scene_name", scene)) if not scene.startswith("scene-") else scene


def metrics(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan),
        where=union > 0,
    )
    occupied_intersection = confusion[:-1, :-1].sum()
    occupied_union = (
        confusion[:-1, :].sum()
        + confusion[:, :-1].sum()
        - occupied_intersection
    )
    return {
        "per_class_iou": {
            name: None if np.isnan(value) else float(value * 100.0)
            for name, value in zip(CLASS_NAMES, iou)
        },
        "miou_non_free": float(np.nanmean(iou[:-1]) * 100.0),
        "occupied_iou": float(occupied_intersection / occupied_union * 100.0),
    }


def main():
    args = parse_args()
    with args.annotation.open("rb") as handle:
        annotation = pickle.load(handle)
    infos = annotation["infos"] if isinstance(annotation, dict) else annotation
    if len(infos) != args.expected_count:
        raise ValueError(
            f"annotation has {len(infos)} records, expected {args.expected_count}"
        )

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    expected_paths = set()
    errors = []
    for info in infos:
        scene = scene_name(info)
        token = str(info["token"])
        pred_path = args.output_root / scene / token / "labels.npz"
        expected_paths.add(pred_path.resolve())
        try:
            with np.load(pred_path) as payload:
                if payload.files != ["semantics"]:
                    raise ValueError(f"keys={payload.files}")
                prediction = payload["semantics"]
            if prediction.shape != (200, 200, 16):
                raise ValueError(f"shape={prediction.shape}")
            if prediction.dtype != np.uint8:
                raise ValueError(f"dtype={prediction.dtype}")
            if prediction.min() < 0 or prediction.max() > 17:
                raise ValueError(
                    f"range=[{prediction.min()}, {prediction.max()}]"
                )
            with np.load(resolve_gt_path(args.repo_root, info)) as gt:
                target = gt["semantics"]
                mask = gt["mask_camera"].astype(bool)
            valid = mask & (target < len(CLASS_NAMES))
            encoded = (
                target[valid].astype(np.int64) * len(CLASS_NAMES)
                + prediction[valid].astype(np.int64)
            )
            confusion += np.bincount(
                encoded, minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        except Exception as exc:
            errors.append({"path": str(pred_path), "error": str(exc)})
            if len(errors) >= 20:
                break

    actual_paths = {
        path.resolve() for path in args.output_root.glob("scene-*/*/labels.npz")
    }
    extras = sorted(str(path) for path in actual_paths - expected_paths)
    if errors or extras or len(actual_paths) != args.expected_count:
        raise RuntimeError(
            f"validation failed: count={len(actual_paths)}/{args.expected_count}, "
            f"errors={errors[:3]}, extras={extras[:3]}"
        )

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    payload = {
        "status": "success",
        "track": "upstream",
        "subtrack": "pointcloud_fusion",
        "source_model": "fusionocc",
        "corruption_modality": args.modality,
        "corruption": args.corruption,
        "severity": args.severity,
        "frame_count": len(actual_paths),
        "shape": [200, 200, 16],
        "dtype": "uint8",
        "class_names": list(CLASS_NAMES),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "code_commit": commit,
        "annotation": str(args.annotation.resolve()),
        "output_root": str(args.output_root.resolve()),
        "mask_policy": "raw_prediction; metrics use GT mask_camera",
        "metrics": metrics(confusion),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(args.done_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
