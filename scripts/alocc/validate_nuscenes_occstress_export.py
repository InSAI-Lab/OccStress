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
        description="Validate one canonical ALOcc OccStress-nuScenes export."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--alocc-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def resolve_gt_path(alocc_root, info):
    path = Path(info["occ_path"])
    if not path.is_absolute():
        path = alocc_root / path
    if path.suffix != ".npz":
        path = path / "labels.npz"
    return path.resolve()


def metrics_from_confusion(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = (
        confusion.sum(axis=0)
        + confusion.sum(axis=1)
        - intersection
    )
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
        "occupied_iou": float(
            occupied_intersection / occupied_union * 100.0
        ),
    }


def main():
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")

    with args.annotation.open("rb") as handle:
        annotation = pickle.load(handle)
    all_infos = (
        annotation["infos"] if isinstance(annotation, dict) else annotation
    )
    if len(all_infos) < args.expected_count:
        raise ValueError(
            f"annotation contains {len(all_infos)} records, "
            f"expected {args.expected_count}"
        )
    info_by_key = {
        (info["scene_name"], info["token"]): info for info in all_infos
    }

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    missing = []
    invalid = []
    actual_paths = {
        path.resolve()
        for path in args.output_root.glob("scene-*/*/labels.npz")
    }
    if len(actual_paths) != args.expected_count:
        raise RuntimeError(
            f"found {len(actual_paths)} predictions, "
            f"expected {args.expected_count}"
        )

    if args.expected_count == len(all_infos):
        infos = all_infos
    else:
        infos = []
        unknown = []
        for path in sorted(actual_paths):
            relative = path.relative_to(args.output_root.resolve())
            key = (relative.parts[0], relative.parts[1])
            info = info_by_key.get(key)
            if info is None:
                unknown.append(str(path))
            else:
                infos.append(info)
        if unknown:
            raise RuntimeError(
                f"{len(unknown)} exported tokens are absent from annotation: "
                f"{unknown[:3]}"
            )

    expected_paths = set()
    for info in infos:
        scene_name = info["scene_name"]
        sample_token = info["token"]
        pred_path = (
            args.output_root / scene_name / sample_token / "labels.npz"
        )
        expected_paths.add(pred_path.resolve())
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue

        try:
            with np.load(pred_path) as payload:
                if payload.files != ["semantics"]:
                    raise ValueError(
                        f"keys={payload.files}, expected ['semantics']"
                    )
                prediction = payload["semantics"]
            if prediction.shape != (200, 200, 16):
                raise ValueError(f"shape={prediction.shape}")
            if prediction.dtype != np.uint8:
                raise ValueError(f"dtype={prediction.dtype}")
            if prediction.min() < 0 or prediction.max() >= len(CLASS_NAMES):
                raise ValueError(
                    f"label range=[{prediction.min()}, {prediction.max()}]"
                )

            gt_path = resolve_gt_path(args.alocc_root, info)
            with np.load(gt_path) as gt_payload:
                target = gt_payload["semantics"]
                mask = gt_payload["mask_camera"].astype(bool)
            valid = mask & (target < len(CLASS_NAMES))
            encoded = (
                target[valid].astype(np.int64) * len(CLASS_NAMES)
                + prediction[valid].astype(np.int64)
            )
            confusion += np.bincount(
                encoded, minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        except Exception as exc:
            invalid.append({"path": str(pred_path), "error": str(exc)})

    extras = sorted(str(path) for path in actual_paths - expected_paths)
    if missing or invalid or extras:
        raise RuntimeError(
            "ALOcc export validation failed: "
            f"missing={len(missing)}, invalid={len(invalid)}, "
            f"extras={len(extras)}; "
            f"examples={missing[:2] + [x['path'] for x in invalid[:2]] + extras[:2]}"
        )

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(args.alocc_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    payload = {
        "status": "success",
        "source_model": "alocc",
        "track": "upstream",
        "subtrack": "camera_only",
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
        "output_root": str(args.output_root.resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "metrics": metrics_from_confusion(confusion),
    }
    atomic_json(args.done_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
