#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


CLASS_NAMES = [
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
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Directly verify SDGOcc results.pkl against nuScenes occupancy GT "
            "using the same timestamp-sorted dataset order and camera-mask mIoU "
            "as SDGOcc self evaluation."
        )
    )
    parser.add_argument("--results", required=True, help="Path to SDGOcc results.pkl")
    parser.add_argument("--ann", required=True, help="Path to bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument(
        "--gt-root",
        default="data/nuscenes/gts",
        help="GT root laid out as <scene>/<token>/labels.npz",
    )
    parser.add_argument(
        "--order",
        choices=["timestamp", "raw"],
        default="timestamp",
        help="results.pkl follows MMDetection3D dataset order, which is timestamp-sorted for NuScenes.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N ordered samples.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every Nth ordered sample. Use with --limit for low-I/O probes.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["camera", "lidar", "none"],
        default="camera",
        help="Visibility mask for the mIoU sanity check.",
    )
    parser.add_argument("--json-out", help="Optional path to write the sanity summary JSON.")
    return parser.parse_args()


def load_infos(path, order):
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    if order == "timestamp":
        missing = [idx for idx, info in enumerate(infos) if "timestamp" not in info]
        if missing:
            raise KeyError(f"{path} has infos without timestamp, first missing index={missing[0]}")
        infos = sorted(infos, key=lambda info: info["timestamp"])
    return infos


def load_results(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def scene_from_info(info):
    if info.get("scene_name"):
        return info["scene_name"]
    occ_path = info.get("occ_path", "")
    for part in occ_path.split("/"):
        if part.startswith("scene-"):
            return part
    raise KeyError(f"Cannot infer scene_name from info keys={list(info)}")


def hist_info(pred, gt, n_classes):
    keep = (gt >= 0) & (gt < n_classes)
    return np.bincount(
        n_classes * gt[keep].astype(np.int64) + pred[keep].astype(np.int64),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)


def per_class_iou(hist):
    return np.nan_to_num(np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)))


def main():
    args = parse_args()
    results_path = Path(args.results)
    ann_path = Path(args.ann)
    gt_root = Path(args.gt_root)
    infos = load_infos(ann_path, args.order)
    results = load_results(results_path)
    if len(infos) != len(results):
        raise ValueError(f"Length mismatch: {len(infos)} infos vs {len(results)} results")

    indices = list(range(0, len(results), args.stride))
    if args.limit > 0:
        indices = indices[: args.limit]

    n_classes = len(CLASS_NAMES)
    hist = np.zeros((n_classes, n_classes), dtype=np.int64)
    samples = []
    mask_key = {"camera": "mask_camera", "lidar": "mask_lidar", "none": None}[args.mask_mode]

    for result_idx in indices:
        info = infos[result_idx]
        scene = scene_from_info(info)
        token = info["token"]
        pred = np.asarray(results[result_idx])
        if pred.shape != (200, 200, 16):
            raise ValueError(f"Unexpected prediction shape at index {result_idx}: {pred.shape}")
        label_path = gt_root / scene / token / "labels.npz"
        with np.load(label_path) as labels:
            gt = labels["semantics"]
            if mask_key is None:
                mask = np.ones(gt.shape, dtype=bool)
            else:
                mask = labels[mask_key].astype(bool)
        if gt.shape != pred.shape or mask.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch at index {result_idx}: pred={pred.shape} gt={gt.shape} mask={mask.shape}"
            )
        sample_hist = hist_info(pred[mask], gt[mask], n_classes)
        hist += sample_hist
        sample_iou = per_class_iou(sample_hist)
        samples.append(
            {
                "index": result_idx,
                "scene": scene,
                "token": token,
                "sample_miou_0_16": round(float(sample_iou[:17].mean() * 100), 4),
            }
        )

    iou = per_class_iou(hist)
    summary = {
        "results": str(results_path),
        "ann": str(ann_path),
        "gt_root": str(gt_root),
        "order": args.order,
        "mask_mode": args.mask_mode,
        "sample_count": len(indices),
        "miou_0_16": round(float(iou[:17].mean() * 100), 4),
        "per_class_iou_0_16": {
            name: round(float(value * 100), 4) for name, value in zip(CLASS_NAMES[:17], iou[:17])
        },
        "sample_probe": samples[:10],
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
