#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


POINTCLOUD_CORRUPTIONS = [
    "beam_missing",
    "cross_sensor",
    "crosstalk",
    "fog",
    "incomplete_echo",
    "motion_blur",
    "snow",
    "wet_ground",
]
POINTCLOUD_SEVERITIES = ["light", "moderate", "heavy"]
FREE_LABEL = 17


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert saved SDGOCC predictions into OccStress upstream labels.npz layout."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-model", default="sdgocc")
    parser.add_argument("--subtrack", default="pointcloud_fusion")
    parser.add_argument("--results-root", default="EXIST/3D/SDGOCC/work_dir/r50_fuse/occstress_upstream")
    parser.add_argument("--ann-cache-root", default="data/nuscenes/sdgocc-nusc-c")
    parser.add_argument("--clean-ann", default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument("--clean-results", default="clean/results.pkl")
    parser.add_argument(
        "--upstream-root",
        help=(
            "Output root for converted labels. Defaults to "
            "<root>/data/OccStress/occ/upstream/<subtrack>/<source-model>."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=["camera", "lidar", "none"],
        default="camera",
        help="Set voxels outside the selected official visibility mask to the free label.",
    )
    parser.add_argument(
        "--mask-root",
        default="data/nuscenes/gts",
        help="Full nuScenes occupancy GT root containing scene/token/labels.npz masks.",
    )
    parser.add_argument(
        "--preserve-ann-order",
        action="store_true",
        help=(
            "Use raw annotation order. By default, match MMDetection3D "
            "NuScenesDataset order by sorting infos by timestamp before zipping "
            "with results.pkl."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_infos(path, *, sort_by_timestamp):
    with path.open("rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    if sort_by_timestamp:
        missing = [idx for idx, info in enumerate(infos) if "timestamp" not in info]
        if missing:
            raise KeyError(f"{path} has infos without timestamp, first missing index={missing[0]}")
        infos = sorted(infos, key=lambda info: info["timestamp"])
    return infos


def load_results(path):
    with path.open("rb") as f:
        return pickle.load(f)


def extract_pred(result):
    if isinstance(result, dict):
        for key in ("pred_occ", "occ_pred", "pred", "semantics"):
            if key in result:
                result = result[key]
                break
        else:
            raise KeyError(f"Cannot find occupancy prediction key in result dict keys={list(result)}")
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    arr = np.asarray(result)
    if arr.shape != (200, 200, 16):
        raise ValueError(f"Unexpected prediction shape {arr.shape}; expected (200, 200, 16)")
    return arr.astype(np.uint8, copy=False)


def resolve_label_path(info, root, mask_root):
    scene_name = info.get("scene_name")
    token = info.get("token")
    candidates = []
    if mask_root is not None and scene_name and token:
        candidates.append(mask_root / scene_name / token / "labels.npz")
    occ_path = info.get("occ_path")
    if occ_path:
        occ_path = Path(occ_path)
        if not occ_path.is_absolute():
            occ_path = root / occ_path
        candidates.append(occ_path if occ_path.suffix == ".npz" else occ_path / "labels.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find occupancy labels for scene={scene_name} token={token}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def load_valid_mask(info, root, mask_root, mask_mode):
    if mask_mode == "none":
        return None
    label_path = resolve_label_path(info, root, mask_root)
    mask_key = "mask_camera" if mask_mode == "camera" else "mask_lidar"
    with np.load(label_path) as labels:
        if mask_key not in labels:
            raise KeyError(f"{label_path} does not contain {mask_key}")
        return labels[mask_key].astype(bool)


def apply_valid_mask(pred, valid_mask):
    if valid_mask is None:
        return pred
    if valid_mask.shape != pred.shape:
        raise ValueError(f"Mask shape {valid_mask.shape} does not match prediction shape {pred.shape}")
    pred = pred.copy()
    pred[~valid_mask] = FREE_LABEL
    return pred


def convert_one(results_path, ann_path, out_root, overwrite, root, mask_root, mask_mode, sort_by_timestamp):
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)

    infos = load_infos(ann_path, sort_by_timestamp=sort_by_timestamp)
    results = load_results(results_path)
    if len(infos) != len(results):
        raise ValueError(f"Length mismatch: {ann_path} has {len(infos)} infos, {results_path} has {len(results)} results")

    written = 0
    skipped = 0
    for info, result in zip(infos, results):
        scene_name = info.get("scene_name")
        token = info.get("token")
        if not scene_name or not token:
            raise KeyError(f"Missing scene_name/token in ann info keys={list(info)}")
        out_path = out_root / scene_name / token / "labels.npz"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred = extract_pred(result)
        pred = apply_valid_mask(pred, load_valid_mask(info, root, mask_root, mask_mode))
        np.savez_compressed(out_path, semantics=pred)
        written += 1
    return {
        "results": str(results_path),
        "ann": str(ann_path),
        "out_root": str(out_root),
        "ann_order": "timestamp_sorted" if sort_by_timestamp else "raw",
        "mask_mode": mask_mode,
        "mask_root": str(mask_root) if mask_root is not None else None,
        "written": written,
        "skipped": skipped,
    }


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    results_root = (root / args.results_root).resolve()
    ann_cache_root = (root / args.ann_cache_root).resolve()
    mask_root = None
    if args.mask_mode != "none":
        mask_root = Path(args.mask_root)
        if not mask_root.is_absolute():
            mask_root = root / mask_root
        mask_root = mask_root.resolve()
    if args.upstream_root:
        upstream_root = Path(args.upstream_root)
        if not upstream_root.is_absolute():
            upstream_root = root / upstream_root
        upstream_root = upstream_root.resolve()
    else:
        upstream_root = root / "data" / "OccStress" / "occ" / "upstream" / args.subtrack / args.source_model
    sort_by_timestamp = not args.preserve_ann_order

    summaries = []
    summaries.append(
        convert_one(
            results_root / args.clean_results,
            root / args.clean_ann,
            upstream_root / "clean",
            args.overwrite,
            root,
            mask_root,
            args.mask_mode,
            sort_by_timestamp,
        )
    )

    for corruption in POINTCLOUD_CORRUPTIONS:
        for severity in POINTCLOUD_SEVERITIES:
            summaries.append(
                convert_one(
                    results_root / "pointcloud" / corruption / severity / "results.pkl",
                    ann_cache_root / "pointcloud" / corruption / severity / "bevdetv2-nuscenes_infos_val.pkl",
                    upstream_root / corruption / severity,
                    args.overwrite,
                    root,
                    mask_root,
                    args.mask_mode,
                    sort_by_timestamp,
                )
            )

    summary_path = upstream_root / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(
            {
                "count": len(summaries),
                "ann_order": "timestamp_sorted" if sort_by_timestamp else "raw",
                "mask_mode": args.mask_mode,
                "mask_root": str(mask_root) if mask_root is not None else None,
                "items": summaries,
            },
            f,
            indent=2,
        )
    print(json.dumps({"summary": str(summary_path), "count": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
