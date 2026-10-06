#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path

import mmcv


IMAGE_CORRUPTIONS = {
    "Brightness",
    "CameraCrash",
    "ColorQuant",
    "Fog",
    "FrameLost",
    "LowLight",
    "MotionBlur",
    "Snow",
}
IMAGE_SEVERITIES = {"easy", "mid", "hard"}

POINTCLOUD_CORRUPTIONS = {
    "beam_missing",
    "cross_sensor",
    "crosstalk",
    "fog",
    "incomplete_echo",
    "motion_blur",
    "snow",
    "wet_ground",
}
POINTCLOUD_SEVERITIES = {"light", "moderate", "heavy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FusionOcc nuScenes-C annotation file by rewriting clean validation paths."
    )
    parser.add_argument("--src", required=True, help="Source clean FusionOcc ann file.")
    parser.add_argument("--dst", required=True, help="Destination rewritten ann file.")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "pointcloud"],
        help="Which modality to corrupt in the rewritten ann file.",
    )
    parser.add_argument("--corruption", required=True, help="nuScenes-C corruption family.")
    parser.add_argument("--severity", required=True, help="nuScenes-C severity level.")
    parser.add_argument(
        "--image-root",
        default="./data/nuScenes-C/raw/image/nuScenes-c",
        help="Root of corrupted nuScenes-C image data.",
    )
    parser.add_argument(
        "--pointcloud-root",
        default="./data/nuScenes-C/raw/pointcloud/nuScenes-C",
        help="Root of corrupted nuScenes-C pointcloud data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of infos to keep for smoke tests. 0 means all.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip existence checks for rewritten target files.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.modality == "image":
        if args.corruption not in IMAGE_CORRUPTIONS:
            raise ValueError(f"Unknown image corruption: {args.corruption}")
        if args.severity not in IMAGE_SEVERITIES:
            raise ValueError(f"Unknown image severity: {args.severity}")
    else:
        if args.corruption not in POINTCLOUD_CORRUPTIONS:
            raise ValueError(f"Unknown pointcloud corruption: {args.corruption}")
        if args.severity not in POINTCLOUD_SEVERITIES:
            raise ValueError(f"Unknown pointcloud severity: {args.severity}")


def rewrite_image_info(info, corruption, severity, image_root):
    rewritten = copy.deepcopy(info)
    for cam_name, cam_info in rewritten["cams"].items():
        filename = Path(cam_info["data_path"]).name
        cam_info["data_path"] = str(image_root / corruption / severity / cam_name / filename)
    return rewritten


def rewrite_pointcloud_info(info, corruption, severity, pointcloud_root):
    rewritten = copy.deepcopy(info)
    filename = Path(rewritten["lidar_path"]).name
    rewritten["lidar_path"] = str(
        pointcloud_root / corruption / severity / "samples" / "LIDAR_TOP" / filename
    )
    return rewritten


def verify_paths(infos, modality):
    missing = []
    if modality == "image":
        for info in infos:
            for cam_info in info["cams"].values():
                path = Path(cam_info["data_path"])
                if not path.exists():
                    missing.append(str(path))
    else:
        for info in infos:
            path = Path(info["lidar_path"])
            if not path.exists():
                missing.append(str(path))
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} rewritten files. First examples:\n{sample}")


def main():
    args = parse_args()
    validate_args(args)

    src = Path(args.src)
    dst = Path(args.dst)
    image_root = Path(args.image_root)
    pointcloud_root = Path(args.pointcloud_root)

    obj = mmcv.load(str(src))
    if not isinstance(obj, dict) or "infos" not in obj:
        raise TypeError(f"Unexpected ann file structure in {src}")

    src_infos = obj["infos"]
    limit = args.limit if args.limit > 0 else len(src_infos)
    rewrite_fn = (
        lambda info: rewrite_image_info(info, args.corruption, args.severity, image_root)
        if args.modality == "image"
        else lambda info: rewrite_pointcloud_info(info, args.corruption, args.severity, pointcloud_root)
    )
    if args.modality == "image":
        rewritten_infos = [
            rewrite_image_info(info, args.corruption, args.severity, image_root)
            for info in src_infos[:limit]
        ]
    else:
        rewritten_infos = [
            rewrite_pointcloud_info(info, args.corruption, args.severity, pointcloud_root)
            for info in src_infos[:limit]
        ]

    if not args.skip_check:
        verify_paths(rewritten_infos, args.modality)

    out = dict(obj)
    out["infos"] = rewritten_infos
    meta = dict(out.get("metadata", {}))
    meta["robustocc_nuscenes_c"] = {
        "modality": args.modality,
        "corruption": args.corruption,
        "severity": args.severity,
        "source_ann": str(src),
        "sample_count": len(rewritten_infos),
    }
    out["metadata"] = meta

    dst.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(out, str(dst))

    print(f"source: {src}")
    print(f"output: {dst}")
    print(f"modality: {args.modality}")
    print(f"setting: {args.corruption}/{args.severity}")
    print(f"infos: {len(rewritten_infos)}")


if __name__ == "__main__":
    main()
