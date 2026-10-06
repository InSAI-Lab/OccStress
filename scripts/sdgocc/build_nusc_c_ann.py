#!/usr/bin/env python3
import argparse
import copy
import pickle
from pathlib import Path


IMAGE_CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Rewrite SDGOCC nuScenes val annotations to nuScenes-C image/pointcloud paths."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=root / "data" / "nuscenes" / "bevdetv2-nuscenes_infos_val.pkl",
    )
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--modality", choices=("image", "pointcloud"), required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root / "data" / "nuScenes-C" / "raw" / "image" / "nuScenes-c",
    )
    parser.add_argument(
        "--pointcloud-root",
        type=Path,
        default=root / "data" / "nuScenes-C" / "raw" / "pointcloud" / "nuScenes-C",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def load_infos(src: Path):
    with src.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict):
        if "infos" in obj:
            return obj, obj["infos"]
        if "data_list" in obj:
            return obj, obj["data_list"]
    if isinstance(obj, list):
        return obj, obj
    raise TypeError(f"Unsupported annotation payload type: {type(obj)!r}")


def rewrite_image_info(info: dict, image_root: Path, corruption: str, severity: str, skip_check: bool):
    for cam_name in IMAGE_CAMERAS:
        cam = info["cams"][cam_name]
        file_name = Path(cam["data_path"]).name
        new_path = image_root / corruption / severity / cam_name / file_name
        if not skip_check and not new_path.exists():
            raise FileNotFoundError(new_path)
        cam["data_path"] = str(new_path)


def rewrite_pointcloud_info(
    info: dict, pointcloud_root: Path, corruption: str, severity: str, skip_check: bool
):
    file_name = Path(info["lidar_path"]).name
    new_path = pointcloud_root / corruption / severity / "samples" / "LIDAR_TOP" / file_name
    if not skip_check and not new_path.exists():
        raise FileNotFoundError(new_path)
    info["lidar_path"] = str(new_path)


def main() -> None:
    args = parse_args()
    payload, infos = load_infos(args.src.resolve())
    limit = args.limit if args.limit > 0 else len(infos)
    rewritten = []

    for info in infos[:limit]:
        new_info = copy.deepcopy(info)
        if args.modality == "image":
            rewrite_image_info(
                new_info,
                args.image_root.resolve(),
                args.corruption,
                args.severity,
                args.skip_check,
            )
        else:
            rewrite_pointcloud_info(
                new_info,
                args.pointcloud_root.resolve(),
                args.corruption,
                args.severity,
                args.skip_check,
            )
        rewritten.append(new_info)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        if "infos" in payload:
            payload["infos"] = rewritten
        else:
            payload["data_list"] = rewritten
        out_obj = payload
    else:
        out_obj = rewritten

    with args.dst.open("wb") as handle:
        pickle.dump(out_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"source: {args.src.resolve()}")
    print(f"output: {args.dst.resolve()}")
    print(f"modality: {args.modality}")
    print(f"setting: {args.corruption}/{args.severity}")
    print(f"infos: {len(rewritten)}")


if __name__ == "__main__":
    main()
