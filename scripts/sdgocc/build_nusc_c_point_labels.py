#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    repo = root / "EXIST" / "3D" / "SDGOCC"
    parser = argparse.ArgumentParser(
        description="Generate SDGOCC point_label/*.npy for one nuScenes-C pointcloud setting."
    )
    parser.add_argument(
        "--corruption-root",
        type=Path,
        required=True,
        help="Setting root like .../raw/pointcloud/nuScenes-C/cross_sensor/heavy",
    )
    parser.add_argument(
        "--clean-data-root",
        type=Path,
        default=root / "data" / "nuscenes",
        help="Clean nuScenes root that contains v1.0-trainval metadata JSONs.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=repo / "projects" / "nuscenes.yaml",
        help="Path to SDGOCC nuscenes.yaml with learning_map.",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_lookup_table(mapping_file: Path) -> np.ndarray:
    with mapping_file.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    lut = np.zeros(256, dtype=np.uint8)
    for src_label, dst_label in cfg["learning_map"].items():
        lut[int(src_label)] = int(dst_label)
    return lut


def build_lidarseg_map(clean_data_root: Path, version: str) -> dict:
    version_root = clean_data_root / version
    sample_data_path = version_root / "sample_data.json"
    lidarseg_path = version_root / "lidarseg.json"
    if not sample_data_path.exists():
        raise FileNotFoundError(sample_data_path)
    if not lidarseg_path.exists():
        raise FileNotFoundError(lidarseg_path)

    with sample_data_path.open("r", encoding="utf-8") as handle:
        sample_data = json.load(handle)
    with lidarseg_path.open("r", encoding="utf-8") as handle:
        lidarseg = json.load(handle)

    token_to_filename = {row["token"]: Path(row["filename"]).name for row in sample_data}
    filename_to_lidarseg = {}
    for row in lidarseg:
        sample_name = token_to_filename.get(row["sample_data_token"])
        if sample_name is None:
            continue
        filename_to_lidarseg[sample_name] = Path(row["filename"]).name
    return filename_to_lidarseg


def main() -> None:
    args = parse_args()
    corruption_root = args.corruption_root.resolve()
    samples_root = corruption_root / "samples" / "LIDAR_TOP"
    lidarseg_root = corruption_root / "lidarseg" / args.version
    out_root = corruption_root / "point_label" / "LIDAR_TOP"

    if not samples_root.exists():
        raise FileNotFoundError(samples_root)
    if not lidarseg_root.exists():
        raise FileNotFoundError(lidarseg_root)

    lut = build_lookup_table(args.mapping_file.resolve())
    filename_to_lidarseg = build_lidarseg_map(args.clean_data_root.resolve(), args.version)

    sample_files = sorted(samples_root.glob("*.pcd.bin"))
    if args.limit > 0:
        sample_files = sample_files[: args.limit]

    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for sample_path in sample_files:
        out_path = out_root / sample_path.name.replace(".pcd.bin", ".npy")
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        lidarseg_name = filename_to_lidarseg.get(sample_path.name)
        if lidarseg_name is None:
            raise KeyError(f"Missing lidarseg mapping for {sample_path.name}")
        lidarseg_path = lidarseg_root / lidarseg_name
        if not lidarseg_path.exists():
            raise FileNotFoundError(lidarseg_path)

        raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
        mapped_labels = lut[raw_labels]
        np.save(out_path, mapped_labels)
        written += 1

    total_outputs = len(list(out_root.glob("*.npy")))
    print(f"corruption_root={corruption_root}")
    print(f"samples={len(sample_files)}")
    print(f"written={written}")
    print(f"skipped={skipped}")
    print(f"total_point_labels={total_outputs}")


if __name__ == "__main__":
    main()
