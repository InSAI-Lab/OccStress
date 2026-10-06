#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import mmcv
import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2] / "EXIST" / "3D" / "SDGOCC"
    default_data_root = Path(os.environ.get("SDGOCC_DATA_ROOT", repo_root / "data" / "nuscenes"))
    default_mapping = repo_root / "projects" / "nuscenes.yaml"

    parser = argparse.ArgumentParser(
        description="Generate SDGOCC point_label/*.npy from nuScenes lidarseg labels."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="nuScenes data root, e.g. /path/to/data/nuscenes",
    )
    parser.add_argument(
        "--version",
        default=os.environ.get("SDGOCC_VERSION", "v1.0-trainval"),
        help="nuScenes version to use",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=default_mapping,
        help="Path to nuscenes.yaml with learning_map",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated labels; defaults to <data-root>/point_label",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Only process the first N samples; 0 means all samples",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose NuScenes logging",
    )
    return parser.parse_args()


def build_lookup_table(learning_map: dict) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for src_label, dst_label in learning_map.items():
        lut[int(src_label)] = int(dst_label)
    return lut


def ensure_prereqs(data_root: Path, version: str, mapping_file: Path) -> None:
    missing = []
    for path in (
        data_root / "samples" / "LIDAR_TOP",
        data_root / version,
        data_root / version / "sample_data.json",
        data_root / version / "lidarseg.json",
        data_root / "lidarseg" / version,
        mapping_file,
    ):
        if not path.exists():
            missing.append(path)

    if missing:
        print("missing prerequisites for point_label generation:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    mapping_file = args.mapping_file.resolve()
    out_dir = (args.out_dir or (data_root / "point_label")).resolve()

    ensure_prereqs(data_root, args.version, mapping_file)

    with mapping_file.open("r", encoding="utf-8") as handle:
        mapping_cfg = yaml.safe_load(handle)
    learning_map = mapping_cfg["learning_map"]
    lut = build_lookup_table(learning_map)

    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=args.verbose)
    samples = nusc.sample[: args.max_samples] if args.max_samples > 0 else nusc.sample

    written = 0
    skipped = 0

    print(f"data_root={data_root}")
    print(f"version={args.version}")
    print(f"mapping_file={mapping_file}")
    print(f"out_dir={out_dir}")
    print(f"samples={len(samples)}")

    for sample in mmcv.track_iter_progress(samples):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", lidar_token)
        lidarseg = nusc.get("lidarseg", lidar_token)

        sample_rel = Path(sample_data["filename"])
        try:
            sample_rel = sample_rel.relative_to("samples")
        except ValueError as exc:
            raise RuntimeError(f"unexpected sample_data filename: {sample_data['filename']}") from exc

        output_path = out_dir / sample_rel.parent / sample_rel.name.replace(".pcd.bin", ".npy")
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        lidarseg_path = data_root / lidarseg["filename"]
        raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
        mapped_labels = lut[raw_labels]

        mmcv.mkdir_or_exist(output_path.parent)
        np.save(output_path, mapped_labels)
        written += 1

    expected = len(samples)
    actual = sum(1 for _ in out_dir.rglob("*.npy"))
    print(f"written={written}")
    print(f"skipped={skipped}")
    print(f"total_point_labels={actual}")
    print(f"expected_for_requested_split={expected}")


if __name__ == "__main__":
    main()
