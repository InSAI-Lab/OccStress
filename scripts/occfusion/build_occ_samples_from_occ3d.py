#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ann_file, "rb") as f:
        data = pickle.load(f)

    data_list = data["data_list"]
    built = 0
    skipped = 0
    for info in data_list:
        token = info["token"]
        lidar_name = Path(info["lidar_points"]["lidar_path"]).name
        out_path = out_dir / f"{lidar_name}.npy"
        if out_path.exists():
            skipped += 1
            continue

        occ_path = data_root / "gts" / token[0:0]  # keep Path type initialized
        if "occ_path" in info and info["occ_path"]:
            occ_path = Path(info["occ_path"])
            if not occ_path.is_absolute():
                occ_path = (Path.cwd() / occ_path).resolve()
        else:
            scene_token = info.get("scene_token")
            if scene_token is None:
                raise KeyError(f"Missing occ_path and scene_token for token {token}")
            occ_path = data_root / "gts" / scene_token / token

        if occ_path.is_dir():
            occ_file = occ_path / "labels.npz"
        else:
            occ_file = occ_path

        occ = np.load(occ_file)
        semantics = occ["semantics"].astype(np.uint8)

        # Build official-style sparse occupancy labels:
        # 0 is empty, 1..16 are semantic classes, and 17 ("others" in Occ3D)
        # is dropped because official OccFusion config uses 16 semantic classes.
        coords = np.argwhere((semantics > 0) & (semantics < 17))
        labels = semantics[coords[:, 0], coords[:, 1], coords[:, 2]]
        sparse = np.concatenate([coords.astype(np.int16), labels[:, None].astype(np.int16)], axis=1)
        np.save(out_path, sparse)
        built += 1

    print(f"built={built} skipped={skipped} out_dir={out_dir}")


if __name__ == "__main__":
    main()
