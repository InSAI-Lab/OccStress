#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


def default_root() -> Path:
    env_root = os.environ.get("OCCSTRESS_CODE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def build_scene_infos(gts_root: Path):
    scene_infos = {}
    sample_count = 0

    proc = subprocess.run(
        ["find", "-L", str(gts_root), "-type", "f", "-name", "labels.npz"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in proc.stdout.splitlines() if line]

    for path_str in sorted(paths):
        labels_path = Path(path_str)
        rel = labels_path.relative_to(gts_root)
        if len(rel.parts) != 3:
            continue
        scene_name, sample_token, file_name = rel.parts
        if file_name != "labels.npz":
            continue
        samples = scene_infos.setdefault(scene_name, {})
        rel_path = Path("gts") / scene_name / sample_token / "labels.npz"
        samples[sample_token] = {"gt_path": rel_path.as_posix()}
        sample_count += 1

    return scene_infos, sample_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a minimal ViewFormer Occ3D annotations.json from data/nuscenes/gts."
    )
    parser.add_argument("--root", default=str(default_root()))
    parser.add_argument("--gts-root")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    gts_root = Path(args.gts_root).resolve() if args.gts_root else (root / "data" / "nuscenes" / "gts").resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else root / "data" / "nuscenes" / "occ3d-nus" / "annotations.json"
    )

    if not gts_root.is_dir():
        raise SystemExit(f"gts root not found: {gts_root}")

    scene_infos, sample_count = build_scene_infos(gts_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scene_infos": scene_infos,
        "metadata": {
            "source": str(gts_root),
            "format": "minimal_viewformer_occ3d",
            "scene_count": len(scene_infos),
            "sample_count": sample_count,
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"wrote {output_path}")
    print(f"scenes={len(scene_infos)}")
    print(f"samples={sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
