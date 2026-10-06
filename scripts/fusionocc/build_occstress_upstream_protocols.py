#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from occstress.datasets.paths import data_root


IMAGE_SETTINGS = {
    "Brightness": ("easy", "mid", "hard"),
    "CameraCrash": ("easy", "mid", "hard"),
    "ColorQuant": ("easy", "mid", "hard"),
    "Fog": ("easy", "mid", "hard"),
    "FrameLost": ("easy", "mid", "hard"),
    "LowLight": ("easy", "mid", "hard"),
    "MotionBlur": ("easy", "mid", "hard"),
    "Snow": ("easy", "mid", "hard"),
}
POINTCLOUD_SETTINGS = {
    "beam_missing": ("light", "moderate", "heavy"),
    "cross_sensor": ("light", "moderate", "heavy"),
    "crosstalk": ("light", "moderate", "heavy"),
    "fog": ("light", "moderate", "heavy"),
    "incomplete_echo": ("light", "moderate", "heavy"),
    "motion_blur": ("light", "moderate", "heavy"),
    "snow": ("light", "moderate", "heavy"),
    "wet_ground": ("light", "moderate", "heavy"),
}
FRAME_PROTOCOLS = ("current", "history_k1", "all_frame")
BACKBONE_STEM = "H4_F6_val_backbone"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build all FusionOcc camera+LiDAR upstream OccStress-nuScenes protocols."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--source-model", default="fusionocc")
    parser.add_argument("--subtrack", default="pointcloud_fusion")
    parser.add_argument("--occ-root", type=Path)
    parser.add_argument("--occstress-root", type=Path)
    parser.add_argument("--backbone-protocol", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--target-root",
        type=Path,
        help="Optional clean GT override; otherwise preserve the release backbone targets.",
    )
    parser.add_argument(
        "--only", choices=["clean", "image", "pointcloud", "all"], default="all"
    )
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_builder(args, corruption, severity, frame_protocol=None):
    root = args.root.resolve()
    shared = data_root(args.occstress_root, dataset='nuscenes', code_root=root)
    occ_root = (
        args.occ_root.expanduser().absolute()
        if args.occ_root
        else shared
        / "occ/upstream/OccStress-nuScenes"
        / args.subtrack
        / args.source_model
    )
    command = [
        sys.executable,
        str(root / "scripts" / "build_upstream_occstress_protocol.py"),
        "--root",
        str(root),
        '--dataset', 'nuscenes',
        '--occstress-root', str(shared),
        "--backbone-protocol",
        str(
            args.backbone_protocol or shared
            / "protocols/manual/OccStress-nuScenes/clean"
            / f"{BACKBONE_STEM}.pkl"
        ),
        "--subtrack",
        args.subtrack,
        "--source-model",
        args.source_model,
        "--corruption",
        corruption,
        "--severity",
        severity,
        "--clean-input-root",
        str(occ_root / "clean"),
    ]
    if args.target_root:
        command += ["--target-root", str(args.target_root.resolve())]
    if args.overwrite:
        command.append("--overwrite")
    if corruption != "clean":
        command += [
            "--corrupted-input-root",
            str(occ_root / corruption / severity),
            "--frame-protocol",
            frame_protocol,
            "--history-k",
            "1",
        ]
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    args.root = args.root.resolve()
    shared = data_root(args.occstress_root, dataset='nuscenes', code_root=args.root)
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else args.root
        / "EXIST"
        / "3D"
        / "FusionOcc"
        / "ckpts"
        / "FusionOcc_BaseWoMask.pth"
    )
    selected = []
    if args.only in {"clean", "all"}:
        run_builder(args, "clean", "clean")
        selected.append(("clean", "clean", "clean"))
    commands = []
    for modality, settings in (
        ("image", IMAGE_SETTINGS),
        ("pointcloud", POINTCLOUD_SETTINGS),
    ):
        if args.only not in {modality, "all"}:
            continue
        for corruption, severities in settings.items():
            for severity in severities:
                for frame_protocol in FRAME_PROTOCOLS:
                    commands.append((corruption, severity, frame_protocol))
                    selected.append((corruption, severity, frame_protocol))
    if commands:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(run_builder, args, *command) for command in commands
            ]
            for future in futures:
                future.result()

    protocol_root = (
        shared
        / "protocols/upstream/OccStress-nuScenes"
        / args.subtrack
        / args.source_model
    )
    occ_root = (
        args.occ_root.expanduser().absolute()
        if args.occ_root
        else shared
        / "occ/upstream/OccStress-nuScenes"
        / args.subtrack
        / args.source_model
    )
    manifest = {
        "track": "upstream",
        "subtrack": args.subtrack,
        "source_model": args.source_model,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "model_modality": "camera+lidar",
        "corruption_modalities": ["image", "pointcloud"],
        "image_settings": IMAGE_SETTINGS,
        "pointcloud_settings": POINTCLOUD_SETTINGS,
        "frame_protocols": list(FRAME_PROTOCOLS),
        "occ_setting_count": 49,
        "protocol_count": 145,
        "selected_build_count": len(selected),
        "occ_root": str(occ_root),
        "prediction_mask_policy": "raw FusionOcc argmax; evaluation uses GT mask_camera",
        "protocol_root": str(protocol_root),
        "backbone_protocol": str(
            args.backbone_protocol or shared
            / "protocols/manual/OccStress-nuScenes/clean"
            / f"{BACKBONE_STEM}.pkl"
        ),
    }
    meta_path = (
        shared
        / "meta/OccStress-nuScenes/upstream"
        / args.subtrack
        / args.source_model
        / "manifest.json"
    )
    atomic_json(meta_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
