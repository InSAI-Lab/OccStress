#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


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
FRAME_PROTOCOLS = ["current", "history_k1", "all_frame"]
BACKBONE_STEM = "H4_F6_val_backbone"


def parse_args():
    parser = argparse.ArgumentParser(description="Build SDGOCC pointcloud-fusion upstream OccStress protocols.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-model", default="sdgocc")
    parser.add_argument("--subtrack", default="pointcloud_fusion")
    parser.add_argument(
        "--occ-root",
        help=(
            "Converted upstream occupancy root. Defaults to "
            "<root>/data/OccStress/occ/upstream/<subtrack>/<source-model>."
        ),
    )
    parser.add_argument("--target-root", default="data/nuscenes/gts")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rel(path, root):
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_meta(root, subtrack, source_model, occ_root):
    meta_root = root / "data" / "OccStress" / "meta" / "upstream" / subtrack / source_model
    protocol_root = root / "data" / "OccStress" / "protocols" / "upstream" / subtrack / source_model
    meta_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "track": "upstream",
        "subtrack": subtrack,
        "source_model": source_model,
        "clean_reference": True,
        "corruption_modality": "pointcloud",
        "camera_input": "clean",
        "corruption_family_count": len(POINTCLOUD_CORRUPTIONS),
        "corruption_families": POINTCLOUD_CORRUPTIONS,
        "severity_count": len(POINTCLOUD_SEVERITIES),
        "severities": POINTCLOUD_SEVERITIES,
        "frame_protocol_count": len(FRAME_PROTOCOLS),
        "frame_protocols": FRAME_PROTOCOLS,
        "occ_setting_count": 1 + len(POINTCLOUD_CORRUPTIONS) * len(POINTCLOUD_SEVERITIES),
        "protocol_count": 1 + len(POINTCLOUD_CORRUPTIONS) * len(POINTCLOUD_SEVERITIES) * len(FRAME_PROTOCOLS),
        "backbone_protocol": "data/OccStress/protocols/manual/clean/H4_F6_val_backbone.pkl",
        "clean_occ_root": rel(occ_root / "clean", root),
        "protocol_root": rel(protocol_root, root),
        "occ_layout": {
            "clean": f"data/OccStress/occ/upstream/{subtrack}/{source_model}/clean/<scene>/<token>/labels.npz",
            "corrupted": (
                f"data/OccStress/occ/upstream/{subtrack}/{source_model}/"
                "<pointcloud_corruption>/<severity>/<scene>/<token>/labels.npz"
            ),
        },
        "protocol_layout": {
            "clean": f"data/OccStress/protocols/upstream/{subtrack}/{source_model}/clean/H4_F6_val_backbone.pkl",
            "corrupted": (
                f"data/OccStress/protocols/upstream/{subtrack}/{source_model}/"
                "<pointcloud_corruption>/<severity>/<frame_protocol>_H4_F6_val_backbone.pkl"
            ),
        },
    }
    with (meta_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    rows = [
        {
            "setting_name": "clean",
            "corruption_modality": "clean",
            "corruption": "clean",
            "severity": "clean",
            "occ_root": rel(occ_root / "clean", root),
            "protocol_count_expected": 1,
            "status": "",
            "notes": "camera clean, pointcloud clean",
        }
    ]
    for corruption in POINTCLOUD_CORRUPTIONS:
        for severity in POINTCLOUD_SEVERITIES:
            rows.append(
                {
                    "setting_name": f"{corruption}/{severity}",
                    "corruption_modality": "pointcloud",
                    "corruption": corruption,
                    "severity": severity,
                    "occ_root": rel(occ_root / corruption / severity, root),
                    "protocol_count_expected": len(FRAME_PROTOCOLS),
                    "status": "",
                    "notes": "camera clean, pointcloud corrupted",
                }
            )
    write_csv(
        meta_root / "occ_settings_template.csv",
        ["setting_name", "corruption_modality", "corruption", "severity", "occ_root", "protocol_count_expected", "status", "notes"],
        rows,
    )

    rows = [
        {
            "protocol_name": f"clean_{BACKBONE_STEM}",
            "corruption_modality": "clean",
            "corruption": "clean",
            "severity": "clean",
            "frame_protocol": "clean",
            "protocol_path": rel(protocol_root / "clean" / f"{BACKBONE_STEM}.pkl", root),
            "status": "",
            "notes": "",
        }
    ]
    for corruption in POINTCLOUD_CORRUPTIONS:
        for severity in POINTCLOUD_SEVERITIES:
            for frame_protocol in FRAME_PROTOCOLS:
                rows.append(
                    {
                        "protocol_name": f"{corruption}_{severity}_{frame_protocol}_{BACKBONE_STEM}",
                        "corruption_modality": "pointcloud",
                        "corruption": corruption,
                        "severity": severity,
                        "frame_protocol": frame_protocol,
                        "protocol_path": rel(
                            protocol_root / corruption / severity / f"{frame_protocol}_{BACKBONE_STEM}.pkl",
                            root,
                        ),
                        "status": "",
                        "notes": "",
                    }
                )
    write_csv(
        meta_root / "protocol_results_template.csv",
        ["protocol_name", "corruption_modality", "corruption", "severity", "frame_protocol", "protocol_path", "status", "notes"],
        rows,
    )

    rows = [{"corruption": "clean_ref", "avg_miou": "", "avg_delta_vs_clean": "", "status": "", "notes": ""}]
    for corruption in POINTCLOUD_CORRUPTIONS:
        rows.append({"corruption": corruption, "avg_miou": "", "avg_delta_vs_clean": "", "status": "", "notes": ""})
    rows.append({"corruption": "pointcloud_corruption_avg", "avg_miou": "", "avg_delta_vs_clean": "", "status": "", "notes": ""})
    write_csv(meta_root / "family_summary_template.csv", ["corruption", "avg_miou", "avg_delta_vs_clean", "status", "notes"], rows)


def run_builder(root, subtrack, source_model, overwrite, target_root, occ_root):
    builder = root / "scripts" / "build_upstream_occstress_protocol.py"
    backbone = root / "data" / "OccStress" / "protocols" / "manual" / "clean" / f"{BACKBONE_STEM}.pkl"
    common = [
        sys.executable,
        str(builder),
        "--root",
        str(root),
        "--backbone-protocol",
        str(backbone),
        "--subtrack",
        subtrack,
        "--source-model",
        source_model,
    ]
    if target_root is not None:
        common += ["--target-root", str(target_root)]
    if overwrite:
        common.append("--overwrite")

    subprocess.run(common + ["--corruption", "clean", "--clean-input-root", str(occ_root / "clean")], check=True)
    for corruption in POINTCLOUD_CORRUPTIONS:
        for severity in POINTCLOUD_SEVERITIES:
            for frame_protocol in FRAME_PROTOCOLS:
                subprocess.run(
                    common
                    + [
                        "--corruption",
                        corruption,
                        "--severity",
                        severity,
                        "--frame-protocol",
                        frame_protocol,
                        "--history-k",
                        "1",
                        "--clean-input-root",
                        str(occ_root / "clean"),
                        "--corrupted-input-root",
                        str(occ_root / corruption / severity),
                    ],
                    check=True,
                )


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    if args.occ_root:
        occ_root = Path(args.occ_root)
        if not occ_root.is_absolute():
            occ_root = root / occ_root
        occ_root = occ_root.resolve()
    else:
        occ_root = root / "data" / "OccStress" / "occ" / "upstream" / args.subtrack / args.source_model
    target_root = Path(args.target_root).resolve() if args.target_root else None
    write_meta(root, args.subtrack, args.source_model, occ_root)
    run_builder(root, args.subtrack, args.source_model, args.overwrite, target_root, occ_root)
    print(
        json.dumps(
            {
                "subtrack": args.subtrack,
                "source_model": args.source_model,
                "protocol_count": 1 + len(POINTCLOUD_CORRUPTIONS) * len(POINTCLOUD_SEVERITIES) * len(FRAME_PROTOCOLS),
                "protocol_root": str(root / "data" / "OccStress" / "protocols" / "upstream" / args.subtrack / args.source_model),
                "meta_root": str(root / "data" / "OccStress" / "meta" / "upstream" / args.subtrack / args.source_model),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
