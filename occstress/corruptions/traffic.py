#!/usr/bin/env python3
import argparse
import collections
import json
import os
import sys

import numpy as np
from occstress.datasets.construction import (
    clean_nuscenes_reference, manual_meta, manual_output, portable_reference,
)

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from occstress.corruptions import semantic


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the traffic subset by left-right mirroring occupancy.")
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Shared OccStress root; defaults to OCCSTRESS_DATA_ROOT.")
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--sample-token", type=str, default="")
    parser.add_argument("--sample-token-file", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_samples(metadata, args):
    token_set = semantic.load_sample_token_set(args.sample_token_file)
    for sample in metadata["samples"]:
        scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
        if args.scene_name and scene_name != args.scene_name:
            continue
        if args.sample_token and sample["token"] != args.sample_token:
            continue
        if token_set is not None and sample["token"] not in token_set:
            continue
        yield sample


def class_histogram(semantics):
    valid = semantics != semantic.CLASS_TO_ID["free"]
    classes, counts = np.unique(semantics[valid], return_counts=True)
    return {semantic.OCC_CLASS_NAMES[int(cls_id)]: int(cnt) for cls_id, cnt in zip(classes, counts)}


def process_sample(sample, metadata, args, summary):
    scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
    occ_path = os.path.join(args.data_root, "gts", scene_name, sample["token"], "labels.npz")
    if not os.path.exists(occ_path):
        return False

    scene_out_dir = manual_output(args.output_root, "occ", "traffic", scene_name, sample["token"])
    event_path = manual_output(args.output_root, "events", "traffic", scene_name, f"{sample['token']}.json")
    out_npz = os.path.join(scene_out_dir, "labels.npz")
    if not args.overwrite and os.path.exists(out_npz) and os.path.exists(event_path):
        return False

    occ = np.load(occ_path)
    semantics = occ["semantics"]
    mirrored_semantics = np.ascontiguousarray(semantics[:, ::-1, :])
    mirrored_mask_lidar = np.ascontiguousarray(occ["mask_lidar"][:, ::-1, :])
    mirrored_mask_camera = np.ascontiguousarray(occ["mask_camera"][:, ::-1, :])
    changed_mask = semantics != mirrored_semantics

    os.makedirs(scene_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(event_path), exist_ok=True)
    np.savez_compressed(
        out_npz,
        semantics=mirrored_semantics,
        mask_lidar=mirrored_mask_lidar,
        mask_camera=mirrored_mask_camera,
    )

    event = {
        "type": "traffic",
        "scene_name": scene_name,
        "sample_token": sample["token"],
        "mirror_axis": "y",
        "operation": "left_right_mirror",
        "changed_voxels_total": int(changed_mask.sum()),
        "class_histogram": class_histogram(mirrored_semantics),
        "occ_path_clean": clean_nuscenes_reference(scene_name, sample["token"]),
        "occ_path_corrupt": portable_reference(scene_out_dir / "labels.npz", args.output_root),
    }
    with open(event_path, "w") as f:
        json.dump(event, f, indent=2)

    if args.save_previews:
        semantic.save_preview(semantics, mirrored_semantics, changed_mask, os.path.join(scene_out_dir, "preview_bev.png"))

    summary["processed"] += 1
    summary["changed_voxels_total"] += int(changed_mask.sum())
    summary["class_histogram_total"].update(class_histogram(mirrored_semantics))
    return True


def main():
    args = parse_args()
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = semantic.collect_indices(nusc_root)
    summary = {
        "type": "traffic",
        "processed": 0,
        "changed_voxels_total": 0,
        "class_histogram_total": collections.Counter(),
    }
    for sample in iter_samples(metadata, args):
        process_sample(sample, metadata, args, summary)
        if args.max_samples > 0 and summary["processed"] >= args.max_samples:
            break

    summary["class_histogram_total"] = dict(summary["class_histogram_total"])
    summary_dir = manual_meta(args.output_root)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "traffic_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
