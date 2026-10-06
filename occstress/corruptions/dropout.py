#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import hashlib
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

REALIZATION_SEED = None


DROPOUT_CONFIGS = {
    "easy": {
        "num_regions_min": 1,
        "num_regions_max": 2,
        "half_size_min": [6, 6, 2],
        "half_size_max": [9, 9, 3],
        "min_removed_voxels": 320,
        "min_occupied_density": 0.12,
        "min_spacing_voxels": 0.0,
        "shape_mode": "cuboid",
    },
    "mid": {
        "num_regions_min": 3,
        "num_regions_max": 4,
        "half_size_min": [12, 12, 4],
        "half_size_max": [18, 18, 6],
        "min_removed_voxels": 1200,
        "min_occupied_density": 0.07,
        "min_spacing_voxels": 9.0,
        "shape_mode": "irregular_blob",
        "target_removed_fraction_range": [0.70, 0.90],
        "blob_count_min": 3,
        "blob_count_max": 5,
    },
    "hard": {
        "num_regions_min": 5,
        "num_regions_max": 8,
        "half_size_min": [18, 18, 5],
        "half_size_max": [26, 26, 8],
        "min_removed_voxels": 2500,
        "min_occupied_density": 0.04,
        "min_spacing_voxels": 10.0,
        "shape_mode": "irregular_blob",
        "target_removed_fraction_range": [0.90, 1.00],
        "blob_count_min": 4,
        "blob_count_max": 8,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the dropout subset in frame-level storage format.")
    parser.add_argument("--severity", type=str, choices=sorted(DROPOUT_CONFIGS), required=True)
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Shared OccStress root; defaults to OCCSTRESS_DATA_ROOT.")
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--sample-token", type=str, default="")
    parser.add_argument("--sample-token-file", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--realization-seed",
        type=int,
        default=None,
        help="Optional independent corruption-realization seed. Omit to reproduce the released deterministic data.",
    )
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(*parts):
    h = hashlib.sha1()
    if REALIZATION_SEED is not None:
        h.update(b"realization|")
        h.update(str(REALIZATION_SEED).encode("utf-8"))
        h.update(b"|")
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest()[:16], 16)


def choose_count(sample_token, severity, low, high):
    if low == high:
        return low
    rng = np.random.default_rng(stable_seed("dropout_num_regions", severity, sample_token))
    return int(rng.integers(low, high + 1))


def choose_half_size(sample_token, severity, step_idx, config):
    low = np.asarray(config["half_size_min"], dtype=np.int64)
    high = np.asarray(config["half_size_max"], dtype=np.int64)
    rng = np.random.default_rng(stable_seed("dropout_half_size", severity, sample_token, step_idx))
    return rng.integers(low, high + 1, size=3, endpoint=False)


def choose_fraction(sample_token, severity, step_idx, tag, low, high):
    rng = np.random.default_rng(stable_seed(tag, severity, sample_token, step_idx))
    return float(rng.uniform(low, high))


def choose_blob_count(sample_token, severity, step_idx, low, high):
    if low == high:
        return low
    rng = np.random.default_rng(stable_seed("dropout_blob_count", severity, sample_token, step_idx))
    return int(rng.integers(low, high + 1))


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


def clip_bounds(center, half_size, shape):
    low = np.maximum(center - half_size, 0)
    high = np.minimum(center + half_size + 1, np.asarray(shape, dtype=np.int64))
    return low, high


def region_removed_indices(semantics, changed_mask, low, high):
    region = semantics[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    region_changed = changed_mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    occupied = region != semantic.CLASS_TO_ID["free"]
    removable = occupied & (~region_changed)
    rel = np.argwhere(removable)
    if len(rel) == 0:
        return np.empty((0, 3), dtype=np.int64), 0.0, {}
    coords = rel + low[None, :]
    volume = int(np.prod(high - low))
    density = float(len(coords) / max(1, volume))
    classes, counts = np.unique(semantics[coords[:, 0], coords[:, 1], coords[:, 2]], return_counts=True)
    hist = {semantic.OCC_CLASS_NAMES[int(cls_id)]: int(cnt) for cls_id, cnt in zip(classes, counts)}
    return coords.astype(np.int64), density, hist


def choose_fallback_center(semantics, changed_mask, occupied_indices):
    if len(occupied_indices) == 0:
        return None
    valid = []
    for center in occupied_indices:
        if changed_mask[center[0], center[1], center[2]]:
            continue
        valid.append(center)
    if not valid:
        valid = occupied_indices
    valid = np.asarray(valid, dtype=np.int64)
    coords = valid.astype(np.float32)
    scene_center = np.asarray(semantics.shape, dtype=np.float32) / 2.0
    scores = np.sum((coords - scene_center[None, :]) ** 2, axis=1)
    return valid[int(np.argmin(scores))]


def irregular_removed_indices(sample_token, severity, step_idx, removable_coords, low, high, config):
    if len(removable_coords) == 0:
        return np.empty((0, 3), dtype=np.int64), 0.0
    local = removable_coords - low[None, :]
    region_shape = (high - low).astype(np.float32)
    frac_low, frac_high = config["target_removed_fraction_range"]
    target_frac = choose_fraction(sample_token, severity, step_idx, "dropout_removed_frac", frac_low, frac_high)
    target_remove = max(config["min_removed_voxels"], int(round(len(removable_coords) * target_frac)))
    target_remove = min(target_remove, len(removable_coords))

    blob_count = choose_blob_count(
        sample_token,
        severity,
        step_idx,
        config["blob_count_min"],
        config["blob_count_max"],
    )
    rng = np.random.default_rng(stable_seed("dropout_irregular_blob", severity, sample_token, step_idx))
    blob_centers = []
    blob_scales = []
    for _ in range(blob_count):
        center = region_shape * rng.uniform(0.2, 0.8, size=3)
        scale = np.maximum(1.0, region_shape * rng.uniform(0.18, 0.38, size=3))
        blob_centers.append(center)
        blob_scales.append(scale)

    local_f = local.astype(np.float32)
    scores = np.zeros(len(local_f), dtype=np.float32)
    for center, scale in zip(blob_centers, blob_scales):
        norm = ((local_f - center[None, :]) / scale[None, :]) ** 2
        scores = np.maximum(scores, np.exp(-0.5 * norm.sum(axis=1)))
    scores += rng.uniform(0.0, 0.18, size=len(scores)).astype(np.float32)
    chosen = np.argsort(scores)[-target_remove:]
    return np.unique(removable_coords[chosen], axis=0), target_frac


def process_sample(sample, metadata, args, summary):
    severity = args.severity
    config = DROPOUT_CONFIGS[severity]
    scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
    occ_path = os.path.join(args.data_root, "gts", scene_name, sample["token"], "labels.npz")
    if not os.path.exists(occ_path):
        return False

    scene_out_dir = manual_output(args.output_root, "occ", "dropout", severity, scene_name, sample["token"])
    event_path = manual_output(args.output_root, "events", "dropout", severity, scene_name, f"{sample['token']}.json")
    if not args.overwrite and os.path.exists(os.path.join(scene_out_dir, "labels.npz")) and os.path.exists(event_path):
        return False

    occ = np.load(occ_path)
    semantics = occ["semantics"].copy()
    original = semantics.copy()
    changed_mask = np.zeros_like(semantics, dtype=bool)
    occupied_indices = np.argwhere(semantics != semantic.CLASS_TO_ID["free"])
    if len(occupied_indices) == 0:
        return False

    rng = np.random.default_rng(stable_seed("dropout_seed_order", severity, sample["token"]))
    seed_order = rng.permutation(len(occupied_indices))
    requested_num_regions = choose_count(sample["token"], severity, config["num_regions_min"], config["num_regions_max"])
    num_regions = requested_num_regions
    chosen_centers = []
    events = []
    cursor = 0
    min_spacing = float(config["min_spacing_voxels"])

    for step_idx in range(num_regions):
        chosen = None
        half_size = choose_half_size(sample["token"], severity, step_idx, config)
        max_trials = min(len(seed_order) - cursor, 256)
        for trial_offset in range(max_trials):
            center = occupied_indices[seed_order[cursor + trial_offset]]
            if min_spacing > 0.0 and chosen_centers:
                min_dist = min(float(np.linalg.norm(center.astype(np.float32) - prev)) for prev in chosen_centers)
                if min_dist < min_spacing:
                    continue
            low, high = clip_bounds(center, half_size, semantics.shape)
            removable, density, hist = region_removed_indices(semantics, changed_mask, low, high)
            if len(removable) < config["min_removed_voxels"]:
                continue
            if density < config["min_occupied_density"]:
                continue
            shape_mode = config.get("shape_mode", "cuboid")
            if shape_mode == "irregular_blob":
                removed, removed_fraction = irregular_removed_indices(
                    sample["token"], severity, step_idx, removable, low, high, config
                )
            else:
                removed = removable
                removed_fraction = 1.0
            if len(removed) < config["min_removed_voxels"]:
                continue
            chosen = {
                "center": center.astype(np.int64),
                "half_size": half_size.astype(np.int64),
                "low": low.astype(np.int64),
                "high": high.astype(np.int64),
                "removed": removed,
                "density": density,
                "hist": hist,
                "shape_mode": shape_mode,
                "removed_fraction": removed_fraction,
            }
            cursor += trial_offset + 1
            break
        if chosen is None:
            fallback_center = choose_fallback_center(semantics, changed_mask, occupied_indices)
            if fallback_center is None:
                break
            fallback_half_size = np.maximum(np.asarray(config["half_size_min"], dtype=np.int64) // 2, 1)
            low, high = clip_bounds(fallback_center, fallback_half_size, semantics.shape)
            removable, density, hist = region_removed_indices(semantics, changed_mask, low, high)
            if len(removable) == 0:
                break
            min_removed = max(32, config["min_removed_voxels"] // 8)
            if len(removable) < min_removed:
                # Fall back to deleting all currently available occupied voxels in the smallest valid crop.
                min_removed = len(removable)
            shape_mode = "cuboid"
            removed = removable
            if len(removed) > min_removed:
                center_f = fallback_center.astype(np.float32)
                score = np.sum((removed.astype(np.float32) - center_f[None, :]) ** 2, axis=1)
                removed = removed[np.argsort(score)[:min_removed]]
            chosen = {
                "center": fallback_center.astype(np.int64),
                "half_size": fallback_half_size.astype(np.int64),
                "low": low.astype(np.int64),
                "high": high.astype(np.int64),
                "removed": np.unique(removed, axis=0),
                "density": density,
                "hist": hist,
                "shape_mode": shape_mode,
                "removed_fraction": float(len(removed) / max(1, len(removable))),
                "fallback": True,
            }

        removed = chosen["removed"]
        semantics[removed[:, 0], removed[:, 1], removed[:, 2]] = semantic.CLASS_TO_ID["free"]
        changed_mask[removed[:, 0], removed[:, 1], removed[:, 2]] = True
        chosen_centers.append(chosen["center"].astype(np.float32))
        events.append(
            {
                "center_voxel": chosen["center"].tolist(),
                "half_size_voxels": chosen["half_size"].tolist(),
                "bounds": {
                    "low_inclusive": chosen["low"].tolist(),
                    "high_exclusive": chosen["high"].tolist(),
                },
                "removed_voxels": int(len(removed)),
                "occupied_density": float(chosen["density"]),
                "class_histogram": chosen["hist"],
                "shape_mode": chosen["shape_mode"],
                "removed_fraction": float(chosen["removed_fraction"]),
                "fallback": bool(chosen.get("fallback", False)),
            }
        )

    if not events:
        return False

    os.makedirs(scene_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(event_path), exist_ok=True)
    np.savez_compressed(
        os.path.join(scene_out_dir, "labels.npz"),
        semantics=semantics,
        mask_lidar=occ["mask_lidar"],
        mask_camera=occ["mask_camera"],
    )

    event = {
        "type": "dropout",
        "severity": severity,
        "sample_token": sample["token"],
        "scene_name": scene_name,
        "num_regions_requested": requested_num_regions,
        "num_regions_applied": len(events),
        "changed_voxels_total": int(changed_mask.sum()),
        "events": events,
        "occ_path_clean": clean_nuscenes_reference(scene_name, sample["token"]),
        "occ_path_corrupt": portable_reference(scene_out_dir / "labels.npz", args.output_root),
    }
    if args.realization_seed is not None:
        event["realization_seed"] = args.realization_seed
    with open(event_path, "w") as f:
        json.dump(event, f, indent=2)

    if args.save_previews:
        semantic.save_preview(original, semantics, changed_mask, os.path.join(scene_out_dir, "preview_bev.png"))

    summary["processed"] += 1
    summary["changed_voxels_total"] += int(changed_mask.sum())
    summary["region_events_total"] += len(events)
    return True


def main():
    global REALIZATION_SEED
    args = parse_args()
    REALIZATION_SEED = args.realization_seed
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = semantic.collect_indices(nusc_root)
    summary = {
        "severity": args.severity,
        "processed": 0,
        "changed_voxels_total": 0,
        "region_events_total": 0,
    }
    if args.realization_seed is not None:
        summary["realization_seed"] = args.realization_seed
    for sample in iter_samples(metadata, args):
        process_sample(sample, metadata, args, summary)
        if args.max_samples > 0 and summary["processed"] >= args.max_samples:
            break

    meta_dir = manual_meta(args.output_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / f"dropout_{args.severity}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
