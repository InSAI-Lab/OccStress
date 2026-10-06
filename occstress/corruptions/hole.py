#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import collections
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


PRIMARY_DYNAMIC_CLASSES = set(semantic.CONFUSION_GROUPS["vehicle"])
FALLBACK_DYNAMIC_CLASSES = set(semantic.CONFUSION_GROUPS["vehicle"] + semantic.CONFUSION_GROUPS["vru"])

HOLE_CONFIGS = {
    "easy": {
        "num_objects_min": 1,
        "num_objects_max": 2,
        "min_support_voxels": 240,
        "min_interior_voxels": 24,
        "removed_ratio_range": [0.58, 0.78],
        "min_spacing_voxels": 0.0,
    },
    "mid": {
        "num_objects_min": 2,
        "num_objects_max": 3,
        "min_support_voxels": 220,
        "min_interior_voxels": 24,
        "removed_ratio_range": [0.78, 0.94],
        "min_spacing_voxels": 4.0,
    },
    "hard": {
        "num_objects_min": 4,
        "num_objects_max": 6,
        "min_support_voxels": 180,
        "min_interior_voxels": 18,
        "removed_ratio_range": [0.94, 1.00],
        "min_spacing_voxels": 3.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the hole subset in frame-level storage format.")
    parser.add_argument("--severity", type=str, choices=sorted(HOLE_CONFIGS), required=True)
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


def choose_count(sample_token, severity, low, high, tag):
    if low == high:
        return low
    rng = np.random.default_rng(stable_seed(tag, severity, sample_token))
    return int(rng.integers(low, high + 1))


def choose_ratio(sample_token, severity, step_idx, ratio_range):
    low, high = ratio_range
    rng = np.random.default_rng(stable_seed("hole_ratio", severity, sample_token, step_idx))
    return float(rng.uniform(low, high))


def erode_one_voxel(support):
    support = np.asarray(support, dtype=np.int64)
    support_set = {tuple(idx.tolist()) for idx in support}
    neighbors = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int64,
    )
    interior = []
    for idx in support:
        keep = True
        for delta in neighbors:
            nxt = tuple((idx + delta).tolist())
            if nxt not in support_set:
                keep = False
                break
        if keep:
            interior.append(idx)
    if not interior:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(interior, dtype=np.int64)


def largest_connected_component(indices):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.int64)
    remaining = {tuple(idx.tolist()) for idx in indices}
    best = []
    while remaining:
        seed = np.asarray(next(iter(remaining)), dtype=np.int64)
        component = semantic.connected_component(indices, seed)
        if len(component) > len(best):
            best = component
        for idx in component:
            remaining.discard(tuple(idx.tolist()))
    if len(best) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(best, dtype=np.int64)


def build_generic_fallback_candidate(semantics):
    non_free = np.argwhere(semantics != semantic.CLASS_TO_ID["free"])
    if len(non_free) == 0:
        return None
    cls_ids, counts = np.unique(semantics[non_free[:, 0], non_free[:, 1], non_free[:, 2]], return_counts=True)
    cls_ids = [int(x) for x in cls_ids if int(x) != semantic.CLASS_TO_ID["free"]]
    if not cls_ids:
        return None
    best_support = np.empty((0, 3), dtype=np.int64)
    best_interior = np.empty((0, 3), dtype=np.int64)
    best_class = None
    for cls_id in cls_ids:
        cls_indices = np.argwhere(semantics == cls_id)
        component = largest_connected_component(cls_indices)
        interior = erode_one_voxel(component)
        if len(interior) == 0:
            continue
        if len(interior) > len(best_interior) or (len(interior) == len(best_interior) and len(component) > len(best_support)):
            best_support = component
            best_interior = interior
            best_class = semantic.OCC_CLASS_NAMES[cls_id]
    if best_class is None or len(best_support) == 0 or len(best_interior) == 0:
        return None
    return {
        "source_class": best_class,
        "support": best_support,
        "interior": best_interior,
        "fallback_mode": "occupied_component",
    }


def build_local_crop_fallback(semantics):
    occupied = np.argwhere(semantics != semantic.CLASS_TO_ID["free"])
    if len(occupied) == 0:
        return None
    occupied_set = {tuple(idx.tolist()) for idx in occupied}
    best_center = None
    best_score = -1
    half = np.array([2, 2, 1], dtype=np.int64)
    for idx in occupied:
        low = np.maximum(idx - half, 0)
        high = np.minimum(idx + half + 1, np.asarray(semantics.shape, dtype=np.int64))
        score = 0
        for x in range(low[0], high[0]):
            for y in range(low[1], high[1]):
                for z in range(low[2], high[2]):
                    if (x, y, z) in occupied_set:
                        score += 1
        if score > best_score:
            best_score = score
            best_center = idx
    if best_center is None:
        return None
    low = np.maximum(best_center - half, 0)
    high = np.minimum(best_center + half + 1, np.asarray(semantics.shape, dtype=np.int64))
    support = []
    interior = []
    for x in range(low[0], high[0]):
        for y in range(low[1], high[1]):
            for z in range(low[2], high[2]):
                if semantics[x, y, z] == semantic.CLASS_TO_ID["free"]:
                    continue
                support.append([x, y, z])
                if low[0] < x < high[0] - 1 and low[1] < y < high[1] - 1 and low[2] < z < high[2] - 1:
                    interior.append([x, y, z])
    if not support or not interior:
        return None
    support = np.asarray(support, dtype=np.int64)
    interior = np.asarray(interior, dtype=np.int64)
    source_classes, counts = np.unique(semantics[support[:, 0], support[:, 1], support[:, 2]], return_counts=True)
    source_class = semantic.OCC_CLASS_NAMES[int(source_classes[int(np.argmax(counts))])]
    return {
        "source_class": source_class,
        "support": support,
        "interior": interior,
        "fallback_mode": "local_crop",
    }


def build_dynamic_candidates(
    sample_token,
    scene_name,
    semantics,
    metadata,
    global_to_state,
    config,
    allowed_classes=None,
    support_config=None,
    min_support_voxels=None,
    min_interior_voxels=None,
):
    candidates = []
    anns = metadata["anns_by_sample"].get(sample_token, [])
    if allowed_classes is None:
        allowed_classes = PRIMARY_DYNAMIC_CLASSES
    if support_config is None:
        support_config = {
            "support_mode": "completed_instance",
            "bbox_expand_voxels": 1,
        }
    if min_support_voxels is None:
        min_support_voxels = config["min_support_voxels"]
    if min_interior_voxels is None:
        min_interior_voxels = config["min_interior_voxels"]
    for ann in anns:
        inst = metadata["instance_by_token"][ann["instance_token"]]
        category_name = metadata["category_by_token"][inst["category_token"]]["name"]
        occ_class = semantic.nusc_category_to_occ(category_name)
        if occ_class not in allowed_classes:
            continue
        occ_id = semantic.CLASS_TO_ID[occ_class]
        source_indices = np.argwhere(semantics == occ_id)
        if len(source_indices) == 0:
            continue
        inside = semantic.box_mask_from_annotation(source_indices, ann, global_to_state)
        if int(inside.sum()) < min_support_voxels:
            continue
        candidate = semantic.AnnotationCandidate(
            ann_token=ann["token"],
            instance_token=ann["instance_token"],
            category_name=category_name,
            occ_class=occ_class,
            voxel_count=int(inside.sum()),
            translation=ann["translation"],
            size=ann["size"],
            rotation=ann["rotation"],
        )
        support = semantic.build_support(semantics, candidate, global_to_state, support_config)
        interior = erode_one_voxel(support)
        if len(support) < min_support_voxels or len(interior) < min_interior_voxels:
            continue
        candidates.append((candidate, support, interior))
    candidates.sort(key=lambda x: (-len(x[1]), -len(x[2]), x[0].instance_token))
    return candidates


def carve_central_void(interior, remove_ratio):
    interior = np.asarray(interior, dtype=np.int64)
    if len(interior) == 0:
        return interior
    center = interior.astype(np.float32).mean(axis=0)
    score = np.sum((interior.astype(np.float32) - center[None, :]) ** 2, axis=1)
    num_remove = max(1, int(round(len(interior) * remove_ratio)))
    chosen = np.argsort(score)[:num_remove]
    return np.unique(interior[chosen], axis=0)


def support_center(support):
    support = np.asarray(support, dtype=np.float32)
    return support.mean(axis=0)


def process_sample(sample, metadata, args, summary):
    severity = args.severity
    config = HOLE_CONFIGS[severity]
    scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
    occ_path = os.path.join(args.data_root, "gts", scene_name, sample["token"], "labels.npz")
    if not os.path.exists(occ_path):
        return False
    lidar_sd = metadata["lidar_sd_by_sample"].get(sample["token"])
    if lidar_sd is None:
        return False

    scene_out_dir = manual_output(args.output_root, "occ", "hole", severity, scene_name, sample["token"])
    event_path = manual_output(args.output_root, "events", "hole", severity, scene_name, f"{sample['token']}.json")
    if not args.overwrite and os.path.exists(os.path.join(scene_out_dir, "labels.npz")) and os.path.exists(event_path):
        return False

    occ = np.load(occ_path)
    semantics = occ["semantics"].copy()
    original = semantics.copy()
    changed_mask = np.zeros_like(semantics, dtype=bool)

    ego_pose = metadata["ego_pose_by_token"][lidar_sd["ego_pose_token"]]
    global_to_state = semantic.invert_transform(semantic.make_transform(ego_pose["translation"], ego_pose["rotation"]))
    candidates = build_dynamic_candidates(sample["token"], scene_name, semantics, metadata, global_to_state, config)
    if not candidates:
        fallback_support = {
            "support_mode": "completed_instance",
            "bbox_expand_voxels": 2,
        }
        candidates = build_dynamic_candidates(
            sample["token"],
            scene_name,
            semantics,
            metadata,
            global_to_state,
            config,
            allowed_classes=FALLBACK_DYNAMIC_CLASSES,
            support_config=fallback_support,
            min_support_voxels=max(60, config["min_support_voxels"] // 3),
            min_interior_voxels=max(4, config["min_interior_voxels"] // 4),
        )
    if REALIZATION_SEED is not None and candidates:
        rng = np.random.default_rng(
            stable_seed("hole_candidate_order", severity, sample["token"])
        )
        candidates = [candidates[index] for index in rng.permutation(len(candidates))]
    generic_fallback = None
    local_crop_fallback = None
    if not candidates:
        generic_fallback = build_generic_fallback_candidate(semantics)
        if generic_fallback is None:
            local_crop_fallback = build_local_crop_fallback(semantics)
    if not candidates and generic_fallback is None and local_crop_fallback is None:
        return False

    requested_num_objects = choose_count(
        sample["token"],
        severity,
        config["num_objects_min"],
        config["num_objects_max"],
        "hole_num_objects",
    )
    num_objects = min(requested_num_objects, len(candidates)) if candidates else 0
    if num_objects == 0:
        num_objects = 1

    events = []
    used_instances = set()
    chosen_centers = []
    min_spacing = float(config.get("min_spacing_voxels", 0.0))
    for step_idx in range(num_objects):
        chosen = None
        for candidate, support, interior in candidates:
            if candidate.instance_token in used_instances:
                continue
            removed_ratio = choose_ratio(sample["token"], severity, step_idx, config["removed_ratio_range"])
            removed = carve_central_void(
                interior,
                removed_ratio,
            )
            overlap = changed_mask[removed[:, 0], removed[:, 1], removed[:, 2]].mean() if len(removed) > 0 else 0.0
            if overlap > 0.05:
                continue
            center = support_center(support)
            if min_spacing > 0.0 and chosen_centers:
                min_dist = min(float(np.linalg.norm(center - prev_center)) for prev_center in chosen_centers)
                if min_dist < min_spacing:
                    continue
            chosen = (candidate, support, interior, removed)
            break
        if chosen is None:
            if not events and (generic_fallback is not None or local_crop_fallback is not None):
                fallback = generic_fallback if generic_fallback is not None else local_crop_fallback
                support = fallback["support"]
                interior = fallback["interior"]
                removed = carve_central_void(
                    interior,
                    choose_ratio(sample["token"], severity, step_idx, config["removed_ratio_range"]),
                )
                if len(removed) == 0:
                    break
                semantics[removed[:, 0], removed[:, 1], removed[:, 2]] = semantic.CLASS_TO_ID["free"]
                changed_mask[removed[:, 0], removed[:, 1], removed[:, 2]] = True
                events.append(
                    {
                        "source_class": fallback["source_class"],
                        "instance_token": "",
                        "ann_token": "",
                        "support_voxels": int(len(support)),
                        "interior_voxels": int(len(interior)),
                        "removed_voxels": int(len(removed)),
                        "removed_ratio": float(len(removed) / max(1, len(interior))),
                        "shell_thickness_voxels": 1,
                        "fallback": True,
                        "fallback_mode": fallback["fallback_mode"],
                    }
                )
            break

        candidate, support, interior, removed = chosen
        semantics[removed[:, 0], removed[:, 1], removed[:, 2]] = semantic.CLASS_TO_ID["free"]
        changed_mask[removed[:, 0], removed[:, 1], removed[:, 2]] = True
        used_instances.add(candidate.instance_token)
        chosen_centers.append(support_center(support))
        events.append(
            {
                "source_class": candidate.occ_class,
                "instance_token": candidate.instance_token,
                "ann_token": candidate.ann_token,
                "support_voxels": int(len(support)),
                "interior_voxels": int(len(interior)),
                "removed_voxels": int(len(removed)),
                "removed_ratio": float(len(removed) / max(1, len(interior))),
                "shell_thickness_voxels": 1,
                "fallback": len(support) < config["min_support_voxels"] or len(interior) < config["min_interior_voxels"],
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
        "type": "hole",
        "severity": severity,
        "sample_token": sample["token"],
        "scene_name": scene_name,
        "num_objects_requested": requested_num_objects,
        "num_objects_applied": len(events),
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
    summary["object_events_total"] += len(events)
    return True


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
        "object_events_total": 0,
    }
    if args.realization_seed is not None:
        summary["realization_seed"] = args.realization_seed
    for sample in iter_samples(metadata, args):
        process_sample(sample, metadata, args, summary)
        if args.max_samples > 0 and summary["processed"] >= args.max_samples:
            break

    meta_dir = manual_meta(args.output_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / f"hole_{args.severity}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
