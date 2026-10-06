#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import collections
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
from occstress.datasets.construction import (
    clean_nuscenes_reference, manual_meta, manual_output, portable_reference,
)

REALIZATION_SEED = None
from PIL import Image


OCC_CLASS_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(OCC_CLASS_NAMES)}

CONFUSION_GROUPS = {
    "vehicle": ["car", "truck", "bus", "trailer", "construction_vehicle"],
    "vru": ["bicycle", "motorcycle", "pedestrian"],
    "roadside": ["barrier", "traffic_cone"],
}

STATIC_GROUPS = {
    "ground": ["driveable_surface", "other_flat", "sidewalk", "terrain"],
    "structure": ["manmade", "vegetation", "others"],
}

CLASS_TO_GROUP = {
    cls_name: group_name
    for group_name, group in CONFUSION_GROUPS.items()
    for cls_name in group
}

STATIC_CLASS_TO_GROUP = {
    cls_name: group_name
    for group_name, group in STATIC_GROUPS.items()
    for cls_name in group
}

EASY_NEIGHBORS = {
    "car": ["truck"],
    "truck": ["car", "bus"],
    "bus": ["truck"],
}

PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=np.float32)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4], dtype=np.float32)

OCC3D_COLOR_MAP = np.array(
    [
        [0, 0, 0],
        [112, 128, 144],
        [255, 61, 99],
        [255, 158, 0],
        [255, 140, 0],
        [233, 150, 70],
        [0, 0, 230],
        [220, 20, 60],
        [255, 69, 0],
        [47, 79, 79],
        [165, 42, 42],
        [0, 207, 191],
        [139, 137, 137],
        [75, 0, 75],
        [150, 240, 80],
        [230, 230, 250],
        [0, 175, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

SEVERITY_CONFIGS = {
    "easy": {
        "dynamic_groups": {"vehicle"},
        "dynamic_group_probs": {"vehicle": 1.0},
        "dynamic_classes": {"car", "truck", "bus"},
        "dynamic_min_voxels": 420,
        "dynamic_num_objects_min": 1,
        "dynamic_num_objects_max": 2,
        "bbox_expand_voxels": 2,
        "support_mode": "completed_instance",
        "mapping_mode": "nearest",
        "static_groups": {"ground", "structure"},
        "static_group_probs": {"ground": 0.6, "structure": 0.4},
        "static_num_regions_min": 1,
        "static_num_regions_max": 2,
        "static_min_voxels": {"ground": 260, "structure": 220},
        "static_half_size_voxels": [7, 7, 3],
    },
    "mid": {
        "dynamic_groups": {"vehicle", "vru", "roadside"},
        "dynamic_group_probs": {"vehicle": 0.75, "vru": 0.15, "roadside": 0.10},
        "dynamic_classes": set(CONFUSION_GROUPS["vehicle"] + CONFUSION_GROUPS["vru"] + CONFUSION_GROUPS["roadside"]),
        "dynamic_min_voxels": 80,
        "dynamic_num_objects_min": 2,
        "dynamic_num_objects_max": 3,
        "bbox_expand_voxels": 2,
        "support_mode": "completed_instance",
        "mapping_mode": "group",
        "static_groups": {"ground", "structure"},
        "static_group_probs": {"ground": 0.5, "structure": 0.5},
        "static_num_regions_min": 2,
        "static_num_regions_max": 2,
        "static_min_voxels": {"ground": 340, "structure": 280},
        "static_half_size_voxels": [11, 11, 4],
    },
    "hard": {
        "dynamic_groups": {"vehicle", "vru", "roadside"},
        "dynamic_group_probs": {"vehicle": 0.55, "vru": 0.30, "roadside": 0.15},
        "dynamic_classes": set(CONFUSION_GROUPS["vehicle"] + CONFUSION_GROUPS["vru"] + CONFUSION_GROUPS["roadside"]),
        "dynamic_min_voxels": 36,
        "dynamic_num_objects_min": 3,
        "dynamic_num_objects_max": 5,
        "bbox_expand_voxels": 2,
        "support_mode": "completed_instance",
        "mapping_mode": "group",
        "group_caps": {"vehicle": 3},
        "require_non_vehicle_if_possible": True,
        "static_groups": {"ground", "structure"},
        "static_group_probs": {"ground": 0.45, "structure": 0.55},
        "static_num_regions_min": 2,
        "static_num_regions_max": 3,
        "static_min_voxels": {"ground": 420, "structure": 340},
        "static_half_size_voxels": [15, 15, 5],
    },
}


@dataclass
class AnnotationCandidate:
    ann_token: str
    instance_token: str
    category_name: str
    occ_class: str
    voxel_count: int
    translation: list
    size: list
    rotation: list


def candidate_group(candidate):
    return CLASS_TO_GROUP[candidate.occ_class]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the semantic subset in frame-level storage format.")
    parser.add_argument("--severity", type=str, choices=sorted(SEVERITY_CONFIGS), required=True)
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Shared OccStress root; defaults to OCCSTRESS_DATA_ROOT.")
    parser.add_argument("--scene-name", type=str, default="", help="Optional scene name filter, e.g. scene-0001.")
    parser.add_argument("--sample-token", type=str, default="", help="Optional single sample token override.")
    parser.add_argument("--sample-token-file", type=str, default="",
                        help="Optional txt/json file listing sample tokens to process.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional count of successfully generated samples.")
    parser.add_argument(
        "--realization-seed",
        type=int,
        default=None,
        help="Optional independent corruption-realization seed. Omit to reproduce the released deterministic data.",
    )
    parser.add_argument("--save-previews", action="store_true", help="Save BEV previews for inspection.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def load_json(root, name):
    with open(os.path.join(root, name), "r") as f:
        return json.load(f)


def load_sample_token_set(path):
    if not path:
        return None
    with open(path, "r") as f:
        if path.endswith(".json"):
            data = json.load(f)
            if isinstance(data, dict):
                if "sample_tokens" in data:
                    data = data["sample_tokens"]
                else:
                    raise ValueError(f"Unsupported sample token json schema: {path}")
            return {str(x) for x in data}
        return {line.strip() for line in f if line.strip()}


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


def quat_to_rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(translation, rotation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_to_rot(rotation)
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def invert_transform(t):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def get_voxel_centers(indices):
    return PC_RANGE[:3] + (indices.astype(np.float32) + 0.5) * VOXEL_SIZE


def nusc_category_to_occ(category_name):
    if category_name == "vehicle.car":
        return "car"
    if category_name == "vehicle.truck":
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name == "vehicle.trailer":
        return "trailer"
    if category_name == "vehicle.construction":
        return "construction_vehicle"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name == "movable_object.barrier":
        return "barrier"
    if category_name == "movable_object.trafficcone":
        return "traffic_cone"
    return None


def box_mask_from_annotation(indices_xyz, ann, global_to_state, expand_voxels=0):
    centers = get_voxel_centers(indices_xyz)
    ann_global = make_transform(ann["translation"], ann["rotation"])
    ann_state = global_to_state @ ann_global
    rot = ann_state[:3, :3]
    center = ann_state[:3, 3]
    half_extent = np.array([ann["size"][1], ann["size"][0], ann["size"][2]], dtype=np.float32) / 2.0
    if expand_voxels > 0:
        half_extent = half_extent + VOXEL_SIZE * float(expand_voxels)
    local = (centers - center[None, :]) @ rot
    return np.all(np.abs(local) <= (half_extent[None, :] * 1.05), axis=1)


def infer_completed_instance_support(semantics, inside_indices, source_id):
    static_ids = {
        CLASS_TO_ID["driveable_surface"],
        CLASS_TO_ID["other_flat"],
        CLASS_TO_ID["sidewalk"],
        CLASS_TO_ID["terrain"],
        CLASS_TO_ID["manmade"],
        CLASS_TO_ID["vegetation"],
        CLASS_TO_ID["free"],
    }
    inside_values = semantics[inside_indices[:, 0], inside_indices[:, 1], inside_indices[:, 2]]
    seeds = inside_indices[inside_values == source_id]
    if len(seeds) == 0:
        return np.empty((0, 3), dtype=np.int64)
    candidates = inside_indices[np.array([v not in static_ids for v in inside_values], dtype=bool)]
    candidate_set = {tuple(idx.tolist()) for idx in candidates}
    queue = collections.deque(tuple(idx.tolist()) for idx in seeds if tuple(idx.tolist()) in candidate_set)
    visited = set(queue)
    neighbors = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int64,
    )
    while queue:
        curr = np.array(queue.popleft(), dtype=np.int64)
        for delta in neighbors:
            nxt = tuple((curr + delta).tolist())
            if nxt in candidate_set and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return np.array(sorted(visited), dtype=np.int64)


def connected_component(indices, seed_idx):
    idx_set = {tuple(idx.tolist()) for idx in indices}
    queue = collections.deque([tuple(seed_idx.tolist())])
    visited = set(queue)
    neighbors = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int64,
    )
    while queue:
        curr = np.array(queue.popleft(), dtype=np.int64)
        for delta in neighbors:
            nxt = tuple((curr + delta).tolist())
            if nxt in idx_set and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return np.array(sorted(visited), dtype=np.int64)


def pick_target_class(source_class, severity, mapping_mode):
    if mapping_mode == "nearest":
        candidates = EASY_NEIGHBORS[source_class]
    else:
        candidates = [x for x in CONFUSION_GROUPS[CLASS_TO_GROUP[source_class]] if x != source_class]
    rng = np.random.default_rng(stable_seed("semantic_target", severity, source_class))
    return candidates[int(rng.integers(0, len(candidates)))]


def pick_static_target_class(source_class, severity):
    group = STATIC_GROUPS[STATIC_CLASS_TO_GROUP[source_class]]
    candidates = [x for x in group if x != source_class]
    rng = np.random.default_rng(stable_seed("semantic_static_target", severity, source_class))
    return candidates[int(rng.integers(0, len(candidates)))]


def choose_count(sample_token, severity, low, high, tag):
    if low == high:
        return low
    rng = np.random.default_rng(stable_seed(tag, severity, sample_token))
    return int(rng.integers(low, high + 1))


def weighted_group_choice(sample_token, severity, step_idx, available_groups, group_probs, tag):
    weights = np.array([group_probs[g] for g in available_groups], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    rng = np.random.default_rng(stable_seed(tag, severity, sample_token, step_idx))
    idx = int(rng.choice(len(available_groups), p=weights))
    return available_groups[idx]


def build_preview(semantics):
    valid = semantics != CLASS_TO_ID["free"]
    z = np.arange(semantics.shape[2], dtype=np.float32).reshape(1, 1, -1)
    score = valid.astype(np.float32) * (z + 1.0)
    selected = np.argmax(score, axis=2)
    bev = np.take_along_axis(semantics, selected[:, :, None], axis=2).squeeze(-1)
    return OCC3D_COLOR_MAP[bev][::-1, ::-1]


def save_preview(clean_semantics, corrupted_semantics, changed_mask, out_path):
    def resize_vis(img, scale=4):
        pil = Image.fromarray(img)
        return np.array(pil.resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST))

    clean = resize_vis(build_preview(clean_semantics))
    corrupted = resize_vis(build_preview(corrupted_semantics))
    bev = changed_mask.any(axis=2).astype(np.uint8)
    changed = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.uint8)
    changed[bev == 1] = np.array([255, 0, 0], dtype=np.uint8)
    changed = resize_vis(changed[::-1, ::-1])
    gap = np.full((clean.shape[0], 16, 3), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([clean, gap, corrupted, gap, changed], axis=1)).save(out_path)


def collect_indices(nusc_root):
    sample_rows = load_json(nusc_root, "sample.json")
    scene_rows = load_json(nusc_root, "scene.json")
    sample_data_rows = load_json(nusc_root, "sample_data.json")
    ego_pose_rows = load_json(nusc_root, "ego_pose.json")
    instance_rows = load_json(nusc_root, "instance.json")
    category_rows = load_json(nusc_root, "category.json")
    ann_rows = load_json(nusc_root, "sample_annotation.json")

    scene_by_token = {row["token"]: row for row in scene_rows}
    ego_pose_by_token = {row["token"]: row for row in ego_pose_rows}
    instance_by_token = {row["token"]: row for row in instance_rows}
    category_by_token = {row["token"]: row for row in category_rows}
    anns_by_sample = collections.defaultdict(list)
    for ann in ann_rows:
        anns_by_sample[ann["sample_token"]].append(ann)

    lidar_sd_by_sample = {}
    for row in sample_data_rows:
        if row["is_key_frame"] and "LIDAR_TOP" in row["filename"] and row["sample_token"] not in lidar_sd_by_sample:
            lidar_sd_by_sample[row["sample_token"]] = row

    return {
        "samples": sample_rows,
        "scene_by_token": scene_by_token,
        "ego_pose_by_token": ego_pose_by_token,
        "instance_by_token": instance_by_token,
        "category_by_token": category_by_token,
        "anns_by_sample": anns_by_sample,
        "lidar_sd_by_sample": lidar_sd_by_sample,
    }


def build_candidates(sample_token, scene_name, semantics, metadata, global_to_state, severity, candidate_classes=None, min_voxels=None):
    config = SEVERITY_CONFIGS[severity]
    candidate_classes = candidate_classes if candidate_classes is not None else config["dynamic_classes"]
    min_voxels = int(min_voxels if min_voxels is not None else config["dynamic_min_voxels"])
    candidates = []
    anns = metadata["anns_by_sample"].get(sample_token, [])
    for ann in anns:
        inst = metadata["instance_by_token"][ann["instance_token"]]
        category_name = metadata["category_by_token"][inst["category_token"]]["name"]
        occ_class = nusc_category_to_occ(category_name)
        if occ_class is None or occ_class not in candidate_classes:
            continue
        occ_id = CLASS_TO_ID[occ_class]
        source_indices = np.argwhere(semantics == occ_id)
        if len(source_indices) == 0:
            continue
        inside = box_mask_from_annotation(source_indices, ann, global_to_state)
        if int(inside.sum()) < min_voxels:
            continue
        candidates.append(
            AnnotationCandidate(
                ann_token=ann["token"],
                instance_token=ann["instance_token"],
                category_name=category_name,
                occ_class=occ_class,
                voxel_count=int(inside.sum()),
                translation=ann["translation"],
                size=ann["size"],
                rotation=ann["rotation"],
            )
        )
    candidates.sort(key=lambda x: (-x.voxel_count, x.instance_token, x.ann_token))
    return candidates


def build_support(semantics, candidate, global_to_state, config):
    source_id = CLASS_TO_ID[candidate.occ_class]
    ann = {
        "translation": candidate.translation,
        "size": candidate.size,
        "rotation": candidate.rotation,
    }
    source_indices = np.argwhere(semantics == source_id)
    inside = box_mask_from_annotation(source_indices, ann, global_to_state)
    eligible = source_indices[inside]
    if config["support_mode"] == "completed_instance":
        nonfree_indices = np.argwhere(semantics != CLASS_TO_ID["free"])
        inside_nonfree = nonfree_indices[box_mask_from_annotation(nonfree_indices, ann, global_to_state)]
        completed = infer_completed_instance_support(semantics, inside_nonfree, source_id)
        if len(completed) >= len(eligible):
            eligible = completed
    if config["bbox_expand_voxels"] > 0:
        expanded_inside = box_mask_from_annotation(
            source_indices,
            ann,
            global_to_state,
            expand_voxels=config["bbox_expand_voxels"],
        )
        expanded_source = source_indices[expanded_inside]
        if len(expanded_source) > 0:
            eligible = np.unique(np.concatenate([eligible, expanded_source], axis=0), axis=0)
    return eligible


def choose_static_region(sample_token, severity, semantics, changed_mask, config, min_voxels_scale=1.0, half_size_override=None, overlap_thresh=0.05):
    available_by_group = collections.defaultdict(list)
    half_size = np.asarray(half_size_override if half_size_override is not None else config["static_half_size_voxels"], dtype=np.int64)

    for group_name in sorted(config["static_groups"]):
        min_voxels = max(24, int(round(config["static_min_voxels"][group_name] * float(min_voxels_scale))))
        for source_class in STATIC_GROUPS[group_name]:
            source_id = CLASS_TO_ID[source_class]
            source_indices = np.argwhere(semantics == source_id)
            if len(source_indices) < min_voxels:
                continue
            seed_order = np.arange(len(source_indices), dtype=np.int64)
            rng = np.random.default_rng(stable_seed("semantic_static_seed_order", severity, sample_token, group_name, source_class))
            rng.shuffle(seed_order)
            for seed_pos in seed_order[: min(96, len(seed_order))]:
                seed_idx = source_indices[int(seed_pos)]
                inside = np.all(np.abs(source_indices - seed_idx[None, :]) <= half_size[None, :], axis=1)
                support = source_indices[inside]
                if len(support) < min_voxels:
                    continue
                support = connected_component(support, seed_idx)
                if len(support) < min_voxels:
                    continue
                overlap = changed_mask[support[:, 0], support[:, 1], support[:, 2]].mean() if len(support) > 0 else 0.0
                if overlap > overlap_thresh:
                    continue
                available_by_group[group_name].append(
                    {
                        "group": group_name,
                        "source_class": source_class,
                        "support": support,
                        "seed_idx": seed_idx,
                        "half_size": half_size.copy(),
                        "score": int(len(support)),
                    }
                )
                break

    for group_name in available_by_group:
        available_by_group[group_name].sort(key=lambda x: (-x["score"], x["source_class"]))

    if not available_by_group:
        return None

    available_groups = sorted(available_by_group.keys())
    chosen_group = weighted_group_choice(
        sample_token,
        severity,
        "static_group",
        available_groups,
        config["static_group_probs"],
        "semantic_static_group_choice",
    )
    return available_by_group[chosen_group][0]


def process_sample(sample, metadata, args, summary):
    severity = args.severity
    config = SEVERITY_CONFIGS[severity]
    scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
    occ_path = os.path.join(args.data_root, "gts", scene_name, sample["token"], "labels.npz")
    if not os.path.exists(occ_path):
        return False
    lidar_sd = metadata["lidar_sd_by_sample"].get(sample["token"])
    if lidar_sd is None:
        return False

    scene_out_dir = manual_output(args.output_root, "occ", "semantic", severity, scene_name, sample["token"])
    event_path = manual_output(args.output_root, "events", "semantic", severity, scene_name, f"{sample['token']}.json")
    if not args.overwrite and os.path.exists(os.path.join(scene_out_dir, "labels.npz")) and os.path.exists(event_path):
        return False

    occ = np.load(occ_path)
    semantics = occ["semantics"].copy()
    original = semantics.copy()

    ego_pose = metadata["ego_pose_by_token"][lidar_sd["ego_pose_token"]]
    global_to_state = invert_transform(make_transform(ego_pose["translation"], ego_pose["rotation"]))
    candidates = build_candidates(sample["token"], scene_name, semantics, metadata, global_to_state, severity)

    num_objects = choose_count(
        sample["token"],
        severity,
        config["dynamic_num_objects_min"],
        config["dynamic_num_objects_max"],
        "semantic_num_dynamic_objects",
    )
    num_regions = choose_count(
        sample["token"],
        severity,
        config["static_num_regions_min"],
        config["static_num_regions_max"],
        "semantic_num_static_regions",
    )

    selected_candidates = []
    changed_mask = np.zeros_like(semantics, dtype=bool)
    object_events = []
    region_events = []
    group_counts = collections.Counter()
    available_by_group = collections.defaultdict(list)
    for candidate in candidates:
        available_by_group[candidate_group(candidate)].append(candidate)

    def try_pick_from_group(group_name):
        for candidate in available_by_group.get(group_name, []):
            if any(candidate.instance_token == prev.instance_token for prev in selected_candidates):
                continue
            support = build_support(semantics, candidate, global_to_state, config)
            if len(support) < config["dynamic_min_voxels"]:
                continue
            overlap = changed_mask[support[:, 0], support[:, 1], support[:, 2]].mean() if len(support) > 0 else 0.0
            if overlap > 0.05:
                continue
            return candidate, support
        return None, None

    target_groups = []
    require_non_vehicle = config.get("require_non_vehicle_if_possible", False)
    if require_non_vehicle:
        non_vehicle_groups = [g for g in available_by_group.keys() if g != "vehicle" and len(available_by_group[g]) > 0]
        if non_vehicle_groups:
            forced_group = weighted_group_choice(
                sample["token"],
                severity,
                "forced_non_vehicle",
                sorted(non_vehicle_groups),
                config["dynamic_group_probs"],
                "semantic_dynamic_group_choice",
            )
            target_groups.append(forced_group)

    while len(target_groups) < num_objects:
        available_groups = []
        for group_name, group_candidates in available_by_group.items():
            if not group_candidates:
                continue
            cap = config.get("group_caps", {}).get(group_name)
            if cap is not None and group_counts[group_name] >= cap:
                continue
            available_groups.append(group_name)
        if not available_groups:
            break
        chosen_group = weighted_group_choice(
            sample["token"],
            severity,
            len(target_groups),
            sorted(available_groups),
            config["dynamic_group_probs"],
            "semantic_dynamic_group_choice",
        )
        target_groups.append(chosen_group)

    for group_name in target_groups:
        candidate, support = try_pick_from_group(group_name)
        if candidate is None:
            fallback_groups = [g for g in sorted(available_by_group.keys()) if g != group_name]
            for fallback_group in fallback_groups:
                cap = config.get("group_caps", {}).get(fallback_group)
                if cap is not None and group_counts[fallback_group] >= cap:
                    continue
                candidate, support = try_pick_from_group(fallback_group)
                if candidate is not None:
                    group_name = fallback_group
                    break
        if candidate is None:
            continue
        target_class = pick_target_class(candidate.occ_class, severity, config["mapping_mode"])
        target_id = CLASS_TO_ID[target_class]
        semantics[support[:, 0], support[:, 1], support[:, 2]] = target_id
        changed_mask[support[:, 0], support[:, 1], support[:, 2]] = True
        selected_candidates.append(candidate)
        group_counts[group_name] += 1
        object_events.append(
            {
                "branch": "instance_aware",
                "ann_token": candidate.ann_token,
                "instance_token": candidate.instance_token,
                "nusc_category_name": candidate.category_name,
                "source_class": candidate.occ_class,
                "target_class": target_class,
                "group": CLASS_TO_GROUP[candidate.occ_class],
                "support_voxels": int(len(support)),
            }
        )

    if not object_events:
        fallback_config = dict(config)
        fallback_config["dynamic_classes"] = set(CONFUSION_GROUPS["vehicle"] + CONFUSION_GROUPS["vru"] + CONFUSION_GROUPS["roadside"])
        fallback_config["dynamic_min_voxels"] = max(16, config["dynamic_min_voxels"] // 4)
        fallback_config["bbox_expand_voxels"] = max(2, config["bbox_expand_voxels"])
        fallback_config["mapping_mode"] = "group"
        fallback_candidates = build_candidates(
            sample["token"],
            scene_name,
            semantics,
            metadata,
            global_to_state,
            severity,
            candidate_classes=fallback_config["dynamic_classes"],
            min_voxels=fallback_config["dynamic_min_voxels"],
        )
        for candidate in fallback_candidates:
            support = build_support(semantics, candidate, global_to_state, fallback_config)
            if len(support) < fallback_config["dynamic_min_voxels"]:
                continue
            overlap = changed_mask[support[:, 0], support[:, 1], support[:, 2]].mean() if len(support) > 0 else 0.0
            if overlap > 0.2:
                continue
            target_class = pick_target_class(candidate.occ_class, severity, fallback_config["mapping_mode"])
            target_id = CLASS_TO_ID[target_class]
            semantics[support[:, 0], support[:, 1], support[:, 2]] = target_id
            changed_mask[support[:, 0], support[:, 1], support[:, 2]] = True
            object_events.append(
                {
                    "branch": "instance_aware",
                    "ann_token": candidate.ann_token,
                    "instance_token": candidate.instance_token,
                    "nusc_category_name": candidate.category_name,
                    "source_class": candidate.occ_class,
                    "target_class": target_class,
                    "group": CLASS_TO_GROUP[candidate.occ_class],
                    "support_voxels": int(len(support)),
                    "fallback": True,
                }
            )
            break

    for _ in range(num_regions):
        region = choose_static_region(sample["token"], severity, semantics, changed_mask, config)
        if region is None:
            break
        target_class = pick_static_target_class(region["source_class"], severity)
        target_id = CLASS_TO_ID[target_class]
        support = region["support"]
        semantics[support[:, 0], support[:, 1], support[:, 2]] = target_id
        changed_mask[support[:, 0], support[:, 1], support[:, 2]] = True
        region_events.append(
            {
                "branch": "static_region",
                "group": region["group"],
                "source_class": region["source_class"],
                "target_class": target_class,
                "support_voxels": int(len(support)),
                "seed_index_xyz": region["seed_idx"].tolist(),
                "half_size_voxels": region["half_size"].tolist(),
            }
        )

    if not region_events:
        fallback_specs = [
            (0.5, None, 0.2),
            (0.25, [4, 4, 2], 0.3),
        ]
        for scale, half_size_override, overlap_thresh in fallback_specs:
            region = choose_static_region(
                sample["token"],
                severity,
                semantics,
                changed_mask,
                config,
                min_voxels_scale=scale,
                half_size_override=half_size_override,
                overlap_thresh=overlap_thresh,
            )
            if region is None:
                continue
            target_class = pick_static_target_class(region["source_class"], severity)
            target_id = CLASS_TO_ID[target_class]
            support = region["support"]
            semantics[support[:, 0], support[:, 1], support[:, 2]] = target_id
            changed_mask[support[:, 0], support[:, 1], support[:, 2]] = True
            region_events.append(
                {
                    "branch": "static_region",
                    "group": region["group"],
                    "source_class": region["source_class"],
                    "target_class": target_class,
                    "support_voxels": int(len(support)),
                    "seed_index_xyz": region["seed_idx"].tolist(),
                    "half_size_voxels": region["half_size"].tolist(),
                    "fallback": True,
                }
            )
            break

    if not object_events and not region_events:
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
        "type": "semantic_hybrid",
        "severity": severity,
        "sample_token": sample["token"],
        "scene_name": scene_name,
        "support_mode": config["support_mode"],
        "bbox_expand_voxels": config["bbox_expand_voxels"],
        "dynamic_group_probs": config["dynamic_group_probs"],
        "static_group_probs": config["static_group_probs"],
        "num_dynamic_objects_requested": num_objects,
        "num_dynamic_objects_applied": len(object_events),
        "num_static_regions_requested": num_regions,
        "num_static_regions_applied": len(region_events),
        "changed_voxels_total": int(changed_mask.sum()),
        "objects": object_events,
        "regions": region_events,
        "occ_path_clean": clean_nuscenes_reference(scene_name, sample["token"]),
        "occ_path_corrupt": portable_reference(scene_out_dir / "labels.npz", args.output_root),
    }
    if args.realization_seed is not None:
        event["realization_seed"] = args.realization_seed
    with open(event_path, "w") as f:
        json.dump(event, f, indent=2)

    if args.save_previews:
        save_preview(original, semantics, changed_mask, os.path.join(scene_out_dir, "preview_bev.png"))

    summary["processed"] += 1
    summary["changed_voxels_total"] += int(changed_mask.sum())
    summary["object_events_total"] += len(object_events)
    summary["static_region_events_total"] += len(region_events)
    return True


def iter_samples(metadata, args):
    token_set = load_sample_token_set(args.sample_token_file)
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
    metadata = collect_indices(nusc_root)
    summary = {
        "severity": args.severity,
        "processed": 0,
        "changed_voxels_total": 0,
        "object_events_total": 0,
        "static_region_events_total": 0,
    }
    if args.realization_seed is not None:
        summary["realization_seed"] = args.realization_seed

    for sample in iter_samples(metadata, args):
        process_sample(sample, metadata, args, summary)
        if args.max_samples > 0 and summary["processed"] >= args.max_samples:
            break

    meta_dir = manual_meta(args.output_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / f"semantic_{args.severity}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
