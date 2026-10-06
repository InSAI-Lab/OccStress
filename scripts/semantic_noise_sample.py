#!/usr/bin/env python3
import argparse
import collections
import json
import os
import random
from dataclasses import dataclass

import numpy as np
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

CLASS_TO_GROUP = {
    cls_name: group_name
    for group_name, group in CONFUSION_GROUPS.items()
    for cls_name in group
}

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

PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=np.float32)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4], dtype=np.float32)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one object-aware semantic noise sample.")
    parser.add_argument(
        "--sample-token",
        type=str,
        default="e93e98b63d3b40209056d129dc53ceee",
        help="nuScenes sample token.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/nuscenes",
        help="nuScenes root containing gts and v1.0-trainval.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/semantic_sample",
        help="Where to write the corrupted sample and preview files.",
    )
    parser.add_argument(
        "--relabel-ratio",
        type=float,
        default=0.6,
        help="Ratio of eligible voxels to relabel inside the selected instance support.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="",
        help="Optional target class. If unset, the script samples one from the same semantic group.",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=20,
        help="Minimum matched voxel count for a candidate instance.",
    )
    parser.add_argument(
        "--support-mode",
        type=str,
        default="completed_instance",
        choices=["semantic_only", "completed_instance"],
        help="How to define the object support to relabel.",
    )
    parser.add_argument(
        "--bbox-expand-voxels",
        type=int,
        default=0,
        help="Expand the annotation box by N voxels and include added source-class voxels.",
    )
    return parser.parse_args()


def load_json(root, name):
    with open(os.path.join(root, name), "r") as f:
        return json.load(f)


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


def get_voxel_centers(indices):
    centers = PC_RANGE[:3] + (indices.astype(np.float32) + 0.5) * VOXEL_SIZE
    return centers


def box_mask_from_annotation(indices_xyz, ann, global_to_state, expand_voxels=0):
    centers = get_voxel_centers(indices_xyz)
    ann_global = make_transform(ann["translation"], ann["rotation"])
    ann_state = global_to_state @ ann_global

    rot = ann_state[:3, :3]
    center = ann_state[:3, 3]

    # nuScenes size is [width, length, height]. Convert to lidar local extents [x, y, z].
    half_extent = np.array([ann["size"][1], ann["size"][0], ann["size"][2]], dtype=np.float32) / 2.0
    if expand_voxels > 0:
        half_extent = half_extent + VOXEL_SIZE * float(expand_voxels)
    local = (centers - center[None, :]) @ rot
    inside = np.all(np.abs(local) <= (half_extent[None, :] * 1.05), axis=1)
    return inside


def bev_vis(semantics):
    valid = semantics != CLASS_TO_ID["free"]
    z = np.arange(semantics.shape[2], dtype=np.float32).reshape(1, 1, -1)
    score = valid.astype(np.float32) * (z + 1.0)
    selected = np.argmax(score, axis=2)
    bev = np.take_along_axis(semantics, selected[:, :, None], axis=2).squeeze(-1)
    img = OCC3D_COLOR_MAP[bev]
    return img[::-1, ::-1]


def changed_vis(changed_mask):
    bev = changed_mask.any(axis=2).astype(np.uint8)
    img = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.uint8)
    img[bev == 1] = np.array([255, 0, 0], dtype=np.uint8)
    return img[::-1, ::-1]


def resize_vis(img, scale=4):
    pil = Image.fromarray(img)
    return np.array(pil.resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST))


def save_preview(clean_semantics, corrupted_semantics, changed_mask, out_path):
    clean = resize_vis(bev_vis(clean_semantics))
    corrupted = resize_vis(bev_vis(corrupted_semantics))
    changed = resize_vis(changed_vis(changed_mask))
    gap = np.full((clean.shape[0], 16, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([clean, gap, corrupted, gap, changed], axis=1)
    Image.fromarray(canvas).save(out_path)


def save_zoom_preview(clean_semantics, corrupted_semantics, changed_mask, out_path, margin=12):
    footprint = changed_mask.any(axis=2)
    ys, xs = np.where(footprint)
    if len(xs) == 0:
        return

    x0 = max(0, xs.min() - margin)
    x1 = min(footprint.shape[0], xs.max() + margin + 1)
    y0 = max(0, ys.min() - margin)
    y1 = min(footprint.shape[1], ys.max() + margin + 1)

    clean_bev = bev_vis(clean_semantics)[x0:x1, y0:y1]
    corrupted_bev = bev_vis(corrupted_semantics)[x0:x1, y0:y1]
    changed_bev = changed_vis(changed_mask)[x0:x1, y0:y1]

    clean_bev = resize_vis(clean_bev, scale=12)
    corrupted_bev = resize_vis(corrupted_bev, scale=12)
    changed_bev = resize_vis(changed_bev, scale=12)

    gap = np.full((clean_bev.shape[0], 24, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([clean_bev, gap, corrupted_bev, gap, changed_bev], axis=1)
    Image.fromarray(canvas).save(out_path)


def save_object_only_preview(clean_semantics, corrupted_semantics, changed_mask, out_path, margin=8):
    footprint = changed_mask.any(axis=2)
    ys, xs = np.where(footprint)
    if len(xs) == 0:
        return

    x0 = max(0, xs.min() - margin)
    x1 = min(footprint.shape[0], xs.max() + margin + 1)
    y0 = max(0, ys.min() - margin)
    y1 = min(footprint.shape[1], ys.max() + margin + 1)

    clean_bev = bev_vis(clean_semantics)[x0:x1, y0:y1].copy()
    corrupted_bev = bev_vis(corrupted_semantics)[x0:x1, y0:y1].copy()
    object_mask = footprint[x0:x1, y0:y1][::-1, ::-1]

    clean_bev[~object_mask] = 0
    corrupted_bev[~object_mask] = 0

    clean_bev = resize_vis(clean_bev, scale=18)
    corrupted_bev = resize_vis(corrupted_bev, scale=18)
    gap = np.full((clean_bev.shape[0], 24, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([clean_bev, gap, corrupted_bev], axis=1)
    Image.fromarray(canvas).save(out_path)


def pick_target_class(source_class, rng, explicit_target=""):
    group = CONFUSION_GROUPS[CLASS_TO_GROUP[source_class]]
    candidates = [x for x in group if x != source_class]
    if explicit_target:
        if explicit_target not in candidates:
            raise ValueError(f"target class {explicit_target} is not allowed for source {source_class}")
        return explicit_target
    return rng.choice(candidates)


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
    inside_values = semantics[
        inside_indices[:, 0],
        inside_indices[:, 1],
        inside_indices[:, 2],
    ]
    seeds = inside_indices[inside_values == source_id]
    if len(seeds) == 0:
        return np.empty((0, 3), dtype=np.int64)

    candidates = inside_indices[np.array([v not in static_ids for v in inside_values], dtype=bool)]
    candidate_set = {tuple(idx.tolist()) for idx in candidates}
    queue = collections.deque(tuple(idx.tolist()) for idx in seeds if tuple(idx.tolist()) in candidate_set)
    visited = set(queue)
    neighbors = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
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


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")

    sample_rows = load_json(nusc_root, "sample.json")
    scene_rows = load_json(nusc_root, "scene.json")
    sample_data_rows = load_json(nusc_root, "sample_data.json")
    ego_pose_rows = load_json(nusc_root, "ego_pose.json")
    calibrated_sensor_rows = load_json(nusc_root, "calibrated_sensor.json")
    instance_rows = load_json(nusc_root, "instance.json")
    category_rows = load_json(nusc_root, "category.json")
    ann_rows = load_json(nusc_root, "sample_annotation.json")

    sample_by_token = {row["token"]: row for row in sample_rows}
    scene_by_token = {row["token"]: row for row in scene_rows}
    ego_pose_by_token = {row["token"]: row for row in ego_pose_rows}
    calibrated_by_token = {row["token"]: row for row in calibrated_sensor_rows}
    instance_by_token = {row["token"]: row for row in instance_rows}
    category_by_token = {row["token"]: row for row in category_rows}

    sample = sample_by_token[args.sample_token]
    scene_name = scene_by_token[sample["scene_token"]]["name"]
    occ_path = os.path.join(args.data_root, "gts", scene_name, args.sample_token, "labels.npz")
    occ = np.load(occ_path)
    semantics = occ["semantics"].copy()
    original = semantics.copy()

    lidar_sd = None
    for row in sample_data_rows:
        if row["sample_token"] != args.sample_token:
            continue
        if not row["is_key_frame"]:
            continue
        if "LIDAR_TOP" in row["filename"]:
            lidar_sd = row
            break
    if lidar_sd is None:
        raise RuntimeError(f"failed to locate LIDAR_TOP sample_data for {args.sample_token}")

    ego_pose = ego_pose_by_token[lidar_sd["ego_pose_token"]]
    calibrated = calibrated_by_token[lidar_sd["calibrated_sensor_token"]]
    # Occ3D occupancy is aligned more naturally with the ego frame than the lidar frame.
    # For object-aware semantic corruption we therefore map annotations from global to ego.
    global_to_state = invert_transform(make_transform(ego_pose["translation"], ego_pose["rotation"]))

    candidates = []
    for ann in ann_rows:
        if ann["sample_token"] != args.sample_token:
            continue
        inst = instance_by_token[ann["instance_token"]]
        category_name = category_by_token[inst["category_token"]]["name"]
        occ_class = nusc_category_to_occ(category_name)
        if occ_class is None or occ_class not in CLASS_TO_GROUP:
            continue
        occ_id = CLASS_TO_ID[occ_class]
        indices = np.argwhere(semantics == occ_id)
        if len(indices) == 0:
            continue
        inside = box_mask_from_annotation(indices, ann, global_to_state)
        voxel_count = int(inside.sum())
        if voxel_count < args.min_voxels:
            continue
        candidates.append(
            AnnotationCandidate(
                ann_token=ann["token"],
                instance_token=ann["instance_token"],
                category_name=category_name,
                occ_class=occ_class,
                voxel_count=voxel_count,
                translation=ann["translation"],
                size=ann["size"],
                rotation=ann["rotation"],
            )
        )

    if not candidates:
        raise RuntimeError("no eligible object-aware semantic corruption candidate was found for this sample")

    chosen = max(candidates, key=lambda x: x.voxel_count)
    source_id = CLASS_TO_ID[chosen.occ_class]
    target_class = pick_target_class(chosen.occ_class, rng, args.target_class)
    target_id = CLASS_TO_ID[target_class]

    indices = np.argwhere(semantics == source_id)
    ann = {
        "translation": chosen.translation,
        "size": chosen.size,
        "rotation": chosen.rotation,
    }
    inside = box_mask_from_annotation(indices, ann, global_to_state)
    eligible = indices[inside]
    if args.support_mode == "completed_instance":
        nonfree_indices = np.argwhere(semantics != CLASS_TO_ID["free"])
        inside_nonfree = nonfree_indices[box_mask_from_annotation(nonfree_indices, ann, global_to_state)]
        completed = infer_completed_instance_support(semantics, inside_nonfree, source_id)
        if len(completed) >= len(eligible):
            eligible = completed
    if args.bbox_expand_voxels > 0:
        expanded_inside = box_mask_from_annotation(
            indices,
            ann,
            global_to_state,
            expand_voxels=args.bbox_expand_voxels,
        )
        expanded_source = indices[expanded_inside]
        if len(expanded_source) > 0:
            eligible = np.unique(np.concatenate([eligible, expanded_source], axis=0), axis=0)
    num_to_change = max(1, int(round(len(eligible) * args.relabel_ratio)))
    selected = np.array(rng.sample(range(len(eligible)), num_to_change), dtype=np.int64)
    changed_indices = eligible[selected]

    changed_mask = np.zeros_like(semantics, dtype=bool)
    semantics[
        changed_indices[:, 0],
        changed_indices[:, 1],
        changed_indices[:, 2],
    ] = target_id
    changed_mask[
        changed_indices[:, 0],
        changed_indices[:, 1],
        changed_indices[:, 2],
    ] = True

    sample_out_dir = os.path.join(args.output_dir, args.sample_token)
    os.makedirs(sample_out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(sample_out_dir, "labels.npz"),
        semantics=semantics,
        mask_lidar=occ["mask_lidar"],
        mask_camera=occ["mask_camera"],
    )

    event = {
        "type": "semantic_instance",
        "mode": "single_frame_demo",
        "sample_token": args.sample_token,
        "scene_name": scene_name,
        "ann_token": chosen.ann_token,
        "instance_token": chosen.instance_token,
        "nusc_category_name": chosen.category_name,
        "source_class": chosen.occ_class,
        "target_class": target_class,
        "group": CLASS_TO_GROUP[chosen.occ_class],
        "eligible_voxels": int(len(eligible)),
        "changed_voxels": int(len(changed_indices)),
        "relabel_ratio": args.relabel_ratio,
        "support_mode": args.support_mode,
        "bbox_expand_voxels": args.bbox_expand_voxels,
        "seed": args.seed,
        "occ_path": occ_path,
    }
    with open(os.path.join(sample_out_dir, "event.json"), "w") as f:
        json.dump(event, f, indent=2)

    save_preview(
        original,
        semantics,
        changed_mask,
        os.path.join(sample_out_dir, "preview_bev.png"),
    )
    save_zoom_preview(
        original,
        semantics,
        changed_mask,
        os.path.join(sample_out_dir, "preview_zoom.png"),
    )
    save_object_only_preview(
        original,
        semantics,
        changed_mask,
        os.path.join(sample_out_dir, "preview_object_only.png"),
    )

    print(json.dumps(event, indent=2))
    print(f"saved corrupted sample to: {sample_out_dir}")


if __name__ == "__main__":
    main()
