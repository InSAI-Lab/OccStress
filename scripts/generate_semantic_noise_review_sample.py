#!/usr/bin/env python3
import argparse
import collections
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import generate_semantic_noise_subset as subset
import semantic_noise_sample as demo


STATIC_GROUPS = {
    "ground": ["driveable_surface", "other_flat", "sidewalk", "terrain"],
    "structure": ["manmade", "vegetation", "others"],
}

STATIC_CLASS_TO_GROUP = {
    cls_name: group_name
    for group_name, group in STATIC_GROUPS.items()
    for cls_name in group
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one hybrid semantic-noise review sample.")
    parser.add_argument(
        "--sample-token",
        type=str,
        default="39b0988d06844e6990fdda1e58b6c0d9",
        help="nuScenes sample token for the review sample.",
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
        default="outputs/semantic_review_sample",
        help="Output directory for the review sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic seed for review sample selection.",
    )
    parser.add_argument(
        "--static-group",
        type=str,
        default="structure",
        choices=["ground", "structure", "auto"],
        help="Which static group to prefer for the local semantic region.",
    )
    parser.add_argument(
        "--static-half-size",
        type=int,
        nargs=3,
        default=[8, 8, 2],
        metavar=("HX", "HY", "HZ"),
        help="Half-size of the static cuboid in voxels.",
    )
    parser.add_argument(
        "--min-static-voxels",
        type=int,
        default=180,
        help="Minimum support size for the static local region.",
    )
    return parser.parse_args()


def pick_static_target(source_class, rng):
    group = STATIC_GROUPS[STATIC_CLASS_TO_GROUP[source_class]]
    candidates = [x for x in group if x != source_class]
    return candidates[int(rng.integers(0, len(candidates)))]


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


def choose_static_region(semantics, changed_mask, rng, static_group, half_size, min_static_voxels):
    class_order = []
    if static_group == "auto":
        class_order = STATIC_GROUPS["structure"] + STATIC_GROUPS["ground"]
    else:
        class_order = STATIC_GROUPS[static_group]

    best = None
    for source_class in class_order:
        source_id = subset.CLASS_TO_ID[source_class]
        source_indices = np.argwhere(semantics == source_id)
        if len(source_indices) < min_static_voxels:
            continue

        seed_order = np.arange(len(source_indices), dtype=np.int64)
        rng.shuffle(seed_order)
        seed_order = seed_order[: min(96, len(seed_order))]

        for seed_pos in seed_order:
            seed_idx = source_indices[seed_pos]
            inside = np.all(np.abs(source_indices - seed_idx[None, :]) <= half_size[None, :], axis=1)
            support = source_indices[inside]
            if len(support) == 0:
                continue
            support = connected_component(support, seed_idx)
            if len(support) < min_static_voxels:
                continue

            overlap = changed_mask[support[:, 0], support[:, 1], support[:, 2]].mean()
            if overlap > 0.05:
                continue

            score = int(len(support))
            if best is None or score > best["score"]:
                best = {
                    "source_class": source_class,
                    "group": STATIC_CLASS_TO_GROUP[source_class],
                    "support": support,
                    "seed_idx": seed_idx,
                    "score": score,
                    "half_size": half_size.copy(),
                }
        if best is not None:
            break
    return best


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = subset.collect_indices(nusc_root)
    sample = next(x for x in metadata["samples"] if x["token"] == args.sample_token)
    scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]

    occ_path = os.path.join(args.data_root, "gts", scene_name, args.sample_token, "labels.npz")
    occ = np.load(occ_path)
    original = occ["semantics"].copy()
    semantics = original.copy()

    lidar_sd = metadata["lidar_sd_by_sample"][args.sample_token]
    ego_pose = metadata["ego_pose_by_token"][lidar_sd["ego_pose_token"]]
    global_to_state = subset.invert_transform(subset.make_transform(ego_pose["translation"], ego_pose["rotation"]))

    candidates = subset.build_candidates(args.sample_token, scene_name, semantics, metadata, global_to_state, "hard")
    if not candidates:
        raise RuntimeError("failed to find an object-aware semantic candidate for the review sample")

    object_candidate = max(candidates, key=lambda x: x.voxel_count)
    object_support = subset.build_support(semantics, object_candidate, global_to_state, subset.SEVERITY_CONFIGS["hard"])
    object_target_class = subset.pick_target_class(object_candidate.occ_class, "hard", "group")
    object_target_id = subset.CLASS_TO_ID[object_target_class]

    object_mask = np.zeros_like(semantics, dtype=bool)
    semantics_object_only = semantics.copy()
    semantics_object_only[
        object_support[:, 0],
        object_support[:, 1],
        object_support[:, 2],
    ] = object_target_id
    object_mask[
        object_support[:, 0],
        object_support[:, 1],
        object_support[:, 2],
    ] = True

    static_region = choose_static_region(
        semantics_object_only,
        object_mask,
        rng,
        args.static_group,
        np.asarray(args.static_half_size, dtype=np.int64),
        args.min_static_voxels,
    )
    if static_region is None:
        raise RuntimeError("failed to find a static local semantic region for the review sample")

    static_target_class = pick_static_target(static_region["source_class"], rng)
    static_target_id = subset.CLASS_TO_ID[static_target_class]
    static_mask = np.zeros_like(semantics, dtype=bool)
    semantics_hybrid = semantics_object_only.copy()
    support = static_region["support"]
    semantics_hybrid[support[:, 0], support[:, 1], support[:, 2]] = static_target_id
    static_mask[support[:, 0], support[:, 1], support[:, 2]] = True

    changed_mask = object_mask | static_mask

    sample_out_dir = os.path.join(args.output_dir, args.sample_token)
    os.makedirs(sample_out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(sample_out_dir, "labels.npz"),
        semantics=semantics_hybrid,
        mask_lidar=occ["mask_lidar"],
        mask_camera=occ["mask_camera"],
    )

    event = {
        "type": "semantic_hybrid_review",
        "mode": "single_frame_demo",
        "sample_token": args.sample_token,
        "scene_name": scene_name,
        "seed": args.seed,
        "support_mode": "completed_instance_plus_static_region",
        "object": {
            "branch": "instance_aware",
            "ann_token": object_candidate.ann_token,
            "instance_token": object_candidate.instance_token,
            "nusc_category_name": object_candidate.category_name,
            "source_class": object_candidate.occ_class,
            "target_class": object_target_class,
            "group": subset.CLASS_TO_GROUP[object_candidate.occ_class],
            "support_voxels": int(len(object_support)),
        },
        "regions": [
            {
                "branch": "static_region",
                "group": static_region["group"],
                "source_class": static_region["source_class"],
                "target_class": static_target_class,
                "support_voxels": int(len(static_region["support"])),
                "seed_index_xyz": static_region["seed_idx"].tolist(),
                "half_size_voxels": static_region["half_size"].tolist(),
            }
        ],
        "changed_voxels_total": int(changed_mask.sum()),
        "occ_path": occ_path,
    }
    with open(os.path.join(sample_out_dir, "event.json"), "w") as f:
        json.dump(event, f, indent=2)

    demo.save_preview(original, semantics_hybrid, changed_mask, os.path.join(sample_out_dir, "preview_bev.png"))
    demo.save_zoom_preview(original, semantics_hybrid, changed_mask, os.path.join(sample_out_dir, "preview_zoom.png"))
    demo.save_preview(original, semantics_object_only, object_mask, os.path.join(sample_out_dir, "preview_dynamic_only.png"))

    semantics_static_only = original.copy()
    semantics_static_only[support[:, 0], support[:, 1], support[:, 2]] = static_target_id
    demo.save_preview(original, semantics_static_only, static_mask, os.path.join(sample_out_dir, "preview_static_only.png"))

    print(json.dumps(event, indent=2))
    print(f"saved hybrid review sample to: {sample_out_dir}")


if __name__ == "__main__":
    main()
