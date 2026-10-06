#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Generate deterministic frame-level manual corruption assets for OccStress-Waymo."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage

from waymo_occstress_common import (
    OCC3D_CLASSES, SEVERITIES, atomic_json, map_waymo_semantics,
    mirror_y_semantics, stable_seed,
)


FREE = 17
CONNECTIVITY = ndimage.generate_binary_structure(3, 1)
CONFUSION_GROUPS = {
    "vehicle": (3, 4, 5, 9, 10),
    "vru": (2, 6, 7),
    "roadside": (1, 8),
}
STATIC_GROUPS = {
    "ground": (11, 12, 13, 14),
    "structure": (15, 16, 0),
}
SEMANTIC_CONFIGS = {
    "easy": {
        "dynamic_groups": ("vehicle",),
        "dynamic_group_probs": (1.0,),
        "dynamic_classes": (3, 4, 10),
        "dynamic_min_voxels": 420,
        "dynamic_num_objects": (1, 2),
        "mapping_mode": "nearest",
        "static_group_probs": {"ground": 0.6, "structure": 0.4},
        "static_num_regions": (1, 2),
        "static_min_voxels": {"ground": 260, "structure": 220},
        "static_half_size": (7, 7, 3),
    },
    "mid": {
        "dynamic_groups": ("vehicle", "vru", "roadside"),
        "dynamic_group_probs": (0.75, 0.15, 0.10),
        "dynamic_classes": tuple(
            cls_id for group in ("vehicle", "vru", "roadside")
            for cls_id in CONFUSION_GROUPS[group]),
        "dynamic_min_voxels": 80,
        "dynamic_num_objects": (2, 3),
        "mapping_mode": "group",
        "static_group_probs": {"ground": 0.5, "structure": 0.5},
        "static_num_regions": (2, 2),
        "static_min_voxels": {"ground": 340, "structure": 280},
        "static_half_size": (11, 11, 4),
    },
    "hard": {
        "dynamic_groups": ("vehicle", "vru", "roadside"),
        "dynamic_group_probs": (0.55, 0.30, 0.15),
        "dynamic_classes": tuple(
            cls_id for group in ("vehicle", "vru", "roadside")
            for cls_id in CONFUSION_GROUPS[group]),
        "dynamic_min_voxels": 36,
        "dynamic_num_objects": (3, 5),
        "mapping_mode": "group",
        "static_group_probs": {"ground": 0.45, "structure": 0.55},
        "static_num_regions": (2, 3),
        "static_min_voxels": {"ground": 420, "structure": 340},
        "static_half_size": (15, 15, 5),
    },
}
HOLE_CONFIGS = {
    "easy": {
        "num_objects": (1, 2), "min_support": 240, "min_interior": 24,
        "removed_ratio": (0.58, 0.78),
    },
    "mid": {
        "num_objects": (2, 3), "min_support": 220, "min_interior": 24,
        "removed_ratio": (0.78, 0.94),
    },
    "hard": {
        "num_objects": (4, 6), "min_support": 180, "min_interior": 18,
        "removed_ratio": (0.94, 1.00),
    },
}
DROPOUT_CONFIGS = {
    "easy": {
        "num_regions": (1, 2), "half_min": (6, 6, 2),
        "half_max": (9, 9, 3), "min_removed": 320,
        "min_density": 0.12, "shape": "cuboid",
    },
    "mid": {
        "num_regions": (3, 4), "half_min": (12, 12, 4),
        "half_max": (18, 18, 6), "min_removed": 1200,
        "min_density": 0.07, "shape": "irregular",
        "removed_fraction": (0.70, 0.90), "blob_count": (3, 5),
    },
    "hard": {
        "num_regions": (5, 8), "half_min": (18, 18, 5),
        "half_max": (26, 26, 8), "min_removed": 2500,
        "min_density": 0.04, "shape": "irregular",
        "removed_fraction": (0.90, 1.00), "blob_count": (4, 8),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--task", required=True,
        help="traffic or FAMILY:SEVERITY, e.g. semantic:hard")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scene")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_semantics(path):
    with np.load(path) as labels:
        raw = labels["voxel_label"]
        infov = labels["infov"] if "infov" in labels else np.ones(raw.shape, dtype=bool)
    return map_waymo_semantics(raw), infov


def choose_count(rng, bounds):
    low, high = bounds
    return low if low == high else int(rng.integers(low, high + 1))


def component_candidates(semantics, class_ids, min_voxels):
    candidates = []
    for cls_id in class_ids:
        labels, count = ndimage.label(semantics == cls_id, CONNECTIVITY)
        if not count:
            continue
        sizes = np.bincount(labels.ravel())
        for component_id in np.flatnonzero(sizes >= min_voxels):
            if component_id == 0:
                continue
            coords = np.argwhere(labels == component_id)
            candidates.append((int(cls_id), coords))
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    return candidates


def class_group(cls_id, groups):
    for group, class_ids in groups.items():
        if cls_id in class_ids:
            return group
    return None


def semantic_target(cls_id, severity, mapping_mode, rng):
    if mapping_mode == "nearest":
        nearest = {3: (10,), 4: (10,), 10: (4, 3)}
        choices = nearest.get(cls_id)
        if choices:
            return int(choices[int(rng.integers(0, len(choices)))])
    group = class_group(cls_id, CONFUSION_GROUPS)
    if group is None:
        group = class_group(cls_id, STATIC_GROUPS)
        choices = STATIC_GROUPS[group] if group else (0,)
    else:
        choices = CONFUSION_GROUPS[group]
    choices = tuple(target for target in choices if target != cls_id)
    return int(choices[int(rng.integers(0, len(choices)))])


def choose_static_region(semantics, changed, config, rng):
    available = []
    half = np.asarray(config["static_half_size"], dtype=np.int64)
    shape = np.asarray(semantics.shape, dtype=np.int64)
    for group, class_ids in STATIC_GROUPS.items():
        minimum = config["static_min_voxels"][group]
        for cls_id in class_ids:
            source = np.argwhere(semantics == cls_id)
            if len(source) < minimum:
                continue
            order = rng.permutation(len(source))[:min(96, len(source))]
            for position in order:
                center = source[int(position)]
                low = np.maximum(center - half, 0)
                high = np.minimum(center + half + 1, shape)
                view = semantics[
                    low[0]:high[0], low[1]:high[1], low[2]:high[2]]
                labels, _ = ndimage.label(view == cls_id, CONNECTIVITY)
                local_center = center - low
                component_id = labels[tuple(local_center)]
                if component_id == 0:
                    continue
                relative = np.argwhere(labels == component_id)
                support = relative + low
                if len(support) < minimum:
                    continue
                overlap = changed[
                    support[:, 0], support[:, 1], support[:, 2]].mean()
                if overlap <= 0.05:
                    available.append((group, cls_id, center, support))
                    break
    if not available:
        return None
    groups = sorted({item[0] for item in available})
    weights = np.asarray(
        [config["static_group_probs"][group] for group in groups],
        dtype=np.float64)
    weights /= weights.sum()
    group = groups[int(rng.choice(len(groups), p=weights))]
    matches = [item for item in available if item[0] == group]
    return matches[int(rng.integers(0, len(matches)))]


def semantic_local_fallback(semantics, config, rng):
    occupied = np.argwhere(semantics != FREE)
    if not len(occupied):
        return None
    shape = np.asarray(semantics.shape)
    half = np.maximum(
        np.asarray(config["static_half_size"], dtype=np.int64) // 2, 1)
    order = rng.permutation(len(occupied))[:min(256, len(occupied))]
    for position in order:
        center = occupied[int(position)]
        cls_id = int(semantics[tuple(center)])
        low = np.maximum(center - half, 0)
        high = np.minimum(center + half + 1, shape)
        view = semantics[
            low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        labels, _ = ndimage.label(view == cls_id, CONNECTIVITY)
        component_id = labels[tuple(center - low)]
        support = np.argwhere(labels == component_id) + low
        if len(support) >= 24:
            return cls_id, center, support
    center = occupied[int(order[0])]
    return int(semantics[tuple(center)]), center, center[None]


def semantic_corruption(semantics, severity, token):
    config = SEMANTIC_CONFIGS[severity]
    output = semantics.copy()
    rng = np.random.default_rng(stable_seed("semantic", severity, token))
    changed = np.zeros(output.shape, dtype=bool)
    events = []

    candidates = component_candidates(
        output, config["dynamic_classes"], config["dynamic_min_voxels"])
    rng.shuffle(candidates)
    requested = choose_count(rng, config["dynamic_num_objects"])
    for cls_id, support in candidates[:requested]:
        target = semantic_target(
            cls_id, severity, config["mapping_mode"], rng)
        output[support[:, 0], support[:, 1], support[:, 2]] = target
        changed[support[:, 0], support[:, 1], support[:, 2]] = True
        events.append({
            "branch": "connected_dynamic_component",
            "source_class": OCC3D_CLASSES[cls_id],
            "target_class": OCC3D_CLASSES[target],
            "changed_voxels": int(len(support)),
        })

    for _ in range(choose_count(rng, config["static_num_regions"])):
        region = choose_static_region(output, changed, config, rng)
        if region is None:
            break
        group, cls_id, center, support = region
        target = semantic_target(cls_id, severity, "group", rng)
        output[support[:, 0], support[:, 1], support[:, 2]] = target
        changed[support[:, 0], support[:, 1], support[:, 2]] = True
        events.append({
            "branch": "connected_static_region",
            "group": group,
            "source_class": OCC3D_CLASSES[cls_id],
            "target_class": OCC3D_CLASSES[target],
            "seed_index_xyz": center.tolist(),
            "half_size_voxels": list(config["static_half_size"]),
            "changed_voxels": int(len(support)),
        })
    if not events:
        fallback = semantic_local_fallback(output, config, rng)
        if fallback is not None:
            cls_id, center, support = fallback
            target = semantic_target(cls_id, severity, "group", rng)
            fallback_half = np.maximum(
                np.asarray(config["static_half_size"]) // 2, 1)
            output[support[:, 0], support[:, 1], support[:, 2]] = target
            changed[support[:, 0], support[:, 1], support[:, 2]] = True
            events.append({
                "branch": "local_connected_fallback",
                "source_class": OCC3D_CLASSES[cls_id],
                "target_class": OCC3D_CLASSES[target],
                "seed_index_xyz": center.tolist(),
                "half_size_voxels": fallback_half.tolist(),
                "changed_voxels": int(len(support)),
            })
    return output, changed, events


def component_interior(coords):
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    mask = np.zeros(high - low, dtype=bool)
    relative = coords - low
    mask[relative[:, 0], relative[:, 1], relative[:, 2]] = True
    interior = ndimage.binary_erosion(
        mask, structure=CONNECTIVITY, iterations=1, border_value=0)
    return np.argwhere(interior) + low


def local_hole_fallback(semantics):
    occupied = np.argwhere(semantics != FREE)
    if not len(occupied):
        return None
    scene_center = np.asarray(semantics.shape, dtype=np.float32) / 2
    center = occupied[np.argmin(
        np.sum((occupied - scene_center[None]) ** 2, axis=1))]
    half = np.array([3, 3, 2])
    low = np.maximum(center - half, 0)
    high = np.minimum(center + half + 1, semantics.shape)
    view = semantics[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    support = np.argwhere(view != FREE) + low
    interior = component_interior(support) if len(support) else support
    return support, interior


def hole_corruption(semantics, severity, token):
    config = HOLE_CONFIGS[severity]
    output = semantics.copy()
    changed = np.zeros(output.shape, dtype=bool)
    rng = np.random.default_rng(stable_seed("hole", severity, token))
    candidates = component_candidates(
        output, CONFUSION_GROUPS["vehicle"], config["min_support"])
    candidates = [
        (cls_id, support, component_interior(support))
        for cls_id, support in candidates
    ]
    candidates = [
        candidate for candidate in candidates
        if len(candidate[2]) >= config["min_interior"]
    ]
    if not candidates:
        fallback_min = max(60, config["min_support"] // 3)
        candidates = component_candidates(
            output,
            CONFUSION_GROUPS["vehicle"] + CONFUSION_GROUPS["vru"],
            fallback_min,
        )
        candidates = [
            (cls_id, support, component_interior(support))
            for cls_id, support in candidates
        ]
        candidates = [
            candidate for candidate in candidates if len(candidate[2]) >= 4]
    rng.shuffle(candidates)
    requested = choose_count(rng, config["num_objects"])
    events = []
    for cls_id, support, interior in candidates[:requested]:
        ratio = float(rng.uniform(*config["removed_ratio"]))
        center = interior.astype(np.float32).mean(axis=0)
        distance = np.sum(
            (interior.astype(np.float32) - center[None]) ** 2, axis=1)
        count = max(1, int(round(len(interior) * ratio)))
        selected = interior[np.argsort(distance)[:count]]
        output[selected[:, 0], selected[:, 1], selected[:, 2]] = FREE
        changed[selected[:, 0], selected[:, 1], selected[:, 2]] = True
        events.append({
            "branch": "connected_dynamic_component",
            "source_class": OCC3D_CLASSES[cls_id],
            "support_voxels": int(len(support)),
            "interior_voxels": int(len(interior)),
            "removed_voxels": int(len(selected)),
            "removed_ratio": float(len(selected) / max(1, len(interior))),
        })
    if not events:
        fallback = local_hole_fallback(output)
        if fallback is not None and len(fallback[1]):
            support, interior = fallback
            ratio = float(rng.uniform(*config["removed_ratio"]))
            count = max(1, int(round(len(interior) * ratio)))
            selected = interior[:count]
            output[selected[:, 0], selected[:, 1], selected[:, 2]] = FREE
            changed[selected[:, 0], selected[:, 1], selected[:, 2]] = True
            events.append({
                "branch": "local_occupied_fallback",
                "support_voxels": int(len(support)),
                "interior_voxels": int(len(interior)),
                "removed_voxels": int(len(selected)),
            })
    return output, changed, events


def irregular_blob(removable, low, high, config, rng):
    fraction = float(rng.uniform(*config["removed_fraction"]))
    target = min(
        len(removable),
        max(config["min_removed"], int(round(len(removable) * fraction))),
    )
    local = removable - low
    shape = (high - low).astype(np.float32)
    blobs = choose_count(rng, config["blob_count"])
    scores = np.zeros(len(removable), dtype=np.float32)
    for _ in range(blobs):
        center = shape * rng.uniform(0.2, 0.8, size=3)
        scale = np.maximum(1.0, shape * rng.uniform(0.18, 0.38, size=3))
        norm = ((local.astype(np.float32) - center[None]) / scale[None]) ** 2
        scores = np.maximum(scores, np.exp(-0.5 * norm.sum(axis=1)))
    scores += rng.uniform(0.0, 0.18, size=len(scores))
    return removable[np.argsort(scores)[-target:]]


def dropout_corruption(semantics, severity, token):
    config = DROPOUT_CONFIGS[severity]
    output = semantics.copy()
    changed = np.zeros(output.shape, dtype=bool)
    occupied = np.argwhere(output != FREE)
    if not len(occupied):
        return output, changed, []
    rng = np.random.default_rng(stable_seed("dropout", severity, token))
    order = rng.permutation(len(occupied))
    shape = np.asarray(output.shape)
    events = []
    cursor = 0
    for _ in range(choose_count(rng, config["num_regions"])):
        selected = None
        for offset in range(min(256, len(order) - cursor)):
            center = occupied[order[cursor + offset]]
            half = rng.integers(
                np.asarray(config["half_min"]),
                np.asarray(config["half_max"]) + 1,
            )
            low = np.maximum(center - half, 0)
            high = np.minimum(center + half + 1, shape)
            view = output[
                low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            available = (view != FREE) & (~changed[
                low[0]:high[0], low[1]:high[1], low[2]:high[2]])
            removable = np.argwhere(available) + low
            density = len(removable) / max(1, int(np.prod(high - low)))
            if (len(removable) < config["min_removed"]
                    or density < config["min_density"]):
                continue
            if config["shape"] == "irregular":
                removed = irregular_blob(removable, low, high, config, rng)
            else:
                removed = removable
            selected = (center, half, low, high, removed, density, False)
            cursor += offset + 1
            break
        if selected is None:
            center = occupied[int(rng.integers(0, len(occupied)))]
            half = np.maximum(np.asarray(config["half_min"]) // 2, 1)
            low = np.maximum(center - half, 0)
            high = np.minimum(center + half + 1, shape)
            view = output[
                low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            available = (view != FREE) & (~changed[
                low[0]:high[0], low[1]:high[1], low[2]:high[2]])
            removed = np.argwhere(available) + low
            if not len(removed):
                break
            density = len(removed) / max(1, int(np.prod(high - low)))
            selected = (center, half, low, high, removed, density, True)
        center, half, low, high, removed, density, fallback = selected
        if not len(removed):
            continue
        output[removed[:, 0], removed[:, 1], removed[:, 2]] = FREE
        changed[removed[:, 0], removed[:, 1], removed[:, 2]] = True
        events.append({
            "center_voxel": center.tolist(),
            "half_size_voxels": half.tolist(),
            "bounds": {
                "low_inclusive": low.tolist(),
                "high_exclusive": high.tolist(),
            },
            "removed_voxels": int(len(removed)),
            "occupied_density": float(density),
            "shape_mode": config["shape"],
            "fallback": fallback,
        })
    return output, changed, events


def output_paths(root, task, frame):
    if task == "traffic":
        base = root / "occ" / "manual" / "traffic"
        event = root / "events" / "manual" / "traffic"
    else:
        family, severity = task.split(":")
        base = root / "occ" / "manual" / family / severity
        event = root / "events" / "manual" / family / severity
    return (
        base / frame["scene_name"] / frame["token"] / "labels.npz",
        event / frame["scene_name"] / f"{frame['token']}.json",
    )


def process_one(frame, root, task, overwrite):
    output_path, event_path = output_paths(root, task, frame)
    if output_path.exists() and event_path.exists() and not overwrite:
        event = json.loads(event_path.read_text(encoding="utf-8"))
        return {
            "status": "skipped",
            "token": frame["token"],
            "changed_voxels": int(event["changed_voxels"]),
            "occupied_voxels": int(event.get("occupied_voxels", 0)),
            "normalization_voxels": int(
                event.get("normalization_voxels", 0)),
        }
    semantics, infov = load_semantics(frame["occ_path"])
    occupied_voxels = int((semantics != FREE).sum())
    if task == "traffic":
        output = mirror_y_semantics(semantics)
        changed = output != semantics
        events = [{
            "branch": "mirror_y",
            "axis": 1,
            "changed_voxels": int(changed.sum()),
        }]
    else:
        family, severity = task.split(":")
        if family == "semantic":
            output, changed, events = semantic_corruption(
                semantics, severity, frame["token"])
        elif family == "hole":
            output, changed, events = hole_corruption(
                semantics, severity, frame["token"])
        elif family == "dropout":
            output, changed, events = dropout_corruption(
                semantics, severity, frame["token"])
        else:
            raise ValueError(f"unsupported task: {task}")
    normalization_voxels = int(np.count_nonzero(
        (semantics != FREE) | (output != FREE)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(temporary, semantics=output, infov=infov)
    os.replace(temporary, output_path)
    values, counts = np.unique(output, return_counts=True)
    atomic_json(event_path, {
        "schema_version": 1,
        "dataset": "Occ3D-Waymo",
        "task": task,
        "token": frame["token"],
        "source_path": frame["occ_path"],
        "output_path": str(output_path),
        "changed_voxels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "occupied_voxels": occupied_voxels,
        "changed_fraction_occupied": float(
            changed.sum() / max(1, occupied_voxels)),
        "normalization_voxels": normalization_voxels,
        "changed_fraction_support": float(
            changed.sum() / max(1, normalization_voxels)),
        "adaptation_policy": (
            "6-neighbor voxel components replace nuScenes instance boxes; "
            "all severity ranges otherwise follow OccStress-nuScenes manual track"
        ),
        "events": events,
        "class_histogram": {
            OCC3D_CLASSES[int(value)]: int(count)
            for value, count in zip(values, counts)
        },
    })
    return {
        "status": "success", "token": frame["token"],
        "changed_voxels": int(changed.sum()),
        "occupied_voxels": occupied_voxels,
        "normalization_voxels": normalization_voxels,
    }


def main():
    args = parse_args()
    if args.task != "traffic":
        family, severity = args.task.split(":")
        if family not in {"semantic", "hole", "dropout"} or severity not in SEVERITIES:
            raise ValueError(f"invalid task: {args.task}")
    frames = json.loads(args.frames_json.read_text(encoding="utf-8"))
    if args.scene:
        frames = [frame for frame in frames if frame["scene_name"] == args.scene]
    if args.max_frames:
        frames = frames[:args.max_frames]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one, frame, args.output_root, args.task, args.overwrite)
            for frame in frames
        ]
        for future in as_completed(futures):
            results.append(future.result())
    success = sum(result["status"] == "success" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    changed_counts = [
        result["changed_voxels"] for result in results
        if "changed_voxels" in result
    ]
    occupied_total = sum(
        result.get("occupied_voxels", 0) for result in results)
    normalization_total = sum(
        result.get("normalization_voxels", 0) for result in results)
    changed_total = sum(changed_counts)
    summary = {
        "task": args.task,
        "requested_frames": len(frames),
        "success": success,
        "skipped": skipped,
        "complete": success + skipped,
        "changed_voxels": changed_total,
        "changed_voxels_min": min(changed_counts) if changed_counts else 0,
        "changed_voxels_mean": (
            float(np.mean(changed_counts)) if changed_counts else 0.0),
        "changed_voxels_max": max(changed_counts) if changed_counts else 0,
        "zero_changed_frames": sum(
            count == 0 for count in changed_counts),
        "changed_fraction_occupied": float(
            changed_total / max(1, occupied_total)),
        "changed_fraction_support": float(
            changed_total / max(1, normalization_total)),
    }
    slug = args.task.replace(":", "_")
    atomic_json(args.output_root / "meta" / "manual" / f"{slug}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
