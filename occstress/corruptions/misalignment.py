#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from occstress.corruptions import semantic


MISALIGNMENT_CONFIGS = {
    "easy": {
        "mode": "constant_bias",
        "affected_frames": 3,
        "dx_range": 0.8,
        "dy_range": 0.8,
        "yaw_deg_range": 4.0,
    },
    "mid": {
        "mode": "drift",
        "affected_frames": 4,
        "dx_range": 2.4,
        "dy_range": 2.4,
        "yaw_deg_range": 12.0,
    },
    "hard": {
        "mode": "drift",
        "affected_frames": None,
        "dx_range": 4.5,
        "dy_range": 4.5,
        "yaw_deg_range": 20.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate misalignment review samples.")
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--output-root", type=str, default="outputs/misalignment_review")
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--scene-name", type=str, default="scene-0001")
    parser.add_argument("--anchor-token", type=str, default="")
    return parser.parse_args()


def z_rotation_deg(yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    out = np.eye(4, dtype=np.float32)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    return out


def make_delta(dx_m, dy_m, yaw_deg):
    out = z_rotation_deg(yaw_deg)
    out[0, 3] = dx_m
    out[1, 3] = dy_m
    return out


def find_anchor_with_history(metadata, scene_name, history_length):
    samples_by_token = {row["token"]: row for row in metadata["samples"]}
    for sample in metadata["samples"]:
        if metadata["scene_by_token"][sample["scene_token"]]["name"] != scene_name:
            continue
        history = []
        cursor = sample
        ok = True
        for _ in range(history_length):
            prev_token = cursor["prev"]
            if not prev_token:
                ok = False
                break
            history.append(prev_token)
            cursor = samples_by_token[prev_token]
        if ok:
            history.reverse()
            return sample["token"], history
    raise RuntimeError(f"Could not find anchor with history_length={history_length} in {scene_name}.")


def collect_history(anchor_token, metadata, history_length):
    samples_by_token = {row["token"]: row for row in metadata["samples"]}
    cursor = samples_by_token[anchor_token]
    history = []
    for _ in range(history_length):
        prev_token = cursor["prev"]
        if not prev_token:
            raise RuntimeError(f"Anchor {anchor_token} does not have enough history for H={history_length}.")
        history.append(prev_token)
        cursor = samples_by_token[prev_token]
    history.reverse()
    return history


def get_clean_rts(anchor_token, history_tokens, metadata):
    anchor_lidar = metadata["lidar_sd_by_sample"][anchor_token]
    anchor_pose = metadata["ego_pose_by_token"][anchor_lidar["ego_pose_token"]]
    t_global_from_anchor = semantic.make_transform(anchor_pose["translation"], anchor_pose["rotation"])
    t_anchor_from_global = semantic.invert_transform(t_global_from_anchor)

    rts = []
    for token in history_tokens:
        lidar = metadata["lidar_sd_by_sample"][token]
        pose = metadata["ego_pose_by_token"][lidar["ego_pose_token"]]
        t_global_from_hist = semantic.make_transform(pose["translation"], pose["rotation"])
        rts.append((t_anchor_from_global @ t_global_from_hist).astype(np.float32))
    return np.stack(rts, axis=0)


def make_affected_mask(history_length, severity):
    if severity == "hard":
        return np.ones(history_length, dtype=np.uint8)
    count = MISALIGNMENT_CONFIGS[severity]["affected_frames"]
    mask = np.zeros(history_length, dtype=np.uint8)
    mask[:count] = 1
    return mask


def build_delta_series(anchor_token, severity, history_length, affected_mask):
    config = MISALIGNMENT_CONFIGS[severity]
    rng = np.random.default_rng(semantic.stable_seed("misalignment", severity, anchor_token, history_length))
    max_dx = float(rng.uniform(-config["dx_range"], config["dx_range"]))
    max_dy = float(rng.uniform(-config["dy_range"], config["dy_range"]))
    max_yaw = float(rng.uniform(-config["yaw_deg_range"], config["yaw_deg_range"]))

    dx = np.zeros(history_length, dtype=np.float32)
    dy = np.zeros(history_length, dtype=np.float32)
    yaw_deg = np.zeros(history_length, dtype=np.float32)
    delta_rt = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], history_length, axis=0)

    affected_indices = np.where(affected_mask == 1)[0]
    if len(affected_indices) == 0:
        return delta_rt, dx, dy, yaw_deg

    if config["mode"] == "constant_bias":
        for idx in affected_indices:
            dx[idx] = max_dx
            dy[idx] = max_dy
            yaw_deg[idx] = max_yaw
            delta_rt[idx] = make_delta(dx[idx], dy[idx], yaw_deg[idx])
    else:
        max_idx = float(affected_indices.max() + 1)
        for idx in affected_indices:
            alpha = float((idx + 1) / max_idx)
            scale = alpha ** 1.25
            dx[idx] = max_dx * scale
            dy[idx] = max_dy * scale
            yaw_deg[idx] = max_yaw * scale
            delta_rt[idx] = make_delta(dx[idx], dy[idx], yaw_deg[idx])

    return delta_rt, dx, dy, yaw_deg


def voxelize_transformed(semantics, rt):
    occ_id_free = semantic.CLASS_TO_ID["free"]
    indices = np.argwhere(semantics != occ_id_free)
    out = np.full_like(semantics, occ_id_free)
    if len(indices) == 0:
        return out
    centers = semantic.get_voxel_centers(indices)
    hom = np.concatenate([centers, np.ones((len(centers), 1), dtype=np.float32)], axis=1)
    warped = (rt @ hom.T).T[:, :3]
    warped_idx = np.floor((warped - semantic.PC_RANGE[:3]) / semantic.VOXEL_SIZE).astype(np.int64)
    valid = np.all((warped_idx >= 0) & (warped_idx < np.asarray(semantics.shape, dtype=np.int64)[None, :]), axis=1)
    warped_idx = warped_idx[valid]
    src_idx = indices[valid]
    out[warped_idx[:, 0], warped_idx[:, 1], warped_idx[:, 2]] = semantics[src_idx[:, 0], src_idx[:, 1], src_idx[:, 2]]
    return out


def aggregate_history(scene_name, history_tokens, rt_series, data_root):
    occ_id_free = semantic.CLASS_TO_ID["free"]
    shape = None
    aggregate = None
    for token, rt in zip(history_tokens, rt_series):
        occ_path = Path(data_root) / "gts" / scene_name / token / "labels.npz"
        occ = np.load(occ_path)
        warped = voxelize_transformed(occ["semantics"], rt)
        if aggregate is None:
            aggregate = np.full_like(warped, occ_id_free)
            shape = warped.shape
        valid = warped != occ_id_free
        aggregate[valid] = warped[valid]
    if aggregate is None:
        raise RuntimeError("No history frames aggregated.")
    return aggregate


def save_review_preview(anchor_semantics, clean_hist, mis_hist, out_path):
    def resize_vis(img, scale=4):
        pil = Image.fromarray(img)
        return np.array(pil.resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST))

    anchor_vis = resize_vis(semantic.build_preview(anchor_semantics))
    clean_vis = resize_vis(semantic.build_preview(clean_hist))
    mis_vis = resize_vis(semantic.build_preview(mis_hist))
    diff = (clean_hist != mis_hist).any(axis=2).astype(np.uint8)
    diff_vis = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    diff_vis[diff == 1] = np.array([255, 0, 0], dtype=np.uint8)
    diff_vis = resize_vis(diff_vis[::-1, ::-1])
    gap = np.full((anchor_vis.shape[0], 16, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([anchor_vis, gap, clean_vis, gap, mis_vis, gap, diff_vis], axis=1)
    Image.fromarray(canvas).save(out_path)


def generate_one(severity, anchor_token, history_tokens, scene_name, metadata, args, index):
    config = MISALIGNMENT_CONFIGS[severity]
    repo_root = Path.cwd()
    sample_id = f"{anchor_token}__H{len(history_tokens)}__misalignment_{severity}__{config['mode']}"
    out_dir = Path(args.output_root) / severity / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rt_clean = get_clean_rts(anchor_token, history_tokens, metadata)
    affected_mask = make_affected_mask(len(history_tokens), severity)
    delta_rt, dx_m, dy_m, yaw_deg = build_delta_series(anchor_token, severity, len(history_tokens), affected_mask)
    rt_misaligned = np.matmul(delta_rt, rt_clean)

    cache_path = out_dir / "matrices.npz"
    np.savez_compressed(
        cache_path,
        rt_clean=rt_clean,
        delta_rt=delta_rt,
        rt_misaligned=rt_misaligned,
        affected_mask=affected_mask,
        dx_m=dx_m,
        dy_m=dy_m,
        yaw_deg=yaw_deg,
    )

    anchor_occ_path = Path(args.data_root) / "gts" / scene_name / anchor_token / "labels.npz"
    anchor_occ = np.load(anchor_occ_path)
    clean_history = aggregate_history(scene_name, history_tokens, rt_clean, args.data_root)
    misaligned_history = aggregate_history(scene_name, history_tokens, rt_misaligned, args.data_root)

    preview_path = out_dir / "preview.png"
    save_review_preview(anchor_occ["semantics"], clean_history, misaligned_history, preview_path)

    event = {
        "type": "misalignment",
        "severity": severity,
        "mode": config["mode"],
        "sample_id": sample_id,
        "scene_name": scene_name,
        "anchor_token": anchor_token,
        "history_length": len(history_tokens),
        "history_tokens": history_tokens,
        "affected_mask": affected_mask.tolist(),
        "dx_m": [float(x) for x in dx_m],
        "dy_m": [float(y) for y in dy_m],
        "yaw_deg": [float(v) for v in yaw_deg],
        "cache_path": os.path.relpath(cache_path, repo_root),
        "anchor_occ_path": os.path.relpath(anchor_occ_path, repo_root),
    }
    event_path = out_dir / "event.json"
    with open(event_path, "w") as f:
        json.dump(event, f, indent=2)

    index[severity] = {
        "sample_id": sample_id,
        "preview": os.path.relpath(preview_path, repo_root),
        "event": os.path.relpath(event_path, repo_root),
        "cache": os.path.relpath(cache_path, repo_root),
    }


def main():
    args = parse_args()
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = semantic.collect_indices(nusc_root)

    if args.anchor_token:
        anchor_token = args.anchor_token
        history_tokens = collect_history(anchor_token, metadata, args.history_length)
        scene_name = metadata["scene_by_token"][next(s for s in metadata["samples"] if s["token"] == anchor_token)["scene_token"]]["name"]
    else:
        anchor_token, history_tokens = find_anchor_with_history(metadata, args.scene_name, args.history_length)
        scene_name = args.scene_name

    index = {
        "scene_name": scene_name,
        "anchor_token": anchor_token,
        "history_tokens": history_tokens,
    }
    for severity in ["easy", "mid", "hard"]:
        generate_one(severity, anchor_token, history_tokens, scene_name, metadata, args, index)

    out_root = Path(args.output_root)
    with open(out_root / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
