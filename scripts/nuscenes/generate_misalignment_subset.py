#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from occstress.corruptions import misalignment as misalign
from occstress.corruptions import semantic
from occstress.datasets.construction import (
    clean_nuscenes_reference, manual_meta, manual_output, portable_reference,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the misalignment subset as clip-scoped event/cache records.")
    parser.add_argument("--severity", type=str, choices=sorted(misalign.MISALIGNMENT_CONFIGS), required=True)
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--anchor-token", type=str, default="")
    parser.add_argument("--anchor-token-file", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_anchor_samples(metadata, args):
    samples_by_token = {row["token"]: row for row in metadata["samples"]}
    token_set = semantic.load_sample_token_set(args.anchor_token_file)
    for sample in metadata["samples"]:
        scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
        if args.scene_name and scene_name != args.scene_name:
            continue
        if args.anchor_token and sample["token"] != args.anchor_token:
            continue
        if token_set is not None and sample["token"] not in token_set:
            continue
        cursor = sample
        ok = True
        history = []
        for _ in range(args.history_length):
            prev_token = cursor["prev"]
            if not prev_token:
                ok = False
                break
            history.append(prev_token)
            cursor = samples_by_token[prev_token]
        if not ok:
            continue
        history.reverse()
        yield sample["token"], scene_name, history


def write_sample(severity, anchor_token, scene_name, history_tokens, metadata, args, summary):
    config = misalign.MISALIGNMENT_CONFIGS[severity]
    sample_id = f"{anchor_token}__H{len(history_tokens)}__misalignment_{severity}__{config['mode']}"

    event_dir = manual_output(args.output_root, "events", "misalignment", severity, scene_name)
    cache_dir = manual_output(args.output_root, "cache", "misalignment", severity, scene_name)
    event_path = event_dir / f"{sample_id}.json"
    cache_path = cache_dir / f"{sample_id}.npz"
    if not args.overwrite and event_path.exists() and cache_path.exists():
        return False

    rt_clean = misalign.get_clean_rts(anchor_token, history_tokens, metadata)
    affected_mask = misalign.make_affected_mask(len(history_tokens), severity)
    delta_rt, dx_m, dy_m, yaw_deg = misalign.build_delta_series(anchor_token, severity, len(history_tokens), affected_mask)
    rt_misaligned = delta_rt @ rt_clean

    event_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "rt_clean": rt_clean,
        "delta_rt": delta_rt,
        "rt_misaligned": rt_misaligned,
        "affected_mask": affected_mask,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "yaw_deg": yaw_deg,
    }
    import numpy as np
    np.savez_compressed(cache_path, **cache_payload)

    history_occ_paths = [clean_nuscenes_reference(scene_name, token) for token in history_tokens]
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
        "anchor_occ_path": clean_nuscenes_reference(scene_name, anchor_token),
        "history_occ_paths": history_occ_paths,
        "cache_path": portable_reference(cache_path, args.output_root),
    }
    with open(event_path, "w") as f:
        json.dump(event, f, indent=2)

    summary["processed"] += 1
    summary["affected_frames_total"] += int(affected_mask.sum())
    summary["mode_counts"][config["mode"]] = summary["mode_counts"].get(config["mode"], 0) + 1
    return True


def main():
    args = parse_args()
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = semantic.collect_indices(nusc_root)
    summary = {
        "severity": args.severity,
        "history_length": args.history_length,
        "processed": 0,
        "affected_frames_total": 0,
        "mode_counts": {},
    }

    for anchor_token, scene_name, history_tokens in iter_anchor_samples(metadata, args):
        write_sample(args.severity, anchor_token, scene_name, history_tokens, metadata, args, summary)
        if args.max_samples > 0 and summary["processed"] >= args.max_samples:
            break

    meta_dir = manual_meta(args.output_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / f"misalignment_{args.severity}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
