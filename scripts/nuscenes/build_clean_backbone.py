#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scripts.nuscenes import generate_protocols as proto
from occstress.corruptions import semantic
from occstress.datasets.construction import manual_meta, portable_reference, write_protocol
from occstress.protocols.layout import resolve_manual_protocol_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a clean backbone protocol from the canonical val split.")
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--ann-file", type=str, required=True)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=6)
    parser.add_argument("--name", type=str, default="clean_H4_F6_val_backbone")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = semantic.collect_indices(nusc_root)
    samples_by_token = {row["token"]: row for row in metadata["samples"]}
    with open(args.ann_file, "rb") as handle:
        allowed = {info["token"] for info in pickle.load(handle)["infos"]}

    records = []
    union_tokens = set()
    history_only_tokens = set()
    for sample in metadata["samples"]:
        if sample["token"] not in allowed:
            continue
        history_tokens = proto.history_tokens_for_anchor(sample, samples_by_token, args.history_length)
        future_tokens = proto.future_tokens_for_anchor(sample, samples_by_token, args.future_length)
        if history_tokens is None or future_tokens is None:
            continue
        scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
        rec = {
            "sample_id": f"{sample['token']}__H{args.history_length}__F{args.future_length}__clean",
            "scene_name": scene_name,
            "scene_token": sample["scene_token"],
            "anchor_token": sample["token"],
            "anchor_timestamp": sample["timestamp"],
            "history_length": args.history_length,
            "history_tokens": history_tokens,
            "future_length": args.future_length,
            "future_tokens": future_tokens,
            "current_input": {
                "token": sample["token"],
                "source": "clean",
                "occ_path": proto.clean_occ_relpath(args.data_root, scene_name, sample["token"]),
            },
            "target": {
                "token": sample["token"],
                "source": "clean",
                "occ_path": proto.clean_occ_relpath(args.data_root, scene_name, sample["token"]),
            },
            "future_targets": [
                {
                    "index": idx,
                    "token": token,
                    "source": "clean",
                    "occ_path": proto.clean_occ_relpath(args.data_root, scene_name, token),
                }
                for idx, token in enumerate(future_tokens)
            ],
            "history": [
                {
                    "index": idx,
                    "token": token,
                    "occ_source": "clean",
                    "occ_path": proto.clean_occ_relpath(args.data_root, scene_name, token),
                    "rt_source": "clean",
                }
                for idx, token in enumerate(history_tokens)
            ],
            "corruption": {
                "type": "clean",
            },
        }
        records.append(rec)
        union_tokens.add(sample["token"])
        union_tokens.update(history_tokens)
        history_only_tokens.update(history_tokens)

    if not records:
        raise ValueError("No complete anchors found in the supplied validation split")
    out_path = resolve_manual_protocol_path(args.name, occstress_root=args.output_root)
    write_protocol(out_path, records, overwrite=args.overwrite)

    meta_dir = manual_meta(args.output_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": args.name,
        "ann_file": Path(args.ann_file).name,
        "history_length": args.history_length,
        "future_length": args.future_length,
        "num_samples": len(records),
        "protocol_path": portable_reference(out_path, args.output_root),
        "content_frame_tokens": len(union_tokens),
        "history_frame_tokens": len(history_only_tokens),
    }
    with open(meta_dir / f"{args.name}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(meta_dir / f"{args.name}_content_frame_tokens.txt", "w") as f:
        for token in sorted(union_tokens):
            f.write(f"{token}\n")
    with open(meta_dir / f"{args.name}_history_frame_tokens.txt", "w") as f:
        for token in sorted(history_only_tokens):
            f.write(f"{token}\n")

    print(json.dumps(manifest, indent=2))
    print(f"backbone written to: {out_path}")


if __name__ == "__main__":
    main()
