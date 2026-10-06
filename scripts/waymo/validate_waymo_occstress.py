#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate canonical OccStress-Waymo metadata, protocols, and optional assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-anchors", type=int, default=5978)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--clean-occ-cache",
        type=Path,
        default=os.environ.get("OCCSTRESS_WAYMO_CLEAN_OCC_CACHE"),
    )
    parser.add_argument(
        "--asset-cache",
        type=Path,
        default=os.environ.get("OCCSTRESS_WAYMO_ASSET_CACHE"),
    )
    return parser.parse_args()


def protocol_files(root):
    return sorted((root / "protocols" / "manual").rglob("*.pkl"))


def resolve_occ(path):
    path = Path(path)
    return path if path.suffix == ".npz" else path / "labels.npz"


def resolve_ref_occ(ref, scene_name, clean_occ_cache, asset_cache):
    source = ref.get("occ_source", ref.get("source"))
    path = str(ref["occ_path"])
    if source is None:
        source = "corrupted" if "/occ/manual/" in path else "clean"
    if source == "clean" and clean_occ_cache:
        cached = (
            clean_occ_cache / str(scene_name).zfill(3) /
            ref["token"] / "labels.npz"
        )
        if cached.exists():
            return cached
    marker = "/occ/manual/"
    if source != "clean" and asset_cache and marker in path:
        relative = path.split(marker, 1)[1]
        cached = asset_cache / "occ" / "manual" / relative
        if cached.exists():
            return cached
    return resolve_occ(path)


def validate_npz(path):
    with np.load(path) as labels:
        key = "semantics" if "semantics" in labels else "voxel_label"
        semantics = labels[key]
        if semantics.shape != (200, 200, 16):
            raise ValueError(f"{path}: invalid shape {semantics.shape}")
        if not np.issubdtype(semantics.dtype, np.integer):
            raise ValueError(f"{path}: non-integer semantics")
        if key == "semantics" and (semantics.min() < 0 or semantics.max() > 17):
            raise ValueError(f"{path}: canonical labels outside [0,17]")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    meta_root = args.root / "meta" / "manual"
    dataset_path = meta_root / "dataset.json"
    class_mapping_path = meta_root / "class_mapping.json"
    dataset = json.loads(dataset_path.read_text())
    class_mapping = json.loads(class_mapping_path.read_text())
    if dataset["anchor_count"] != args.expected_anchors:
        raise ValueError("dataset anchor count mismatch")
    if dataset["grid_shape"] != [200, 200, 16]:
        raise ValueError(f"invalid dataset grid: {dataset['grid_shape']}")
    if len(class_mapping.get("classes", [])) != 18:
        raise ValueError("class mapping must contain 18 target classes")
    files = protocol_files(args.root)
    if len(files) != 38:
        raise ValueError(f"expected 38 protocols, found {len(files)}")
    reference = None
    checked_npz = set()
    for path in files:
        with path.open("rb") as stream:
            records = pickle.load(stream)
        if len(records) != args.expected_anchors:
            raise ValueError(f"{path}: expected {args.expected_anchors}, got {len(records)}")
        anchors = [record["anchor_token"] for record in records]
        if reference is None:
            reference = anchors
        elif anchors != reference:
            raise ValueError(f"{path}: anchor order mismatch")
        selected = records[:args.max_records] if args.max_records else records
        for record in selected:
            if len(record["history"]) != 4 or len(record["future_targets"]) != 6:
                raise ValueError(f"{path}: invalid H4/F6 record")
            refs = record["history"] + [record["current_input"], record["target"]]
            refs += record["future_targets"]
            for ref in refs:
                occ = resolve_ref_occ(
                    ref,
                    record["scene_name"],
                    args.clean_occ_cache,
                    args.asset_cache,
                )
                if args.require_assets and not occ.exists():
                    raise FileNotFoundError(occ)
                if occ.exists() and occ not in checked_npz:
                    validate_npz(occ)
                    checked_npz.add(occ)
    payload = {
        "status": "success",
        "dataset": dataset["dataset"],
        "dataset_metadata": str(dataset_path.resolve()),
        "dataset_metadata_sha256": sha256(dataset_path),
        "class_mapping": str(class_mapping_path.resolve()),
        "class_mapping_sha256": sha256(class_mapping_path),
        "frame_count": dataset["frame_count"],
        "scene_count": dataset["scene_count"],
        "protocol_count": len(files),
        "anchors_per_protocol": len(reference),
        "history_frames": dataset["history_length"],
        "future_frames": dataset["future_length"],
        "grid_shape": dataset["grid_shape"],
        "require_assets": args.require_assets,
        "records_checked_per_protocol": (
            args.max_records or args.expected_anchors),
        "validated_npz": len(checked_npz),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(
            args.output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
