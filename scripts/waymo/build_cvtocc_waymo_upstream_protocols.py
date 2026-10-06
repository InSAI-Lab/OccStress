#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate CVT-Occ Waymo exports and build the 73 upstream protocols."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import tempfile
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from occstress.datasets.paths import data_root
from occstress.datasets.construction import portable_reference, write_protocol

CORRUPTIONS = (
    "CameraCrash",
    "FrameLost",
    "ColorQuant",
    "Brightness",
    "LowLight",
    "Fog",
    "MotionBlur",
    "Snow",
)
SEVERITIES = ("easy", "mid", "hard")
PATTERNS = {
    "current_only": ((False, False, False, False), True),
    "recent_burst": ((False, False, False, True), True),
    "history_only": ((True, True, True, True), False),
}
VOXEL_SHAPE = (200, 200, 16)
ADAPTER_VERSION = "cvtocc-waymo-2hz-v1"
EXPECTED_SCENES = 202
EXPECTED_FRAMES = 7998
EXPECTED_ANCHORS = 5978


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occstress-root", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--occ-root", type=Path)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--expected-anchors", type=int, default=EXPECTED_ANCHORS)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument(
        "--corruption",
        action="append",
        choices=CORRUPTIONS,
        help=(
            "Build only the selected completed corruption family. Repeat for "
            "multiple families. The default remains the full benchmark."
        ),
    )
    parser.add_argument(
        "--severity",
        action="append",
        choices=SEVERITIES,
        help=(
            "Build only the selected completed severity. Requires "
            "--corruption and may be repeated."
        ),
    )
    parser.add_argument(
        "--skip-file-validation",
        action="store_true",
        help=(
            "Trust validated per-scene .done.json files during a partial "
            "family build. Full benchmark builds always scan exported files."
        ),
    )
    args = parser.parse_args()
    args.occstress_root = data_root(args.occstress_root, dataset="waymo")
    root = args.occstress_root
    meta = root / "meta/OccStress-Waymo/upstream/camera_only/cvtocc"
    args.backbone = args.backbone or root / "protocols/manual/OccStress-Waymo/clean/H4_F6_val_backbone.pkl"
    args.index_root = args.index_root or meta / "frame_index"
    args.occ_root = args.occ_root or root / "occ/upstream/OccStress-Waymo/camera_only/cvtocc"
    args.protocol_root = args.protocol_root or root / "protocols/upstream/OccStress-Waymo/camera_only/cvtocc"
    args.manifest = args.manifest or meta / "protocols.json"
    for path in (args.backbone, args.occ_root, args.protocol_root, args.manifest):
        portable_reference(path, root)
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def settings(clean_only: bool = False) -> list[tuple[str, str]]:
    output = [("clean", "clean")]
    if clean_only:
        return output
    output.extend(
        (corruption, severity)
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    )
    return output


def setting_root(occ_root: Path, corruption: str, severity: str) -> Path:
    if corruption == "clean":
        return occ_root / "clean"
    return occ_root / corruption / severity


def scene_indexes(index_root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for done_path in sorted(
        index_root.glob("[0-9][0-9][0-9].done.json")
    ):
        try:
            done = load_json(done_path)
            index_path = done_path.with_name(
                done_path.name[: -len(".done.json")] + ".json"
            )
            index = load_json(index_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if done.get("status") != "success":
            continue
        scene = done_path.name[:3]
        if not index.get("frames"):
            continue
        output[scene] = index
    return output


def export_progress(
    index_root: Path,
    occ_root: Path,
    expected_scenes: int,
    expected_frames: int,
    requested_settings: list[tuple[str, str]],
) -> tuple[dict[str, Any], bool]:
    indexes = scene_indexes(index_root)
    indexed_frames = sum(len(index["frames"]) for index in indexes.values())
    complete = 0
    valid = 0
    checkpoint_digests = Counter()
    for corruption, severity in requested_settings:
        root = setting_root(occ_root, corruption, severity)
        for scene, index in indexes.items():
            done_path = root / scene / ".done.json"
            if not done_path.is_file():
                continue
            complete += 1
            try:
                done = load_json(done_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            expected = len(index["frames"])
            if (
                done.get("status") != "success"
                or int(done.get("frames", -1)) != expected
                or tuple(done.get("voxel_shape", ())) != VOXEL_SHAPE
                or done.get("adapter_version") != ADAPTER_VERSION
            ):
                continue
            checkpoint_digests[str(done.get("checkpoint_sha256"))] += 1
            valid += 1
    expected_tasks = expected_scenes * len(requested_settings)
    payload = {
        "checkpoint_sha256_counts": dict(checkpoint_digests),
        "adapter_version": ADAPTER_VERSION,
        "complete_scene_settings": complete,
        "expected_scene_settings": expected_tasks,
        "indexed_frames": indexed_frames,
        "indexed_scenes": len(indexes),
        "valid_scene_settings": valid,
    }
    ready = (
        len(indexes) == expected_scenes
        and indexed_frames == expected_frames
        and valid == expected_tasks
        and len(checkpoint_digests) == 1
    )
    return payload, ready


def validate_export_files(
    indexes: dict[str, dict[str, Any]],
    occ_root: Path,
    requested_settings: list[tuple[str, str]],
) -> int:
    missing = []
    count = 0
    sample_paths = []
    for corruption, severity in requested_settings:
        root = setting_root(occ_root, corruption, severity)
        for scene, index in sorted(indexes.items()):
            frames = index["frames"]
            for frame in frames:
                path = root / scene / frame["token"] / "labels.npz"
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
                count += 1
            sample_paths.extend(
                (
                    root / scene / frames[0]["token"] / "labels.npz",
                    root / scene / frames[-1]["token"] / "labels.npz",
                )
            )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CVT-Occ outputs are missing; first={missing[:8]}"
        )
    for path in sample_paths:
        with np.load(path) as payload:
            semantics = payload["semantics"]
        if semantics.shape != VOXEL_SHAPE:
            raise ValueError(f"{path}: invalid shape {semantics.shape}")
        if not np.issubdtype(semantics.dtype, np.integer):
            raise ValueError(f"{path}: non-integer labels {semantics.dtype}")
        if semantics.min(initial=0) < 0 or semantics.max(initial=0) > 17:
            raise ValueError(f"{path}: labels outside [0, 17]")
    return count


def observed_path(
    occ_root: Path,
    scene: str,
    token: str,
    corruption: str,
    severity: str,
    occstress_root=None,
) -> str:
    return portable_reference(
        (
            setting_root(occ_root, corruption, severity)
            / scene
            / token
            / "labels.npz"
        ), occstress_root
    )


def rewrite_observed_frame(
    frame: dict[str, Any],
    occ_root: Path,
    scene: str,
    root_corruption: str,
    root_severity: str,
    occstress_root=None,
) -> None:
    frame["occ_path"] = observed_path(
        occ_root,
        scene,
        frame["token"],
        root_corruption,
        root_severity,
        occstress_root,
    )
    frame["occ_source"] = "cvtocc"
    frame["source"] = "cvtocc"
    frame["prediction_variant"] = (
        "clean"
        if root_corruption == "clean"
        else f"{root_corruption}/{root_severity}"
    )
    frame["event_path"] = None


def build_protocol(
    backbone: list[dict[str, Any]],
    occ_root: Path,
    corruption: str,
    severity: str,
    pattern: str,
    occstress_root=None,
) -> list[dict[str, Any]]:
    if pattern == "clean":
        history_mask = (False, False, False, False)
        current_corrupted = False
    else:
        history_mask, current_corrupted = PATTERNS[pattern]
    records = []
    for source in backbone:
        if int(source["history_length"]) != 4:
            raise ValueError(f"{source['sample_id']}: expected H4")
        if int(source["future_length"]) != 6:
            raise ValueError(f"{source['sample_id']}: expected F6")
        if len(source['history']) != 4 or len(source['future_targets']) != 6:
            raise ValueError(f"{source['sample_id']}: missing observed/future frames")
        record = copy.deepcopy(source)
        scene = str(record["scene_name"]).zfill(3)
        record.update(
            {
                "track": "upstream",
                "subtrack": "camera_only",
                "source_model": "cvtocc",
                "upstream_adapter_version": ADAPTER_VERSION,
                "reference_protocol": "manual/OccStress-Waymo/clean/H4_F6_val_backbone",
                "frame_protocol": pattern,
                "corruption_family": corruption,
                "severity": None if corruption == "clean" else severity,
                "upstream_generation_policy": (
                    "scene_continuous_clean_and_corrupted_cvtocc_states_"
                    "selected_by_downstream_temporal_mask"
                ),
            }
        )
        for index, frame in enumerate(record["history"]):
            active = corruption != "clean" and history_mask[index]
            rewrite_observed_frame(
                frame,
                occ_root,
                scene,
                corruption if active else "clean",
                severity if active else "clean",
                occstress_root,
            )
        active = corruption != "clean" and current_corrupted
        rewrite_observed_frame(
            record["current_input"],
            occ_root,
            scene,
            corruption if active else "clean",
            severity if active else "clean",
            occstress_root,
        )
        record["target"]["occ_source"] = "clean"
        for future in record["future_targets"]:
            future["occ_source"] = "clean"
        record["sample_id"] = (
            f"{record['anchor_token']}__H4__F6__cvtocc__"
            f"{corruption}__{severity}__{pattern}"
        )
        record["corruption"] = {
            "type": corruption,
            "severity": None if corruption == "clean" else severity,
            "frame_protocol": pattern,
            "history_active_mask": list(history_mask),
            "current_active": bool(current_corrupted),
        }
        for frame in [record['target'], *record['future_targets']]:
            path = Path(frame['occ_path'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Use a namespaced, portable clean backbone for release construction')
        records.append(record)
    return records


def output_path(
    protocol_root: Path,
    corruption: str,
    severity: str,
    pattern: str,
) -> Path:
    if corruption == "clean":
        return protocol_root / "clean/H4_F6_val_backbone.pkl"
    return (
        protocol_root
        / corruption
        / severity
        / f"{pattern}_H4_F6_val_backbone.pkl"
    )


def main() -> None:
    args = parse_args()
    if args.clean_only and (args.corruption or args.severity):
        raise ValueError(
            "--clean-only cannot be combined with --corruption or --severity"
        )
    if args.severity and not args.corruption:
        raise ValueError("--severity requires --corruption")
    if args.skip_file_validation and not args.corruption:
        raise ValueError(
            "--skip-file-validation is restricted to partial --corruption builds"
        )
    selected_corruptions = tuple(
        dict.fromkeys(args.corruption or CORRUPTIONS)
    )
    selected_severities = tuple(
        dict.fromkeys(args.severity or SEVERITIES)
    )
    requested_settings = [("clean", "clean")]
    if not args.clean_only:
        requested_settings.extend(
            (corruption, severity)
            for corruption in selected_corruptions
            for severity in selected_severities
        )
    while True:
        progress, ready = export_progress(
            args.index_root,
            args.occ_root,
            args.expected_scenes,
            args.expected_frames,
            requested_settings,
        )
        progress["status"] = "ready" if ready else "waiting"
        print(json.dumps(progress, sort_keys=True), flush=True)
        if ready:
            break
        if not args.wait:
            raise RuntimeError("CVT-Occ Waymo export is incomplete")
        time.sleep(args.poll_seconds)

    indexes = scene_indexes(args.index_root)
    if args.skip_file_validation:
        exported_files = args.expected_frames * len(requested_settings)
    else:
        exported_files = validate_export_files(
            indexes,
            args.occ_root,
            requested_settings,
        )
    with args.backbone.open("rb") as handle:
        backbone = pickle.load(handle)
    if len(backbone) != args.expected_anchors:
        raise ValueError(
            f"expected {args.expected_anchors} anchors, got {len(backbone)}"
        )
    scenes = Counter(str(record["scene_name"]).zfill(3) for record in backbone)
    if set(scenes) != set(indexes):
        raise ValueError("backbone and CVT-Occ export scene sets differ")

    requested = [("clean", "clean", "clean")]
    if not args.clean_only:
        requested.extend(
            (corruption, severity, pattern)
            for corruption in selected_corruptions
            for severity in selected_severities
            for pattern in PATTERNS
        )
    protocols = []
    for corruption, severity, pattern in requested:
        records = build_protocol(
            backbone, args.occ_root, corruption, severity, pattern, args.occstress_root
        )
        path = output_path(
            args.protocol_root, corruption, severity, pattern
        )
        write_protocol(path, records, overwrite=args.overwrite)
        protocols.append(
            {
                "anchors": len(records),
                "corruption": corruption,
                "path": portable_reference(path, args.occstress_root),
                "pattern": pattern,
                "severity": severity,
                "sha256": sha256_file(path),
            }
        )

    checkpoint_counts = progress["checkpoint_sha256_counts"]
    manifest = {
        "anchors_per_protocol": len(backbone),
        "adapter_version": ADAPTER_VERSION,
        "checkpoint_sha256": next(iter(checkpoint_counts)),
        "dataset": "OccStress-Waymo",
        "exported_npz_files": exported_files,
        "frame_count": sum(
            len(index["frames"]) for index in indexes.values()
        ),
        "future_horizons_seconds": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "observed_offsets_seconds": [-2.0, -1.5, -1.0, -0.5, 0.0],
        "protocol_count": len(protocols),
        "protocols": protocols,
        "scene_count": len(indexes),
        "source_model": "CVT-Occ",
        "status": "success",
        "upstream_generation_policy": (
            "Clean and corrupted occupancy states are exported with "
            "scene-continuous CVT-Occ inference, then selected by the "
            "H4/current temporal mask. This matches the nuScenes upstream "
            "track convention and is not per-anchor sensor-stream rerunning."
        ),
        "temporal_patterns": {
            name: {
                "current_active": current,
                "history_active_mask": list(history),
            }
            for name, (history, current) in PATTERNS.items()
        },
    }
    if args.clean_only:
        manifest_path = args.manifest.with_name("protocols.clean.json")
    elif args.corruption:
        family_slug = "-".join(
            corruption.lower() for corruption in selected_corruptions
        )
        severity_slug = "-".join(selected_severities)
        manifest_path = args.manifest.with_name(
            f"protocols.partial.{family_slug}.{severity_slug}.json"
        )
    else:
        manifest_path = args.manifest
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "anchors_per_protocol": len(backbone),
                "manifest": str(manifest_path.resolve()),
                "protocol_count": len(protocols),
                "status": "success",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
