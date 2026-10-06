#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate camera-corrupted EFFOcc exports and build 73 OccStress-Waymo protocols."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import tempfile
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from occstress.datasets.paths import data_root
from occstress.datasets.construction import portable_reference, write_protocol

from effocc_waymo_camera_corruptions import SEVERITY


ADAPTER_VERSION = "effocc-waymo-robobev-camera-v1"
CORRUPTIONS = (
    "Brightness",
    "LowLight",
    "Fog",
    "Snow",
    "MotionBlur",
    "ColorQuant",
    "CameraCrash",
    "FrameLost",
)
SEVERITIES = ("easy", "mid", "hard")
SUBTRACK = "camera_fusion"
PATTERNS = {
    "current_only": ((False, False, False, False), True),
    "recent_burst": ((False, False, False, True), True),
    "history_only": ((True, True, True, True), False),
}
VOXEL_SHAPE = (200, 200, 16)
EXPECTED_SCENES = 202
EXPECTED_FRAMES = 7_998
EXPECTED_ANCHORS = 5_978


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occstress-root", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--occ-root", type=Path)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--expected-anchors", type=int, default=EXPECTED_ANCHORS)
    parser.add_argument("--spec-only", action="store_true")
    parser.add_argument("--skip-file-validation", action="store_true")
    args = parser.parse_args()
    args.occstress_root = data_root(args.occstress_root, dataset="waymo")
    root = args.occstress_root
    meta = root / "meta/OccStress-Waymo/upstream" / SUBTRACK / "effocc"
    args.backbone = args.backbone or root / "protocols/manual/OccStress-Waymo/clean/H4_F6_val_backbone.pkl"
    args.index_root = args.index_root or meta / "frame_index"
    args.occ_root = args.occ_root or root / "occ/upstream/OccStress-Waymo" / SUBTRACK / "effocc"
    args.protocol_root = args.protocol_root or root / "protocols/upstream/OccStress-Waymo" / SUBTRACK / "effocc"
    args.manifest = args.manifest or meta / ("protocol-specs.json" if args.spec_only else "protocols.json")
    for path in (args.backbone, args.occ_root, args.protocol_root, args.manifest):
        portable_reference(path, root)
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def settings() -> list[tuple[str, str]]:
    return [("clean", "clean")] + [
        (corruption, severity)
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ]


def protocol_specs() -> list[dict[str, str]]:
    output = [
        {
            "corruption": "clean",
            "severity": "clean",
            "pattern": "clean",
        }
    ]
    output.extend(
        {
            "corruption": corruption,
            "severity": severity,
            "pattern": pattern,
        }
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
        for pattern in PATTERNS
    )
    if len(output) != 73:
        raise AssertionError(f"expected 73 protocols, got {len(output)}")
    return output


def setting_root(root: Path, corruption: str, severity: str) -> Path:
    if corruption == "clean":
        return root / "clean"
    return root / corruption / severity


def output_path(
    root: Path,
    corruption: str,
    severity: str,
    pattern: str,
) -> Path:
    if corruption == "clean":
        return root / "clean/H4_F6_val_backbone.pkl"
    return (
        root
        / corruption
        / severity
        / f"{pattern}_H4_F6_val_backbone.pkl"
    )


def scene_indexes(index_root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted(index_root.glob("[0-9][0-9][0-9].json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("frames"):
            continue
        output[path.stem] = payload
    return output


def export_progress(
    indexes: dict[str, dict[str, Any]],
    occ_root: Path,
) -> dict[str, Any]:
    valid = 0
    checkpoints = Counter()
    for corruption, severity in settings():
        root = setting_root(occ_root, corruption, severity)
        for scene, index in indexes.items():
            done_path = root / scene / ".done.json"
            if not done_path.is_file():
                continue
            try:
                done = json.loads(done_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                done.get("status") != "success"
                or done.get("adapter_version") != ADAPTER_VERSION
                or int(done.get("frames", -1)) != len(index["frames"])
                or tuple(done.get("voxel_shape", ())) != VOXEL_SHAPE
            ):
                continue
            digest = done.get("checkpoint_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                continue
            checkpoints[str(done.get("checkpoint_sha256"))] += 1
            valid += 1
    return {
        "checkpoint_sha256_counts": dict(checkpoints),
        "expected_scene_settings": len(indexes) * len(settings()),
        "indexed_frames": sum(len(index["frames"]) for index in indexes.values()),
        "indexed_scenes": len(indexes),
        "valid_scene_settings": valid,
    }


def validate_export_files(
    indexes: dict[str, dict[str, Any]],
    occ_root: Path,
) -> int:
    count = 0
    for corruption, severity in settings():
        root = setting_root(occ_root, corruption, severity)
        for scene, index in indexes.items():
            for frame in index["frames"]:
                path = root / scene / frame["token"] / "labels.npz"
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
                count += 1
            for frame in (index["frames"][0], index["frames"][-1]):
                path = root / scene / frame["token"] / "labels.npz"
                with np.load(path) as payload:
                    semantics = payload["semantics"]
                if semantics.shape != VOXEL_SHAPE:
                    raise ValueError(f"{path}: invalid shape {semantics.shape}")
                if not np.issubdtype(semantics.dtype, np.integer):
                    raise ValueError(f"{path}: non-integer labels")
                if semantics.min(initial=0) < 0 or semantics.max(initial=0) > 17:
                    raise ValueError(f"{path}: labels outside [0, 17]")
    return count


def rewrite_observed_frame(
    frame: dict[str, Any],
    occ_root: Path,
    scene: str,
    corruption: str,
    severity: str,
    occstress_root=None,
) -> None:
    frame["occ_path"] = portable_reference(
        (
            setting_root(occ_root, corruption, severity)
            / scene
            / frame["token"]
            / "labels.npz"
        ), occstress_root
    )
    frame["occ_source"] = "effocc"
    frame["source"] = "effocc"
    frame["prediction_variant"] = (
        "clean" if corruption == "clean" else f"{corruption}/{severity}"
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
        current_active = False
    else:
        history_mask, current_active = PATTERNS[pattern]
    records = []
    for source in backbone:
        if int(source["history_length"]) != 4:
            raise ValueError(f"{source['sample_id']}: expected H4")
        if int(source["future_length"]) != 6:
            raise ValueError(f"{source['sample_id']}: expected F6")
        if len(source["history"]) != 4 or len(source["future_targets"]) != 6:
            raise ValueError(f"{source['sample_id']}: missing observed/future frames")
        record = copy.deepcopy(source)
        scene = str(record["scene_name"]).zfill(3)
        record.update(
            {
                "track": "upstream",
                "subtrack": "camera_fusion",
                "source_model": "effocc",
                "source_modality": "camera_lidar",
                "corrupted_modality": (
                    None if corruption == "clean" else "camera"
                ),
                "upstream_adapter_version": ADAPTER_VERSION,
                "reference_protocol": "manual/OccStress-Waymo/clean/H4_F6_val_backbone",
                "frame_protocol": pattern,
                "corruption_family": corruption,
                "severity": None if corruption == "clean" else severity,
                "upstream_generation_policy": (
                    "scene_continuous_clean_lidar_and_camera_corrupted_"
                    "effocc_states_"
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
        active = corruption != "clean" and current_active
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
            f"{record['anchor_token']}__H4__F6__effocc_camera__"
            f"{corruption}__{severity}__{pattern}"
        )
        record["corruption"] = {
            "type": corruption,
            "severity": None if corruption == "clean" else severity,
            "frame_protocol": pattern,
            "history_active_mask": list(history_mask),
            "current_active": bool(current_active),
            "corrupted_modality": (
                None if corruption == "clean" else "camera"
            ),
        }
        for frame in [record["target"], *record["future_targets"]]:
            path = Path(frame["occ_path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Use a namespaced, portable clean backbone for release construction")
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    specs = protocol_specs()
    outputs = [args.manifest]
    if not args.spec_only:
        outputs.extend(output_path(args.protocol_root, **spec) for spec in specs)
    if not args.overwrite:
        existing = next((path for path in outputs if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"{existing}; use --overwrite only for intentional regeneration")
    base_manifest = {
        "adapter_version": ADAPTER_VERSION,
        "anchor_count": args.expected_anchors,
        "backbone": portable_reference(args.backbone, args.occstress_root),
        "corruption_families": list(CORRUPTIONS),
        "dataset": "Occ3D-Waymo",
        "expected_frames": args.expected_frames,
        "expected_scenes": args.expected_scenes,
        "input_window": {
            "history": 4,
            "current": 1,
            "offsets_seconds": [-2.0, -1.5, -1.0, -0.5, 0.0],
        },
        "occupancy_setting_count": len(settings()),
        "protocol_count": len(specs),
        "robobev_severity": SEVERITY,
        "corruption_implementation": str(
            Path(__file__).resolve().with_name(
                "effocc_waymo_camera_corruptions.py"
            )
        ),
        "corruption_implementation_sha256": sha256_file(
            Path(__file__).resolve().with_name(
                "effocc_waymo_camera_corruptions.py"
            )
        ),
        "severities": list(SEVERITIES),
        "source_model": "EFFOcc",
        "source_modality": "camera_lidar",
        "corrupted_modality": "camera",
        "status": "specified",
        "subtrack": "camera_fusion",
        "target_policy": "clean current and six clean future occupancy targets",
        "track": "upstream",
    }
    if args.spec_only:
        base_manifest["protocols"] = [
            {
                **spec,
                "path": portable_reference(
                    output_path(
                        args.protocol_root,
                        spec["corruption"],
                        spec["severity"],
                        spec["pattern"],
                    ), args.occstress_root
                ),
            }
            for spec in specs
        ]
        atomic_json(args.manifest, base_manifest)
        print(json.dumps(base_manifest, indent=2, sort_keys=True))
        return

    indexes = scene_indexes(args.index_root)
    progress = export_progress(indexes, args.occ_root)
    ready = (
        progress["indexed_scenes"] == args.expected_scenes
        and progress["indexed_frames"] == args.expected_frames
        and progress["valid_scene_settings"]
        == progress["expected_scene_settings"]
        and len(progress["checkpoint_sha256_counts"]) == 1
    )
    if not ready:
        raise RuntimeError(f"EFFOcc Waymo export is incomplete: {progress}")
    exported_files = (
        None
        if args.skip_file_validation
        else validate_export_files(indexes, args.occ_root)
    )

    with args.backbone.open("rb") as handle:
        backbone = pickle.load(handle)
    if len(backbone) != args.expected_anchors:
        raise ValueError(
            f"expected {args.expected_anchors} anchors, got {len(backbone)}"
        )
    scenes = {str(record["scene_name"]).zfill(3) for record in backbone}
    if scenes != set(indexes):
        raise ValueError("backbone and EFFOcc export scene sets differ")

    protocols = []
    for spec in specs:
        records = build_protocol(
            backbone,
            args.occ_root,
            spec["corruption"],
            spec["severity"],
            spec["pattern"],
            args.occstress_root,
        )
        path = output_path(
            args.protocol_root,
            spec["corruption"],
            spec["severity"],
            spec["pattern"],
        )
        write_protocol(path, records, overwrite=args.overwrite)
        protocols.append(
            {
                **spec,
                "anchors": len(records),
                "path": portable_reference(path, args.occstress_root),
                "sha256": sha256_file(path),
            }
        )

    base_manifest.update(
        {
            "backbone_sha256": sha256_file(args.backbone),
            "checkpoint_sha256": next(
                iter(progress["checkpoint_sha256_counts"])
            ),
            "export_progress": progress,
            "exported_occupancy_files": exported_files,
            "protocols": protocols,
            "status": "success",
            "temporal_patterns": {
                name: {
                    "history_active_mask": list(history),
                    "current_active": current,
                }
                for name, (history, current) in PATTERNS.items()
            },
        }
    )
    atomic_json(args.manifest, base_manifest)
    print(
        json.dumps(
            {
                "anchors": len(backbone),
                "manifest": str(args.manifest.resolve()),
                "protocol_count": len(protocols),
                "status": "success",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
