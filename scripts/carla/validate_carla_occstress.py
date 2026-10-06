#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate OccStress-CARLA clean data, manual assets, controls, and 38 protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WAYMO_SCRIPT_DIR = SCRIPT_DIR.parent / "waymo"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WAYMO_SCRIPT_DIR))

from build_carla_occstress import (  # noqa: E402
    UNIOCC_TO_OCC3D,
    carla_command,
    map_uniocc_semantics,
)
from generate_carla_occstress_assets import (  # noqa: E402
    ALL_TASKS,
    run_corruption,
)
from waymo_occstress_common import (  # noqa: E402
    CONTENT_FAMILIES,
    FUTURE_OFFSETS_SECONDS,
    OBSERVED_OFFSETS_SECONDS,
    SEVERITIES,
    TEMPORAL_PATTERNS,
    mirror_y_semantics,
    relative_xy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-anchors", type=int, default=330)
    parser.add_argument("--expected-frames", type=int, default=360)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_spec(path: Path, root: Path) -> tuple[str, str | None, str]:
    relative = path.relative_to(root / "protocols" / "manual")
    parts = relative.parts
    if parts[0] == "clean":
        return "clean", None, "clean"
    if parts[0] == "traffic":
        return "traffic", None, "all_frame"
    if len(parts) != 3:
        raise ValueError(f"invalid protocol path: {path}")
    return parts[0], parts[1], parts[2].removesuffix(
        "_H4_F6_val_backbone.pkl")


def expected_protocol_specs() -> set[tuple[str, str | None, str]]:
    specs = {("clean", None, "clean"), ("traffic", None, "all_frame")}
    specs.update(
        (family, severity, pattern)
        for family in CONTENT_FAMILIES + ("misalignment",)
        for severity in SEVERITIES
        for pattern in TEMPORAL_PATTERNS
    )
    return specs


def asset_path(root: Path, task: str, frame: dict) -> Path:
    if task == "traffic":
        base = root / "occ" / "manual" / "traffic"
    else:
        family, severity = task.split(":")
        base = root / "occ" / "manual" / family / severity
    return base / frame["scene_name"] / frame["token"] / "labels.npz"


def event_path(root: Path, task: str, frame: dict) -> Path:
    if task == "traffic":
        base = root / "events" / "manual" / "traffic"
    else:
        family, severity = task.split(":")
        base = root / "events" / "manual" / family / severity
    return base / frame["scene_name"] / f"{frame['token']}.json"


def load_occ(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        if set(payload.files) != {"semantics", "infov"}:
            raise ValueError(f"{path}: expected semantics and infov")
        semantics = payload["semantics"]
        infov = payload["infov"]
    if semantics.shape != (200, 200, 16):
        raise ValueError(f"{path}: invalid semantics shape {semantics.shape}")
    if infov.shape != semantics.shape:
        raise ValueError(f"{path}: invalid infov shape {infov.shape}")
    if not np.issubdtype(semantics.dtype, np.integer):
        raise ValueError(f"{path}: semantics must be integer")
    if semantics.min() < 0 or semantics.max() > 17:
        raise ValueError(f"{path}: semantics outside Occ3D [0,17]")
    if infov.dtype != np.bool_:
        raise ValueError(f"{path}: infov must be boolean")
    return semantics, infov


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def validate_controls(base_info: dict, anchor_tokens: set[str]) -> int:
    checked = 0
    for scene_frames in base_info["infos"].values():
        poses = {
            frame["token"]: np.asarray(frame["ego2global"], dtype=np.float64)
            for frame in scene_frames
        }
        by_token = {frame["token"]: frame for frame in scene_frames}
        ordered = sorted(scene_frames, key=lambda frame: frame["timestamp"])
        index_by_token = {
            frame["token"]: index for index, frame in enumerate(ordered)
        }
        for token in sorted(anchor_tokens):
            if token not in by_token:
                continue
            index = index_by_token[token]
            frame = by_token[token]
            future = ordered[index + 1:index + 7]
            if len(future) != 6:
                raise ValueError(f"{token}: incomplete future controls")
            cumulative = np.stack([
                relative_xy(poses[token], poses[target["token"]])
                for target in future
            ])
            trajectory = np.asarray(
                frame["gt_ego_fut_trajs"], dtype=np.float32)
            if trajectory.shape != (6, 2):
                raise ValueError(f"{token}: invalid future trajectory")
            if not np.allclose(np.cumsum(trajectory, axis=0), cumulative, atol=1e-5):
                raise ValueError(f"{token}: trajectory/pose mismatch")
            if not np.array_equal(
                np.asarray(frame["gt_ego_fut_masks"]), np.ones(6)
            ):
                raise ValueError(f"{token}: future control mask is not complete")
            if not np.array_equal(
                np.asarray(frame["gt_ego_fut_cmd"]), carla_command(cumulative)
            ):
                raise ValueError(f"{token}: command/trajectory mismatch")
            checked += 1
    if checked != len(anchor_tokens):
        raise ValueError(
            f"validated {checked}/{len(anchor_tokens)} anchor controls")
    return checked


def validate_protocols(
    root: Path, frames: list[dict], expected_anchors: int
) -> tuple[set[str], int]:
    protocol_files = sorted((root / "protocols" / "manual").rglob("*.pkl"))
    specs = {protocol_spec(path, root) for path in protocol_files}
    expected_specs = expected_protocol_specs()
    if specs != expected_specs:
        raise ValueError(
            f"protocol spec mismatch; missing={sorted(expected_specs - specs)}, "
            f"extra={sorted(specs - expected_specs)}"
        )

    frame_by_token = {frame["token"]: frame for frame in frames}
    reference_anchors = None
    all_anchor_tokens: set[str] = set()
    records_checked = 0
    for path in protocol_files:
        family, severity, pattern = protocol_spec(path, root)
        with path.open("rb") as stream:
            records = pickle.load(stream)
        if len(records) != expected_anchors:
            raise ValueError(
                f"{path}: expected {expected_anchors}, got {len(records)}")
        anchors = [record["anchor_token"] for record in records]
        if len(set(anchors)) != expected_anchors:
            raise ValueError(f"{path}: duplicate anchors")
        if reference_anchors is None:
            reference_anchors = anchors
        elif anchors != reference_anchors:
            raise ValueError(f"{path}: anchor order mismatch")

        active = set(TEMPORAL_PATTERNS.get(pattern, ()))
        for record in records:
            anchor = record["anchor_token"]
            current = frame_by_token[anchor]
            if record["dataset"] != "UniOcc-CARLA":
                raise ValueError(f"{path}: dataset mismatch")
            if len(record["history"]) != 4 or len(record["future_targets"]) != 6:
                raise ValueError(f"{path}: invalid H4/F6")
            if record["observed_offsets_seconds"] != list(
                OBSERVED_OFFSETS_SECONDS
            ):
                raise ValueError(f"{path}: observed offsets mismatch")
            if record["future_offsets_seconds"] != list(
                FUTURE_OFFSETS_SECONDS
            ):
                raise ValueError(f"{path}: future offsets mismatch")

            input_refs = record["history"] + [record["current_input"]]
            input_offsets = (-4, -3, -2, -1, 0)
            for ref, offset in zip(input_refs, input_offsets):
                frame = frame_by_token[ref["token"]]
                expected = frame["occ_path"]
                expected_source = "clean"
                if family in CONTENT_FAMILIES and offset in active:
                    expected = asset_path(
                        root, f"{family}:{severity}", frame)
                    expected_source = "corrupted"
                elif family == "traffic":
                    expected = asset_path(root, "traffic", frame)
                    expected_source = "corrupted"
                if not same_path(ref["occ_path"], expected):
                    raise ValueError(
                        f"{path}: wrong input path for {ref['token']}")
                if ref.get("occ_source") != expected_source:
                    raise ValueError(
                        f"{path}: wrong input source for {ref['token']}")

            target_expected = current["occ_path"]
            target_source = "clean"
            if family == "traffic":
                target_expected = asset_path(root, "traffic", current)
                target_source = "corrupted"
            if not same_path(record["target"]["occ_path"], target_expected):
                raise ValueError(f"{path}: target mismatch at {anchor}")
            if record["target"].get("occ_source") != target_source:
                raise ValueError(f"{path}: target source mismatch at {anchor}")
            for ref in record["future_targets"]:
                frame = frame_by_token[ref["token"]]
                expected = (
                    asset_path(root, "traffic", frame)
                    if family == "traffic" else frame["occ_path"]
                )
                if not same_path(ref["occ_path"], expected):
                    raise ValueError(
                        f"{path}: future target mismatch at {anchor}")

            if family == "misalignment":
                metadata = record["misalignment"]
                expected_mask = [
                    int(offset in active) for offset in (-4, -3, -2, -1)
                ]
                if metadata["affected_mask"] != expected_mask:
                    raise ValueError(f"{path}: misalignment mask mismatch")
                if metadata["current_active"] != (0 in active):
                    raise ValueError(f"{path}: current alignment mismatch")
                deltas = np.asarray(metadata["delta_rt"])
                current_delta = np.asarray(metadata["current_delta_rt"])
                for active_value, delta in zip(expected_mask, deltas):
                    if not active_value and not np.allclose(delta, np.eye(4)):
                        raise ValueError(f"{path}: inactive history has delta")
                if 0 not in active and not np.allclose(
                    current_delta, np.eye(4)
                ):
                    raise ValueError(f"{path}: inactive current has delta")
            if family == "traffic":
                transform = record.get("traffic_transform")
                if transform is None:
                    raise ValueError(f"{path}: missing traffic transform")
                mirror = np.diag([1.0, -1.0, 1.0, 1.0])
                if not np.array_equal(
                    np.asarray(transform["mirror_matrix"]), mirror
                ):
                    raise ValueError(f"{path}: traffic mirror matrix mismatch")
                if transform["relative_pose_policy"] != (
                    "M @ relative_pose @ M"
                ):
                    raise ValueError(f"{path}: traffic pose policy mismatch")
                source_trajectory = np.asarray(
                    current["gt_ego_fut_trajs"], dtype=np.float32)
                expected_trajectory = source_trajectory.copy()
                expected_trajectory[:, 1] *= -1
                if not np.array_equal(
                    np.asarray(transform["current_gt_ego_fut_trajs"]),
                    expected_trajectory,
                ):
                    raise ValueError(
                        f"{path}: traffic trajectory mismatch at {anchor}")
                source_command = np.asarray(
                    current["gt_ego_fut_cmd"], dtype=np.float32)
                if not np.array_equal(
                    np.asarray(transform["current_gt_ego_fut_cmd"]),
                    source_command[[1, 0, 2]],
                ):
                    raise ValueError(
                        f"{path}: traffic command mismatch at {anchor}")
            records_checked += 1
            all_anchor_tokens.add(anchor)
    return all_anchor_tokens, records_checked


def validate_assets(
    root: Path,
    frames: list[dict],
    require_assets: bool,
    check_determinism: bool,
) -> tuple[dict, int, list[str]]:
    if not require_assets:
        return {}, 0, []
    task_totals = defaultdict(lambda: {
        "changed": 0,
        "occupied": 0,
        "support": 0,
        "bytes": 0,
        "zero_changed_frames": 0,
    })
    validated = 0
    labels_present = set()
    deterministic_indices = {0, len(frames) // 2, len(frames) - 1}

    for frame_index, frame in enumerate(frames):
        clean, clean_infov = load_occ(Path(frame["occ_path"]))
        labels_present.update(np.unique(clean).astype(int).tolist())
        with np.load(frame["raw_occ_path"]) as raw_payload:
            mapped = map_uniocc_semantics(raw_payload["occ_label"])
            raw_infov = np.asarray(raw_payload["occ_mask_camera"], dtype=bool)
        if not np.array_equal(clean, mapped):
            raise ValueError(f"{frame['token']}: canonical mapping mismatch")
        if not np.array_equal(clean_infov, raw_infov):
            raise ValueError(f"{frame['token']}: canonical infov mismatch")
        validated += 1

        for task in ALL_TASKS:
            output_file = asset_path(root, task, frame)
            metadata_file = event_path(root, task, frame)
            if not output_file.is_file() or not metadata_file.is_file():
                missing = output_file if not output_file.is_file() else metadata_file
                raise FileNotFoundError(missing)
            output, output_infov = load_occ(output_file)
            if not np.array_equal(clean_infov, output_infov):
                raise ValueError(f"{output_file}: infov changed")
            changed = int(np.count_nonzero(output != clean))
            event = json.loads(metadata_file.read_text(encoding="utf-8"))
            if event["changed_voxels"] != changed:
                raise ValueError(f"{metadata_file}: changed count mismatch")
            if changed == 0:
                task_totals[task]["zero_changed_frames"] += 1
            occupied = int(np.count_nonzero(clean != 17))
            support = int(np.count_nonzero((clean != 17) | (output != 17)))
            if event["occupied_voxels"] != occupied:
                raise ValueError(f"{metadata_file}: occupied count mismatch")
            if event["normalization_voxels"] != support:
                raise ValueError(f"{metadata_file}: support count mismatch")
            if task == "traffic" and not np.array_equal(
                output, mirror_y_semantics(clean)
            ):
                raise ValueError(f"{output_file}: traffic mirror mismatch")
            if check_determinism and frame_index in deterministic_indices:
                regenerated, _, _ = run_corruption(
                    clean, task, frame["token"])
                if not np.array_equal(output, regenerated):
                    raise ValueError(
                        f"{output_file}: deterministic regeneration mismatch")
            task_totals[task]["changed"] += changed
            task_totals[task]["occupied"] += occupied
            task_totals[task]["support"] += support
            task_totals[task]["bytes"] += output_file.stat().st_size
            validated += 1

    summaries = {}
    for task, totals in sorted(task_totals.items()):
        summaries[task] = {
            **totals,
            "changed_fraction_occupied": float(
                totals["changed"] / max(1, totals["occupied"])),
            "changed_fraction_support": float(
                totals["changed"] / max(1, totals["support"])),
        }
        if totals["zero_changed_frames"]:
            raise ValueError(
                f"{task}: {totals['zero_changed_frames']} unchanged frames")

    severity_warnings = []
    for family in CONTENT_FAMILIES:
        values = [
            summaries[f"{family}:{severity}"]["changed_fraction_support"]
            for severity in SEVERITIES
        ]
        if not (values[0] <= values[1] <= values[2]):
            severity_warnings.append(
                f"{family} aggregate changed fractions are not monotonic: {values}"
            )
    expected_labels = set(UNIOCC_TO_OCC3D.values())
    if not labels_present.issubset(expected_labels):
        raise ValueError(
            f"clean labels outside class map: {sorted(labels_present)}")
    return summaries, validated, severity_warnings


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    meta = root / "meta" / "manual"
    dataset_path = meta / "dataset.json"
    mapping_path = meta / "class_mapping.json"
    frames_path = meta / "frames_H4_F6_val.json"
    base_info_path = meta / "carla_base_info.pkl"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    with base_info_path.open("rb") as stream:
        base_info = pickle.load(stream)

    if dataset["frame_count"] != args.expected_frames:
        raise ValueError("dataset frame count mismatch")
    if dataset["anchor_count"] != args.expected_anchors:
        raise ValueError("dataset anchor count mismatch")
    if dataset["window_definition"] != "4 history + current + 6 future":
        raise ValueError("dataset window definition mismatch")
    if dataset["coordinate_convention"] != (
        "CARLA left-handed ego-local x-forward/y-right/z-up"
    ):
        raise ValueError("dataset coordinate convention mismatch")
    if dataset["grid_shape"] != [200, 200, 16]:
        raise ValueError("dataset grid mismatch")
    if len(mapping["classes"]) != 18:
        raise ValueError("class mapping must define 18 target classes")
    if mapping["mapping"] != {
        str(key): value for key, value in UNIOCC_TO_OCC3D.items()
    }:
        raise ValueError("class mapping mismatch")
    if len(frames) != args.expected_frames:
        raise ValueError("frames metadata count mismatch")

    anchors, protocol_records = validate_protocols(
        root, frames, args.expected_anchors)
    controls_checked = validate_controls(base_info, anchors)
    asset_summaries, validated_npz, warnings = validate_assets(
        root, frames, args.require_assets, args.check_determinism)
    payload = {
        "status": "success",
        "dataset": dataset["dataset"],
        "benchmark_name": dataset["benchmark_name"],
        "root": str(root),
        "dataset_metadata_sha256": sha256(dataset_path),
        "class_mapping_sha256": sha256(mapping_path),
        "frame_count": len(frames),
        "scene_count": dataset["scene_count"],
        "protocol_count": len(
            list((root / "protocols" / "manual").rglob("*.pkl"))),
        "anchors_per_protocol": len(anchors),
        "protocol_records_checked": protocol_records,
        "anchor_controls_checked": controls_checked,
        "history_frames": dataset["history_length"],
        "current_frames": 1,
        "future_frames": dataset["future_length"],
        "grid_shape": dataset["grid_shape"],
        "require_assets": args.require_assets,
        "determinism_checked": args.check_determinism,
        "validated_npz": validated_npz,
        "asset_summaries": asset_summaries,
        "warnings": warnings,
    }
    if args.output_json:
        atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
