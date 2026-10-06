#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Build and validate both 73-protocol CARLA upstream tracks."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from occstress.datasets.paths import data_root, dataset_name, resolve_occstress_path
from occstress.protocols.validation import anchor_key, load_records, validate_records

DATASET = dataset_name('carla')

CAMERA_CORRUPTIONS = (
    "CameraCrash",
    "FrameLost",
    "MotionBlur",
    "ColorQuant",
    "Brightness",
    "LowLight",
    "Fog",
    "Snow",
)
CAMERA_SEVERITIES = ("easy", "mid", "hard")
LIDAR_CORRUPTIONS = (
    "beam_missing",
    "cross_sensor",
    "crosstalk",
    "fog",
    "incomplete_echo",
    "motion_blur",
    "snow",
    "wet_ground",
)
LIDAR_SEVERITIES = ("light", "moderate", "heavy")
FRAME_PROTOCOLS = ("current", "history_k1", "all_frame")
EXPECTED_ANCHORS = 330


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--occstress-root', dest='occstress_root', type=Path,
                        help='Shared OccStress root; overrides OCCSTRESS_DATA_ROOT.')
    parser.add_argument('--external-root', type=Path, help='External GT root; overrides OCCSTRESS_EXTERNAL_ROOT.')
    parser.add_argument("--builder", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build_upstream_occstress_protocol.py")
    parser.add_argument(
        "--track",
        choices=("camera", "lidar", "fusion_camera", "both"),
        default="both",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def track_spec(track: str) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    if track == "camera":
        return (
            "camera_only", "flashocc_carla",
            CAMERA_CORRUPTIONS, CAMERA_SEVERITIES,
        )
    if track == "lidar":
        return (
            "pointcloud_fusion", "effocc_carla",
            LIDAR_CORRUPTIONS, LIDAR_SEVERITIES,
        )
    if track == "fusion_camera":
        return (
            "pointcloud_fusion", "effocc_carla",
            CAMERA_CORRUPTIONS, CAMERA_SEVERITIES,
        )
    raise ValueError(track)


def run_builder(
    args: argparse.Namespace,
    subtrack: str,
    source_model: str,
    corruption: str,
    severity: str = "clean",
    frame_protocol: str = "",
) -> None:
    upstream_root = (
        args.occstress_root / 'occ/upstream' / DATASET / subtrack / source_model
    )
    command = [
        args.python,
        str(args.builder),
        '--root', str(CODE_ROOT),
        '--dataset', 'carla',
        '--occstress-root', str(args.occstress_root),
        "--backbone-protocol", str(
            args.occstress_root
            / 'protocols/manual' / DATASET / 'clean/H4_F6_val_backbone.pkl'
        ),
        "--clean-input-root", str(upstream_root / "clean"),
        "--subtrack", subtrack,
        "--source-model", source_model,
        "--corruption", corruption,
    ]
    if args.external_root:
        command.extend(['--external-root', str(args.external_root)])
    if corruption != "clean":
        command.extend([
            "--severity", severity,
            "--frame-protocol", frame_protocol,
            "--history-k", "1",
            "--corrupted-input-root", str(
                upstream_root / corruption / severity
            ),
        ])
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def expected_protocols(
    occstress_root: Path,
    subtrack: str,
    source_model: str,
    corruptions: tuple[str, ...],
    severities: tuple[str, ...],
) -> list[tuple[Path, str, str, str]]:
    root = occstress_root / 'protocols/upstream' / DATASET / subtrack / source_model
    output = [(
        root / "clean/H4_F6_val_backbone.pkl",
        "clean", "clean", "clean",
    )]
    for corruption in corruptions:
        for severity in severities:
            for frame_protocol in FRAME_PROTOCOLS:
                output.append((
                    root / corruption / severity
                    / f"{frame_protocol}_H4_F6_val_backbone.pkl",
                    corruption,
                    severity,
                    frame_protocol,
                ))
    return output


def validate_record(
    record: dict,
    corruption: str,
    severity: str,
    frame_protocol: str,
    source_model: str,
    *, occstress_root=None, external_root=None, checked=None,
) -> None:
    if record.get('dataset') != DATASET:
        raise ValueError('upstream protocol lost ' + DATASET + ' dataset identity')
    if record.get("history_length") != 4 or record.get("future_length") != 6:
        raise ValueError("upstream protocol violates H4/F6")
    if len(record["history"]) != 4 or len(record["future_targets"]) != 6:
        raise ValueError("invalid history or future target count")
    if record.get("source_model") != source_model:
        raise ValueError("source model mismatch")

    refs = [*record["history"], record["current_input"]]
    if corruption == "clean":
        expected_variants = ["clean"] * 5
    else:
        corrupted = f"{corruption}/{severity}"
        expected_variants = {
            "current": ["clean", "clean", "clean", "clean", corrupted],
            "history_k1": [
                "clean", "clean", "clean", corrupted, corrupted
            ],
            "all_frame": [corrupted, corrupted, corrupted, corrupted, "clean"],
        }[frame_protocol]
    observed = [ref.get("prediction_variant") for ref in refs]
    if observed != expected_variants:
        raise ValueError(
            f"temporal placement mismatch: {observed} != {expected_variants}"
        )
    for ref in [*refs, record['target'], *record['future_targets']]:
        value = Path(ref['occ_path'])
        if value.is_absolute():
            raise ValueError('Expected a portable release path: ' + str(value))
        path = resolve_occstress_path(value, occstress_root=occstress_root,
                                      external_root=external_root, dataset='carla')
        if checked is not None and path in checked:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        if checked is not None:
            checked.add(path)


def validate_track(
    args: argparse.Namespace,
    track: str,
    subtrack: str,
    source_model: str,
    corruptions: tuple[str, ...],
    severities: tuple[str, ...],
) -> dict:
    protocols = expected_protocols(
        args.occstress_root, subtrack, source_model, corruptions, severities
    )
    if len(protocols) != 73:
        raise RuntimeError(f"expected 73 protocols, got {len(protocols)}")
    reference_anchors = None
    total_records = 0
    backbone_path = args.occstress_root / 'protocols/manual' / DATASET / 'clean/H4_F6_val_backbone.pkl'
    backbone = load_records(backbone_path, trusted_pickle=True)
    validate_records(backbone, expected_anchors=EXPECTED_ANCHORS)
    checked = set()
    for path, corruption, severity, frame_protocol in protocols:
        with path.open("rb") as stream:
            records = pickle.load(stream)
        if len(records) != EXPECTED_ANCHORS:
            raise ValueError(f"{path}: expected {EXPECTED_ANCHORS} anchors")
        validate_records(records, expected_anchors=EXPECTED_ANCHORS)
        if [anchor_key(row) for row in records] != [anchor_key(row) for row in backbone]:
            raise ValueError('Anchor order differs from the clean backbone')
        anchors = [record["anchor_token"] for record in records]
        if len(set(anchors)) != EXPECTED_ANCHORS:
            raise ValueError(f"{path}: duplicate anchors")
        if reference_anchors is None:
            reference_anchors = anchors
        elif anchors != reference_anchors:
            raise ValueError(f"{path}: anchor ordering mismatch")
        for record, clean in zip(records, backbone):
            if record['target'] != clean['target'] or record['future_targets'] != clean['future_targets']:
                raise ValueError('Clean targets changed')
            validate_record(
                record, corruption, severity, frame_protocol, source_model,
                occstress_root=args.occstress_root, external_root=args.external_root, checked=checked,
            )
        total_records += len(records)
    return {
        "status": "success",
        "track": track,
        "subtrack": subtrack,
        "source_model": source_model,
        "protocol_count": len(protocols),
        "anchor_count_per_protocol": EXPECTED_ANCHORS,
        "total_protocol_records": total_records,
        "corruptions": list(corruptions),
        "severities": list(severities),
        "frame_protocols": list(FRAME_PROTOCOLS),
        "protocol_root": str(
            args.occstress_root / 'protocols/upstream' / DATASET / subtrack / source_model
        ),
    }


def main() -> int:
    args = parse_args()
    args.occstress_root = data_root(args.occstress_root, dataset='carla', code_root=CODE_ROOT)
    if args.external_root:
        args.external_root = args.external_root.expanduser().resolve()
    args.builder = args.builder.resolve()
    selected = ("camera", "lidar") if args.track == "both" else (args.track,)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    backbone = args.occstress_root / 'protocols/manual' / DATASET / 'clean/H4_F6_val_backbone.pkl'
    validate_records(load_records(backbone, trusted_pickle=True), expected_anchors=EXPECTED_ANCHORS)
    summary = args.summary or args.occstress_root / 'meta' / DATASET / ('upstream/protocol_validation_' + args.track + '.json')
    if summary.exists() and not args.overwrite:
        raise FileExistsError(summary)
    build_tasks = []
    for track in selected:
        subtrack, source_model, corruptions, severities = track_spec(track)
        if not args.overwrite:
            for path, *_ in expected_protocols(args.occstress_root, subtrack, source_model, corruptions, severities):
                if path.exists():
                    raise FileExistsError(path)
        build_tasks.append((subtrack, source_model, "clean", "clean", ""))
        for corruption in corruptions:
            for severity in severities:
                for frame_protocol in FRAME_PROTOCOLS:
                    build_tasks.append((
                        subtrack, source_model, corruption,
                        severity, frame_protocol,
                    ))

    def build(task: tuple[str, str, str, str, str]) -> None:
        run_builder(args, *task)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(build, build_tasks))

    summaries = {}
    for track in selected:
        subtrack, source_model, corruptions, severities = track_spec(track)
        summaries[track] = validate_track(
            args, track, subtrack, source_model, corruptions, severities
        )
    payload = {
        "status": "success",
        'dataset': DATASET,
        'occstress_root': str(args.occstress_root),
        "tracks": summaries,
    }
    atomic_json(summary.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
