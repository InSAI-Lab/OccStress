#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from occstress.datasets.paths import data_root, dataset_name
from occstress.protocols.validation import anchor_key, load_records, validate_records

DATASET = dataset_name('nuscenes')
EXPECTED_ANCHORS = 4519
EXPECTED_FRAMES = 6019

CORRUPTIONS = (
    "Brightness",
    "CameraCrash",
    "ColorQuant",
    "Fog",
    "FrameLost",
    "LowLight",
    "MotionBlur",
    "Snow",
)
SEVERITIES = ("easy", "mid", "hard")
FRAME_PROTOCOLS = ("current", "history_k1", "all_frame")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build all ALOcc camera-only OccStress-nuScenes protocols from trusted local assets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument('--occstress-root', type=Path, help='Shared OccStress root; overrides OCCSTRESS_DATA_ROOT.')
    parser.add_argument('--export-root', type=Path, help='ALOcc export root mounted inside the shared root.')
    parser.add_argument('--backbone-protocol', type=Path, help='Trusted clean release backbone.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def setting_root(occ_root, corruption, severity):
    if corruption == "clean":
        return occ_root / "clean"
    return occ_root / corruption / severity


def require_exports(occ_root):
    settings = [("clean", "clean")]
    settings.extend(
        (corruption, severity)
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    )
    done_payloads = []
    for corruption, severity in settings:
        done_path = (
            setting_root(occ_root, corruption, severity) / ".done.json"
        )
        try:
            payload = json.loads(done_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid export marker {done_path}: {exc}")
        native = payload.get('status') == 'success' and payload.get('frame_count') == EXPECTED_FRAMES
        released = payload.get('status') == 'payload_validated' and payload.get('files') == EXPECTED_FRAMES
        if not (native or released):
            raise RuntimeError(f"incomplete export marker: {done_path}")
        done_payloads.append(payload)
    return done_payloads


def run_builder(
    builder,
    root,
    backbone,
    clean_root,
    corruption,
    severity=None,
    frame_protocol=None,
    *, occstress_root=None, overwrite=False,
):
    command = [
        sys.executable,
        str(builder),
        "--root",
        str(root),
        '--dataset', 'nuscenes',
        '--occstress-root', str(data_root(occstress_root, code_root=root)),
        "--backbone-protocol",
        str(backbone),
        "--subtrack",
        "camera_only",
        "--source-model",
        "alocc",
        "--corruption",
        corruption,
        "--clean-input-root",
        str(clean_root),
    ]
    if overwrite:
        command.append('--overwrite')
    if corruption != "clean":
        corrupted_root = clean_root.parent / corruption / severity
        command.extend(
            [
                "--severity",
                severity,
                "--frame-protocol",
                frame_protocol,
                "--history-k",
                "1",
                "--corrupted-input-root",
                str(corrupted_root),
            ]
        )
    subprocess.run(command, check=True)


def protocol_paths(protocol_root, stem='H4_F6_val_backbone'):
    paths = [protocol_root / 'clean' / f'{stem}.pkl']
    paths.extend(
        protocol_root
        / corruption
        / severity
        / f"{frame_protocol}_{stem}.pkl"
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
        for frame_protocol in FRAME_PROTOCOLS
    )
    return paths


def main():
    args = parse_args()
    root = args.root.resolve()
    shared = data_root(args.occstress_root, dataset='nuscenes', code_root=root)
    occ_root = (args.export_root.expanduser().absolute() if args.export_root else
                shared / 'occ/upstream' / DATASET / 'camera_only/alocc')
    protocol_root = shared / 'protocols/upstream' / DATASET / 'camera_only/alocc'
    meta_root = shared / 'meta' / DATASET / 'upstream/camera_only/alocc'
    backbone = (args.backbone_protocol.expanduser().absolute() if args.backbone_protocol else
                shared / 'protocols/manual' / DATASET / 'clean/H4_F6_val_backbone.pkl')
    reference = load_records(backbone, trusted_pickle=True)
    validate_records(reference, expected_anchors=EXPECTED_ANCHORS)
    paths = protocol_paths(protocol_root, backbone.stem)
    if not args.overwrite:
        existing = [path for path in [*paths, meta_root / 'manifest.json', meta_root / 'protocol_build.done.json'] if path.exists()]
        if existing:
            raise FileExistsError('Pass --overwrite explicitly to replace: ' + str(existing[0]))
    builder = root / "scripts" / "build_upstream_occstress_protocol.py"

    done_payloads = require_exports(occ_root)
    clean_root = occ_root / "clean"
    run_builder(
        builder,
        root,
        backbone,
        clean_root,
        corruption="clean",
        occstress_root=shared, overwrite=args.overwrite,
    )
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            for frame_protocol in FRAME_PROTOCOLS:
                run_builder(
                    builder,
                    root,
                    backbone,
                    clean_root,
                    corruption,
                    severity,
                    frame_protocol,
                    occstress_root=shared, overwrite=args.overwrite,
                )

    invalid = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                records = pickle.load(handle)
            validate_records(records, expected_anchors=EXPECTED_ANCHORS)
            if [anchor_key(row) for row in records] != [anchor_key(row) for row in reference]:
                raise ValueError('Anchor order differs from clean backbone')
            for row, clean in zip(records, reference):
                if row['target'] != clean['target'] or row['future_targets'] != clean['future_targets']:
                    raise ValueError('Clean targets changed')
        except Exception as exc:
            invalid.append(f"{path}: {exc}")
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} invalid protocols: {invalid[:3]}"
        )

    provenance = {
        (
            payload.get('checkpoint_sha256'),
            payload.get('code_commit'),
            payload.get('config'),
        )
        for payload in done_payloads
    }
    if len(provenance) != 1:
        raise RuntimeError(f"export provenance differs: {provenance}")
    checkpoint_sha, code_commit, config = provenance.pop()
    manifest = {
        "status": "success",
        "track": "upstream",
        "subtrack": "camera_only",
        "source_model": "alocc",
        "clean_reference": True,
        "corruption_family_count": len(CORRUPTIONS),
        "corruption_families": list(CORRUPTIONS),
        "severity_count": len(SEVERITIES),
        "severities": list(SEVERITIES),
        "frame_protocol_count": len(FRAME_PROTOCOLS),
        "frame_protocols": list(FRAME_PROTOCOLS),
        "occ_setting_count": 1 + len(CORRUPTIONS) * len(SEVERITIES),
        "protocol_count": len(paths),
        "records_per_protocol": EXPECTED_ANCHORS,
        'dataset': DATASET,
        'export_receipt_statuses': sorted({row['status'] for row in done_payloads}),
        'checkpoint_provenance_available': checkpoint_sha is not None,
        "checkpoint_sha256": checkpoint_sha,
        "code_commit": code_commit,
        "config": config,
        "backbone_protocol": str(backbone),
        "clean_occ_root": str(clean_root),
        "protocol_root": str(protocol_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(meta_root / "manifest.json", manifest)
    atomic_json(meta_root / "protocol_build.done.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
