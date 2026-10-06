#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import copy
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
OCCSTRESS_CODE_ROOT = SCRIPT_PATH.parent.parent
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(OCCSTRESS_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(OCCSTRESS_CODE_ROOT))

from occstress.protocols.temporal import history_mask_for_protocol, target_active_for_protocol
from occstress.datasets.paths import data_root, dataset_key, dataset_name, resolve_occstress_path
from occstress.protocols.validation import load_records, validate_records
from occstress.results.io import write_json
from scripts.occstress_layout import resolve_upstream_protocol_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one nested upstream OccStress protocol from a clean backbone protocol."
    )
    parser.add_argument('--dataset', type=dataset_key, default='nuscenes', choices=['nuscenes', 'waymo', 'carla'])
    parser.add_argument('--backbone-protocol', help='Trusted local clean H4/F6 pickle; defaults to the namespaced manual backbone.')
    parser.add_argument("--clean-input-root", required=True)
    parser.add_argument("--corrupted-input-root", default="")
    parser.add_argument("--subtrack", choices=["camera_only", "camera_fusion", "pointcloud_fusion"], required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--corruption", required=True, help="Use 'clean' for the upstream clean reference protocol.")
    parser.add_argument("--severity", default="clean")
    parser.add_argument("--frame-protocol", choices=["current", "history_k1", "all_frame"], default="")
    parser.add_argument("--history-k", type=int, default=1)
    parser.add_argument(
        "--target-root",
        default="",
        help="Optional canonical clean GT root; by default backbone targets are preserved.",
    )
    parser.add_argument("--root", default=str(OCCSTRESS_CODE_ROOT))
    parser.add_argument('--occstress-root', dest='occstress_root',
                        help='Shared OccStress root; overrides OCCSTRESS_DATA_ROOT. The old flag is an alias, not a per-dataset root.')
    parser.add_argument('--external-root', help='External assets root; overrides OCCSTRESS_EXTERNAL_ROOT.')
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_root(path, root):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    # Preserve logical paths below dataset mounts, not their symlink targets.
    return Path(os.path.abspath(path))


def occ_path_from_root(root, scene_name, token):
    for value in (scene_name, token):
        if not value or Path(value).name != value or value in {'.', '..'}:
            raise ValueError('Scene and frame identifiers must be single path components')
    return str(root / scene_name / token / "labels.npz")


def rewrite_target_paths(record, target_root):
    if target_root is None:
        return
    scene_name = record["scene_name"]
    target = record.get("target")
    if target and target.get("token"):
        target["occ_path"] = occ_path_from_root(target_root, scene_name, target["token"])
    for future in record.get("future_targets", []):
        token = future.get("token")
        if token:
            future["occ_path"] = occ_path_from_root(target_root, scene_name, token)


def require_exists(path, missing, sample_id, role, checked=None):
    if checked is not None and path in checked:
        return True
    if not path.is_file():
        missing.append((sample_id, role, str(path)))
        return False
    if checked is not None:
        checked.add(path)
    return True


def build_record(record, args, clean_root, corrupted_root, missing):
    sample_id = record["sample_id"]
    out = copy.deepcopy(record)
    out["track"] = "upstream"
    out["subtrack"] = args.subtrack
    out["source_model"] = args.source_model
    out["reference_protocol"] = "clean/H4_F6_val_backbone"
    rewrite_target_paths(out, args.target_root_resolved)

    history_length = int(record["history_length"])
    if args.corruption == "clean":
        history_mask = [1] * history_length
        current_corrupted = True
    else:
        history_mask = history_mask_for_protocol(history_length, args.frame_protocol, args.history_k)
        current_corrupted = target_active_for_protocol(args.frame_protocol)

    for idx, hist in enumerate(out["history"]):
        scene_name = out["scene_name"]
        token = hist["token"]
        use_corrupted = bool(history_mask[idx])
        chosen_root = clean_root if args.corruption == "clean" or not use_corrupted else corrupted_root
        chosen_path = Path(occ_path_from_root(chosen_root, scene_name, token))
        require_exists(chosen_path, missing, sample_id, f"history[{idx}]", getattr(args, 'checked_paths', None))
        hist["occ_path"] = str(chosen_path)
        hist["occ_source"] = args.source_model
        hist["prediction_variant"] = "clean" if chosen_root == clean_root else f"{args.corruption}/{args.severity}"
        hist["event_path"] = None

    current_token = out["current_input"]["token"]
    current_root = clean_root if args.corruption == "clean" or not current_corrupted else corrupted_root
    current_path = Path(occ_path_from_root(current_root, out["scene_name"], current_token))
    require_exists(current_path, missing, sample_id, "current_input", getattr(args, 'checked_paths', None))
    out["current_input"]["occ_path"] = str(current_path)
    out["current_input"]["source"] = args.source_model
    out["current_input"]["prediction_variant"] = (
        "clean" if current_root == clean_root else f"{args.corruption}/{args.severity}"
    )
    out["current_input"]["event_path"] = None

    if args.corruption == "clean":
        out["sample_id"] = f"{record['anchor_token']}__H{record['history_length']}__F{record['future_length']}__{args.source_model}_clean"
        out["corruption"] = {
            "type": "clean",
            "severity": "clean",
            "frame_protocol": "clean",
            "k": None,
        }
    else:
        out["sample_id"] = (
            f"{record['anchor_token']}__H{record['history_length']}__F{record['future_length']}__"
            f"{args.source_model}_{args.corruption}_{args.severity}__{args.frame_protocol}"
        )
        out["corruption"] = {
            "type": args.corruption,
            "severity": args.severity,
            "frame_protocol": args.frame_protocol,
            "k": args.history_k,
        }

    return out


def protocol_output_path(args, backbone_path, root):
    stem = backbone_path.stem
    if args.corruption == "clean":
        protocol_stem = stem
    else:
        prefix = args.frame_protocol
        # Preserve the filenames used by the published STCOcc catalog.
        if (dataset_key(args.dataset) == 'nuscenes' and args.subtrack == 'camera_only'
                and args.source_model == 'stcocc'):
            prefix = {'current': 'current_only', 'history_k1': 'recent_burst',
                      'all_frame': 'history_only'}[prefix]
        protocol_stem = f"{prefix}_{stem}"
    severity = "clean" if args.corruption == "clean" else args.severity
    return resolve_upstream_protocol_path(
        args.subtrack, args.source_model, args.corruption, severity,
        protocol_stem, root=root, must_exist=False, dataset=args.dataset,
        occstress_root=args.output_occstress_root_resolved,
    )


def summary_output_path(args, output_path, root):
    occstress_root = data_root(args.output_occstress_root_resolved, dataset=args.dataset, code_root=root)
    meta_root = occstress_root / 'meta' / dataset_name(args.dataset) / 'upstream' / args.subtrack / args.source_model
    meta_root = meta_root / args.corruption
    if args.corruption != "clean":
        meta_root = meta_root / args.severity
    return meta_root / f"{output_path.stem}.json"


def portable_record_paths(record, args, root, missing):
    namespace = dataset_name(args.dataset)
    if record.get('dataset', namespace) != namespace:
        raise ValueError('Use a release-normalized backbone for ' + namespace)
    shared = args.output_occstress_root_resolved
    external = args.external_root or os.environ.get('OCCSTRESS_EXTERNAL_ROOT')
    external = normalize_root(external, root) if external else None
    frames = [*record['history'], record['current_input'], record['target'], *record['future_targets']]
    for frame in frames:
        value = Path(frame['occ_path'])
        if value.is_absolute():
            value = normalize_root(value, root)
            if value.is_relative_to(shared):
                value = value.relative_to(shared)
            elif external is not None and value.is_relative_to(external):
                value = Path('external') / value.relative_to(external)
            else:
                raise ValueError('Mount occupancy/GT under the shared or external root before building: ' + str(value))
        if value.parts[:2] == ('data', 'OccStress'):
            value = Path(*value.parts[2:])
        if value.parts[:3] not in {('occ', 'upstream', namespace), ('occ', 'manual', namespace)} and value.parts[:2] != ('external', namespace):
            raise ValueError('Occupancy path lacks the selected dataset namespace: ' + str(value))
        resolved = resolve_occstress_path(value, code_root=root, occstress_root=shared,
                                         dataset=args.dataset, external_root=external)
        require_exists(resolved, missing, record['sample_id'], 'occ_path', args.checked_paths)
        frame['occ_path'] = value.as_posix()


def atomic_pickle(path, records, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Fail rather than clobber a concurrent build.
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    args.output_occstress_root_resolved = data_root(args.occstress_root, dataset=args.dataset, code_root=root)
    backbone_path = (normalize_root(args.backbone_protocol, root) if args.backbone_protocol else
                     args.output_occstress_root_resolved / 'protocols/manual' / dataset_name(args.dataset) / 'clean/H4_F6_val_backbone.pkl')
    clean_root = normalize_root(args.clean_input_root, root)
    corrupted_root = normalize_root(args.corrupted_input_root, root) if args.corrupted_input_root else None
    args.target_root_resolved = normalize_root(args.target_root, root) if args.target_root else None
    args.checked_paths = set()
    for value in (args.source_model, args.corruption, args.severity):
        if Path(value).name != value or value in {'.', '..'}:
            raise ValueError('Source, corruption and severity must be single path components')

    if args.corruption == "clean":
        args.severity = "clean"
        args.frame_protocol = ""
    else:
        if not args.frame_protocol:
            raise ValueError("--frame-protocol is required for non-clean upstream protocols")
        if corrupted_root is None:
            raise ValueError("--corrupted-input-root is required for non-clean upstream protocols")

    output_path = protocol_output_path(args, backbone_path, root)
    summary_path = summary_output_path(args, output_path, root)
    for path in (output_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    backbone_records = load_records(backbone_path, trusted_pickle=True)
    validate_records(backbone_records)
    for record in backbone_records:
        if record.get('corruption', {}).get('type') != 'clean':
            raise ValueError('The backbone must be clean, not a corrupted protocol')

    missing = []
    out_records = [
        build_record(record, args, clean_root, corrupted_root, missing) for record in backbone_records
    ]
    for record in out_records:
        portable_record_paths(record, args, root, missing)
    validate_records(out_records, expected_anchors=len(backbone_records))

    if missing:
        preview = "\n".join(f"{sid}\t{role}\t{path}" for sid, role, path in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} upstream prediction files while building protocol.\n{preview}"
        )

    atomic_pickle(output_path, out_records, args.overwrite)

    summary = {
        'dataset': dataset_name(args.dataset),
        "track": "upstream",
        "subtrack": args.subtrack,
        "source_model": args.source_model,
        "corruption": args.corruption,
        "severity": args.severity,
        "frame_protocol": "clean" if args.corruption == "clean" else args.frame_protocol,
        "history_k": None if args.corruption == "clean" else args.history_k,
        "backbone_protocol": str(backbone_path),
        "clean_input_root": str(clean_root),
        "corrupted_input_root": str(corrupted_root) if corrupted_root else None,
        "target_root": str(args.target_root_resolved) if args.target_root_resolved else None,
        "num_records": len(out_records),
        "output_path": str(output_path),
        'occstress_root': str(args.output_occstress_root_resolved),
    }
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
