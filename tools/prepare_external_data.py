#!/usr/bin/env python3
"""Mount authorized source data and prepare only the external CARLA clean GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.datasets.paths import data_root, dataset_key, dataset_name, resolve_occstress_path
from occstress.protocols.validation import load_records, validate_records
from occstress.results.io import write_json
from scripts.carla.build_carla_occstress import EXPECTED_SCENES, map_uniocc_semantics

KINDS = {
    'nuscenes': {'gts', 'world-nuscenes_infos_val.pkl',
                 'nuscenes_infos_val_temporal_v3_scene.pkl', 'native_dataset'},
    'waymo': {'native_gt', 'cam_infos_vali.pkl', 'native_dataset'},
    'carla': {'canonical_gt', 'native_dataset'},
}
ANCHORS = {'nuscenes': 4519, 'waymo': 5978, 'carla': 330}


def external_root(value):
    value = value or os.environ.get('OCCSTRESS_EXTERNAL_ROOT')
    if not value:
        raise ValueError('Set --external-root or OCCSTRESS_EXTERNAL_ROOT')
    return Path(value).expanduser().resolve()


def mount(root, dataset, kind, source):
    if kind not in KINDS[dataset]:
        raise ValueError(f'{kind} is not an external dependency for {dataset}')
    source = Path(source).expanduser().resolve(strict=True)
    if (kind.endswith('.pkl') and (not source.is_file() or not source.stat().st_size)
            or not kind.endswith('.pkl') and not source.is_dir()):
        raise ValueError('Wrong file/directory type: ' + str(source))
    if kind == 'native_gt' and not (source / 'validation-data').is_dir():
        raise ValueError('Waymo --source must contain validation-data (the Occ3D voxel04 root)')
    parent = root / dataset_name(dataset)
    if parent.is_symlink():
        raise ValueError('Use a real dataset namespace directory, not a symlink: ' + str(parent))
    target = parent / kind
    if source == target or source.is_relative_to(target):
        raise ValueError('Mount source must be outside its destination')
    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            return {'status': 'already_mounted', 'kind': kind}
        raise FileExistsError('Existing mount is never replaced: ' + str(target))
    parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())
    return {'status': 'mounted', 'kind': kind, 'read_only_enforced': False}


def carla_frames(shared, raw):
    meta = shared / 'meta/OccStress-CARLA'
    dataset = json.loads((meta / 'dataset.json').read_text())
    controls = json.loads((meta / 'controls.json').read_text())
    if not controls['metadata'].get('model_view') or controls['metadata'].get(
            'coordinate_convention') != 'Occ3D right-handed ego-local x-forward/y-left/z-up':
        raise ValueError('Expected published right-handed CARLA controls')
    expected_hash = dataset.get('source_scene_info_sha256')
    if not expected_hash or hashlib.sha256((raw / 'scene_infos.pkl').read_bytes()).hexdigest() != expected_hash:
        raise ValueError('scene_infos.pkl differs from the published CARLA source revision')
    infos = controls['infos']
    if {scene: len(rows) for scene, rows in infos.items()} != EXPECTED_SCENES:
        raise ValueError('CARLA release must contain the fixed three-scene, 360-frame inventory')
    frames = []
    for scene, rows in sorted(infos.items()):
        for index, row in enumerate(rows):
            token = f"carla-val-{scene.removeprefix('scene_')}-{index:03d}"
            if row['scene_name'] != scene or row['frame_idx'] != index or row['token'] != token:
                raise ValueError('Unexpected CARLA frame order or token')
            expected = f'external/OccStress-CARLA/canonical_gt/{scene}/{token}/labels.npz'
            if row['occ_path'] != expected:
                raise ValueError('Unexpected canonical GT path')
            source = raw / scene / f'{index}.npz'
            if not source.is_file():
                raise FileNotFoundError(source)
            frames.append((source, Path(scene) / token / 'labels.npz'))
    return frames


def carla_gt(shared, external, raw, max_frames=0):
    raw = Path(raw).expanduser().resolve(strict=True)
    frames = carla_frames(shared, raw)
    destination_root = external / 'OccStress-CARLA/canonical_gt'
    if destination_root.resolve() == raw or destination_root.resolve().is_relative_to(raw):
        raise ValueError('Generated GT must stay outside the raw source directory')
    total = len(frames)
    if max_frames < 0:
        raise ValueError('--max-frames must be nonnegative')
    if max_frames:
        frames = frames[:max_frames]
    created = reused = 0
    for source, relative in frames:
        with np.load(source, allow_pickle=False) as payload:
            semantics = map_uniocc_semantics(payload['occ_label'])
            mask = np.asarray(payload['occ_mask_camera'])
        if semantics.shape != (200, 200, 16) or mask.shape != semantics.shape:
            raise ValueError('Unexpected native CARLA grid: ' + str(source))
        if not np.isin(mask, (0, 1)).all():
            raise ValueError('Invalid camera mask: ' + str(source))
        # Native arrays are left-handed; released controls are already right-handed.
        semantics = np.ascontiguousarray(np.flip(semantics, axis=1))
        mask = np.ascontiguousarray(np.flip(mask.astype(bool), axis=1))
        target = destination_root / relative
        if not target.resolve().is_relative_to(external) or target.is_symlink():
            raise ValueError('Refusing to write through an external GT symlink')
        if target.exists():
            with np.load(target, allow_pickle=False) as existing:
                if not (np.array_equal(existing['semantics'], semantics)
                        and np.array_equal(existing['infov'], mask)):
                    raise ValueError('Existing GT differs; never overwritten: ' + str(target))
            reused += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.labels.', suffix='.tmp', dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, semantics=semantics, infov=mask)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        created += 1
    report = {'dataset': 'OccStress-CARLA', 'status': 'complete' if len(frames) == total else 'partial',
              'checked_frames': len(frames), 'expected_frames': total,
              'created': created, 'reused': reused, 'controls_modified': False,
              'conversion': 'UniOcc-to-Occ3D class map; flip axis 1 exactly once'}
    if len(frames) == total:
        write_json(destination_root / '.prepared.json', report)
    return report


def check(shared, external, dataset, trusted_pickle=False):
    name = dataset_name(dataset)
    backbone = shared / 'protocols/manual' / name / 'clean/H4_F6_val_backbone.pkl'
    rows = load_records(backbone, trusted_pickle=trusted_pickle)
    validate_records(rows, expected_anchors=ANCHORS[dataset])
    paths = {frame['occ_path'] for row in rows for frame in
             [*row['history'], row['current_input'], row['target'], *row['future_targets']]}
    missing = []
    for value in sorted(paths):
        path = resolve_occstress_path(value, occstress_root=shared, external_root=external, dataset=dataset)
        if not path.is_file() or not path.stat().st_size:
            missing.append(value)
    metadata = ([external / name / 'world-nuscenes_infos_val.pkl',
                 external / name / 'nuscenes_infos_val_temporal_v3_scene.pkl'] if dataset == 'nuscenes'
                else [shared / 'meta' / name / 'controls.json'])
    for path in metadata:
        if not path.is_file() or not path.stat().st_size:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing external dependencies; first: {missing[:5]}')
    return {'dataset': name, 'status': 'passed', 'anchors': len(rows),
            'distinct_references': len(paths), 'check': 'all clean references exist and are nonempty',
            'voxel_content_checked': False, 'gpu_inference_run': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('mount', 'carla-gt', 'check'):
        item = sub.add_parser(name)
        item.add_argument('--external-root', type=Path)
        if name != 'mount':
            item.add_argument('--occstress-root', type=Path)
        if name != 'carla-gt':
            item.add_argument('--dataset', type=dataset_key, required=True)
    sub.choices['mount'].add_argument('--kind', required=True)
    sub.choices['mount'].add_argument('--source', type=Path, required=True)
    sub.choices['carla-gt'].add_argument('--source-root', type=Path, required=True)
    sub.choices['carla-gt'].add_argument('--max-frames', type=int, default=0,
                                        help='Partial smoke only; 0 prepares all 360 frames')
    sub.choices['check'].add_argument('--trust-pickle', action='store_true')
    args = parser.parse_args()
    external = external_root(args.external_root)
    if args.command == 'mount':
        report = mount(external, args.dataset, args.kind, args.source)
    elif args.command == 'carla-gt':
        report = carla_gt(data_root(args.occstress_root), external, args.source_root, args.max_frames)
    else:
        report = check(data_root(args.occstress_root), external, args.dataset, args.trust_pickle)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
