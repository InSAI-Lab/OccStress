#!/usr/bin/env python3
"""Validate and hash the formal config inheritance graph, without importing CUDA."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.distribution import withheld


def config_closure(path, root, active=None):
    path = path.resolve()
    root = root.resolve()
    path.relative_to(root)
    active = set() if active is None else set(active)
    if path in active:
        raise ValueError(f'Cyclic config inheritance: {path.relative_to(root)}')
    active.add(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    result = {path}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
            bases = ast.literal_eval(node.value)
            for base in [bases] if isinstance(bases, str) else bases:
                result.update(config_closure(path.parent / base, root, active))
    return result


def expected_files(root=ROOT):
    contract_path = root / 'configs/evaluation_contract.json'
    upstream_path = root / 'configs/upstream_methods.json'
    contract = json.loads(contract_path.read_text())
    environments = json.loads((root / 'environments/profiles.json').read_text())
    files = {contract_path, upstream_path, root / 'environments/profiles.json',
             root / 'configs/distribution.json'}
    files.update((root / 'occstress').rglob('*.py'))
    # Dataset adapters affect inputs and must invalidate completed-run receipts.
    files.update(root / path for path in (
        'EXIST/4D/II-World/mmdet3d/datasets/nuscenes_occstress_world_dataset.py',
        'EXIST/4D/II-World/mmdet3d/datasets/waymo_occstress_world_dataset.py',
        'EXIST/4D/GenieDrive/occ_gen/mmdet3d/datasets/nuscenes_occstress_world_dataset.py',
        'EXIST/4D/GenieDrive/occ_gen/mmdet3d/datasets/waymo_occstress_world_dataset.py',
        'EXIST/4D/COME/dataset/dataset.py',
        'EXIST/4D/OccWorld/dataset/dataset.py',
        'EXIST/4D/DOME/dataset/occstress_dataset.py',
        'EXIST/4D/SparseWorld/occstress/carla_dataset.py',
        'EXIST/4D/SparseWorld/occstress/waymo_dataset.py',
        'EXIST/4D/SparseWorld/occstress_adapters/__init__.py',
    ))
    files.update((root / 'configs/suites').glob('*.json'))
    files.update(root / name for name in ('tools/run_formal.py', 'tools/run_manifest.py',
                                         'tools/summarize.py', 'tools/update_release_metadata.py'))
    files.update(root / name for name in (
        'scripts/build_upstream_occstress_protocol.py',
        'tools/prepare_external_data.py',
        'scripts/carla/build_carla_occstress.py',
        'scripts/waymo/waymo_occstress_common.py',
        'scripts/alocc/build_nuscenes_occstress_protocols.py',
        'scripts/stcocc/build_occstress_upstream_protocols_from_nuscc.sh',
    ))
    for name, spec in contract['models'].items():
        if withheld(root, spec['directory']):
            continue
        if name not in environments['models']:
            raise ValueError(f'Missing isolated environment: {name}')
        offsets = spec['input_offsets_seconds']
        canonical = contract['timeline']['canonical_observed_offsets_seconds']
        if offsets != canonical[-len(offsets):]:
            raise ValueError(f'Invalid observed offsets: {name}')
        directory = root / spec['directory']
        configs = set(spec['configs'].values()) | set(spec.get('tokenizer_configs', {}).values())
        configs.update(spec.get('additional_configs', []))
        for config in configs:
            files.update(config_closure(directory / config, root))
        files.update(directory / entry for entry in spec['entrypoints'].values())
    for spec in json.loads(upstream_path.read_text())['sources']:
        if withheld(root, spec['directory']):
            continue
        if spec['environment'] not in environments['models']:
            raise ValueError('Missing upstream environment: ' + spec['id'])
        if spec['corrupted_sensor'] not in spec['sensor_inputs']:
            raise ValueError('Corrupted sensor is not an input: ' + spec['id'])
        files.update(config_closure(root / spec['directory'] / spec['config'], root))
        files.update(root / spec[key] for key in ('exporter', 'protocol_builder'))
    for recipe in environments['recipes'].values():
        files.add(root / recipe['requirements'])
    return {path for path in files if not withheld(root, path.relative_to(root).as_posix())}


def make_lock(root=ROOT):
    return {
        'schema_version': 1,
        'scope': 'Formal config inheritance, declared entrypoints, environment recipes and contracts. Not a checkpoint/data lock or a GPU gate.',
        'files': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(expected_files(root))},
    }


def issues(root=ROOT):
    try:
        current = make_lock(root)
        recorded = json.loads((root / 'configs/formal_configs.lock.json').read_text())
    except (OSError, ValueError, SyntaxError) as error:
        return [str(error)]
    return [f'Formal source drift: {name}'
            for name in sorted(set(current['files']) | set(recorded['files']))
            if current['files'].get(name) != recorded['files'].get(name)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-lock', action='store_true', help='Maintainer only: deliberately accept reviewed source changes.')
    args = parser.parse_args()
    if args.write_lock:
        payload = make_lock()
        (ROOT / 'configs/formal_configs.lock.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(f'Locked {len(payload["files"])} formal source/config files')
    else:
        errors = issues()
        print(json.dumps({'issues': errors, 'gpu_gate': 'not_run'}, indent=2))
        return int(bool(errors))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
