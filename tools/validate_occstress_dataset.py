#!/usr/bin/env python3
"""Validate release manifests without recursively scanning millions of assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occstress.datasets.paths import DATASETS, data_root, dataset_name, resolve_occstress_path


def validate(root, include_counts=False, datasets=None, require_payloads=False):
    root = Path(root).expanduser().resolve()
    issues, notes = [], []
    try:
        catalog = json.loads((root / 'manifests/protocols.json').read_text())
        if not isinstance(catalog, list):
            raise ValueError('Protocol catalog must be a list')
    except (OSError, ValueError) as error:
        return [str(error)], notes
    names = [dataset_name(key) for key in datasets] if datasets else sorted({r['dataset'] for r in catalog})
    for name in names:
        try:
            metadata = json.loads((root / 'meta' / name / 'dataset.json').read_text())
            if metadata['dataset'] != name:
                raise ValueError('Metadata dataset identity mismatch')
            mapping = root / 'meta' / name / 'class_mapping.json'
            if hashlib.sha256(mapping.read_bytes()).hexdigest() != metadata['class_mapping_sha256']:
                raise ValueError('Class mapping checksum mismatch')
            controls = metadata.get('controls', {})
            if controls.get('path'):
                control_path = resolve_occstress_path('meta/' + name + '/' + controls['path'],
                                                      occstress_root=root, dataset=name)
                if hashlib.sha256(control_path.read_bytes()).hexdigest() != controls['sha256']:
                    raise ValueError('Control metadata checksum mismatch')
            rows = [row for row in catalog if row['dataset'] == name]
            if not rows:
                raise ValueError('No catalogued protocols')
            for row in rows:
                path = resolve_occstress_path(row['path'], occstress_root=root, dataset=name)
                if not path.is_file():
                    issues.append('Missing protocol: ' + row['path'])
                if row['anchors'] != metadata['anchor_count']:
                    issues.append('Anchor count mismatch: ' + row['path'])
            notes.append(f'{name}: {len(rows)} catalogued protocols; {metadata["anchor_count"]} anchors each')
            progress_path = root / 'manifests' / (name + '.payload_progress.json')
            progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
            notes.append(f'{name}: payload status={progress.get("status", "not_started")}')
            if require_payloads:
                conditions = progress.get('complete_conditions', [])
                if progress.get('status') != 'complete' or progress.get('pending_conditions') or not conditions:
                    issues.append(name + ': payload assembly incomplete')
                for condition in conditions:
                    marker = json.loads((root / condition / '.done.json').read_text())
                    if marker.get('status') != 'payload_validated':
                        issues.append('Invalid completion marker: ' + condition)
            if include_counts:
                count = sum(1 for _ in (root / 'occ').glob('*/' + name + '/**/labels.npz'))
                notes.append(f'{name}: recursive labels.npz count={count}')
        except (OSError, ValueError, KeyError) as error:
            issues.append(name + ': ' + str(error))
    notes.append('This checks catalog/metadata/layout, not model inference or external GT availability.')
    return issues, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default=str(data_root()))
    parser.add_argument('--dataset', action='append', choices=sorted(DATASETS))
    parser.add_argument('--require-payloads', action='store_true')
    parser.add_argument('--count-files', action='store_true', help='Opt in to an expensive recursive asset count.')
    args = parser.parse_args()
    issues, notes = validate(args.root, args.count_files, args.dataset, args.require_payloads)
    print(json.dumps({'issues': issues, 'notes': notes, 'gpu_gate': 'not_run'}, indent=2))
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
