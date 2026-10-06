#!/usr/bin/env python3
"""Check derived method metadata and selected-source hashes; write only on request."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.distribution import source_status, withheld


def derived_methods(root):
    contract = json.loads((root / 'configs/evaluation_contract.json').read_text())
    return {'schema_version': 1, 'generated_from': 'configs/evaluation_contract.json',
            'future_horizons_seconds': contract['timeline']['future_horizons_seconds'],
            'note': 'Adapter capabilities and internal validation are distinct from public source availability. See distribution.json.',
            'methods': [{'id': name, 'directory': spec['directory'],
                         'entrypoints': sorted(set(spec['entrypoints'].values())),
                         'configuration': spec['configs']['nuscenes'],
                         'input_offsets_seconds': spec['input_offsets_seconds'],
                         'tracks': spec['tracks'], 'status': contract['status'],
                         'source_status': source_status(root, spec['directory'])}
                        for name, spec in contract['models'].items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Maintainer only: record reviewed code changes')
    args = parser.parse_args()
    path = ROOT / 'configs/methods.json'
    expected = derived_methods(ROOT)
    errors = []
    for relative, group in [('environments/profiles.json', 'models'),
                            ('configs/upstream_methods.json', 'sources')]:
        registry_path = ROOT / relative
        registry = json.loads(registry_path.read_text())
        entries = registry[group].values() if isinstance(registry[group], dict) else registry[group]
        for entry in entries:
            status = source_status(ROOT, entry['directory'])
            if args.write:
                entry['source_status'] = status
            elif entry.get('source_status') != status:
                errors.append('Source availability differs from distribution profile: ' + entry['directory'])
        if args.write:
            registry_path.write_text(json.dumps(registry, indent=2) + '\n')
    if args.write:
        path.write_text(json.dumps(expected, indent=2) + '\n')
    elif json.loads(path.read_text()) != expected:
        errors.append('methods.json differs from the authoritative evaluation contract')
    path = ROOT / 'docs/source-imports.json'
    ledger = json.loads(path.read_text())
    for row in ledger['files']:
        if withheld(ROOT, row['path']):
            continue
        actual = hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest()
        if args.write:
            row['release_sha256'] = actual
        elif actual != row.get('release_sha256'):
            errors.append('Imported-source drift: ' + row['path'])
    if args.write:
        path.write_text(json.dumps(ledger, indent=2) + '\n')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Derived metadata and selected-source hashes are current.')


if __name__ == '__main__':
    main()
