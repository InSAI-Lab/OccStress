#!/usr/bin/env python3
"""Read package metadata without importing frameworks or exposing local paths."""
import argparse
from importlib.metadata import distributions
import json
from pathlib import Path


def capture(prefix):
    sites = sorted({p.resolve() for p in prefix.glob('lib/python*/site-packages')})
    versions = {}
    for dist in distributions(path=[str(p) for p in sites]):
        name = dist.metadata.get('Name')
        if name:
            versions.setdefault(name.lower().replace('_', '-'), set()).add(dist.version)
    return {
        'python_series': sorted({p.parent.name.removeprefix('python') for p in sites}),
        'packages': {name: sorted(values) for name, values in sorted(versions.items())},
        'ambiguous_packages': sorted(name for name, values in versions.items() if len(values) > 1),
        'note': 'Observed metadata, not a fresh-install lock or proof of GPU compatibility. Paths and credentials omitted.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', action='append', required=True, metavar='LABEL=PREFIX')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {}
    for value in args.prefix:
        label, prefix = value.split('=', 1)
        result[label] = capture(Path(prefix).expanduser())
        if not result[label]['packages']:
            raise ValueError(f'No package metadata found for {label}')
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
