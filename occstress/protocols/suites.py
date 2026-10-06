"""Expand explicit suite definitions; do not discover experiments by globbing."""
from pathlib import Path

from occstress.protocols.temporal import PATTERNS


def manual_conditions():
    base = Path('protocols/manual')
    rows = [{'id': 'clean', 'family': 'clean', 'kind': 'clean',
             'protocol_path': str(base / 'clean/H4_F6_val_backbone.pkl')}]
    for family in ('misalignment', 'dropout', 'traffic', 'semantic', 'hole'):
        if family == 'traffic':
            rows.append({'id': 'traffic', 'family': family, 'kind': 'traffic',
                         'protocol_path': str(base / 'traffic/H4_F6_val_backbone.pkl')})
            continue
        for severity in ('easy', 'mid', 'hard'):
            for pattern in PATTERNS:
                rows.append({'id': f'{family}_{severity}_{pattern}', 'kind': 'corrupted',
                             'family': family, 'severity': severity, 'pattern': pattern,
                             'protocol_path': str(base / family / severity /
                                                  f'{pattern}_H4_F6_val_backbone.pkl')})
    return rows


def conditions(suite):
    if suite.get('schema_version') != 1:
        raise ValueError('Unsupported suite version')
    rows = manual_conditions() if suite.get('template') == 'manual38' else suite['conditions']
    ids = [row['id'] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Suite condition IDs must be nonempty and unique')
    if sum(row['kind'] == 'clean' for row in rows) != 1:
        raise ValueError('A source-specific suite needs exactly one paired clean condition')
    if any(row['kind'] not in {'clean', 'traffic', 'corrupted'} for row in rows):
        raise ValueError('Unknown condition kind')
    return rows
