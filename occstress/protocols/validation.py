"""Structural validation without loading occupancy assets or model code."""
import json
import pickle
from pathlib import Path


def load_records(path, *, trusted_pickle=False):
    path = Path(path)
    if path.suffix == '.json':
        value = json.loads(path.read_text())
    elif path.suffix == '.pkl' and trusted_pickle:
        with path.open('rb') as handle:
            value = pickle.load(handle)
    else:
        raise ValueError('Use JSON, or explicitly trust a local protocol pickle')
    if isinstance(value, dict):
        value = value['records']
    if not isinstance(value, list):
        raise ValueError('Expected a list of canonical records')
    return value


def anchor_key(record):
    return [str(record['scene_name']), str(record['anchor_token'])]


def validate_records(records, expected_anchors=None):
    seen = set()
    for record in records:
        key = tuple(anchor_key(record))
        if not all(key) or key in seen:
            raise ValueError(f'Empty or duplicate anchor: {key}')
        seen.add(key)
        if len(record['history']) != 4 or len(record['future_targets']) != 6:
            raise ValueError(f'{key}: canonical window must be H4/current/F6')
        current = record['current_input']['token']
        if current != record['anchor_token'] or record['target']['token'] != current:
            raise ValueError(f'{key}: current/target token mismatch')
        tokens = [x['token'] for x in record['history']] + [current]
        tokens += [x['token'] for x in record['future_targets']]
        if len(set(tokens)) != 11 or not all(tokens):
            raise ValueError(f'{key}: repeated/empty tokens or current used as future')
    if not seen or (expected_anchors is not None and len(seen) != expected_anchors):
        raise ValueError(f'Anchor count {len(seen)}, expected {expected_anchors}')
    return {'anchors': len(seen), 'scenes': len({key[0] for key in seen}),
            'asset_validation': 'not_performed'}
