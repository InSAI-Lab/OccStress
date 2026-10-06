"""Versioned results: explicit evaluation identity plus sufficient statistics."""
from occstress.metrics.occupancy import METRIC_POLICY, score_counts, validate_counts
from occstress.protocols.temporal import HORIZONS

IDENTITY_FIELDS = ('model', 'dataset', 'track', 'source', 'protocol_id', 'protocol_sha256',
                   'code_revision', 'checkpoint_sha256', 'class_map_sha256', 'base_info_sha256',
                   'input_offsets_seconds', 'control_policy', 'voxel_mask', 'seed')


def validate_identity(identity):
    for key in IDENTITY_FIELDS:
        if key not in identity or identity[key] is None or identity[key] == '':
            raise ValueError(f'Missing evaluation identity: {key}')
    for key in ('protocol_sha256', 'class_map_sha256', 'base_info_sha256'):
        hashes = [identity[key]]
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError(f'Invalid SHA256: {key}')
    hashes = identity['checkpoint_sha256']
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Record all checkpoint components as a name-to-SHA256 mapping')
    for h in hashes.values():
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('Invalid checkpoint SHA256')
    if identity['input_offsets_seconds'] not in ([-2, -1.5, -1, -0.5, 0], [-1.5, -1, -0.5, 0]):
        raise ValueError('Unsupported observed window')


def build_result(identity, anchor_ids, counts, provenance=None):
    validate_identity(identity)
    if not isinstance(anchor_ids, list) or not anchor_ids:
        raise ValueError('Explicit evaluated anchor IDs are required')
    if any(not isinstance(x, list) or len(x) != 2 or not all(isinstance(v, str) and v for v in x)
           for x in anchor_ids):
        raise ValueError('Each anchor ID is [scene_name, anchor_token]')
    if len({tuple(x) for x in anchor_ids}) != len(anchor_ids):
        raise ValueError('Duplicate anchor IDs')
    counts = validate_counts(counts)
    return {'schema_version': 1, 'status': 'success', 'metric_policy': METRIC_POLICY,
            'identity': identity, 'future_horizons_seconds': HORIZONS,
            'anchor_ids': sorted(anchor_ids), 'evaluated_records': len(anchor_ids),
            'counts': {key: value.tolist() for key, value in counts.items()},
            'scores': score_counts(counts), 'provenance': provenance or {}}


def validate_result(result):
    if result.get('schema_version') != 1 or result.get('status') != 'success':
        raise ValueError('Not a successful version-1 canonical result')
    if result.get('metric_policy') != METRIC_POLICY or result.get('future_horizons_seconds') != HORIZONS:
        raise ValueError('Metric policy or future horizon mismatch')
    checked = build_result(result['identity'], result['anchor_ids'], result['counts'])
    if result['evaluated_records'] != checked['evaluated_records'] or result['scores'] != checked['scores']:
        raise ValueError('Stale scores or anchor count mismatch')
    return result
