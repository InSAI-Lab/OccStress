"""Merge counts before scoring; macro-average only declared paired conditions."""
import numpy as np

from occstress.metrics.occupancy import COUNT_SHAPES
from occstress.protocols.suites import conditions
from occstress.results.schema import build_result, validate_result


def merge_shards(results):
    if not results:
        raise ValueError('No shards supplied')
    identity = results[0]['identity']
    seen, anchors = set(), []
    totals = {key: np.zeros(shape, dtype=np.int64) for key, shape in COUNT_SHAPES.items()}
    for result in results:
        validate_result(result)
        if result['identity'] != identity:
            raise ValueError('Cannot merge different evaluation identities')
        keys = {tuple(key) for key in result['anchor_ids']}
        if seen & keys:
            raise ValueError('Overlapping anchor IDs across shards')
        seen.update(keys)
        anchors.extend(result['anchor_ids'])
        for key in totals:
            totals[key] += np.asarray(result['counts'][key], dtype=np.int64)
            if (totals[key] > 10**15).any():
                raise ValueError('Count overflow limit exceeded')
    return build_result(identity, anchors, totals, {'operation': 'merge_counts', 'shards': len(results)})


def summarize_suite(suite, results, allow_partial=False):
    declared = {row['id']: row for row in conditions(suite)}
    grouped = {}
    for result in results:
        validate_result(result)
        identity = result['identity']
        if any(identity[key] != suite[key] for key in ('dataset', 'track', 'source')):
            raise ValueError('Result dataset/track/source differs from suite')
        pid = identity['protocol_id']
        if pid not in declared:
            raise ValueError(f'Undeclared protocol: {pid}')
        grouped.setdefault(pid, []).append(result)
    merged = {key: merge_shards(rows) for key, rows in grouped.items()}
    missing = sorted(set(declared) - set(merged))
    incomplete = sorted(key for key, value in merged.items() if value['evaluated_records'] != suite['expected_anchors'])
    if (missing or incomplete) and not allow_partial:
        raise ValueError(f'Missing protocols {missing}; incomplete protocols {incomplete}')
    reference = None
    cohort = None
    rows = []
    for pid, result in merged.items():
        common = {k: v for k, v in result['identity'].items() if k not in {'protocol_id', 'protocol_sha256'}}
        if reference is not None and common != reference:
            raise ValueError('Different model/checkpoint/control/class-map/mask identities')
        reference = common
        keys = result['anchor_ids']
        if pid not in incomplete:
            if cohort is not None and keys != cohort:
                raise ValueError('Protocol anchor cohorts differ')
            cohort = keys
        rows.append({**{k: v for k, v in declared[pid].items() if k != 'protocol_path'},
                     'anchors': result['evaluated_records'],
                     'complete': pid not in incomplete, **result['scores']['paper_1_2_3s']})
    clean = next((row for row in rows if row['kind'] == 'clean' and row['complete']), None)
    for row in rows:
        row['miou_drop_pp'] = clean['miou'] - row['miou'] if clean and clean['miou'] is not None and row['miou'] is not None and row['complete'] else None
        row['miou_relative_drop_percent'] = row['miou_drop_pp'] / clean['miou'] * 100 if clean and clean['miou'] and row['miou_drop_pp'] is not None else None
    complete = not missing and not incomplete
    corrupted = [row for row in rows if row['kind'] == 'corrupted']
    robust = {key: float(np.mean([row[key] for row in corrupted]))
              if complete and corrupted and all(row[key] is not None for row in corrupted) else None
              for key in ('miou', 'iou')}
    return {'schema_version': 1, 'suite_id': suite['id'], 'metric_policy': 'occstress-present-gt-v1',
            'score_horizons_seconds': [1, 2, 3], 'identity': reference,
            'complete': complete, 'missing': missing, 'incomplete': incomplete,
            'coverage': f'{len(merged) - len(incomplete)}/{len(declared)}',
            'clean': clean, 'robust_mean_excluding_traffic': robust, 'protocols': rows}
