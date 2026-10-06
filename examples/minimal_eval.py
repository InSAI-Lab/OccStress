#!/usr/bin/env python3
"""CPU-only synthetic scoring example, not a model reproduction."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occstress.metrics.occupancy import confusion_from_labels, from_confusion
from occstress.protocols.validation import anchor_key, load_records, validate_records
from occstress.results.io import content_hash, write_json
from occstress.results.schema import build_result
from occstress.results.summary import summarize_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    records = load_records(Path(__file__).parent / 'fixtures/protocol.json')
    validate_records(records, 1)
    results = []
    for condition in ('clean', 'synthetic_error'):
        target = np.array([0, 1, 4, 4, 17, 17], dtype=np.uint8)
        prediction = target.copy()
        if condition != 'clean':
            prediction[:2] = 17
        semantic = confusion_from_labels(prediction, target)
        binary = confusion_from_labels((prediction != 17).astype(int), (target != 17).astype(int), classes=2)
        identity = {'model': 'synthetic-demo', 'dataset': 'nuscenes', 'track': 'manual', 'source': 'gt',
                    'protocol_id': condition, 'protocol_sha256': content_hash(condition),
                    'code_revision': 'synthetic-not-a-model', 'checkpoint_sha256': {'demo': content_hash('no weights')},
                    'class_map_sha256': content_hash(list(range(18))), 'base_info_sha256': content_hash(records),
                    'input_offsets_seconds': [-2, -1.5, -1, -0.5, 0], 'control_policy': 'synthetic-no-controls',
                    'voxel_mask': 'all-valid', 'seed': 0}
        result = build_result(identity, [anchor_key(r) for r in records],
                              from_confusion(np.repeat(semantic[None], 6, axis=0), np.repeat(binary[None], 6, axis=0)),
                              {'synthetic': True})
        write_json(args.output_dir / f'{condition}.json', result)
        results.append(result)
    suite = {'schema_version': 1, 'id': 'synthetic-two-condition-demo', 'dataset': 'nuscenes',
             'track': 'manual', 'source': 'gt', 'expected_anchors': 1,
             'conditions': [{'id': 'clean', 'kind': 'clean'}, {'id': 'synthetic_error', 'kind': 'corrupted'}]}
    write_json(args.output_dir / 'suite.json', suite)
    write_json(args.output_dir / 'summary.json', summarize_suite(suite, results))
    print(f'Synthetic CPU example: {args.output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
