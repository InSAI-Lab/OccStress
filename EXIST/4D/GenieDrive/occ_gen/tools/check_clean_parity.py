#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare official-loader and OccStress clean metrics.')
    parser.add_argument('official_json')
    parser.add_argument('adapter_json')
    parser.add_argument('--tolerance', type=float, default=0.01)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def load_metrics(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get('metrics', payload)


def main():
    args = parse_args()
    official = load_metrics(args.official_json)
    adapter_payload = load_metrics(args.adapter_json)
    adapter = {}
    if 'horizon_metrics' in adapter_payload:
        for horizon, metrics in adapter_payload['horizon_metrics'].items():
            adapter[f'semantics_miou_time_{horizon}s'] = metrics['miou']
            adapter[f'binary_iou_time_{horizon}s'] = metrics['iou']
    else:
        adapter = adapter_payload

    comparisons = {}
    passed = True
    for horizon in HORIZONS:
        for metric_name in ('semantics_miou', 'binary_iou'):
            key = f'{metric_name}_time_{horizon}s'
            official_value = float(official[key])
            adapter_value = float(adapter[key])
            difference = adapter_value - official_value
            item_passed = (
                math.isfinite(difference) and
                abs(difference) <= args.tolerance)
            comparisons[key] = {
                'official': official_value,
                'adapter': adapter_value,
                'difference': difference,
                'passed': item_passed,
            }
            passed = passed and item_passed

    payload = {
        'passed': passed,
        'tolerance': args.tolerance,
        'comparisons': comparisons,
        'official_source': str(Path(args.official_json).resolve()),
        'adapter_source': str(Path(args.adapter_json).resolve()),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
