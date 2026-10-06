#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import json
from pathlib import Path


TARGETS = {
    'miou': {1.0: 14.93, 2.0: 13.15, 3.0: 11.51, 'average': 13.20},
    'iou': {1.0: 22.96, 2.0: 22.10, 3.0: 21.05, 'average': 22.03},
}


def main():
    parser = argparse.ArgumentParser(
        description='Check SparseWorld clean results against paper Table 1.')
    parser.add_argument(
        'result', nargs='?',
        default='occstress/reproduction/clean.json')
    parser.add_argument('--tolerance', type=float, default=0.30)
    args = parser.parse_args()

    payload = json.loads(Path(args.result).read_text())
    by_time = {
        item['seconds']: item for item in payload['metrics']['horizons']}
    average = payload['metrics']['paper_1s_2s_3s_average']
    checks = []
    for metric, targets in TARGETS.items():
        for key, target in targets.items():
            actual = (
                average[metric] if key == 'average'
                else by_time[key][metric])
            delta = actual - target
            checks.append({
                'metric': metric,
                'time': key,
                'target': target,
                'actual': actual,
                'delta': delta,
                'passed': abs(delta) <= args.tolerance,
            })
    passed = all(item['passed'] for item in checks)
    report = {
        'passed': passed,
        'tolerance': args.tolerance,
        'sample_count': payload['sample_count'],
        'anchor_mode': payload['anchor_mode'],
        'checks': checks,
    }
    output = Path(args.result).with_name('paper_gate.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
