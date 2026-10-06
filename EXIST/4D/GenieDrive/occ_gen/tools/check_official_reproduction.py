#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


TARGETS = {
    'semantics_miou_time_0s': 70.07,
    'binary_iou_time_0s': 63.13,
    'semantics_miou_time_1.0s': 50.47,
    'binary_iou_time_1.0s': 56.87,
    'semantics_miou_time_2.0s': 41.47,
    'binary_iou_time_2.0s': 51.46,
    'semantics_miou_time_3.0s': 35.83,
    'binary_iou_time_3.0s': 47.08,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check GenieDrive official reproduction against paper metrics.')
    parser.add_argument('metrics_json')
    parser.add_argument('--tolerance', type=float, default=0.30)
    parser.add_argument(
        '--reference-metrics',
        help='optional metrics JSON to require numerical consistency with')
    parser.add_argument('--consistency-tolerance', type=float, default=0.01)
    parser.add_argument('--output-json')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.metrics_json) as handle:
        metrics = json.load(handle)['metrics']

    checks = {}
    passed = True
    for name, target in TARGETS.items():
        actual = float(metrics[name])
        delta = actual - target
        item_passed = math.isfinite(actual) and abs(delta) <= args.tolerance
        checks[name] = {
            'actual': actual,
            'target': target,
            'delta': delta,
            'passed': item_passed,
        }
        passed = passed and item_passed

    forecast_miou = [
        float(metrics[f'semantics_miou_time_{horizon}s'])
        for horizon in (1.0, 2.0, 3.0)
    ]
    forecast_iou = [
        float(metrics[f'binary_iou_time_{horizon}s'])
        for horizon in (1.0, 2.0, 3.0)
    ]
    averages = {
        'forecast_miou': sum(forecast_miou) / len(forecast_miou),
        'forecast_iou': sum(forecast_iou) / len(forecast_iou),
    }
    consistency = None
    if args.reference_metrics:
        with open(args.reference_metrics) as handle:
            reference = json.load(handle)['metrics']
        consistency_checks = {}
        consistency_passed = True
        for name in TARGETS:
            actual = float(metrics[name])
            reference_value = float(reference[name])
            delta = actual - reference_value
            item_passed = (
                math.isfinite(actual) and
                math.isfinite(reference_value) and
                abs(delta) <= args.consistency_tolerance)
            consistency_checks[name] = {
                'actual': actual,
                'reference': reference_value,
                'delta': delta,
                'passed': item_passed,
            }
            consistency_passed = consistency_passed and item_passed
        consistency = {
            'passed': consistency_passed,
            'tolerance': args.consistency_tolerance,
            'reference': str(Path(args.reference_metrics).resolve()),
            'checks': consistency_checks,
        }
        passed = passed and consistency_passed

    payload = {
        'passed': passed,
        'tolerance': args.tolerance,
        'checks': checks,
        'averages': averages,
        'consistency': consistency,
        'source': str(Path(args.metrics_json).resolve()),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(rendered + '\n')
        temporary.replace(output)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
