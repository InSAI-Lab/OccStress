#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import json
from pathlib import Path


CORRUPTIONS = [
    'brightness',
    'camera_crash',
    'color_quant',
    'fog',
    'frame_lost',
    'low_light',
    'motion_blur',
    'snow',
]
SEVERITIES = ['easy', 'mid', 'hard']
FRAME_PROTOCOLS = ['history_k1', 'current', 'all_frame']


def main():
    parser = argparse.ArgumentParser(
        description='Build the fixed 73-setting camera OccStress manifest.')
    parser.add_argument(
        '--output', default='occstress/protocols.json')
    args = parser.parse_args()

    protocols = [{
        'id': 'clean',
        'corruption': 'clean',
        'severity': None,
        'frame_protocol': None,
    }]
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            for frame_protocol in FRAME_PROTOCOLS:
                protocols.append({
                    'id': f'{corruption}__{severity}__{frame_protocol}',
                    'corruption': corruption,
                    'severity': severity,
                    'frame_protocol': frame_protocol,
                })
    if len(protocols) != 73:
        raise AssertionError(len(protocols))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema_version': 1,
        'track': 'camera_upstream',
        'history_times_seconds': [-2.0, -1.5, -1.0, -0.5],
        'current_time_seconds': 0.0,
        'future_times_seconds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'protocols': protocols,
    }, indent=2) + '\n')
    print(f'Wrote {len(protocols)} protocols to {output}')


if __name__ == '__main__':
    main()
