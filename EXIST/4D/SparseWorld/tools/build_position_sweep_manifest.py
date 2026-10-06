#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import json
from pathlib import Path


CORRUPTIONS = ('snow', 'motion_blur', 'camera_crash')
POSITIONS = (
    'position_tminus4',
    'position_tminus3',
    'position_tminus2',
    'position_tminus1',
    'position_t',
)


def main():
    parser = argparse.ArgumentParser(
        description='Build the frozen SparseWorld position-sweep task manifest.')
    parser.add_argument(
        '--output', default='occstress/position_sweep_v2_protocols.json')
    args = parser.parse_args()

    protocols = [{
        'id': 'shared_clean',
        'corruption': 'clean',
        'severity': None,
        'frame_protocol': None,
    }]
    for corruption in CORRUPTIONS:
        for position in POSITIONS:
            protocols.append({
                'id': f'{corruption}__hard__{position}',
                'corruption': corruption,
                'severity': 'hard',
                'frame_protocol': position,
            })
    if len(protocols) != 16:
        raise AssertionError(len(protocols))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema_version': 2,
        'track': 'camera_position_sweep_v2',
        'selection_policy': 'predefined_before_v2_results',
        'logical_protocol_count': 18,
        'unique_compute_count': 16,
        'shared_clean_aliases': [
            f'{corruption}__hard__clean' for corruption in CORRUPTIONS
        ],
        'input_times_seconds': [-2.0, -1.5, -1.0, -0.5, 0.0],
        'future_times_seconds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'protocols': protocols,
    }, indent=2) + '\n')
    print(f'Wrote {len(protocols)} unique tasks to {output}')


if __name__ == '__main__':
    main()
