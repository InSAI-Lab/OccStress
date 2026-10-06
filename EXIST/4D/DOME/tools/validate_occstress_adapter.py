#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.occstress_dataset import (  # noqa: E402
    FUTURE_OFFSETS,
    OBSERVED_OFFSETS,
    OccStressProtocolDataset,
    _pose,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate DOME OccStress anchor construction.')
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument('--nuscenes-root')
    parser.add_argument(
        '--protocol',
        action='append',
        required=True,
        help='Named protocol in LABEL=PATH form; may be repeated.')
    parser.add_argument('--max-samples', type=int, default=32)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def parse_protocol(value):
    try:
        label, path = value.split('=', 1)
    except ValueError as error:
        raise ValueError(
            f'invalid protocol specification: {value}') from error
    return label, Path(path).resolve()


def main():
    args = parse_args()
    with open(args.base_info, 'rb') as handle:
        base = pickle.load(handle)
    token_to_info = {
        info['token']: info
        for frames in base['infos'].values()
        for info in frames
    }

    summaries = {}
    for specification in args.protocol:
        label, protocol = parse_protocol(specification)
        dataset = OccStressProtocolDataset(
            protocol_path=protocol,
            base_info=args.base_info,
            occstress_root=args.occstress_root,
            nuscenes_root=args.nuscenes_root,
            max_samples=args.max_samples,
        )
        classes = set()
        for index in range(len(dataset)):
            occupancy, target, metadata = dataset[index]
            record = dataset.records[index]
            if occupancy.shape != (10, 200, 200, 16):
                raise ValueError(
                    f'{label}[{index}] has shape {occupancy.shape}')
            if not np.array_equal(occupancy, target):
                raise ValueError(f'{label}[{index}] changed metric targets')
            if (
                    tuple(metadata['observed_offsets_seconds']) !=
                    OBSERVED_OFFSETS or
                    tuple(metadata['future_offsets_seconds']) !=
                    FUTURE_OFFSETS):
                raise ValueError(
                    f'{label}[{index}] has invalid temporal offsets')
            if metadata['rel_poses'].shape != (10, 2):
                raise ValueError(
                    f'{label}[{index}] has invalid pose conditioning shape')
            if not np.isfinite(metadata['rel_poses']).all():
                raise ValueError(
                    f'{label}[{index}] has non-finite pose conditioning')
            classes.update(int(item) for item in np.unique(occupancy))

            if record['corruption']['type'] == 'clean':
                expected_tokens = (
                    record['history_tokens'][-3:] +
                    [record['anchor_token']] +
                    record['future_tokens'])
                expected_poses = np.stack([
                    _pose(token_to_info[token])
                    for token in expected_tokens
                ])
                if not np.allclose(
                        dataset._sequence_poses(record),
                        expected_poses,
                        atol=1e-5,
                        rtol=0):
                    raise ValueError(
                        f'{label}[{index}] clean poses differ from official data')
            if record['corruption']['type'] == 'misalignment':
                clean_tokens = (
                    record['history_tokens'][-3:] +
                    [record['anchor_token']])
                clean_observed = np.stack([
                    _pose(token_to_info[token])
                    for token in clean_tokens
                ])
                adapted = np.stack(dataset._observed_poses(record))
                if (
                        any(
                            item.get('rt_source') == 'misalignment'
                            for item in record['history'][-3:]) and
                        np.allclose(adapted, clean_observed, atol=1e-5, rtol=0)):
                    raise ValueError(
                        f'{label}[{index}] ignored active misalignment')

        summaries[label] = {
            'protocol': str(protocol),
            'samples_checked': len(dataset),
            'sequence_shape': [10, 200, 200, 16],
            'observed_offsets_seconds': list(OBSERVED_OFFSETS),
            'future_offsets_seconds': list(FUTURE_OFFSETS),
            'class_ids_seen': sorted(classes),
            'passed': True,
        }

    payload = {
        'passed': True,
        'max_samples': args.max_samples,
        'protocols': summaries,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
