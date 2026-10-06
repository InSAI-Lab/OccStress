#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import mmcv
import numpy as np
import pyquaternion
import torch
from mmcv import Config
from nuscenes.utils.geometry_utils import transform_matrix


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Validate GenieDrive OccStress adapter inputs and metadata.')
    parser.add_argument(
        '--config',
        default=str(root / 'configs/world_model/vae_e2e_occstress.py'))
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument(
        '--protocol',
        action='append',
        required=True,
        help='named protocol in LABEL=PATH form; may be repeated')
    parser.add_argument('--max-samples', type=int, default=32)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def parse_protocol(value):
    try:
        label, path = value.split('=', 1)
    except ValueError as error:
        raise ValueError(f'invalid protocol specification: {value}') from error
    return label, str(Path(path).resolve())


def unwrap_tensor(value):
    if hasattr(value, 'data'):
        value = value.data
    if torch.is_tensor(value):
        value = value.cpu().numpy()
    return np.asarray(value)


def rotation_matrix(quaternion):
    return pyquaternion.Quaternion(quaternion).rotation_matrix


def assert_close(name, actual, expected, atol=1e-4):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=atol, rtol=0):
        difference = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape else None)
        raise ValueError(
            f'{name} mismatch: {actual.shape} vs {expected.shape}, '
            f'max difference={difference}')


def validate_clean_metadata(dataset, raw, record, sample_index):
    observe_tokens = record['history_tokens'] + [record['anchor_token']]
    observe_infos = [dataset.token_to_info[token] for token in observe_tokens]
    expected_translations = np.array([
        info['ego2global_translation'] for info in observe_infos
    ])
    assert_close(
        f'clean[{sample_index}].observe_translations',
        raw['protocol_observe_translations'],
        expected_translations)
    for frame_index, (actual, info) in enumerate(zip(
            raw['protocol_observe_rotations'], observe_infos)):
        assert_close(
            f'clean[{sample_index}].observe_rotation[{frame_index}]',
            rotation_matrix(actual),
            rotation_matrix(info['ego2global_rotation']))

    expected_lcf = np.array([
        [
            dataset.token_to_info[token]['gt_ego_lcf_feat'][0],
            dataset.token_to_info[token]['gt_ego_lcf_feat'][1],
            dataset.token_to_info[token]['gt_ego_lcf_feat'][4],
        ]
        for token in record['history_tokens']
    ])
    assert_close(
        f'clean[{sample_index}].observe_lcf',
        raw['protocol_observe_ego_lcf_feat'],
        expected_lcf)

    future_infos = [
        dataset.token_to_info[token] for token in record['future_tokens']
    ]
    pose_infos = [dataset.token_to_info[record['anchor_token']]] + future_infos
    expected_global = np.array([
        transform_matrix(
            info['ego2global_translation'],
            pyquaternion.Quaternion(info['ego2global_rotation']))
        for info in pose_infos
    ])
    assert_close(
        f'clean[{sample_index}].future_global',
        raw['curr_ego_to_global'],
        expected_global)
    assert_close(
        f'clean[{sample_index}].future_translation',
        raw['ego_to_global_translation'],
        np.array([info['ego2global_translation'] for info in pose_infos]))

    control_infos = pose_infos[:-1]
    assert_close(
        f'clean[{sample_index}].future_command',
        raw['gt_ego_fut_cmd'],
        np.array([info['gt_ego_fut_cmd'] for info in control_infos]))
    expected_control_lcf = np.array([
        [
            info['gt_ego_lcf_feat'][0],
            info['gt_ego_lcf_feat'][1],
            info['gt_ego_lcf_feat'][4],
        ]
        for info in control_infos
    ])
    assert_close(
        f'clean[{sample_index}].future_lcf',
        raw['gt_ego_lcf_feat'],
        expected_control_lcf)


def main():
    args = parse_args()
    os.environ['OCCSTRESS_ROOT'] = str(Path(args.occstress_root).resolve())
    os.environ['GENIEDRIVE_VAL_INFO'] = str(Path(args.base_info).resolve())

    from mmdet3d.datasets import build_dataset

    config = Config.fromfile(args.config)
    summaries = {}
    for specification in args.protocol:
        label, protocol_path = parse_protocol(specification)
        dataset_config = config.data.test.copy()
        dataset_config.ann_file = str(Path(args.base_info).resolve())
        dataset_config.protocol_path = protocol_path
        dataset_config.max_samples = args.max_samples
        dataset = build_dataset(dataset_config)
        expected_count = min(
            args.max_samples, len(mmcv.load(protocol_path, file_format='pkl')))
        if len(dataset) != expected_count:
            raise ValueError(
                f'{label} has {len(dataset)} samples, expected {expected_count}')

        class_ids = set()
        for sample_index in range(len(dataset)):
            record = dataset.data_infos[sample_index]
            raw = dataset.get_data_info(sample_index)
            if len(raw['previous_occ_path']) != 3:
                raise ValueError(f'{label}[{sample_index}] does not have H3 input')
            if len(raw['future_occ_path']) != 6:
                raise ValueError(f'{label}[{sample_index}] does not have F6 target')
            if len(raw['occ_index']) != 10 or len(set(raw['occ_index'])) != 10:
                raise ValueError(
                    f'{label}[{sample_index}] has invalid temporal token sequence')

            sample = dataset[sample_index]
            volume = unwrap_tensor(sample['voxel_semantics'])
            if volume.shape != (10, 200, 200, 16):
                raise ValueError(
                    f'{label}[{sample_index}] has volume shape {volume.shape}')
            if not np.isfinite(volume).all():
                raise ValueError(f'{label}[{sample_index}] contains NaN/Inf')
            unique = np.unique(volume)
            if not np.isin(unique, np.r_[np.arange(18), 255]).all():
                raise ValueError(
                    f'{label}[{sample_index}] has invalid class IDs {unique}')
            class_ids.update(int(value) for value in unique)

            for metadata_name in (
                    'curr_to_future_ego_rt',
                    'curr_ego_to_global',
                    'ego_to_global_rotation',
                    'ego_to_global_translation',
                    'gt_ego_fut_trajs',
                    'gt_ego_fut_cmd',
                    'gt_ego_lcf_feat',
                    'protocol_observe_rotations',
                    'protocol_observe_translations',
                    'protocol_observe_ego_lcf_feat'):
                if not np.isfinite(np.asarray(raw[metadata_name])).all():
                    raise ValueError(
                        f'{label}[{sample_index}].{metadata_name} has NaN/Inf')

            if record['corruption']['type'] == 'clean':
                validate_clean_metadata(
                    dataset, raw, record, sample_index)
            elif record['corruption']['type'] == 'misalignment':
                misalignment = record['misalignment']
                affected = np.asarray(
                    misalignment['affected_mask'], dtype=bool)
                clean_history_translations = np.array([
                    dataset.token_to_info[token]['ego2global_translation']
                    for token in record['history_tokens']
                ])
                observed_history_translations = np.asarray(
                    raw['protocol_observe_translations'][-5:-1])
                history_differences = np.linalg.norm(
                    observed_history_translations -
                    clean_history_translations,
                    axis=1)
                if affected.any() and not (
                        history_differences[affected] > 1e-4).any():
                    raise ValueError(
                        f'{label}[{sample_index}] ignored history misalignment')
                if misalignment.get('current_active'):
                    current_delta = np.asarray(
                        misalignment['current_delta_rt'])
                    if np.allclose(current_delta, np.eye(4), atol=1e-4):
                        raise ValueError(
                            f'{label}[{sample_index}] has inactive current delta')
                    clean_current_translation = np.asarray(
                        dataset.token_to_info[record['anchor_token']][
                            'ego2global_translation'])
                    if np.allclose(
                            raw['protocol_observe_translations'][-1],
                            clean_current_translation,
                            atol=1e-4,
                            rtol=0):
                        raise ValueError(
                            f'{label}[{sample_index}] ignored current misalignment')

        summaries[label] = {
            'protocol': protocol_path,
            'samples_checked': len(dataset),
            'volume_shape': [10, 200, 200, 16],
            'future_frames': 6,
            'class_ids_seen': sorted(class_ids),
            'passed': True,
        }

    payload = {
        'passed': True,
        'max_samples': args.max_samples,
        'protocols': summaries,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
