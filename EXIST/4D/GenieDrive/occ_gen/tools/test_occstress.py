#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
OCC3D_CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Evaluate one OccStress protocol.')
    parser.add_argument(
        '--config',
        default=str(root / 'configs/world_model/vae_e2e_occstress.py'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument(
        '--occstress-root',
        help='OccStress data root; inferred for protocols stored below protocols/')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--scene-shard')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scene_shards(value):
    if not value:
        return []
    return [
        str(scene).strip().zfill(3)
        for scene in str(value).split(',')
        if str(scene).strip()
    ]


def git_commit(root):
    override = os.environ.get('GENIEDRIVE_CODE_COMMIT')
    if override:
        return override
    try:
        return subprocess.check_output(
            ['git', '-C', str(root.parent), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'not-a-git-checkout'


def protocol_name(path):
    path = Path(path).resolve()
    parts = path.parts
    if 'protocols' in parts:
        return '/'.join(parts[parts.index('protocols') + 1:])
    return path.name


def infer_occstress_root(protocol):
    protocol = Path(protocol).resolve()
    for parent in protocol.parents:
        if parent.name == 'protocols':
            return parent.parent
    return None


def carla_horizon_metrics(metrics, class_mapping_path):
    with Path(class_mapping_path).open() as handle:
        class_mapping = json.load(handle)
    present = [
        int(index) for index in class_mapping['labels_present']
        if int(index) != 17
    ]
    counts = metrics.get('_aggregate_counts') or {}
    semantic = counts.get('semantic_confusion')
    binary = counts.get('binary_confusion')
    if not semantic or not binary or len(semantic) != 6 or len(binary) != 6:
        raise ValueError('CARLA evaluation requires six raw confusion matrices')

    output = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        confusion = np.asarray(semantic[horizon_index], dtype=np.float64)
        union = (
            confusion.sum(axis=1) + confusion.sum(axis=0)
            - np.diag(confusion))
        per_class = np.divide(
            np.diag(confusion), union,
            out=np.full(18, np.nan, dtype=np.float64),
            where=union > 0) * 100.0
        present_miou = float(np.nanmean(per_class[present]))

        binary_confusion = np.asarray(binary[horizon_index], dtype=np.float64)
        binary_union = (
            binary_confusion[1, :].sum()
            + binary_confusion[:, 1].sum()
            - binary_confusion[1, 1])
        binary_iou = (
            100.0 * binary_confusion[1, 1] / binary_union
            if binary_union > 0 else float('nan'))
        output[str(horizon)] = {
            'miou': present_miou,
            'iou': float(binary_iou),
            'present_class_miou': present_miou,
            'present_class_indices': present,
            'present_class_names': [OCC3D_CLASSES[index] for index in present],
            'per_class_iou': {
                OCC3D_CLASSES[index]: (
                    None if not math.isfinite(per_class[index])
                    else float(per_class[index]))
                for index in range(18)
            },
        }
    return output


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(output.suffix + '.raw.tmp')
    final_tmp = output.with_suffix(output.suffix + '.tmp')

    env = os.environ.copy()
    config_name = Path(args.config).name.lower()
    is_carla = 'carla' in config_name
    is_waymo = 'waymo' in config_name
    env['OCCSTRESS_PROTOCOL'] = str(Path(args.protocol).resolve())
    env['GENIEDRIVE_VAL_INFO'] = str(Path(args.base_info).resolve())
    if is_waymo or is_carla:
        env['OCCSTRESS_WAYMO_PROTOCOL'] = str(Path(args.protocol).resolve())
        env['OCCSTRESS_WAYMO_BASE_INFO'] = str(Path(args.base_info).resolve())
        if args.scene_shard:
            env['OCCSTRESS_WAYMO_SCENE_SHARD'] = str(args.scene_shard)
        else:
            env.pop('OCCSTRESS_WAYMO_SCENE_SHARD', None)
    occstress_root = (
        Path(args.occstress_root).resolve()
        if args.occstress_root else infer_occstress_root(args.protocol))
    if occstress_root is not None:
        env['OCCSTRESS_ROOT'] = str(occstress_root)
        env['OCCSTRESS_DATA_ROOT'] = str(occstress_root)
        if is_waymo:
            env.setdefault(
                'OCCSTRESS_WAYMO_CLEAN_OCC_CACHE',
                str(occstress_root / 'clean_occ'))
            env.setdefault('OCCSTRESS_WAYMO_ASSET_CACHE', str(occstress_root))
    if args.max_samples is None:
        env.pop('OCCSTRESS_MAX_SAMPLES', None)
        env.pop('OCCSTRESS_WAYMO_MAX_SAMPLES', None)
    else:
        env['OCCSTRESS_MAX_SAMPLES'] = str(args.max_samples)
        env['OCCSTRESS_WAYMO_MAX_SAMPLES'] = str(args.max_samples)
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    # The shared MMCV environment may be compiled only for H100 (sm90).
    # The repository's numerically equivalent PyTorch implementation works on
    # both H100 and A100 and avoids silently continuing after an sm80 kernel
    # launch failure.
    env['GENIEDRIVE_MSDA_IMPL'] = env.get(
        'GENIEDRIVE_MSDA_IMPL', 'pytorch')

    command = [
        sys.executable,
        str(root / 'tools/test.py'),
        str(Path(args.config).resolve()),
        str(Path(args.checkpoint).resolve()),
        '--eval',
        'forecasting_miou',
        '--metrics-json',
        str(raw_output),
        '--stream-eval',
        '--gpu-id',
        str(args.gpu_id),
        '--seed',
        str(args.seed),
        '--deterministic',
    ]
    started = time.time()
    subprocess.run(command, cwd=root, env=env, check=True)
    elapsed = time.time() - started

    with raw_output.open() as handle:
        raw = json.load(handle)
    metrics = raw['metrics']
    class_mapping = Path(args.base_info).resolve().with_name(
        'class_mapping.json')
    if is_carla:
        horizon_metrics = carla_horizon_metrics(metrics, class_mapping)
    else:
        horizon_metrics = {}
        for horizon in HORIZONS:
            horizon_metrics[str(horizon)] = {
                'miou': float(metrics[f'semantics_miou_time_{horizon}s']),
                'iou': float(metrics[f'binary_iou_time_{horizon}s']),
            }
    average_miou = (
        sum(value['miou'] for value in horizon_metrics.values()) /
        len(HORIZONS))
    average_iou = (
        sum(value['iou'] for value in horizon_metrics.values()) /
        len(HORIZONS))
    metric_values = [
        value
        for horizon in horizon_metrics.values()
        for value in (horizon['miou'], horizon['iou'])
    ] + [average_miou, average_iou]
    if not all(math.isfinite(value) for value in metric_values):
        raise ValueError('evaluation produced a non-finite metric')

    payload = {
        'status': 'success',
        'method': 'GenieDrive',
        'code_commit': git_commit(root),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256(args.checkpoint),
        'protocol': str(Path(args.protocol).resolve()),
        'protocol_name': protocol_name(args.protocol),
        'protocol_sha256': sha256(args.protocol),
        'dataset': (
            'UniOcc-CARLA' if is_carla else
            'Occ3D-Waymo' if is_waymo else 'Occ3D-nuScenes'),
        'scene_shard': (
            '-'.join(scene_shards(args.scene_shard))
            if args.scene_shard else None),
        'scene_shards': scene_shards(args.scene_shard),
        'base_info': str(Path(args.base_info).resolve()),
        'occstress_root': (
            str(occstress_root) if occstress_root is not None else None),
        'evaluated_records': int(raw['evaluated_records']),
        'observed_offsets_seconds': [-1.5, -1.0, -0.5, 0.0],
        'ignored_offset_seconds': -2.0,
        'control_policy': (
            'carla_gt_future_trajectory_and_command'
            if is_carla else
            'waymo_gt_future_trajectory_and_command' if is_waymo
            else 'official_gt_future_trajectory_and_command'),
        'future_horizons_seconds': list(HORIZONS),
        'horizon_metrics': horizon_metrics,
        'average_miou': average_miou,
        'average_iou': average_iou,
        'aggregate_counts': metrics.get('_aggregate_counts'),
        'elapsed_seconds': elapsed,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'command': command,
        'ms_deform_attn_impl': env['GENIEDRIVE_MSDA_IMPL'],
    }
    if is_carla:
        payload.update({
            'metric_policy': 'present_gt_occupied_classes',
            'coordinate_convention': (
                'Occ3D right-handed ego-local x-forward/y-left/z-up'),
            'zero_shot_source_checkpoint': 'Occ3D-nuScenes',
        })
    if is_waymo or is_carla:
        adapter = root / 'mmdet3d/datasets/waymo_occstress_world_dataset.py'
        payload.update({
            'class_mapping': str(class_mapping),
            'class_mapping_sha256': sha256(class_mapping),
            'entrypoint_sha256': sha256(Path(__file__).resolve()),
            'dataset_adapter_sha256': sha256(adapter),
            'world_config_sha256': sha256(Path(args.config).resolve()),
        })
    with final_tmp.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(final_tmp, output)
    raw_output.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
