#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Evaluate one occupancy-only OccStress protocol with OccWorld."""

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import socket
import subprocess
import sys
import time


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default=str(ROOT / 'config/occworld_occstress.py'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'not-a-git-checkout'


def class_mapping_sha256(protocol):
    override = os.environ.get('OCCSTRESS_CLASS_MAPPING')
    if override:
        return sha256(override)
    path = Path(protocol).resolve()
    for parent in path.parents:
        if parent.name == 'protocols':
            mapping = parent.parent / 'meta/manual/class_mapping.json'
            return sha256(mapping)
    raise ValueError(f'protocol is not below a protocols directory: {path}')


def protocol_inventory(protocol, max_samples):
    with Path(protocol).open('rb') as stream:
        records = pickle.load(stream)
    if max_samples:
        records = records[:max_samples]
    scenes = sorted({
        str(record['scene_name']).zfill(3) for record in records
    })
    return len(records), scenes


def dataset_name(base_info):
    from occstress.datasets.metadata import load_metadata
    base = load_metadata(base_info, trusted_pickle=True)
    name = base.get('metadata', {}).get('dataset', 'UniOcc-CARLA')
    return 'Occ3D-Waymo' if name == 'OccStress-Waymo' else name


def metric_value(metrics, prefix, horizon):
    keys = (
        f'{prefix}_{horizon:g}s',
        f'{prefix}_{horizon:.1f}s',
    )
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f'missing metric keys {keys}')


def main():
    args = parse_args()
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(output.suffix + '.raw.tmp')
    final_tmp = output.with_suffix(output.suffix + '.tmp')
    expected, scene_shards = protocol_inventory(
        args.protocol, args.max_samples)
    dataset = dataset_name(args.base_info)
    if expected == 0:
        raise ValueError('selected protocol has no anchors')

    env = os.environ.copy()
    protocol_env = (
        'OCCSTRESS_WAYMO_PROTOCOL' if dataset == 'Occ3D-Waymo'
        else 'OCCSTRESS_CARLA_PROTOCOL')
    base_env = (
        'OCCSTRESS_WAYMO_BASE_INFO' if dataset == 'Occ3D-Waymo'
        else 'OCCSTRESS_CARLA_BASE_INFO')
    env.update({
        protocol_env: str(Path(args.protocol).resolve()),
        base_env: str(Path(args.base_info).resolve()),
        'OCCSTRESS_OCCWORLD_BATCH_SIZE': str(args.batch_size),
        'OCCSTRESS_OCCWORLD_NUM_WORKERS': str(args.num_workers),
        'OCCSTRESS_OCCWORLD_PROTOCOL_PATH': str(Path(args.protocol).resolve()),
        'OCCSTRESS_OCCWORLD_IMAGESET': str(Path(args.base_info).resolve()),
        'OCCSTRESS_OCCWORLD_FUTURE_ALIGNED': '1',
        'OCCWORLD_CKPT': str(Path(args.checkpoint).resolve()),
        'PYTHONPATH': str(ROOT) + os.pathsep + env.get('PYTHONPATH', ''),
    })
    env.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu_id))
    if args.max_samples:
        env['OCCSTRESS_OCCWORLD_MAX_SAMPLES'] = str(args.max_samples)
    else:
        env.pop('OCCSTRESS_OCCWORLD_MAX_SAMPLES', None)

    command = [
        sys.executable,
        str(ROOT / 'eval_metric_stp3.py'),
        '--py-config', str(Path(args.config).resolve()),
        '--work-dir', str(Path(args.work_dir).resolve()),
        '--resume-from', str(Path(args.checkpoint).resolve()),
        '--metrics-json', str(raw_output),
        '--seed', str(args.seed),
    ]
    started = time.time()
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    raw = json.loads(raw_output.read_text())
    if int(raw['evaluated_records']) != expected:
        raise RuntimeError(
            f'expected {expected} anchors, got {raw["evaluated_records"]}')

    horizon_metrics = {}
    for horizon in HORIZONS:
        values = {
            'miou': metric_value(
                raw['metrics'], 'semantics_miou_time', horizon),
            'iou': metric_value(
                raw['metrics'], 'binary_iou_time', horizon),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f'non-finite metric at {horizon}s: {values}')
        horizon_metrics[str(horizon)] = values

    payload = {
        'status': 'success',
        'method': 'OccWorld',
        'dataset': dataset,
        'code_commit': git_commit(),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': (
            args.checkpoint_sha256 or sha256(args.checkpoint)),
        'zero_shot_source_checkpoint': 'Occ3D-nuScenes',
        'protocol': str(Path(args.protocol).resolve()),
        'protocol_sha256': sha256(args.protocol),
        'class_mapping_sha256': class_mapping_sha256(args.protocol),
        'base_info': str(Path(args.base_info).resolve()),
        'evaluated_records': expected,
        'scene_shard_count': len(scene_shards),
        'scene_shards': scene_shards,
        'observed_offsets_seconds': [-2.0, -1.5, -1.0, -0.5, 0.0],
        'control_policy': (
            'waymo_gt_command_autoregressive_ego_motion'
            if dataset == 'Occ3D-Waymo'
            else 'carla_gt_command_autoregressive_ego_motion'),
        'coordinate_convention': (
            'Occ3D right-handed ego-local x-forward/y-left/z-up'),
        'metric_policy': 'present_gt_occupied_classes',
        'future_horizons_seconds': list(HORIZONS),
        'horizon_metrics': horizon_metrics,
        'average_miou': sum(
            value['miou'] for value in horizon_metrics.values()) / 6,
        'average_iou': sum(
            value['iou'] for value in horizon_metrics.values()) / 6,
        'aggregate_counts': raw['counts'],
        'elapsed_seconds': time.time() - started,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES'],
    }
    final_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(final_tmp, output)
    raw_output.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
