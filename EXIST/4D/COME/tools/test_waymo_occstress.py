#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Evaluate one COME OccStress-Waymo protocol or scene shard."""

import argparse
import hashlib
import json
import math
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(
        root / 'configs/local_eval_controlnet_occstress.py'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--scene-shard')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root):
    try:
        return subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'not-a-git-checkout'


def expected_records(protocol, scene, max_samples):
    with Path(protocol).open('rb') as stream:
        records = pickle.load(stream)
    if scene is not None:
        scene = str(scene).zfill(3)
        records = [
            record for record in records
            if str(record['scene_name']).zfill(3) == scene
        ]
    return min(len(records), max_samples) if max_samples else len(records)


def metric_key(metrics, prefix, horizon):
    candidates = (
        f'{prefix}_{horizon:g}s',
        f'{prefix}_{horizon:.1f}s',
    )
    for candidate in candidates:
        if candidate in metrics:
            return candidate
    raise KeyError(f'missing any of {candidates}')


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(output.suffix + '.raw.tmp')
    final_tmp = output.with_suffix(output.suffix + '.tmp')
    expected = expected_records(
        args.protocol, args.scene_shard, args.max_samples)
    if expected == 0:
        raise ValueError('selected shard has no protocol anchors')

    env = os.environ.copy()
    env.update({
        'OCCSTRESS_WAYMO_PROTOCOL': str(Path(args.protocol).resolve()),
        'OCCSTRESS_WAYMO_BASE_INFO': str(Path(args.base_info).resolve()),
        'PYTHONPATH': str(root) + os.pathsep + env.get('PYTHONPATH', ''),
    })
    env.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu_id))
    if args.scene_shard:
        env['OCCSTRESS_WAYMO_SCENE_SHARD'] = str(args.scene_shard).zfill(3)
    else:
        env.pop('OCCSTRESS_WAYMO_SCENE_SHARD', None)
    if args.max_samples:
        env['OCCSTRESS_WAYMO_MAX_SAMPLES'] = str(args.max_samples)
    else:
        env.pop('OCCSTRESS_WAYMO_MAX_SAMPLES', None)

    command = [
        sys.executable,
        str(root / 'tools/test_diffusion_control.py'),
        '--py-config', str(Path(args.config).resolve()),
        '--work-dir', str(Path(args.work_dir).resolve()),
        '--resume-from', str(Path(args.checkpoint).resolve()),
        '--metrics-json', str(raw_output),
        '--seed', str(args.seed),
    ]
    started = time.time()
    subprocess.run(command, cwd=root, env=env, check=True)
    raw = json.loads(raw_output.read_text())
    if int(raw['evaluated_records']) != expected:
        raise RuntimeError(
            f'expected {expected} anchors, got {raw["evaluated_records"]}')

    horizon_metrics = {}
    for horizon in HORIZONS:
        values = {
            'miou': float(raw['metrics'][metric_key(
                raw['metrics'], 'semantics_miou_time', horizon)]),
            'iou': float(raw['metrics'][metric_key(
                raw['metrics'], 'binary_iou_time', horizon)]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f'non-finite metric at {horizon}s: {values}')
        horizon_metrics[str(horizon)] = values

    payload = {
        'status': 'success',
        'method': 'COME',
        'dataset': 'Occ3D-Waymo',
        'code_commit': git_commit(root),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256(args.checkpoint),
        'protocol': str(Path(args.protocol).resolve()),
        'protocol_sha256': sha256(args.protocol),
        'base_info': str(Path(args.base_info).resolve()),
        'scene_shard': (
            str(args.scene_shard).zfill(3) if args.scene_shard else None),
        'evaluated_records': expected,
        'observed_offsets_seconds': [-1.5, -1.0, -0.5, 0.0],
        'ignored_offset_seconds': -2.0,
        'control_policy': 'waymo_gt_future_trajectory_and_command',
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
