#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Calibrate concurrent DOME evaluations on one H100.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument('--nuscenes-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max-samples', type=int, default=16)
    parser.add_argument('--levels', default='1,2,3,4')
    parser.add_argument('--dataloader-workers', type=int, default=2)
    parser.add_argument('--output-json', required=True)
    parser.add_argument(
        '--config', default=str(root / 'config/train_dome.py'))
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def representative_protocols(root):
    return [
        root / 'protocols/manual/clean/H4_F6_val_backbone.pkl',
        root / (
            'protocols/manual/semantic/hard/'
            'recent_burst_H4_F6_val_backbone.pkl'),
        root / (
            'protocols/upstream/camera_only/stcocc/Snow/hard/'
            'recent_burst_H4_F6_val_backbone.pkl'),
        root / (
            'protocols/upstream/pointcloud_fusion/sdgocc/snow/heavy/'
            'recent_burst_H4_F6_val_backbone.pkl'),
    ]


def gpu_memory_mib(gpu):
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=memory.used',
        '--format=csv,noheader,nounits',
        '-i',
        str(gpu),
    ], text=True)
    return int(result.strip().splitlines()[0])


def run_level(args, level, protocols, root, checkpoint_hash, vae_hash):
    level_root = Path(args.output_root) / f'concurrency_{level}'
    level_root.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    started = time.time()
    for index in range(level):
        protocol = protocols[index % len(protocols)]
        output = level_root / f'task_{index}.json'
        output.unlink(missing_ok=True)
        log_path = level_root / f'task_{index}.log'
        command = [
            sys.executable,
            str(root / 'tools/eval_occstress.py'),
            '--config', str(Path(args.config).resolve()),
            '--checkpoint', str(Path(args.checkpoint).resolve()),
            '--vae-checkpoint', str(Path(args.vae_checkpoint).resolve()),
            '--checkpoint-sha256', checkpoint_hash,
            '--vae-checkpoint-sha256', vae_hash,
            '--base-info', str(Path(args.base_info).resolve()),
            '--occstress-root', str(Path(args.occstress_root).resolve()),
            '--nuscenes-root', str(Path(args.nuscenes_root).resolve()),
            '--protocol', str(protocol.resolve()),
            '--output-json', str(output),
            '--max-samples', str(args.max_samples),
            '--num-workers', str(args.dataloader_workers),
            '--log-every', '0',
        ]
        log_handle = log_path.open('w')
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, output))
        logs.append(log_handle)

    peak_memory = 0
    while any(process.poll() is None for process, _ in processes):
        try:
            peak_memory = max(peak_memory, gpu_memory_mib(args.gpu))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        time.sleep(1)
    elapsed = time.time() - started
    for log_handle in logs:
        log_handle.close()

    failures = []
    for process, output in processes:
        if process.returncode != 0:
            failures.append({
                'output': str(output),
                'returncode': process.returncode,
            })
        elif not output.is_file():
            failures.append({
                'output': str(output),
                'error': 'missing result',
            })
    return {
        'concurrency': level,
        'total_samples': level * args.max_samples,
        'elapsed_seconds': elapsed,
        'throughput_anchors_per_second':
            level * args.max_samples / elapsed,
        'peak_gpu_memory_mib': peak_memory,
        'failures': failures,
        'passed': not failures and peak_memory < 86 * 1024,
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    protocols = representative_protocols(Path(args.occstress_root))
    missing = [str(path) for path in protocols if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing protocols: {missing}')
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    print('hashing checkpoints', flush=True)
    checkpoint_hash = sha256(args.checkpoint)
    vae_hash = sha256(args.vae_checkpoint)

    levels = []
    levels_to_test = [
        int(item) for item in args.levels.split(',') if item.strip()
    ]
    if not levels_to_test or any(level < 1 for level in levels_to_test):
        raise ValueError(f'invalid concurrency levels: {args.levels}')
    for level in levels_to_test:
        result = run_level(
            args, level, protocols, root, checkpoint_hash, vae_hash)
        if levels:
            result['throughput_improvement_over_previous'] = (
                result['throughput_anchors_per_second'] /
                levels[-1]['throughput_anchors_per_second'] - 1)
        levels.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result['passed']:
            break

    selected = (
        levels[0]['concurrency'] if levels[0]['passed'] else None)
    for result in levels[1:]:
        if (
                result['passed'] and
                result['throughput_improvement_over_previous'] >= 0.10):
            selected = result['concurrency']
    payload = {
        'selected_workers_per_gpu': selected,
        'gpu': args.gpu,
        'max_samples': args.max_samples,
        'memory_limit_mib': 86 * 1024,
        'levels': levels,
        'checkpoint_sha256': checkpoint_hash,
        'vae_checkpoint_sha256': vae_hash,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if selected is None:
        raise SystemExit('no concurrency level passed')


if __name__ == '__main__':
    main()
