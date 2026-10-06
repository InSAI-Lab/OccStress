#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Calibrate concurrent OccStress evaluations on one GPU.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--output-json', required=True)
    parser.add_argument(
        '--config',
        default=str(root / 'configs/world_model/vae_e2e_occstress.py'))
    return parser.parse_args()


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
    output = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=memory.used',
        '--format=csv,noheader,nounits',
        '-i',
        str(gpu),
    ], text=True)
    return int(output.strip().splitlines()[0])


def run_level(args, level, protocols, root, output_root):
    level_root = output_root / f'concurrency_{level}'
    level_root.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['OMP_NUM_THREADS'] = '1'
    env['GENIEDRIVE_MSDA_IMPL'] = 'pytorch'

    started = time.time()
    for index, protocol in enumerate(protocols[:level]):
        output = level_root / f'task_{index}.json'
        log_path = level_root / f'task_{index}.log'
        command = [
            sys.executable,
            str(root / 'tools/test_occstress.py'),
            '--config', str(Path(args.config).resolve()),
            '--checkpoint', str(Path(args.checkpoint).resolve()),
            '--protocol', str(protocol.resolve()),
            '--base-info', str(Path(args.base_info).resolve()),
            '--occstress-root', str(Path(args.occstress_root).resolve()),
            '--output-json', str(output),
            '--max-samples', str(args.max_samples),
        ]
        log_handle = log_path.open('w')
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT)
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
            continue
        try:
            with output.open() as handle:
                payload = json.load(handle)
            if (
                    payload.get('status') != 'success' or
                    payload.get('evaluated_records') != args.max_samples):
                failures.append({
                    'output': str(output),
                    'error': 'invalid success payload',
                })
        except (OSError, ValueError) as error:
            failures.append({
                'output': str(output),
                'error': str(error),
            })

    return {
        'concurrency': level,
        'tasks': level,
        'samples_per_task': args.max_samples,
        'total_samples': level * args.max_samples,
        'elapsed_seconds': elapsed,
        'throughput_samples_per_second':
            level * args.max_samples / elapsed,
        'peak_gpu_memory_mib': peak_memory,
        'failures': failures,
        'passed': not failures and peak_memory < 86 * 1024,
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocols = representative_protocols(Path(args.occstress_root))
    missing = [str(path) for path in protocols if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing calibration protocols: {missing}')

    levels = []
    for level in (1, 2, 3):
        levels.append(
            run_level(args, level, protocols, root, output_root))
    if levels[-1]['peak_gpu_memory_mib'] < 60 * 1024:
        levels.append(
            run_level(args, 4, protocols, root, output_root))

    selected = 1 if levels[0]['passed'] else None
    previous = levels[0]
    for level in levels[1:]:
        improvement = (
            level['throughput_samples_per_second'] /
            previous['throughput_samples_per_second'] - 1)
        level['throughput_improvement_over_previous'] = improvement
        if level['passed'] and improvement >= 0.10:
            selected = level['concurrency']
        previous = level

    payload = {
        'selected_workers_per_gpu': selected,
        'gpu': args.gpu,
        'max_samples': args.max_samples,
        'memory_limit_mib': 86 * 1024,
        'level4_test_threshold_mib': 60 * 1024,
        'levels': levels,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit('no concurrency level passed calibration')


if __name__ == '__main__':
    main()
