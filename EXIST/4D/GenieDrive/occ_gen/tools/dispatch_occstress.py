#!/usr/bin/env python3
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Run resumable OccStress protocol workers.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument('--position-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--config', default=str(root / 'configs/world_model/vae_e2e_occstress.py'))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument(
        '--track',
        '--tracks',
        dest='track',
        choices=('all', 'manual', 'stcocc', 'sdgocc', 'camera', 'pointcloud'),
        default='all')
    parser.add_argument('--include-position-sweep', action='store_true')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--retry-failures', type=int, default=1)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='validate and print the protocol inventory without launching workers')
    return parser.parse_args()


def protocol_jobs(args):
    root = Path(args.occstress_root)
    track_roots = {
        'manual': root / 'protocols/manual',
        'camera': root / 'protocols/upstream/camera_only/stcocc',
        'pointcloud': root / 'protocols/upstream/pointcloud_fusion/sdgocc',
    }
    track_aliases = {
        'stcocc': 'camera',
        'sdgocc': 'pointcloud',
    }
    selected_track = track_aliases.get(args.track, args.track)
    selected = (
        track_roots if selected_track == 'all'
        else {selected_track: track_roots[selected_track]})
    jobs = []
    track_counts = {}
    for track, track_root in selected.items():
        protocols = sorted(track_root.rglob('*.pkl'))
        track_counts[track] = len(protocols)
        for protocol in protocols:
            relative = protocol.relative_to(root / 'protocols')
            jobs.append((0 if 'clean' in relative.parts else 2, track, protocol, relative))

    if args.include_position_sweep:
        position_root = Path(args.position_root)
        settings = ('semantic_hard', 'stcocc_snow_hard', 'sdgocc_snow_heavy')
        for setting in settings:
            protocols = sorted((position_root / setting).glob('*.pkl'))
            if len(protocols) != 6:
                raise ValueError(
                    f'{setting} contains {len(protocols)} protocols, expected 6')
            for protocol in protocols:
                relative = Path('position_sweep') / setting / protocol.name
                jobs.append((1, 'position_sweep', protocol, relative))
    expected_counts = {
        'manual': 38,
        'camera': 73,
        'pointcloud': 73,
    }
    for track, count in track_counts.items():
        if count != expected_counts[track]:
            raise ValueError(
                f'{track} contains {count} protocols, '
                f'expected {expected_counts[track]}')
    return sorted(jobs, key=lambda item: (item[0], str(item[3])))


def valid_success(path, max_samples):
    if not path.is_file():
        return False
    try:
        with path.open() as handle:
            payload = json.load(handle)
        expected = max_samples if max_samples is not None else 4519
        if (
                payload.get('status') != 'success' or
                payload.get('evaluated_records') != expected):
            return False
        horizon_metrics = payload.get('horizon_metrics', {})
        metric_values = []
        for horizon in HORIZONS:
            metrics = horizon_metrics.get(str(horizon), {})
            metric_values.extend([metrics.get('miou'), metrics.get('iou')])
        metric_values.extend([
            payload.get('average_miou'),
            payload.get('average_iou'),
        ])
        return all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in metric_values)
    except (OSError, ValueError, TypeError):
        return False


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = protocol_jobs(args)
    if args.dry_run:
        payload = {
            'standard_protocols': sum(
                item[1] != 'position_sweep' for item in inventory),
            'position_sweep_protocols': sum(
                item[1] == 'position_sweep' for item in inventory),
            'protocols': [str(item[3]) for item in inventory],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    jobs = deque()
    skipped = 0
    for _, track, protocol, relative in inventory:
        output = output_root / relative.with_suffix('.json')
        if valid_success(output, args.max_samples):
            skipped += 1
            continue
        jobs.append({
            'track': track,
            'protocol': protocol,
            'relative': relative,
            'output': output,
            'attempt': 0,
        })

    slots = [
        (gpu, slot)
        for gpu in [item.strip() for item in args.gpus.split(',') if item.strip()]
        for slot in range(args.workers_per_gpu)
    ]
    running = {}
    completed = []
    failed = []
    print(f'queued={len(jobs)} skipped={skipped} slots={len(slots)}', flush=True)

    while jobs or running:
        for slot in slots:
            if slot in running or not jobs:
                continue
            job = jobs.popleft()
            job['attempt'] += 1
            job['output'].parent.mkdir(parents=True, exist_ok=True)
            log_path = job['output'].with_suffix('.log')
            command = [
                sys.executable,
                str(root / 'tools/test_occstress.py'),
                '--config', str(Path(args.config).resolve()),
                '--checkpoint', str(Path(args.checkpoint).resolve()),
                '--protocol', str(job['protocol'].resolve()),
                '--base-info', str(Path(args.base_info).resolve()),
                '--occstress-root', str(Path(args.occstress_root).resolve()),
                '--output-json', str(job['output']),
            ]
            if args.max_samples is not None:
                command.extend(['--max-samples', str(args.max_samples)])
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = slot[0]
            env['OMP_NUM_THREADS'] = '1'
            log_handle = log_path.open('a')
            log_handle.write(
                f'\n=== attempt {job["attempt"]} gpu={slot[0]} slot={slot[1]} '
                f'time={time.time()} ===\n')
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT)
            running[slot] = (process, job, log_handle)
            print(f'start gpu={slot[0]} slot={slot[1]} {job["relative"]}', flush=True)

        time.sleep(2)
        for slot, (process, job, log_handle) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            del running[slot]
            if status == 0 and valid_success(job['output'], args.max_samples):
                completed.append(str(job['relative']))
                print(f'done {job["relative"]}', flush=True)
            elif job['attempt'] <= args.retry_failures:
                jobs.append(job)
                print(f'retry {job["relative"]} status={status}', flush=True)
            else:
                failed.append({'protocol': str(job['relative']), 'status': status})
                print(f'failed {job["relative"]} status={status}', flush=True)

        manifest = {
            'queued_remaining': len(jobs),
            'running': [str(item[1]['relative']) for item in running.values()],
            'completed_this_run': completed,
            'skipped_existing': skipped,
            'failed': failed,
            'workers_per_gpu': args.workers_per_gpu,
            'gpus': args.gpus,
        }
        manifest_tmp = output_root / 'dispatch_manifest.json.tmp'
        with manifest_tmp.open('w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(manifest_tmp, output_root / 'dispatch_manifest.json')

    if failed:
        raise SystemExit(f'{len(failed)} protocols failed')


if __name__ == '__main__':
    main()
