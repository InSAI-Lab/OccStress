#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
from collections import deque
import hashlib
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
    parser = argparse.ArgumentParser(
        description='Run resumable DOME OccStress protocol workers.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--vae-checkpoint-sha256')
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument('--nuscenes-root')
    parser.add_argument('--position-root', required=True)
    parser.add_argument(
        '--protocol-root',
        help='Evaluate every PKL below this root instead of a built-in track.')
    parser.add_argument('--expected-protocols', type=int)
    parser.add_argument('--output-root', required=True)
    parser.add_argument(
        '--config', default=str(root / 'config/train_dome.py'))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--dataloader-workers', type=int, default=2)
    parser.add_argument(
        '--track',
        choices=('all', 'manual', 'stcocc', 'sdgocc', 'camera', 'pointcloud'),
        default='all')
    parser.add_argument('--include-position-sweep', action='store_true')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--expected-records', type=int, default=4519)
    parser.add_argument('--retry-failures', type=int, default=1)
    parser.add_argument('--deadline-epoch', type=float)
    parser.add_argument('--min-start-seconds', type=float, default=0)
    parser.add_argument('--num-dispatch-shards', type=int, default=1)
    parser.add_argument('--dispatch-shard-index', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_jobs(args):
    root = Path(args.occstress_root)
    if args.protocol_root:
        protocol_root = Path(args.protocol_root).resolve()
        protocols = sorted(protocol_root.rglob('*.pkl'))
        if (
                args.expected_protocols is not None and
                len(protocols) != args.expected_protocols):
            raise ValueError(
                f'{protocol_root} contains {len(protocols)} protocols, '
                f'expected {args.expected_protocols}')
        jobs = []
        for protocol in protocols:
            relative = protocol.relative_to(protocol_root)
            priority = 0 if 'clean' in relative.parts else 2
            jobs.append((priority, 'custom', protocol, relative))
        return sorted(jobs, key=lambda item: (item[0], str(item[3])))

    track_roots = {
        'manual': root / 'protocols/manual',
        'camera': root / 'protocols/upstream/camera_only/stcocc',
        'pointcloud': root / 'protocols/upstream/pointcloud_fusion/sdgocc',
    }
    aliases = {'stcocc': 'camera', 'sdgocc': 'pointcloud'}
    track = aliases.get(args.track, args.track)
    selected = track_roots if track == 'all' else {track: track_roots[track]}
    jobs = []
    counts = {}
    for track_name, track_root in selected.items():
        protocols = sorted(track_root.rglob('*.pkl'))
        counts[track_name] = len(protocols)
        for protocol in protocols:
            relative = protocol.relative_to(root / 'protocols')
            priority = 0 if 'clean' in relative.parts else 2
            jobs.append((priority, track_name, protocol, relative))

    if args.include_position_sweep:
        position_root = Path(args.position_root)
        settings = (
            'semantic_hard',
            'stcocc_snow_hard',
            'sdgocc_snow_heavy',
        )
        for setting in settings:
            protocols = sorted((position_root / setting).glob('*.pkl'))
            if len(protocols) != 6:
                raise ValueError(
                    f'{setting} contains {len(protocols)} protocols, expected 6')
            for protocol in protocols:
                relative = Path('position_sweep') / setting / protocol.name
                # The three standard clean protocols are evaluated first and
                # are equivalent to the position-sweep clean baselines. Keep
                # those duplicate clean runs behind non-clean protocols when
                # wall time is limited.
                priority = 3 if '_clean_' in protocol.name else 1
                jobs.append(
                    (priority, 'position_sweep', protocol, relative))

    expected = {'manual': 38, 'camera': 73, 'pointcloud': 73}
    for track_name, count in counts.items():
        if count != expected[track_name]:
            raise ValueError(
                f'{track_name} contains {count} protocols, '
                f'expected {expected[track_name]}')
    return sorted(jobs, key=lambda item: (item[0], str(item[3])))


def valid_success(path, max_samples, expected_records):
    if not path.is_file():
        return False
    expected = max_samples if max_samples is not None else expected_records
    try:
        with path.open() as handle:
            payload = json.load(handle)
        if (
                payload.get('status') != 'success' or
                payload.get('method') != 'DOME' or
                payload.get('evaluated_records') != expected):
            return False
        metrics = payload.get('horizon_metrics', {})
        values = []
        for horizon in HORIZONS:
            item = metrics.get(str(horizon), {})
            values.extend((item.get('miou'), item.get('iou')))
        values.extend((
            payload.get('average_miou'),
            payload.get('average_iou'),
        ))
        return all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values)
    except (OSError, ValueError, TypeError):
        return False


def write_manifest(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = protocol_jobs(args)
    if not 0 <= args.dispatch_shard_index < args.num_dispatch_shards:
        raise ValueError('dispatch shard index is outside the shard count')
    inventory = [
        item for index, item in enumerate(inventory)
        if index % args.num_dispatch_shards == args.dispatch_shard_index
    ]
    if args.dry_run:
        print(json.dumps({
            'standard_protocols': sum(
                item[1] != 'position_sweep' for item in inventory),
            'position_sweep_protocols': sum(
                item[1] == 'position_sweep' for item in inventory),
            'protocols': [str(item[3]) for item in inventory],
        }, indent=2, sort_keys=True))
        return

    checkpoint_sha256 = args.checkpoint_sha256
    vae_checkpoint_sha256 = args.vae_checkpoint_sha256
    if checkpoint_sha256 is None or vae_checkpoint_sha256 is None:
        print('hashing checkpoints once for all workers', flush=True)
    checkpoint_sha256 = checkpoint_sha256 or file_sha256(args.checkpoint)
    vae_checkpoint_sha256 = (
        vae_checkpoint_sha256 or file_sha256(args.vae_checkpoint))
    jobs = deque()
    skipped = 0
    for _, track, protocol, relative in inventory:
        output = output_root / relative.with_suffix('.json')
        if valid_success(output, args.max_samples, args.expected_records):
            skipped += 1
            continue
        jobs.append({
            'track': track,
            'protocol': protocol,
            'relative': relative,
            'output': output,
            'attempt': 0,
        })

    gpu_ids = [
        item.strip() for item in args.gpus.split(',') if item.strip()
    ]
    slots = [
        (gpu, index)
        for gpu in gpu_ids
        for index in range(args.workers_per_gpu)
    ]
    if not slots:
        raise ValueError('no GPU worker slots were configured')

    running = {}
    completed = []
    failed = []
    print(
        f'queued={len(jobs)} skipped={skipped} slots={len(slots)}',
        flush=True)
    manifest_path = output_root / (
        f'dispatch_manifest_shard_{args.dispatch_shard_index:02d}_of_'
        f'{args.num_dispatch_shards:02d}.json')
    while jobs or running:
        stop_launching = (
            args.deadline_epoch is not None
            and time.time() + args.min_start_seconds >= args.deadline_epoch
        )
        for slot in slots:
            if slot in running or not jobs or stop_launching:
                continue
            job = jobs.popleft()
            job['attempt'] += 1
            job['output'].parent.mkdir(parents=True, exist_ok=True)
            log_path = job['output'].with_suffix('.log')
            command = [
                sys.executable,
                str(root / 'tools/eval_occstress.py'),
                '--config', str(Path(args.config).resolve()),
                '--checkpoint', str(Path(args.checkpoint).resolve()),
                '--vae-checkpoint', str(Path(args.vae_checkpoint).resolve()),
                '--checkpoint-sha256', checkpoint_sha256,
                '--vae-checkpoint-sha256', vae_checkpoint_sha256,
                '--protocol', str(job['protocol'].resolve()),
                '--base-info', str(Path(args.base_info).resolve()),
                '--occstress-root',
                str(Path(args.occstress_root).resolve()),
                '--output-json', str(job['output']),
                '--num-workers', str(args.dataloader_workers),
            ]
            if args.nuscenes_root:
                command.extend([
                    '--nuscenes-root',
                    str(Path(args.nuscenes_root).resolve()),
                ])
            if args.max_samples is not None:
                command.extend(['--max-samples', str(args.max_samples)])

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = slot[0]
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            log_handle = log_path.open('a')
            log_handle.write(
                f'\n=== attempt={job["attempt"]} gpu={slot[0]} '
                f'slot={slot[1]} time={time.time()} ===\n')
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[slot] = (process, job, log_handle)
            print(
                f'start gpu={slot[0]} slot={slot[1]} '
                f'{job["relative"]}',
                flush=True)

        time.sleep(2)
        for slot, (process, job, log_handle) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            del running[slot]
            if status == 0 and valid_success(
                    job['output'], args.max_samples,
                    args.expected_records):
                completed.append(str(job['relative']))
                print(f'done {job["relative"]}', flush=True)
            elif job['attempt'] <= args.retry_failures:
                jobs.append(job)
                print(
                    f'retry {job["relative"]} status={status}',
                    flush=True)
            else:
                failed.append({
                    'protocol': str(job['relative']),
                    'status': status,
                })
                print(
                    f'failed {job["relative"]} status={status}',
                    flush=True)

        write_manifest(manifest_path, {
            'queued_remaining': len(jobs),
            'deferred_at_deadline': len(jobs) if stop_launching else 0,
            'running': {
                f'{slot[0]}:{slot[1]}': str(item[1]['relative'])
                for slot, item in running.items()
            },
            'completed_this_run': completed,
            'skipped_existing': skipped,
            'failed': failed,
            'workers_per_gpu': args.workers_per_gpu,
            'dataloader_workers': args.dataloader_workers,
            'gpus': gpu_ids,
            'checkpoint_sha256': checkpoint_sha256,
            'vae_checkpoint_sha256': vae_checkpoint_sha256,
            'dispatch_shard_index': args.dispatch_shard_index,
            'num_dispatch_shards': args.num_dispatch_shards,
        })

        if stop_launching and not running:
            print(
                f'deferred={len(jobs)} protocols at dispatch deadline',
                flush=True)
            break

    if failed:
        raise SystemExit(f'{len(failed)} protocols failed')


if __name__ == '__main__':
    main()
