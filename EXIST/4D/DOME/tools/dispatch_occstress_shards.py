#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from dispatch_occstress import HORIZONS, protocol_jobs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one global worker of sharded DOME OccStress evaluation.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--vae-checkpoint-sha256', required=True)
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
        '--config', default=str(ROOT / 'config/train_dome.py'))
    parser.add_argument('--track', choices=(
        'all', 'manual', 'stcocc', 'sdgocc', 'camera', 'pointcloud'),
        default='all')
    parser.add_argument('--include-position-sweep', action='store_true')
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--worker-index', type=int, required=True)
    parser.add_argument('--global-workers', type=int, required=True)
    parser.add_argument(
        '--source-global-workers',
        type=int,
        help='Original global worker count whose task partition is being reassigned.')
    parser.add_argument(
        '--source-worker-indices',
        help='Comma-separated original worker indices to reassign.')
    parser.add_argument('--worker-group', default='main')
    parser.add_argument('--local-concurrency', type=int, default=11)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--retry-failures', type=int, default=1)
    parser.add_argument('--launch-stagger-seconds', type=float, default=1.0)
    parser.add_argument('--coordinator', action='store_true')
    parser.add_argument('--coordinator-timeout-seconds', type=int, default=28800)
    return parser.parse_args()


def final_is_valid(path, expected):
    if not path.is_file():
        return False
    try:
        with path.open() as handle:
            payload = json.load(handle)
        metrics = payload.get('horizon_metrics', {})
        values = [
            value
            for horizon in HORIZONS
            for value in (
                metrics.get(str(horizon), {}).get('miou'),
                metrics.get(str(horizon), {}).get('iou'),
            )
        ]
        return (
            payload.get('status') == 'success'
            and payload.get('method') == 'DOME'
            and payload.get('evaluated_records') == expected
            and all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in values
            )
        )
    except (OSError, ValueError, TypeError):
        return False


def shard_expected(total, num_shards, shard_index):
    if shard_index >= total:
        return 0
    return (total - 1 - shard_index) // num_shards + 1


def shard_is_valid(path, args, protocol, expected):
    if not path.is_file():
        return False
    try:
        with path.open() as handle:
            payload = json.load(handle)
        return (
            payload.get('status') == 'success'
            and payload.get('method') == 'DOME'
            and payload.get('checkpoint_sha256') == args.checkpoint_sha256
            and payload.get('vae_checkpoint_sha256')
            == args.vae_checkpoint_sha256
            and Path(payload.get('protocol', '')).resolve()
            == protocol.resolve()
            and payload.get('num_shards') == args.num_shards
            and payload.get('evaluated_records') == expected
            and 'raw_confusion' in payload
        )
    except (OSError, ValueError, TypeError):
        return False


def build_tasks(args, inventory):
    total_records = args.max_samples if args.max_samples is not None else 4519
    tasks = []
    for _, track, protocol, relative in inventory:
        output = Path(args.output_root).resolve() / relative.with_suffix('.json')
        shard_dir = (
            Path(args.output_root).resolve()
            / '.shards'
            / relative.with_suffix('')
        )
        for shard_index in range(args.num_shards):
            tasks.append({
                'track': track,
                'protocol': protocol.resolve(),
                'relative': relative,
                'output': output,
                'shard_dir': shard_dir,
                'shard_index': shard_index,
                'expected': shard_expected(
                    total_records, args.num_shards, shard_index),
                'attempt': 0,
            })
    return tasks


def select_source_tasks(args, all_tasks):
    if args.source_global_workers is None and args.source_worker_indices is None:
        return all_tasks
    if args.source_global_workers is None or args.source_worker_indices is None:
        raise ValueError(
            '--source-global-workers and --source-worker-indices must be used together')
    if args.source_global_workers <= 0:
        raise ValueError('--source-global-workers must be positive')
    selected_indices = {
        int(value)
        for value in args.source_worker_indices.split(',')
        if value.strip()
    }
    if not selected_indices:
        raise ValueError('--source-worker-indices is empty')
    invalid = sorted(
        index for index in selected_indices
        if index < 0 or index >= args.source_global_workers
    )
    if invalid:
        raise ValueError(f'invalid source worker indices: {invalid}')
    return [
        task
        for task_index, task in enumerate(all_tasks)
        if task_index % args.source_global_workers in selected_indices
    ]


def worker_tasks(args, selected_tasks):
    if args.global_workers <= 0:
        raise ValueError('--global-workers must be positive')
    if not 0 <= args.worker_index < args.global_workers:
        raise ValueError('--worker-index must be in [0, global-workers)')
    return [
        task for task_index, task in enumerate(selected_tasks)
        if task_index % args.global_workers == args.worker_index
    ]


def launch_command(args, task, shard_json):
    command = [
        sys.executable,
        str(ROOT / 'tools/eval_occstress.py'),
        '--config', str(Path(args.config).resolve()),
        '--checkpoint', str(Path(args.checkpoint).resolve()),
        '--vae-checkpoint', str(Path(args.vae_checkpoint).resolve()),
        '--checkpoint-sha256', args.checkpoint_sha256,
        '--vae-checkpoint-sha256', args.vae_checkpoint_sha256,
        '--protocol', str(task['protocol']),
        '--base-info', str(Path(args.base_info).resolve()),
        '--occstress-root', str(Path(args.occstress_root).resolve()),
        '--output-json', str(shard_json),
        '--num-workers', str(args.num_workers),
        '--num-shards', str(args.num_shards),
        '--shard-index', str(task['shard_index']),
    ]
    if args.nuscenes_root:
        command += [
            '--nuscenes-root', str(Path(args.nuscenes_root).resolve())]
    if args.max_samples is not None:
        command += ['--max-samples', str(args.max_samples)]
    return command


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_local_tasks(args, tasks):
    queue = deque(tasks)
    running = {}
    completed = []
    skipped = []
    failed = []
    state_path = (
        Path(args.output_root).resolve()
        / '.workers'
        / f'{args.worker_group}_worker_{args.worker_index:03d}.json'
    )
    while queue or running:
        while queue and len(running) < args.local_concurrency:
            task = queue.popleft()
            if final_is_valid(
                    task['output'],
                    args.max_samples if args.max_samples is not None else 4519):
                skipped.append(str(task['relative']))
                continue
            shard_json = (
                task['shard_dir']
                / f"shard_{task['shard_index']:02d}.json"
            )
            if shard_is_valid(
                    shard_json, args, task['protocol'], task['expected']):
                skipped.append(
                    f"{task['relative']}#{task['shard_index']}")
                continue
            task['attempt'] += 1
            log_path = shard_json.with_suffix('.log')
            shard_json.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open('a')
            log_handle.write(
                f'\n=== worker={args.worker_index} '
                f'attempt={task["attempt"]} time={time.time()} ===\n')
            log_handle.flush()
            process = subprocess.Popen(
                launch_command(args, task, shard_json),
                cwd=ROOT,
                env={
                    **os.environ,
                    'OMP_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1',
                },
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[process.pid] = (process, task, shard_json, log_handle)
            print(
                f"start pid={process.pid} {task['relative']} "
                f"shard={task['shard_index']}/{args.num_shards}",
                flush=True,
            )
            if args.launch_stagger_seconds:
                time.sleep(args.launch_stagger_seconds)

        if running:
            time.sleep(2)
        for pid, (process, task, shard_json, log_handle) in list(
                running.items()):
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            del running[pid]
            if (
                    status == 0
                    and shard_is_valid(
                        shard_json, args, task['protocol'], task['expected'])):
                completed.append(
                    f"{task['relative']}#{task['shard_index']}")
                print(
                    f"done {task['relative']} "
                    f"shard={task['shard_index']}",
                    flush=True,
                )
            elif task['attempt'] <= args.retry_failures:
                queue.append(task)
                print(
                    f"retry status={status} {task['relative']} "
                    f"shard={task['shard_index']}",
                    flush=True,
                )
            else:
                failure = {
                    'protocol': str(task['relative']),
                    'shard_index': task['shard_index'],
                    'status': status,
                }
                failed.append(failure)
                print(f'failed {failure}', flush=True)

        write_json(state_path, {
            'status': 'running' if queue or running else (
                'failed' if failed else 'success'),
            'worker_index': args.worker_index,
            'worker_group': args.worker_group,
            'global_workers': args.global_workers,
            'local_concurrency': args.local_concurrency,
            'queued_remaining': len(queue),
            'running': [
                {
                    'protocol': str(item[1]['relative']),
                    'shard_index': item[1]['shard_index'],
                    'pid': pid,
                }
                for pid, item in running.items()
            ],
            'completed': completed,
            'skipped': skipped,
            'failed': failed,
            'hostname': os.uname().nodename,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        })
    return failed


def coordinate_merges(args, inventory):
    output_root = Path(args.output_root).resolve()
    expected = args.max_samples if args.max_samples is not None else 4519
    pending = deque(inventory)
    failures = []
    started = time.time()
    status_path = output_root / 'sharded_dispatch_manifest.json'
    while pending:
        next_pending = deque()
        progress = 0
        for _, _, protocol, relative in pending:
            output = output_root / relative.with_suffix('.json')
            if final_is_valid(output, expected):
                progress += 1
                continue
            shard_dir = output_root / '.shards' / relative.with_suffix('')
            shard_paths = [
                shard_dir / f'shard_{index:02d}.json'
                for index in range(args.num_shards)
            ]
            if all(path.is_file() for path in shard_paths):
                command = [
                    sys.executable,
                    str(ROOT / 'tools/merge_occstress_shards.py'),
                    '--shard-dir', str(shard_dir),
                    '--output-json', str(output),
                    '--num-shards', str(args.num_shards),
                ]
                result = subprocess.run(
                    command, cwd=ROOT, text=True, capture_output=True)
                if result.returncode == 0 and final_is_valid(output, expected):
                    progress += 1
                    print(f'merged {relative}', flush=True)
                    continue
                failures.append({
                    'protocol': str(relative),
                    'status': result.returncode,
                    'stderr': result.stderr[-2000:],
                })
            next_pending.append((0, '', protocol, relative))
        pending = next_pending
        write_json(status_path, {
            'status': 'running' if pending else 'success',
            'protocol_count': len(inventory),
            'complete': len(inventory) - len(pending),
            'pending': len(pending),
            'merge_failures': failures[-20:],
            'num_shards': args.num_shards,
            'global_workers': args.global_workers,
            'elapsed_seconds': time.time() - started,
        })
        if not pending:
            break
        if time.time() - started > args.coordinator_timeout_seconds:
            raise TimeoutError(
                f'coordinator timed out with {len(pending)} protocols pending')
        time.sleep(10 if progress else 30)


def main():
    args = parse_args()
    if args.local_concurrency <= 0:
        raise ValueError('--local-concurrency must be positive')
    inventory = protocol_jobs(args)
    all_tasks = build_tasks(args, inventory)
    selected_tasks = select_source_tasks(args, all_tasks)
    assigned = worker_tasks(args, selected_tasks)
    print(json.dumps({
        'worker_index': args.worker_index,
        'global_workers': args.global_workers,
        'protocol_count': len(inventory),
        'global_task_count': len(all_tasks),
        'selected_task_count': len(selected_tasks),
        'assigned_task_count': len(assigned),
        'num_shards': args.num_shards,
        'local_concurrency': args.local_concurrency,
        'coordinator': args.coordinator,
        'worker_group': args.worker_group,
    }, indent=2, sort_keys=True), flush=True)
    failures = run_local_tasks(args, assigned)
    if failures:
        raise SystemExit(f'{len(failures)} shard tasks failed')
    if args.coordinator:
        coordinate_merges(args, inventory)


if __name__ == '__main__':
    main()
