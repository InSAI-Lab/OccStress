"""Sequential native tasks with content-checked completion receipts.

Explicit execute only. Parallelism belongs to the caller's GPU scheduler, using
disjoint output paths. The per-output lock prevents duplicate concurrent work.
"""
import os
from pathlib import Path
import shlex
import subprocess

from occstress.adapters.forecasting import command_plan
from occstress.adapters.native_results import extract_counts
from occstress.protocols.temporal import HORIZONS
from occstress.results.io import content_hash, load_json, sha256_file, write_json


def task_plan(task, root):
    if not task.get('id') or task.get('expected_records', 0) <= 0 or not task.get('inputs'):
        raise ValueError('Tasks require id, expected_records and named input artifact paths')
    output = Path(task['native_output']).expanduser()
    if not output.is_absolute():
        raise ValueError('native_output must be absolute (native models run in different directories)')
    if any(not Path(path).expanduser().is_absolute() for path in task['inputs'].values()):
        raise ValueError('Input artifact paths must be absolute')
    if output.resolve() in {Path(p).expanduser().resolve() for p in task['inputs'].values()}:
        raise ValueError('Native output must not overwrite an input artifact')
    env = dict(os.environ)
    env.update(task.get('env', {}))
    directory, command, env = command_plan(task['model'], task['dataset'], task['python'],
                                           task['native_args'], root=root, environ=env)
    return output, directory, command, env


def fingerprint(task, root, env):
    return content_hash({'task': task,
                         'inputs': {key: sha256_file(Path(p).expanduser()) for key, p in task['inputs'].items()},
                         'formal_sources': sha256_file(root / 'configs/formal_configs.lock.json'),
                         'environment': {key: value for key, value in env.items()
                                         if key.startswith(('OCCSTRESS_', 'OCCWORLD_', 'COME_', 'II_',
                                                            'GENIEDRIVE_', 'DOME_', 'OCCSTRESS_WAYMO_', 'OCCSTRESS_CARLA_'))}})


def result_valid(output, expected_records):
    value = load_json(output)
    if value.get('evaluated_records') != expected_records:
        raise ValueError('Native result has an unexpected evaluated-record count')
    if value.get('future_horizons_seconds', HORIZONS) != HORIZONS:
        raise ValueError('Native result has non-F6 horizons')
    extract_counts(value)
    return value


def can_resume(output, receipt_path, digest, expected_records):
    if not output.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = load_json(receipt_path)
        if receipt.get('fingerprint') != digest or receipt.get('native_sha256') != sha256_file(output):
            return False
        result_valid(output, expected_records)
        return receipt.get('status') == 'success'
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_manifest(path, root, execute=False):
    manifest = load_json(path)
    if manifest.get('schema_version') != 1 or not manifest.get('tasks'):
        raise ValueError('Expected version-1 manifest with tasks')
    tasks = manifest['tasks']
    plans = [task_plan(task, root) for task in tasks]
    if len({task['id'] for task in tasks}) != len(tasks) or len({p[0].resolve() for p in plans}) != len(plans):
        raise ValueError('Task IDs and native output paths must be unique')
    for task, (output, directory, command, env) in zip(tasks, plans):
        print(task['id'] + ': ' + shlex.join(command), flush=True)
        if not execute:
            continue
        digest = fingerprint(task, root, env)
        receipt_path = output.with_suffix(output.suffix + '.done.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        lock = output.with_suffix(output.suffix + '.lock')
        # A crash leaves a visible lock; check the owner before manually removing it.
        with lock.open('x') as handle:
            handle.write(str(os.getpid()) + '\n')
        try:
            if can_resume(output, receipt_path, digest, task['expected_records']):
                print(task['id'] + ': verified success, skipped', flush=True)
                continue
            if output.exists():
                raise ValueError(f'Unverified existing output {output}; choose a fresh path or archive it first')
            subprocess.run(command, cwd=directory, env=env, check=True)
            result_valid(output, task['expected_records'])
            if fingerprint(task, root, env) != digest:
                raise ValueError('Input artifacts changed while this task was running')
            write_json(receipt_path, {'schema_version': 1, 'status': 'success',
                                     'fingerprint': digest, 'native_sha256': sha256_file(output)})
        finally:
            lock.unlink()
