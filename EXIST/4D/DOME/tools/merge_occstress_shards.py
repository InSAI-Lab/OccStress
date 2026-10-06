#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_FILES = (
    ROOT / 'dataset/occstress_dataset.py',
    ROOT / 'tools/eval_occstress.py',
    ROOT / 'tools/merge_occstress_shards.py',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge deterministic DOME OccStress evaluation shards.')
    parser.add_argument('--shard-dir', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--keep-raw-confusion', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_metrics(confusion):
    per_class = []
    for class_index in range(17):
        seen = int(confusion[class_index, :].sum())
        positive = int(confusion[:, class_index].sum())
        correct = int(confusion[class_index, class_index])
        if seen == 0:
            per_class.append(100.0)
        else:
            per_class.append(
                100.0 * correct / (seen + positive - correct))
    return float(np.mean(per_class)), per_class


def binary_iou(confusion):
    seen = int(confusion[1, :].sum())
    positive = int(confusion[:, 1].sum())
    correct = int(confusion[1, 1])
    if seen == 0:
        return 100.0
    return 100.0 * correct / (seen + positive - correct)


def horizon_result(semantic, binary):
    result = {}
    for index, horizon in enumerate(HORIZONS):
        miou, per_class = semantic_metrics(semantic[index])
        result[str(horizon)] = {
            'miou': miou,
            'iou': binary_iou(binary[index]),
            'per_class_iou': per_class,
        }
    return result


def load_shards(shard_dir, num_shards):
    payloads = []
    for shard_index in range(num_shards):
        path = shard_dir / f'shard_{shard_index:02d}.json'
        with path.open() as handle:
            payload = json.load(handle)
        if payload.get('status') != 'success':
            raise ValueError(f'{path} is not successful')
        if payload.get('num_shards') != num_shards:
            raise ValueError(f'{path} has wrong num_shards')
        if payload.get('shard_index') != shard_index:
            raise ValueError(f'{path} has wrong shard_index')
        payloads.append((path, payload))
    return payloads


def main():
    args = parse_args()
    shards = load_shards(args.shard_dir.resolve(), args.num_shards)
    invariant_keys = (
        'method',
        'code_commit',
        'config',
        'checkpoint_sha256',
        'vae_checkpoint_sha256',
        'protocol',
        'protocol_name',
        'protocol_sha256',
        'base_info',
        'occstress_root',
        'observed_offsets_seconds',
        'ignored_offset_seconds',
        'control_policy',
        'future_horizons_seconds',
        'sampling',
    )
    reference = shards[0][1]
    for path, payload in shards[1:]:
        mismatched = [
            key for key in invariant_keys
            if payload.get(key) != reference.get(key)
        ]
        if mismatched:
            raise ValueError(f'{path} mismatches shard 0: {mismatched}')

    semantic = np.zeros((6, 18, 18), dtype=np.int64)
    binary = np.zeros((6, 2, 2), dtype=np.int64)
    reconstruction_semantic = np.zeros((6, 18, 18), dtype=np.int64)
    reconstruction_binary = np.zeros((6, 2, 2), dtype=np.int64)
    evaluated_records = 0
    for _, payload in shards:
        raw = payload['raw_confusion']
        semantic += np.asarray(raw['semantic'], dtype=np.int64)
        binary += np.asarray(raw['binary'], dtype=np.int64)
        reconstruction_semantic += np.asarray(
            raw['reconstruction_semantic'], dtype=np.int64)
        reconstruction_binary += np.asarray(
            raw['reconstruction_binary'], dtype=np.int64)
        evaluated_records += int(payload['evaluated_records'])

    expected_records = int(reference['full_protocol_records'])
    if evaluated_records != expected_records:
        raise ValueError(
            f'merged {evaluated_records} records, expected {expected_records}')

    horizon_metrics = horizon_result(semantic, binary)
    reconstruction = horizon_result(
        reconstruction_semantic, reconstruction_binary)['0.5']
    average_miou = float(np.mean(
        [value['miou'] for value in horizon_metrics.values()]))
    average_iou = float(np.mean(
        [value['iou'] for value in horizon_metrics.values()]))
    paper_keys = ('1.0', '2.0', '3.0')

    result = dict(reference)
    result.update({
        'status': 'success',
        'evaluated_records': evaluated_records,
        'num_shards': args.num_shards,
        'shard_index': None,
        'horizon_metrics': horizon_metrics,
        'average_miou': average_miou,
        'average_iou': average_iou,
        'paper_average_miou': float(np.mean(
            [horizon_metrics[key]['miou'] for key in paper_keys])),
        'paper_average_iou': float(np.mean(
            [horizon_metrics[key]['iou'] for key in paper_keys])),
        'current_reconstruction_miou': reconstruction['miou'],
        'current_reconstruction_iou': reconstruction['iou'],
        'elapsed_seconds': float(max(
            payload['elapsed_seconds'] for _, payload in shards)),
        'aggregate_gpu_seconds': float(sum(
            payload['elapsed_seconds'] for _, payload in shards)),
        'anchors_per_second': float(sum(
            payload['anchors_per_second'] for _, payload in shards)),
        'peak_gpu_memory_mib': float(max(
            payload['peak_gpu_memory_mib'] for _, payload in shards)),
        'shard_results': [str(path) for path, _ in shards],
        'hostnames': sorted({
            payload.get('hostname') for _, payload in shards
            if payload.get('hostname')
        }),
        'slurm_job_ids': sorted({
            payload.get('slurm_job_id') for _, payload in shards
            if payload.get('slurm_job_id')
        }),
        'adapter_provenance': {
            'git_commit': reference.get('code_commit'),
            'file_sha256': {
                str(path.relative_to(ROOT)): sha256(path)
                for path in PROVENANCE_FILES
            },
        },
    })
    if args.keep_raw_confusion:
        result['raw_confusion'] = {
            'semantic': semantic.tolist(),
            'binary': binary.tolist(),
            'reconstruction_semantic': reconstruction_semantic.tolist(),
            'reconstruction_binary': reconstruction_binary.tolist(),
        }
    else:
        result.pop('raw_confusion', None)

    values = [
        value
        for item in horizon_metrics.values()
        for value in (item['miou'], item['iou'])
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('merged metrics contain non-finite values')

    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        'status': 'success',
        'output': str(output),
        'evaluated_records': evaluated_records,
        'average_miou': average_miou,
        'average_iou': average_iou,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
