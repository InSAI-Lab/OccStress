#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Tokenize and evaluate one OccStress-Waymo protocol scene shard."""

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-config', default=str(
        root / 'configs/scene_tokenizer/ii_scene_tokenizer_waymo_occstress.py'))
    parser.add_argument('--world-config', default=str(
        root / 'configs/world_model/ii_generate_world_waymo_occstress.py'))
    parser.add_argument('--tokenizer-checkpoint', required=True)
    parser.add_argument('--world-checkpoint', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--clean-token-root', required=True)
    parser.add_argument(
        '--current-token-root',
        help='Existing parent directory containing token_4f; skips current tokenization')
    parser.add_argument('--scene-shard', required=True)
    parser.add_argument('--token-work-root', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--dataset', choices=('waymo', 'carla'))
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep-token-cache', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def class_mapping_metadata(base_info):
    path = Path(base_info).resolve().with_name('class_mapping.json')
    if not path.is_file():
        raise FileNotFoundError(f'class mapping not found: {path}')
    return str(path), sha256(path)


def git_commit(root):
    override = os.environ.get('IIWORLD_CODE_COMMIT')
    if override:
        return override
    try:
        return subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'not-a-git-checkout'


def parse_scenes(value):
    return tuple(
        str(scene).strip().zfill(3)
        for scene in str(value).split(',')
        if str(scene).strip()
    )


def shard_size(protocol, scenes, max_samples):
    with Path(protocol).open('rb') as stream:
        records = pickle.load(stream)
    scene_set = set(scenes)
    records = [
        record for record in records
        if str(record['scene_name']).zfill(3) in scene_set
    ]
    if max_samples:
        records = records[:max_samples]
    return len(records)


def future_token_names(protocol, scenes, max_samples, base_info):
    with Path(protocol).open('rb') as stream:
        records = pickle.load(stream)
    scene_set = set(scenes)
    records = [
        record for record in records
        if str(record['scene_name']).zfill(3) in scene_set
    ]
    if max_samples:
        records = records[:max_samples]
    from occstress.datasets.metadata import load_metadata
    base = load_metadata(base_info, trusted_pickle=True)
    token_to_info = {
        info['token']: info
        for scene_infos in base['infos'].values()
        for info in scene_infos
    }
    return {
        (
            str(record['scene_name']).zfill(3),
            f"{int(token_to_info[frame['token']]['frame_idx']):03d}_04",
        )
        for record in records
        for frame in record['future_targets']
    }


def current_token_names(protocol, scenes, max_samples, base_info):
    with Path(protocol).open('rb') as stream:
        records = pickle.load(stream)
    scene_set = set(scenes)
    records = [
        record for record in records
        if str(record['scene_name']).zfill(3) in scene_set
    ]
    if max_samples:
        records = records[:max_samples]
    from occstress.datasets.metadata import load_metadata
    base = load_metadata(base_info, trusted_pickle=True)
    token_to_info = {
        info['token']: info
        for scene_infos in base['infos'].values()
        for info in scene_infos
    }
    return {
        (
            str(record['scene_name']).zfill(3),
            f"{int(token_to_info[record['anchor_token']]['frame_idx']):03d}_04",
        )
        for record in records
    }


def run(command, root, env):
    subprocess.run(command, cwd=root, env=env, check=True)


def carla_horizon_metrics(metrics, class_mapping_path):
    with Path(class_mapping_path).open() as stream:
        class_mapping = json.load(stream)
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
    dataset_name = args.dataset or (
        'carla' if 'carla' in Path(args.world_config).name.lower() else 'waymo')
    is_carla = dataset_name == 'carla'
    scenes = parse_scenes(args.scene_shard)
    if not scenes:
        raise ValueError('scene_shard must name at least one scene')
    scene_key = '-'.join(scenes)
    expected = shard_size(args.protocol, scenes, args.max_samples)
    if expected == 0:
        raise ValueError(
            f'protocol contains no anchors for scenes {scene_key}')

    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(output.suffix + '.raw.tmp.json')
    final_tmp = output.with_suffix(output.suffix + '.tmp')
    token_root = (
        Path(args.token_work_root).resolve() /
        f'{Path(args.protocol).stem}-{sha256(args.protocol)[:12]}' / scene_key
    )
    generated_current_token_root = token_root / 'current'
    current_token_root = (
        Path(args.current_token_root).resolve()
        if args.current_token_root
        else generated_current_token_root
    )
    # Traffic mirrors targets as well; never reuse unmirrored future tokens.
    with Path(args.protocol).open('rb') as stream:
        protocol_records = pickle.load(stream)
    mirror_modes = {bool(record.get('traffic_mirror', False)) for record in protocol_records}
    if len(mirror_modes) != 1:
        raise ValueError('A protocol must use one consistent traffic-mirror policy')
    tokenizer_digest = sha256(args.tokenizer_checkpoint)
    cache_key = hashlib.sha256((
        sha256(args.base_info) + tokenizer_digest + dataset_name
        + sha256(args.tokenizer_config)
        + sha256(root / 'mmdet3d/datasets/waymo_occstress_world_dataset.py')
        + str(next(iter(mirror_modes)))
    ).encode()).hexdigest()[:20]
    future_token_root = Path(args.clean_token_root).resolve() / cache_key
    if not args.current_token_root:
        current_token_root.mkdir(parents=True, exist_ok=True)
    future_token_root.mkdir(parents=True, exist_ok=True)
    expected_current = current_token_names(
        args.protocol, scenes, args.max_samples, args.base_info)
    expected_future = future_token_names(
        args.protocol, scenes, args.max_samples, args.base_info)

    env = os.environ.copy()
    env.update({
        'OCCSTRESS_WAYMO_PROTOCOL': str(Path(args.protocol).resolve()),
        'OCCSTRESS_DATASET': dataset_name,
        'OCCSTRESS_WAYMO_BASE_INFO': str(Path(args.base_info).resolve()),
        'OCCSTRESS_WAYMO_SCENE_SHARD': ','.join(scenes),
        'OCCSTRESS_IIWORLD_WAYMO_TOKEN_ROOT': str(current_token_root),
        'IIWORLD_WAYMO_CLEAN_TOKEN_ROOT': str(
            future_token_root / 'token_4f'),
        'IIWORLD_TOKENIZER_CHECKPOINT': str(
            Path(args.tokenizer_checkpoint).resolve()),
        'II_WORLD_MS_DEFORM_ATTN_IMPL': os.environ.get(
            'II_WORLD_MS_DEFORM_ATTN_IMPL', 'pytorch'),
        'PYTHONPATH': str(root) + os.pathsep + env.get('PYTHONPATH', ''),
    })
    if args.max_samples:
        env['OCCSTRESS_WAYMO_MAX_SAMPLES'] = str(args.max_samples)
    else:
        env.pop('OCCSTRESS_WAYMO_MAX_SAMPLES', None)

    common = [
        '--launcher', 'none', '--gpu-id', str(args.gpu_id),
        '--seed', str(args.seed), '--deterministic',
    ]
    tokenizer_command = [
        sys.executable, str(root / 'tools/test.py'),
        str(Path(args.tokenizer_config).resolve()),
        str(Path(args.tokenizer_checkpoint).resolve()),
        '--discard-results', *common,
    ]
    world_command = [
        sys.executable, str(root / 'tools/test.py'),
        str(Path(args.world_config).resolve()),
        str(Path(args.world_checkpoint).resolve()),
        '--scene_checkpoints', str(Path(args.tokenizer_checkpoint).resolve()),
        '--eval', 'forecasting_miou',
        '--metrics-json', str(raw_output),
        '--stream-eval', *common,
    ]

    started = time.time()
    try:
        if args.current_token_root:
            missing_current = [
                current_token_root / 'token_4f' / scene / f'{name}.npz'
                for scene, name in expected_current
                if not (
                    current_token_root / 'token_4f' /
                    scene / f'{name}.npz').is_file()
            ]
            if missing_current:
                raise FileNotFoundError(
                    f'{len(missing_current)} existing current tokens missing; '
                    f'first={missing_current[0]}')
        else:
            env['IIWORLD_WAYMO_TOKEN_MODE'] = 'current'
            run(tokenizer_command, root, env)
            token_files = list(
                (current_token_root / 'token_4f').rglob('*.npz'))
            if len(token_files) != expected:
                raise RuntimeError(
                    f'expected {expected} anchor tokens, '
                    f'found {len(token_files)}')
        missing_future = [
            future_token_root / 'token_4f' / scene / f'{name}.npz'
            for scene, name in expected_future
            if not (
                future_token_root / 'token_4f' /
                scene / f'{name}.npz').is_file()
        ]
        if missing_future:
            env['IIWORLD_WAYMO_TOKEN_MODE'] = 'future'
            env['OCCSTRESS_IIWORLD_WAYMO_TOKEN_ROOT'] = str(future_token_root)
            run(tokenizer_command, root, env)
        future_files = list(
            (future_token_root / 'token_4f').rglob('*.npz'))
        missing_future = [
            future_token_root / 'token_4f' / scene / f'{name}.npz'
            for scene, name in expected_future
            if not (
                future_token_root / 'token_4f' /
                scene / f'{name}.npz').is_file()
        ]
        if missing_future:
            raise RuntimeError(
                f'{len(missing_future)} required future tokens are missing; '
                f'cache contains {len(future_files)} files')
        env['IIWORLD_WAYMO_TOKEN_MODE'] = 'current'
        env['OCCSTRESS_IIWORLD_WAYMO_TOKEN_ROOT'] = str(current_token_root)
        run(world_command, root, env)
        raw = json.loads(raw_output.read_text())
        if int(raw['evaluated_records']) != expected:
            raise RuntimeError(
                f'expected {expected} evaluated anchors, got '
                f'{raw["evaluated_records"]}')

        metrics = raw['metrics']
        class_mapping, class_mapping_digest = class_mapping_metadata(
            args.base_info)
        if is_carla:
            horizon_metrics = carla_horizon_metrics(metrics, class_mapping)
        else:
            horizon_metrics = {}
            for horizon in HORIZONS:
                values = {
                    'miou': float(
                        metrics[f'semantics_miou_time_{horizon:.1f}s']),
                    'iou': float(
                        metrics[f'binary_iou_time_{horizon:.1f}s']),
                }
                if not all(
                        math.isfinite(value) for value in values.values()):
                    raise ValueError(
                        f'non-finite metric at {horizon}s: {values}')
                horizon_metrics[str(horizon)] = values
        tokenizer_config = Path(args.tokenizer_config).resolve()
        world_config = Path(args.world_config).resolve()
        dataset_adapter = (
            root / 'mmdet3d/datasets/waymo_occstress_world_dataset.py').resolve()
        entrypoint = Path(__file__).resolve()

        payload = {
            'status': 'success',
            'method': 'II-World',
            'dataset': 'UniOcc-CARLA' if is_carla else 'Occ3D-Waymo',
            'code_commit': git_commit(root),
            'entrypoint_sha256': sha256(entrypoint),
            'dataset_adapter_sha256': sha256(dataset_adapter),
            'tokenizer_config_sha256': sha256(tokenizer_config),
            'world_config_sha256': sha256(world_config),
            'checkpoint': str(Path(args.world_checkpoint).resolve()),
            'checkpoint_sha256': sha256(args.world_checkpoint),
            'tokenizer_checkpoint': str(
                Path(args.tokenizer_checkpoint).resolve()),
            'tokenizer_checkpoint_sha256': tokenizer_digest,
            'current_token_source': (
                'existing_official_cache'
                if args.current_token_root else 'manual_h4_adapter'),
            'current_token_root': str(current_token_root),
            'future_token_source': 'shared_clean_cache',
            'future_cache_key': cache_key,
            'targets_mirrored': next(iter(mirror_modes)),
            'future_token_root': str(future_token_root),
            'required_future_token_count': len(expected_future),
            'protocol': str(Path(args.protocol).resolve()),
            'protocol_sha256': sha256(args.protocol),
            'base_info': str(Path(args.base_info).resolve()),
            'class_mapping': class_mapping,
            'class_mapping_sha256': class_mapping_digest,
            'scene_shard': scene_key,
            'scene_shards': list(scenes),
            'evaluated_records': expected,
            'observed_offsets_seconds': [-2.0, -1.5, -1.0, -0.5, 0.0],
            'control_policy': (
                'carla_gt_future_command_zero_lcf_gt_ego_pose'
                if is_carla else
                'official_waymo_fixed_command_100_zero_lcf_gt_ego_pose'),
            'ms_deform_attn_impl': env['II_WORLD_MS_DEFORM_ATTN_IMPL'],
            'future_horizons_seconds': list(HORIZONS),
            'horizon_metrics': horizon_metrics,
            'average_miou': sum(
                value['miou'] for value in horizon_metrics.values()) / 6,
            'average_iou': sum(
                value['iou'] for value in horizon_metrics.values()) / 6,
            'aggregate_counts': metrics.get('_aggregate_counts'),
            'elapsed_seconds': time.time() - started,
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        }
        if is_carla:
            payload.update({
                'metric_policy': 'present_gt_occupied_classes',
                'coordinate_convention': (
                    'Occ3D right-handed ego-local x-forward/y-left/z-up'),
                'zero_shot_source_checkpoint': 'Occ3D-nuScenes',
            })
        final_tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
        os.replace(final_tmp, output)
    finally:
        raw_output.unlink(missing_ok=True)
        if not args.keep_token_cache and output.exists():
            shutil.rmtree(token_root, ignore_errors=True)

    print(json.dumps({
        'status': payload['status'],
        'scene_shard': payload['scene_shard'],
        'scene_shards': payload['scene_shards'],
        'evaluated_records': payload['evaluated_records'],
        'average_miou': payload['average_miou'],
        'average_iou': payload['average_iou'],
        'elapsed_seconds': payload['elapsed_seconds'],
        'output_json': str(output),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
