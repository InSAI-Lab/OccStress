#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]
FUTURE_SECONDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PAPER_TARGET = {'miou': 13.20, 'iou': 22.03}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stream SparseWorld predictions into OccStress metrics.')
    parser.add_argument(
        '--config',
        default='configs/sparseworld/nuscenes-temporal/'
                'sparseworld-traj-finetune.py')
    parser.add_argument('--checkpoint', default='ckpts/epoch_56.pth')
    parser.add_argument(
        '--ann-file',
        default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl')
    parser.add_argument(
        '--camera-root',
        default=str(Path(os.environ.get('OCCSTRESS_CODE_ROOT',
                         REPO_ROOT.parents[2])) / 'data/nuScenes-C/raw/image/nuScenes-c'))
    parser.add_argument('--manifest', default='occstress/protocols.json')
    parser.add_argument('--protocol-index', type=int)
    parser.add_argument('--protocol-id', default='clean')
    parser.add_argument('--corruption', default='clean')
    parser.add_argument('--severity')
    parser.add_argument('--frame-protocol')
    parser.add_argument('--output-json')
    parser.add_argument('--output-root', default='occstress/results')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--anchor-mode', choices=['occstress', 'official'],
        default='occstress')
    parser.add_argument('--require-paper-match', action='store_true')
    parser.add_argument('--paper-tolerance', type=float, default=0.30)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def resolve_protocol(args):
    if args.protocol_index is None:
        protocol = {
            'id': args.protocol_id,
            'corruption': args.corruption,
            'severity': args.severity,
            'frame_protocol': args.frame_protocol,
        }
    else:
        manifest_path = (REPO_ROOT / args.manifest).resolve()
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        protocols = manifest['protocols']
        if args.protocol_index < 0 or args.protocol_index >= len(protocols):
            raise IndexError(
                f'Protocol index {args.protocol_index} is outside '
                f'[0, {len(protocols) - 1}]')
        protocol = protocols[args.protocol_index]
    if protocol['corruption'] != 'clean':
        if not protocol.get('severity') or not protocol.get('frame_protocol'):
            raise ValueError(f'Incomplete protocol: {protocol}')
    return protocol


def atomic_json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)


def successful_result_exists(
        path, protocol_id, max_samples, anchor_mode):
    if not path.is_file():
        return False
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get('status') != 'success':
        return False
    if payload.get('protocol', {}).get('id') != protocol_id:
        return False
    if payload.get('anchor_mode') != anchor_mode:
        return False
    if max_samples is not None:
        if payload.get('sample_count') != max_samples:
            return False
    elif payload.get('sample_count') != payload.get('dataset_sample_count'):
        return False
    return True


def sha256_file(path):
    digest_path = path.with_suffix(path.suffix + '.sha256')
    if digest_path.is_file():
        fields = digest_path.read_text().strip().split()
        if len(fields) >= 1 and len(fields[0]) == 64:
            return fields[0]
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    tmp = digest_path.with_name(f'.{digest_path.name}.{os.getpid()}.tmp')
    tmp.write_text(f'{value}  {path.name}\n')
    os.replace(tmp, digest_path)
    return value


def sha256_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(
            ['git', *args], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inject_occstress_source(pipeline, protocol, camera_root):
    spec = None
    if protocol['corruption'] != 'clean':
        spec = {
            'root': str(Path(camera_root).resolve()),
            'corruption': protocol['corruption'],
            'severity': protocol['severity'],
            'frame_protocol': protocol['frame_protocol'],
        }
    found = set()
    for transform in pipeline:
        transform_type = transform.get('type')
        if transform_type in {
                'LoadMultiViewImageFromFiles',
                'LoadMultiViewImageFromMultiSweeps'}:
            transform['occstress'] = spec
            found.add(transform_type)
    expected = {
        'LoadMultiViewImageFromFiles',
        'LoadMultiViewImageFromMultiSweeps',
    }
    if found != expected:
        raise RuntimeError(
            f'Could not inject OccStress source into pipeline: found {found}')


def unwrap_array(value):
    while value.__class__.__name__ == 'DataContainer':
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        import torch
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    value = np.asarray(value)
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    return value


def confusion(prediction, target, classes):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != (200, 200, 16):
        raise ValueError(f'Unexpected prediction shape: {prediction.shape}')
    if target.shape != prediction.shape:
        raise ValueError(
            f'Prediction/target shape mismatch: '
            f'{prediction.shape} versus {target.shape}')
    if not np.isfinite(prediction).all():
        raise ValueError('Prediction contains NaN or Inf')
    if prediction.min() < 0 or prediction.max() >= classes:
        raise ValueError(
            f'Prediction labels outside [0, {classes - 1}]: '
            f'{prediction.min()}..{prediction.max()}')
    valid = (target >= 0) & (target < classes)
    encoded = classes * target[valid].astype(np.int64)
    encoded += prediction[valid].astype(np.int64)
    return np.bincount(
        encoded, minlength=classes * classes).reshape(classes, classes)


def metrics_from_hist(semantic_hist, occupancy_hist):
    semantic_denominator = (
        semantic_hist.sum(1) + semantic_hist.sum(0)
        - np.diag(semantic_hist))
    semantic_iou = np.divide(
        np.diag(semantic_hist),
        semantic_denominator,
        out=np.full(semantic_denominator.shape, np.nan, dtype=np.float64),
        where=semantic_denominator != 0,
    )
    occupancy_denominator = (
        occupancy_hist.sum(1) + occupancy_hist.sum(0)
        - np.diag(occupancy_hist))
    occupancy_iou = np.divide(
        np.diag(occupancy_hist),
        occupancy_denominator,
        out=np.full(occupancy_denominator.shape, np.nan, dtype=np.float64),
        where=occupancy_denominator != 0,
    )
    return {
        'miou': float(np.nanmean(semantic_iou[:17]) * 100),
        'iou': float(occupancy_iou[1] * 100),
        'per_class_iou': {
            name: (None if np.isnan(value) else float(value * 100))
            for name, value in zip(CLASS_NAMES, semantic_iou)
        },
    }


def add_prediction(hist_semantic, hist_occupancy, horizon, pred, gt):
    hist_semantic[horizon] += confusion(pred, gt, 18)
    pred_occupied = (pred != 17).astype(np.uint8)
    gt_occupied = (gt != 17).astype(np.uint8)
    hist_occupancy[horizon] += confusion(
        pred_occupied, gt_occupied, 2)


def load_framework():
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.parallel import _functions as mmcv_parallel_functions
    from mmcv.runner import load_checkpoint
    import mmdet
    from mmdet.apis import set_random_seed
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    # MMCV 1.x passes integer device IDs to a private PyTorch helper whose
    # PyTorch 2.8 signature now requires torch.device.
    torch_get_stream = torch.nn.parallel._functions._get_stream

    def get_stream_compat(device):
        if isinstance(device, int):
            device = torch.device('cuda', device)
        return torch_get_stream(device)

    mmcv_parallel_functions._get_stream = get_stream_compat

    if mmdet.__version__ > '2.23.0':
        from mmdet.utils import compat_cfg, setup_multi_processes
    else:
        from mmdet3d.utils import compat_cfg, setup_multi_processes
    return {
        'torch': torch,
        'Config': Config,
        'MMDataParallel': MMDataParallel,
        'load_checkpoint': load_checkpoint,
        'set_random_seed': set_random_seed,
        'replace_ImageToTensor': replace_ImageToTensor,
        'build_dataloader': build_dataloader,
        'build_dataset': build_dataset,
        'build_model': build_model,
        'patch_config': patch_config,
        'compat_cfg': compat_cfg,
        'setup_multi_processes': setup_multi_processes,
    }


def build_runtime(args, protocol, framework):
    config_path = (REPO_ROOT / args.config).resolve()
    cfg = framework['Config'].fromfile(str(config_path))
    cfg = framework['compat_cfg'](cfg)
    cfg = framework['patch_config'](cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if '4D' in cfg.model.type:
        cfg.model.align_after_view_transfromation = True
    cfg.data.test.test_mode = True
    cfg.data.test.occstress_full_anchors = args.anchor_mode == 'occstress'
    ann_file = Path(args.ann_file).expanduser()
    if not ann_file.is_absolute():
        ann_file = REPO_ROOT / ann_file
    ann_file = ann_file.resolve()
    cfg.data.test.ann_file = str(ann_file)
    inject_occstress_source(
        cfg.data.test.pipeline, protocol, args.camera_root)
    cfg.data.test.pipeline = framework['replace_ImageToTensor'](
        cfg.data.test.pipeline)
    cfg.data.workers_per_gpu = args.workers
    framework['setup_multi_processes'](cfg)
    framework['set_random_seed'](args.seed, deterministic=True)

    dataset = framework['build_dataset'](cfg.data.test)
    loader = framework['build_dataloader'](
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )
    model = framework['build_model'](
        cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint_path = (REPO_ROOT / args.checkpoint).resolve()
    checkpoint = framework['load_checkpoint'](
        model, str(checkpoint_path), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get(
        'CLASSES', dataset.CLASSES)
    model = framework['MMDataParallel'](model.cuda(), device_ids=[0])
    model.eval()
    return dataset, loader, model, checkpoint_path, ann_file


def evaluate(args, protocol, output_path):
    framework = load_framework()
    torch = framework['torch']
    if not torch.cuda.is_available():
        raise RuntimeError('SparseWorld OccStress evaluation requires CUDA')

    dataset, loader, model, checkpoint_path, ann_file = build_runtime(
        args, protocol, framework)
    expected_samples = len(dataset)
    sample_limit = (
        expected_samples if args.max_samples is None
        else min(args.max_samples, expected_samples))
    if args.require_paper_match and sample_limit != expected_samples:
        raise ValueError('--require-paper-match requires the full dataset')
    if args.require_paper_match and args.anchor_mode != 'official':
        raise ValueError(
            '--require-paper-match requires --anchor-mode official')

    semantic_hist = np.zeros((6, 18, 18), dtype=np.int64)
    occupancy_hist = np.zeros((6, 2, 2), dtype=np.int64)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    processed = 0

    for data in loader:
        if processed >= sample_limit:
            break
        temporal_gt = data['temporal_semantics'][0]
        with torch.inference_mode():
            result = model(return_loss=False, rescale=True, **data)
        for horizon in range(6):
            pred = unwrap_array(
                result[f'semantic_occ_{horizon + 1}s'][0])
            gt = unwrap_array(
                temporal_gt[horizon + 1]['voxel_semantics'])
            add_prediction(
                semantic_hist, occupancy_hist, horizon, pred, gt)
        processed += 1
        if processed == 1 or processed % 25 == 0:
            elapsed = time.time() - started
            print(
                f'[{protocol["id"]}] {processed}/{sample_limit} '
                f'({processed / elapsed:.2f} samples/s)',
                flush=True)

    if processed != sample_limit:
        raise RuntimeError(
            f'Evaluated {processed} samples, expected {sample_limit}')

    horizon_metrics = [
        metrics_from_hist(semantic_hist[i], occupancy_hist[i])
        for i in range(6)
    ]
    paper_indices = [1, 3, 5]
    paper_metrics = {
        metric: float(np.mean([
            horizon_metrics[index][metric] for index in paper_indices]))
        for metric in ('miou', 'iou')
    }
    gate = {
        'required': args.require_paper_match,
        'targets': PAPER_TARGET,
        'tolerance': args.paper_tolerance,
        'passed': all(
            abs(paper_metrics[key] - target) <= args.paper_tolerance
            for key, target in PAPER_TARGET.items()),
    }
    status = (
        'success'
        if not args.require_paper_match or gate['passed']
        else 'paper_gate_failed')
    adapter_paths = [
        REPO_ROOT / 'tools/occstress_eval.py',
        REPO_ROOT / 'mmdet3d/datasets/pipelines/loading.py',
        REPO_ROOT / 'mmdet3d/datasets/'
                    'nuscenes_dataset_occ_trajectory.py',
    ]
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if args.protocol_index is not None:
        adapter_paths.append(manifest_path)
    result_payload = {
        'status': status,
        'method': 'SparseWorld',
        'evaluated_records': processed,
        'protocol': protocol,
        'anchor_mode': args.anchor_mode,
        'sample_count': processed,
        'dataset_sample_count': expected_samples,
        'input_times_seconds': [0.0, -0.5, -1.0, -1.5, -2.0],
        'future_times_seconds': FUTURE_SECONDS,
        'metrics': {
            'horizons': [
                {'seconds': seconds, **metrics}
                for seconds, metrics in zip(
                    FUTURE_SECONDS, horizon_metrics)
            ],
            'mean_six_frames': {
                key: float(np.mean([
                    item[key] for item in horizon_metrics]))
                for key in ('miou', 'iou')
            },
            'paper_1s_2s_3s_average': paper_metrics,
        },
        'paper_reproduction_gate': gate,
        'semantic_confusion': semantic_hist.tolist(),
        'occupancy_confusion': occupancy_hist.tolist(),
        'runtime': {
            'elapsed_seconds': time.time() - started,
            'samples_per_second': processed / (time.time() - started),
            'peak_gpu_memory_gib': (
                torch.cuda.max_memory_allocated() / 1024 ** 3),
            'gpu': torch.cuda.get_device_name(0),
            'cuda_visible_devices': os.environ.get(
                'CUDA_VISIBLE_DEVICES'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_array_job_id': os.environ.get(
                'SLURM_ARRAY_JOB_ID'),
            'slurm_array_task_id': os.environ.get(
                'SLURM_ARRAY_TASK_ID'),
            'hostname': socket.gethostname(),
        },
        'provenance': {
            'code_commit': git_value('rev-parse', 'HEAD'),
            'code_dirty': bool(git_value('status', '--porcelain')),
            'adapter_sha256': sha256_files(adapter_paths),
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_sha256': sha256_file(checkpoint_path),
            'ann_file': str(ann_file),
            'ann_file_sha256': sha256_file(ann_file),
            'config_path': str((REPO_ROOT / args.config).resolve()),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
        },
    }
    atomic_json_dump(result_payload, output_path)
    print(json.dumps({
        'status': status,
        'protocol': protocol['id'],
        'sample_count': processed,
        'paper_metrics': paper_metrics,
        'mean_six_frames': result_payload['metrics']['mean_six_frames'],
        'output_json': str(output_path),
    }, indent=2), flush=True)
    if status != 'success':
        raise SystemExit(2)


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    protocol = resolve_protocol(args)
    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else (REPO_ROOT / args.output_root /
              f'{protocol["id"]}.json').resolve())
    if not args.force and successful_result_exists(
            output_path, protocol['id'], args.max_samples,
            args.anchor_mode):
        print(f'SKIP existing successful result: {output_path}')
        return
    evaluate(args, protocol, output_path)


if __name__ == '__main__':
    main()
