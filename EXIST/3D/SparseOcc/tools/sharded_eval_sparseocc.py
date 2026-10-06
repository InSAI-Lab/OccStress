#!/usr/bin/env python
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

import importlib


def prepare_cfg(config, ann_file_override=None):
    cfg = Config.fromfile(config)
    if ann_file_override:
        cfg.data.val.ann_file = ann_file_override
    cfg.data.workers_per_gpu = 0
    if hasattr(cfg.model, 'pretrained'):
        cfg.model.pretrained = None
    return cfg


class ShardDataset:
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)
        self.CLASSES = getattr(base, 'CLASSES', None)
        self.PALETTE = getattr(base, 'PALETTE', None)
        self.flag = np.zeros(len(self.indices), dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def setup_modules(force_offline_sweeps=False):
    importlib.import_module('models')
    importlib.import_module('loaders')
    if force_offline_sweeps:
        loading = importlib.import_module('loaders.pipelines.loading')
        loading.get_dist_info = lambda: (0, 2)


def run_infer(args):
    setup_modules(force_offline_sweeps=True)
    cfg = prepare_cfg(args.config, args.ann_file)
    set_random_seed(0, deterministic=True)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / 'preds'
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f'cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}', flush=True)
    dataset = build_dataset(cfg.data.val)
    total = len(dataset)
    indices = list(range(args.shard_id, total, args.num_shards))
    if args.limit and args.limit > 0:
        indices = indices[:args.limit]
    original_len = len(indices)
    if not args.overwrite:
        indices = [i for i in indices if not (pred_dir / f'{i:06d}.pkl').exists()]
    skipped = original_len - len(indices)
    print(f'shard_id={args.shard_id} num_shards={args.num_shards} total={total} shard_len={original_len} pending={len(indices)} skipped_existing={skipped}', flush=True)
    print(f'SKIP_EXISTING_PREDS shard={args.shard_id} skipped={skipped} pending={len(indices)}', flush=True)
    if not indices:
        print(f'SHARD_DONE shard={args.shard_id} done=0 skipped={skipped}', flush=True)
        return
    shard_dataset = ShardDataset(dataset, indices)

    loader = build_dataloader(
        shard_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model = build_model(cfg.model)
    model.cuda()
    model = MMDataParallel(model, [0])
    load_checkpoint(model, args.weights, map_location='cuda', strict=True)
    model.eval()

    done = 0
    t0 = time.time()
    with torch.no_grad():
        for local_i, data in enumerate(loader):
            global_idx = indices[local_i]
            pred_path = pred_dir / f'{global_idx:06d}.pkl'
            if pred_path.exists() and not args.overwrite:
                done += 1
                continue
            result = model(return_loss=False, rescale=True, **data)
            if isinstance(result, list):
                if len(result) != 1:
                    raise RuntimeError(f'expected batch result len=1, got {len(result)}')
                result = result[0]
            tmp = pred_path.with_suffix('.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, pred_path)
            done += 1
            if done == 1 or done % args.print_freq == 0 or done == len(indices):
                dt = time.time() - t0
                print(f'progress shard={args.shard_id} {done}/{len(indices)} elapsed={dt:.1f}s', flush=True)
    print(f'SHARD_DONE shard={args.shard_id} done={done}', flush=True)


def dense_from_sparse(result, dense_shape, free_id):
    from models.utils import sparse2dense
    sem_pred = torch.from_numpy(result['sem_pred'])
    occ_loc = torch.from_numpy(result['occ_loc'].astype(np.int64))
    dense, _ = sparse2dense(occ_loc, sem_pred, dense_shape=list(dense_shape), empty_value=free_id)
    return dense.squeeze(0).numpy().astype(np.int64)


def compute_miou(dataset, results, class_names):
    num_classes = len(class_names)
    free_id = num_classes - 1
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    pred_dir = None
    for i, (info, result) in enumerate(zip(dataset.data_infos, results)):
        occ_path = os.path.join(dataset.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
        gt = np.load(occ_path, allow_pickle=True)['semantics'].astype(np.int64)
        pred = dense_from_sparse(result, gt.shape, free_id)
        mask = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
        binc = np.bincount(num_classes * gt[mask] + pred[mask], minlength=num_classes ** 2)
        hist += binc.reshape(num_classes, num_classes)
        if (i + 1) % 500 == 0 or i + 1 == len(results):
            print(f'mIoU accumulation {i+1}/{len(results)}', flush=True)
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = np.divide(np.diag(hist), denom, out=np.full(num_classes, np.nan), where=denom > 0)
    return {
        'mIoU': float(np.nanmean(iou[:free_id])),
        'mIoU_percent': float(np.nanmean(iou[:free_id]) * 100.0),
        'per_class_iou': {name: (None if np.isnan(v) else float(v)) for name, v in zip(class_names, iou)},
    }


def run_evaluate(args):
    setup_modules()
    cfg = prepare_cfg(args.config, args.ann_file)
    dataset = build_dataset(cfg.data.val)
    total = len(dataset)
    pred_dir = Path(args.out_dir) / 'preds'
    missing = [i for i in range(total) if not (pred_dir / f'{i:06d}.pkl').exists()]
    if missing:
        print(f'MISSING_COUNT={len(missing)}')
        print('MISSING_FIRST=' + ','.join(map(str, missing[:50])))
        raise SystemExit(2)
    results = []
    for i in range(total):
        with open(pred_dir / f'{i:06d}.pkl', 'rb') as f:
            results.append(pickle.load(f))
    print(f'loaded_predictions={len(results)}', flush=True)
    metrics = dataset.evaluate(results, jsonfile_prefix=None)
    from configs.r50_nuimg_704x256_8f import occ_class_names
    miou = compute_miou(dataset, results, occ_class_names)
    metrics.update(miou)
    out = Path(args.out_dir) / 'metrics.json'
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print('METRICS_JSON', out)
    for k, v in metrics.items():
        if k != 'per_class_iou':
            print(f'{k}: {v}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--shard-id', type=int, required=True)
    p.add_argument('--num-shards', type=int, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--print-freq', type=int, default=20)
    p.add_argument('--ann-file', default=None)
    p.set_defaults(func=run_infer)

    p = sub.add_parser('evaluate')
    p.add_argument('--config', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--ann-file', default=None)
    p.set_defaults(func=run_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
