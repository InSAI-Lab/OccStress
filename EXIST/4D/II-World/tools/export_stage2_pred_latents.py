import argparse
import os
import sys
import warnings

import mmcv
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mmcv import Config, DictAction
    from mmcv.runner import get_dist_info, init_dist, load_checkpoint
except Exception:
    from mmengine.config import Config, DictAction
    from mmengine.dist import get_dist_info, init_dist
    from mmengine.runner import load_checkpoint

from mmcv.parallel import DataContainer

from mmdet.apis import set_random_seed
try:
    from mmdet.utils import setup_multi_processes, compat_cfg
except Exception:
    from mmdet3d.utils import setup_multi_processes, compat_cfg

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export frozen stage2 predicted latents.')
    parser.add_argument('config', help='stage2 config')
    parser.add_argument('checkpoint', help='stage2 checkpoint')
    parser.add_argument('--out-dir', required=True, help='output cache root')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--horizon-index', type=int, default=0, help='0 means 0.5s future')
    parser.add_argument('--key', default='pred_latent')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def unwrap_img_metas(img_metas):
    if isinstance(img_metas, DataContainer):
        img_metas = img_metas.data
    if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
        img_metas = img_metas[0]
    return img_metas


def get_dataset_loader_cfg(cfg, split, distributed):
    data_cfg = getattr(cfg.data, split)
    if split in ['val', 'test']:
        data_cfg.test_mode = True
    loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=distributed,
        shuffle=False,
        persistent_workers=False,
    )
    loader_cfg.update(cfg.data.get(f'{split}_dataloader', {}))
    if split == 'test':
        loader_cfg.update(cfg.data.get('test_dataloader', {}))
    return data_cfg, loader_cfg


def cache_path(out_dir, meta):
    scene_name = str(meta['scene_name'])
    sample_idx = meta['sample_idx']
    return os.path.join(out_dir, scene_name, f'{sample_idx}.npz')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    else:
        rank, world_size = 0, 1
        cfg.gpu_ids = [args.gpu_id]

    set_random_seed(args.seed, deterministic=args.deterministic)

    data_cfg, loader_cfg = get_dataset_loader_cfg(cfg, args.split, distributed)
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(dataset, **loader_cfg)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if 'test_mode' in cfg.model:
        cfg.model.test_mode = False
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    exported = 0
    skipped = 0
    predict_future_frame = args.horizon_index + 1

    if rank == 0:
        mmcv.mkdir_or_exist(args.out_dir)
        print(
            f'Exporting split={args.split} horizon_index={args.horizon_index} '
            f'world_size={world_size} out_dir={args.out_dir}')

    with torch.no_grad():
        for data in mmcv.track_iter_progress(data_loader):
            latent = data['latent'].cuda(non_blocking=True).float()
            img_metas = unwrap_img_metas(data['img_metas'])
            if len(img_metas) != latent.shape[0]:
                raise RuntimeError(f'img_metas batch mismatch: {len(img_metas)} vs {latent.shape[0]}')

            sample_dict = model.forward_sample(
                latent,
                img_metas,
                predict_future_frame=predict_future_frame,
                train=False,
            )
            pred_latents = sample_dict['pred_latents'][:, args.horizon_index].detach().cpu().numpy().astype(np.float32)

            for pred_latent, meta in zip(pred_latents, img_metas):
                path = cache_path(args.out_dir, meta)
                if args.skip_existing and os.path.exists(path):
                    skipped += 1
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(
                    path,
                    **{
                        args.key: pred_latent,
                        'horizon_index': np.array(args.horizon_index, dtype=np.int64),
                        'sample_idx': np.array(str(meta['sample_idx'])),
                        'scene_name': np.array(str(meta['scene_name'])),
                    },
                )
                exported += 1

    if distributed:
        torch.distributed.barrier()
    print(f'rank={rank} exported={exported} skipped={skipped}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
