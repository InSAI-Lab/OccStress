# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import torch

# -------------------------
# OpenMMLab 2.0 compatible imports
# -------------------------
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import load_checkpoint
from mmengine.fileio import dump as mm_dump

# mmcv 2.x still provides these
from mmcv.cnn import fuse_conv_bn

# DataParallel / DDP wrappers: prefer MMEngine, fallback to torch
try:
    from mmengine.model.wrappers import MMDataParallel, MMDistributedDataParallel
except Exception:
    MMDataParallel = torch.nn.DataParallel
    MMDistributedDataParallel = torch.nn.parallel.DistributedDataParallel

# fp16 wrapper removed in OpenMMLab 2.0; keep as no-op for compatibility
def wrap_fp16_model(model):
    return model

# -------------------------
# mmdet / mmdet3d imports (OpenMMLab 2.0)
# -------------------------
import mmdet
from mmdet.apis import set_random_seed

# replace_ImageToTensor path changed across versions; keep robust
try:
    from mmdet.datasets import replace_ImageToTensor
except Exception:
    # mmdet3 sometimes moves it
    from mmdet.datasets.transforms import replace_ImageToTensor  # type: ignore

from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# setup_multi_processes / compat_cfg live in mmdet for v3, fallback to mmdet3d
try:
    from mmdet.utils import setup_multi_processes, compat_cfg
except Exception:
    from mmdet3d.utils import setup_multi_processes, compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--scene_checkpoints', default='ckpts/ii_scene_tokenizer_4f.pth')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Whether to fuse conv and bn')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='(Deprecated) ids of gpus to use')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use (non-distributed)')
    parser.add_argument('--format-only', action='store_true', help='Format results without evaluation')
    parser.add_argument('--eval', type=str, default='bbox', nargs='+', help='evaluation metrics')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where results will be saved')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results')
    parser.add_argument('--no-aavt', action='store_true', help='Do not align after view transformer.')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic cudnn')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the config')
    parser.add_argument('--options', nargs='+', action=DictAction, help='(deprecated) use --eval-options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot be both specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def _get_test_dataset_and_loader_cfg(cfg, distributed: bool):
    """
    Support BOTH:
      - old style: cfg.data.test + cfg.data.test_dataloader
      - new style: cfg.test_dataloader (MMEngine)
    """
    # new style
    if hasattr(cfg, 'test_dataloader'):
        test_loader_cfg = cfg.test_dataloader.to_dict() if hasattr(cfg.test_dataloader, 'to_dict') else dict(cfg.test_dataloader)
        dataset_cfg = test_loader_cfg.get('dataset', None)
        if dataset_cfg is None:
            raise KeyError('cfg.test_dataloader.dataset is required (new style config).')
        test_loader_cfg.setdefault('sampler', {})  # keep mmengine happy if needed
        test_loader_cfg['dataset'] = dataset_cfg
        test_loader_cfg['persistent_workers'] = test_loader_cfg.get('persistent_workers', False)
        # ensure dist flag compatible with legacy builder usage
        test_loader_cfg['dist'] = distributed
        return dataset_cfg, test_loader_cfg

    # old style
    if not hasattr(cfg, 'data') or not hasattr(cfg.data, 'test'):
        raise KeyError('Config must have either cfg.test_dataloader or cfg.data.test.')

    cfg.data.test.test_mode = True
    # replace pipeline when samples_per_gpu > 1
    td = cfg.data.get('test_dataloader', {})
    if td.get('samples_per_gpu', 1) > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    test_loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=distributed,
        shuffle=False
    )
    test_loader_cfg.update(td)
    return cfg.data.test, test_loader_cfg


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        'Please specify at least one operation with --out/--eval/--format-only/--show/--show-dir'

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl/pickle file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # gpu ids
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # distributed
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # dataset / dataloader cfg
    dataset_cfg, test_loader_cfg = _get_test_dataset_and_loader_cfg(cfg, distributed)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build dataset & dataloader (use mmdet3d legacy builders)
    dataset = build_dataset(dataset_cfg)

    # build_dataloader import moved/removed across versions; use mmengine if available
    try:
        from mmdet3d.datasets import build_dataloader  # legacy
        data_loader = build_dataloader(dataset, **test_loader_cfg)
    except Exception:
        # MMEngine dataloader path
        from mmengine.dataset import build_dataloader as mmengine_build_dataloader
        # mmengine expects 'dataset' object in cfg-like dict
        data_loader = mmengine_build_dataloader(dict(dataset=dataset, **test_loader_cfg))

    # build the model
    if hasattr(cfg, 'model'):
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        if isinstance(cfg.model, dict) and 'test_mode' in cfg.model:
            cfg.model.test_mode = True
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    else:
        raise KeyError('cfg.model is required.')

    # fp16 (OpenMMLab2 uses AMP; keep no-op for compatibility)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # class / palette info
    meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    if 'CLASSES' in meta:
        model.CLASSES = meta['CLASSES']
    else:
        model.CLASSES = getattr(dataset, 'CLASSES', None)
    if 'PALETTE' in meta:
        model.PALETTE = meta['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    # run test
    if not distributed:
        model = model.cuda()
        model = MMDataParallel(model, device_ids=cfg.gpu_ids) if MMDataParallel is not torch.nn.DataParallel else model
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, batch_size=1)
    else:
        model = model.cuda()
        model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, batch_size=1)

    # rank0 postprocess
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mm_dump(outputs, args.out)

        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy() if hasattr(cfg, 'evaluation') else {}
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
