import argparse
import os

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import setup_multi_processes
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export latent tokens without collecting test outputs')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--save-root', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    split_cfg = cfg.data[args.split]
    split_cfg.test_mode = True
    split_cfg.ann_file = args.ann_file
    if cfg.data.get('test_dataloader', {}).get('samples_per_gpu', 1) > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)

    dataloader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=0 if distributed else cfg.data.get('workers_per_gpu', 4),
        dist=distributed,
        shuffle=False,
        persistent_workers=False,
    )
    if 'test_dataloader' in cfg.data:
        dataloader_cfg.update(cfg.data.test_dataloader)

    dataset = build_dataset(split_cfg)
    try:
        from mmdet3d.datasets import build_dataloader
        data_loader = build_dataloader(dataset, **dataloader_cfg)
    except Exception:
        from mmengine.dataset import build_dataloader as mmengine_build_dataloader
        data_loader = mmengine_build_dataloader(dict(dataset=dataset, **dataloader_cfg))

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.save_results = True
    cfg.model.save_root_override = args.save_root
    cfg.model.compare_history_at_test = False
    cfg.model.save_aligned_history_vis = False
    cfg.model.return_aligned_history_metrics = False
    cfg.model.eval_aligned_history = False
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = model.cuda()
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    model.eval()
    rank, _ = get_dist_info()
    if rank == 0:
        print(f'Exporting tokens to {args.save_root} from {args.ann_file}')

    for data in data_loader:
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

    if rank == 0:
        print('Token export complete.')


if __name__ == '__main__':
    main()
