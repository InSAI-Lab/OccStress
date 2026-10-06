import os
import pickle
import importlib
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

base_ann = 'data/nuscenes/nuscenes_infos_val_sweep.pkl'
smoke_ann = 'data/nuscenes/nuscenes_infos_val_sweep_smoke1.pkl'
if not os.path.exists(smoke_ann):
    with open(base_ann, 'rb') as f:
        obj = pickle.load(f)
    smoke_infos = obj['infos'][:2]
    smoke_obj = dict(obj)
    smoke_obj['infos'] = smoke_infos
    with open(smoke_ann, 'wb') as f:
        pickle.dump(smoke_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('wrote smoke_ann', smoke_ann, [(x.get('scene_name'), x.get('token'), len(x.get('sweeps', []))) for x in smoke_infos])
else:
    print('using smoke_ann', smoke_ann)

cfg = Config.fromfile('configs/r50_nuimg_704x256_8f.py')
cfg.data.val.ann_file = smoke_ann
cfg.data.workers_per_gpu = 0
cfg.model.pretrained = None
importlib.import_module('models')
importlib.import_module('loaders')
set_random_seed(0, deterministic=True)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('building dataset')
dataset = build_dataset(cfg.data.val)
print('dataset_len', len(dataset))
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, shuffle=False, seed=0)
print('building model')
model = build_model(cfg.model)
model.cuda()
model = MMDataParallel(model, [0])
ckpt = 'checkpoints/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth'
print('loading', ckpt)
load_checkpoint(model, ckpt, map_location='cuda', strict=True)
print('running single_gpu_test')
results = single_gpu_test(model, loader)
print('num_results', len(results))
first = results[0]
if isinstance(first, dict):
    print('result_keys', sorted(first.keys()))
    for k, v in first.items():
        if hasattr(v, 'shape'):
            print('shape', k, tuple(v.shape))
else:
    print('result_type', type(first).__name__)
print('SMOKE_OK')
