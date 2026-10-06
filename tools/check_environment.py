#!/usr/bin/env python3
"""Check an isolated model's import ownership; CUDA kernels are opt-in."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.distribution import require_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    profiles = json.loads((ROOT / 'environments/profiles.json').read_text())
    model = profiles['models'][args.model]
    try:
        require_source(ROOT, model['directory'])
    except ValueError as error:
        parser.error(str(error))
    recipe = profiles['recipes'][model['recipe']]
    directory = ROOT / model['directory']
    os.environ.update(model['env'])
    os.environ['OCCSTRESS_CODE_ROOT'] = str(ROOT)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(sys.prefix) / 'cache/matplotlib'))
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if model.get('dependency_directory'):
        sys.path.insert(0, str(ROOT / model['dependency_directory']))
    sys.path[:0] = [str(directory), str(ROOT / 'scripts/waymo'), str(ROOT / 'scripts/carla'), str(ROOT)]
    os.chdir(directory)
    if model.get('dependency_directory'):
        # Native exporters import mmdet3d before the CVT plugin registers overrides.
        importlib.import_module('mmdet3d.datasets')
        importlib.import_module('mmdet3d.models')
    package = importlib.import_module(model['package'])
    location = Path(package.__file__).resolve()
    if directory.resolve() not in location.parents:
        raise RuntimeError('Imported the wrong model package: ' + str(location))
    if model.get('dependency_directory'):
        dependency = importlib.import_module('mmdet3d')
        expected = (ROOT / model['dependency_directory']).resolve()
        if expected not in Path(dependency.__file__).resolve().parents:
            raise RuntimeError('mmdet3d is not from the declared dependency')
    check_registry = args.cuda or model.get('cpu_registry_check', True)
    if model['package'] == 'mmdet3d' and check_registry:
        importlib.import_module('mmdet3d.models')
        importlib.import_module('mmdet3d.datasets')
    if model['package'] == 'model' and check_registry:
        dataset_package = importlib.import_module('dataset')
        if directory.resolve() not in Path(dataset_package.__file__).resolve().parents:
            raise RuntimeError('Imported a dataset registry from another model')
    if args.model in ('come', 'dome'):
        importlib.import_module('xformers.ops')
    import torch
    import mmcv
    problems = []
    if torch.__version__.split('+')[0] != recipe['torch'].split('+')[0]:
        problems.append('torch differs from the recipe')
    if mmcv.__version__ != recipe['mmcv']:
        problems.append('mmcv differs from the recipe')
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('No visible CUDA device')
        from mmcv.ops import nms
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 2., 2.]], device='cuda')
        nms(boxes, torch.tensor([0.9, 0.8], device='cuda'), 0.5)
        if args.model in ('come', 'dome'):
            from xformers.ops import memory_efficient_attention
            query = torch.randn((1, 32, 4, 32), device='cuda', dtype=torch.float16)
            attention = memory_efficient_attention(query, query, query)
            if not torch.isfinite(attention).all():
                raise RuntimeError('Non-finite xformers CUDA output')
        torch.cuda.synchronize()
    print(json.dumps({'model': args.model, 'package': str(location),
                      'python': sys.version.split()[0], 'torch': torch.__version__,
                      'cuda_build': torch.version.cuda, 'mmcv': mmcv.__version__,
                      'version_warnings': problems, 'cuda_nms_test': args.cuda,
                      'cuda_attention_test': args.cuda and args.model in ('come', 'dome'),
                      'registry_import': 'checked' if check_registry else 'deferred: eager CUDA JIT in native ray metrics',
                      'checkpoint_and_full_model_gate': 'not_run'}, indent=2))
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
