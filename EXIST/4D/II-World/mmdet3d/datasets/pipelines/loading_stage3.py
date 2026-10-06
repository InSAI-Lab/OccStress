import os

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadStage3PredLatent(object):
    """Load cached stage2 predicted latent for decoder-only stage3 training.

    Expected cache layout:
        data_path/<scene_name>/<sample_idx>.npz

    The cache is intentionally keyed by the current sample. For the first
    pred-only experiment each file contains the 0.5s predicted latent.
    """

    def __init__(self,
                 data_path,
                 key='pred_latent',
                 to_float32=True,
                 dataset_type='occ3d'):
        self.data_path = data_path
        self.key = key
        self.to_float32 = to_float32
        self.dataset_type = dataset_type

    def _cache_path(self, results):
        scene_name = str(results['scene_name'])
        sample_idx = results['sample_idx']
        if self.dataset_type == 'waymo':
            scene_name = scene_name.zfill(3)
            sample_idx = os.path.splitext(os.path.basename(results['occ_path']))[0]
        return os.path.join(self.data_path, scene_name, f'{sample_idx}.npz')

    def __call__(self, results):
        path = self._cache_path(results)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Stage3 pred latent cache not found: {path}')

        data = np.load(path)
        if self.key not in data:
            raise KeyError(f'Key {self.key!r} not found in {path}; keys={list(data.keys())}')

        pred_latent = data[self.key]
        if self.to_float32:
            pred_latent = pred_latent.astype(np.float32, copy=False)
        results['pred_latent'] = pred_latent
        results['pred_latent_cache_path'] = path
        return results
