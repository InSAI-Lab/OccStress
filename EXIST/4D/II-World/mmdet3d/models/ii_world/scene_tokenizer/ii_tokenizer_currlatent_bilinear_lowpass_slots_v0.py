import math
import os.path
import time

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS

from .ii_tokenizer_v0_original import IISceneTokenizerV0Original


@DETECTORS.register_module()
class IISceneTokenizerCurrLatentBilinearLowpassSlotsV0(IISceneTokenizerV0Original):
    """V0 baseline ablation using current-latent low-pass maps as history slots.

    The clean v0 baseline normally feeds four detached history BEV slots into the
    inter-scene VQ path. This ablation removes real history and instead feeds
    progressively smoothed copies of the current BEV latent. It isolates the
    effect of bilinear-interpolation-like low-pass statistics from true temporal
    content.
    """

    def __init__(self,
                 currlatent_lowpass_passes=(1, 2, 3, 4),
                 currlatent_detach_slots=True,
                 save_root_override=None,
                 export_only=False,
                 **kwargs):
        super().__init__(**kwargs)
        if len(currlatent_lowpass_passes) != self.frame_number:
            raise ValueError(
                f'currlatent_lowpass_passes must have {self.frame_number} entries, '
                f'got {len(currlatent_lowpass_passes)}')
        self.currlatent_lowpass_passes = tuple(float(x) for x in currlatent_lowpass_passes)
        if any(num_passes < 0 for num_passes in self.currlatent_lowpass_passes):
            raise ValueError('currlatent_lowpass_passes must be non-negative.')
        self.currlatent_detach_slots = currlatent_detach_slots
        self.export_only = export_only
        if save_root_override is not None:
            self.save_root = save_root_override
            mmcv.mkdir_or_exist(self.save_root)

    def _bilinear_lowpass_once(self, bev):
        channels = bev.shape[1]
        kernel = bev.new_tensor([[1.0, 2.0, 1.0],
                                 [2.0, 4.0, 2.0],
                                 [1.0, 2.0, 1.0]]) / 16.0
        kernel = kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
        return F.conv2d(bev, kernel, padding=1, groups=channels)

    def _lowpass_repeated(self, bev, num_passes):
        out = bev
        whole_passes = int(math.floor(float(num_passes)))
        frac_pass = float(num_passes) - whole_passes
        for _ in range(whole_passes):
            out = self._bilinear_lowpass_once(out)
        if frac_pass > 0:
            out_next = self._bilinear_lowpass_once(out)
            out = out * (1.0 - frac_pass) + out_next * frac_pass
        return out

    def align_bev(self, curr_bev, img_metas):
        del img_metas
        slot_source = curr_bev.detach() if self.currlatent_detach_slots else curr_bev
        slots = [
            self._lowpass_repeated(slot_source, num_passes)
            for num_passes in self.currlatent_lowpass_passes
        ]
        return torch.stack(slots, dim=1).clone()

    def forward_test(self, voxel_semantics, img_metas, **kwargs):
        if not self.export_only:
            return super().forward_test(voxel_semantics, img_metas, **kwargs)

        start_time = time.time()
        curr_bev, _ = self.forward_encoder(voxel_semantics)
        sampled_bev = self.align_bev(curr_bev, img_metas)
        z_sampled, _, _ = self.vq(curr_bev, sampled_bev, is_voxel=False)
        end_time = time.time()

        if self.save_results:
            save_tokens = z_sampled.detach().cpu().numpy()
            for save_token, img_meta in zip(save_tokens, img_metas):
                if self.results_type != 'waymo':
                    scene_name = str(img_meta['scene_name'])
                    sample_idx = str(img_meta['sample_idx'])
                    out_dir = os.path.join(self.save_root, 'token_4f', scene_name)
                    mmcv.mkdir_or_exist(out_dir)
                    np.savez(os.path.join(out_dir, f'{sample_idx}.npz'), token=save_token)
                else:
                    scene_name = str(img_meta['scene_name']).zfill(3)
                    occ_path_idx = img_meta['occ_path'].split('/')[-1].split('.')[0]
                    out_dir = os.path.join(self.save_root, 'token_4f', scene_name)
                    mmcv.mkdir_or_exist(out_dir)
                    np.savez(os.path.join(out_dir, f'{occ_path_idx}.npz'), token=save_token)

        return [dict(
            index=[img_meta['index'] for img_meta in img_metas],
            time=end_time - start_time,
        )]
