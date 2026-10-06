"""No-history-slot ablation for the locked v0 scene tokenizer baseline."""

import os.path
import time

import mmcv
import numpy as np
from mmdet.models import DETECTORS

from .ii_tokenizer_v0_original import IISceneTokenizerV0Original


@DETECTORS.register_module()
class IISceneTokenizerV0NoHistorySlots(IISceneTokenizerV0Original):
    """Keep the v0 architecture, but remove information from temporal slots.

    The recover-time branch in the vector quantizer still receives four slots,
    so parameter count and loss definitions stay aligned with the baseline.
    Only the history-slot content is ablated.
    """

    def __init__(self, save_root_override=None, export_only=False, **kwargs):
        super().__init__(**kwargs)
        self.export_only = export_only
        if save_root_override is not None:
            self.save_root = save_root_override
            mmcv.mkdir_or_exist(self.save_root)

    def align_bev(self, curr_bev, img_metas):
        del img_metas
        batch_size, channels, width, height = curr_bev.shape
        return curr_bev.new_zeros(
            (batch_size, self.frame_number, channels, width, height),
        )

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
