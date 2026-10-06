# OccStress adapter/portability modifications; see docs/source-imports.json.
# Copyright (c) OpenMMLab. All rights reserved.
import os
import time

import mmcv
import numpy as np

from mmdet.models import DETECTORS
from mmcv.parallel import DataContainer

from .ii_tokenizer import IISceneTokenizer


@DETECTORS.register_module()
class OccStressIISceneTokenizer(IISceneTokenizer):
    def __init__(self, save_root=None, save_only_anchor=True, **kwargs):
        super().__init__(**kwargs)
        if save_root is not None:
            self.save_root = save_root
            mmcv.mkdir_or_exist(self.save_root)
        self.save_only_anchor = save_only_anchor

    def forward_test(self, voxel_semantics, img_metas, **kwargs):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data

        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        voxel_semantics = self.normalize_batched_tensor(voxel_semantics, len(img_metas))
        if voxel_semantics.dim() == 4:
            voxel_semantics = voxel_semantics.unsqueeze(1)

        if (
            self.vote_mode_at_test != 'none'
            or self.compare_history_at_test
            or self.eval_aligned_history
            or self.disable_history_at_test
        ):
            outputs = super().forward_test(voxel_semantics, img_metas, **kwargs)
            outputs[0]['targ_semantics'] = voxel_semantics.cpu().numpy().astype(np.uint8)
            return outputs

        bs, t, w, h, d = voxel_semantics.shape
        start_time = time.time()
        curr_bev, shapes = self.forward_encoder(voxel_semantics)
        aligned = self.align_bev(curr_bev, img_metas)
        sampled_bev = aligned[0] if isinstance(aligned, tuple) else aligned
        z_sampled, loss, info = self.vq(curr_bev, sampled_bev, is_voxel=False)
        end_time = time.time()
        logits = self.forward_decoder(z_sampled, shapes, (bs, 1, w, h, d))

        output_dict = dict()
        pred = logits.softmax(-1).argmax(-1).cpu().numpy()
        if self.save_results:
            img_meta = img_metas[0]
            should_save = True
            if self.save_only_anchor:
                should_save = bool(img_meta.get('occstress_save_token', True))
            if should_save:
                save_token = z_sampled[0].cpu().numpy()
                sample_name = img_meta.get('protocol_sample_id', img_meta['sample_idx'])
                scene_name = str(img_meta['scene_name'])
                mmcv.mkdir_or_exist(os.path.join(self.save_root, 'token_4f', scene_name))
                np.savez(os.path.join(self.save_root, 'token_4f', scene_name, f'{sample_name}.npz'), token=save_token)

        output_dict['semantics'] = pred.astype(np.uint8)
        output_dict['targ_semantics'] = voxel_semantics.cpu().numpy().astype(np.uint8)
        output_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        output_dict['time'] = end_time - start_time
        return [output_dict]
