import math

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS

from .ii_tokenizer_v0_original import IISceneTokenizerV0Original


@DETECTORS.register_module()
class IISceneTokenizerHistorySlotBilinearNoAlignV0(IISceneTokenizerV0Original):
    """V0 tokenizer ablation: low-pass raw history slots without ego alignment.

    Baseline ``align_bev`` warps cached history slots into the current ego frame
    using grid_sample, which also introduces bilinear smoothing. This ablation
    removes the warp entirely and applies an explicit bilinear-like low-pass to
    each cached history slot before the inter-scene VQ path.
    """

    def __init__(self,
                 history_slot_lowpass_passes=(0, 0.25, 0.5, 0.75),
                 history_slot_detach_slots=True,
                 **kwargs):
        super().__init__(**kwargs)
        if len(history_slot_lowpass_passes) != self.frame_number:
            raise ValueError(
                f'history_slot_lowpass_passes must have {self.frame_number} entries, '
                f'got {len(history_slot_lowpass_passes)}')
        self.history_slot_lowpass_passes = tuple(float(x) for x in history_slot_lowpass_passes)
        if any(num_passes < 0 for num_passes in self.history_slot_lowpass_passes):
            raise ValueError('history_slot_lowpass_passes must be non-negative.')
        self.history_slot_detach_slots = bool(history_slot_detach_slots)

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
        # Keep the same packed history cache layout as the v0 baseline:
        # [B, frame_number * C, 1, H, W] in the align_bev internal layout.
        curr_bev_hw = curr_bev.permute(0, 1, 3, 2).unsqueeze(2)
        bs, channels, z_dim, height, width = curr_bev_hw.shape
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])

        if self.history_bev is None:
            self.history_bev = curr_bev_hw.repeat(1, self.frame_number, 1, 1, 1).clone()

        if start_of_sequence.sum() > 0:
            self.history_bev[start_of_sequence] = curr_bev_hw[start_of_sequence].repeat(
                1, self.frame_number, 1, 1, 1)

        history_bev = self.history_bev.detach() if self.history_slot_detach_slots else self.history_bev
        history_slots = history_bev.reshape(bs, self.frame_number, channels, z_dim, height, width)

        lowpass_slots = []
        for slot_idx, num_passes in enumerate(self.history_slot_lowpass_passes):
            slot = history_slots[:, slot_idx, :, 0]
            lowpass_slots.append(self._lowpass_repeated(slot, num_passes))

        sampled_bev = torch.stack(lowpass_slots, dim=1)
        sampled_bev = sampled_bev.permute(0, 1, 2, 4, 3).clone()

        # No align/warp: roll the raw packed history cache directly.
        self.history_bev = torch.cat(
            [curr_bev_hw, self.history_bev.detach()[:, :-self.vq_channel]], dim=1).detach().clone()

        return sampled_bev
