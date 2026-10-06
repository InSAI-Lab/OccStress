"""Residual low-pass inter VQ ablation for the locked v0 tokenizer."""

import torch
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_v0_original import IntraInterVectorQuantizerV0Original


@BACKBONES.register_module()
class IntraInterVectorQuantizerV0RestLowpassInter(
        IntraInterVectorQuantizerV0Original):
    """Apply the temporal/inter VQ branch directly on low-pass residuals.

    Baseline inter quantizes ``z_rest + sampled_bev[:, i]`` after the intra
    branch. This diagnostic removes history-slot content from the inter branch
    and instead quantizes progressively low-passed copies of the current
    residual, i.e. ``z_rest_i = lowpass(z_rest, p_i)``.
    """

    def __init__(self, *args, rest_lowpass_passes=(0, 0, 1, 2), **kwargs):
        super().__init__(*args, **kwargs)
        if len(rest_lowpass_passes) != self.recover_time:
            raise ValueError(
                'rest_lowpass_passes must have recover_time entries, '
                f'got {len(rest_lowpass_passes)} vs {self.recover_time}')
        self.rest_lowpass_passes = tuple(int(x) for x in rest_lowpass_passes)
        if any(x < 0 for x in self.rest_lowpass_passes):
            raise ValueError('rest_lowpass_passes must be non-negative.')

    def _lowpass_once(self, bev):
        channels = bev.shape[1]
        kernel = bev.new_tensor([[1.0, 2.0, 1.0],
                                 [2.0, 4.0, 2.0],
                                 [1.0, 2.0, 1.0]]) / 16.0
        kernel = kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
        return F.conv2d(bev, kernel, padding=1, groups=channels)

    def _lowpass_repeated(self, bev, num_passes):
        out = bev
        for _ in range(num_passes):
            out = self._lowpass_once(out)
        return out

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False,
                return_logits=False, is_voxel=False):
        del sampled_bev
        curr_bev = self.quant_conv(curr_bev)
        _, _, width, height = curr_bev.shape
        shapes = []
        for i in range(self.recover_stage):
            width_i = width // (2 ** (self.recover_stage - i - 1))
            height_i = height // (2 ** (self.recover_stage - i - 1))
            shapes.append((width_i, height_i))

        mean_vq_loss = 0
        z_rest = curr_bev.clone()
        z_hat = torch.zeros_like(z_rest)
        for i in range(self.recover_stage):
            rest_i_shape = shapes[i]
            z_rest_i = F.interpolate(
                z_rest, rest_i_shape, mode='bilinear', align_corners=False)

            z_q_i, loss_i, info = self.forward_quantizer(
                z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            perplexity, min_encodings, min_encoding_indices = info

            z_q_i = F.interpolate(
                z_q_i, (width, height), mode='bilinear', align_corners=False)
            z_hat = z_hat + self.recover_scale_conv[i](z_q_i)
            z_rest = z_rest - z_q_i
            mean_vq_loss += loss_i

        mean_vq_loss *= 1 / self.recover_stage

        for i, num_passes in enumerate(self.rest_lowpass_passes):
            z_rest_i = self._lowpass_repeated(z_rest, num_passes)
            z_q_i, _, _ = self.forward_quantizer(
                z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_hat = z_hat + self.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, (
            perplexity, min_encodings, min_encoding_indices)
