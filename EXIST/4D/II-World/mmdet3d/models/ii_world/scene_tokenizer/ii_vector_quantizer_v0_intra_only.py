"""Ablation quantizer that removes the temporal inter branch only."""

import torch
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_v0_original import IntraInterVectorQuantizerV0Original


@BACKBONES.register_module()
class IntraOnlyVectorQuantizerV0(IntraInterVectorQuantizerV0Original):
    """Run baseline intra residual VQ, then keep the baseline post-quant conv.

    This isolates the contribution of the temporal inter branch. The module
    still owns the parent ``recover_time_conv`` parameters for checkpoint
    compatibility, but they are intentionally unused in ``forward``.
    """

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False,
                return_logits=False, is_voxel=False):
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
        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, (
            perplexity, min_encodings, min_encoding_indices)
