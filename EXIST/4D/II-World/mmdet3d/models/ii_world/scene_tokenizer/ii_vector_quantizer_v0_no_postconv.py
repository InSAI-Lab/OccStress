"""Ablation quantizer that skips the final post-quant 1x1 projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_v0_original import IntraInterVectorQuantizerV0Original


@BACKBONES.register_module()
class IntraInterVectorQuantizerV0NoPostConv(IntraInterVectorQuantizerV0Original):
    """Keep the baseline VQ path but return the pre-postconv latent.

    The module still owns ``post_quant_conv`` through the parent class, so a
    baseline checkpoint can be loaded without changing the state-dict layout.
    Only the forward output is changed for this diagnostic ablation.
    """

    def __init__(self, *args, keep_post_quant_conv_for_ckpt=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_post_quant_conv_for_ckpt = keep_post_quant_conv_for_ckpt
        if not keep_post_quant_conv_for_ckpt:
            self.post_quant_conv = nn.Identity()

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

        for i in range(self.recover_time):
            z_rest_i = z_rest + sampled_bev[:, i]
            z_q_i, _, _ = self.forward_quantizer(
                z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_hat = z_hat + self.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        return self.post_quant_conv(z_hat), mean_vq_loss, (
            perplexity, min_encodings, min_encoding_indices)
