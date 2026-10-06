import torch
import torch.nn as nn

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_fusion_base import HistoryFusionVectorQuantizerBase


@BACKBONES.register_module()
class IntraInterVectorQuantizerRefineFusion(HistoryFusionVectorQuantizerBase):
    def __init__(self, refine_hidden=None, **kwargs):
        super().__init__(**kwargs)
        refine_hidden = refine_hidden or self.e_dim
        self.refine_proj = nn.Sequential(
            nn.Conv2d(self.e_dim * 3, refine_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_hidden, self.e_dim, 1),
        )
        self.refine_gate = nn.Sequential(
            nn.Conv2d(self.e_dim * 3, refine_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_hidden, 1, 1),
        )
        self.refine_out = nn.Conv2d(self.e_dim, self.e_dim, 1)

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev, z_rest, z_hat, mean_vq_loss, last_info = self._run_intra_quantization(
            curr_bev, temp, rescale_logits, return_logits, is_voxel)

        hist_mean = sampled_bev.mean(dim=1)
        refine_input = torch.cat([z_hat, hist_mean, z_hat - hist_mean], dim=1)
        refine_feat = self.refine_proj(refine_input)
        refine_gate = torch.sigmoid(self.refine_gate(refine_input))
        refine_q, _, _ = self.forward_quantizer(
            z_rest + refine_feat, temp, rescale_logits, return_logits, is_voxel)
        z_hat = z_hat + refine_gate * self.refine_out(refine_q)

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, last_info
