import torch
import torch.nn as nn

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_fusion_base import HistoryFusionVectorQuantizerBase


@BACKBONES.register_module()
class IntraInterVectorQuantizerGateFusion(HistoryFusionVectorQuantizerBase):
    def __init__(self, gate_hidden=None, **kwargs):
        super().__init__(**kwargs)
        gate_hidden = gate_hidden or self.e_dim
        self.temporal_gate = nn.Sequential(
            nn.Conv2d(self.e_dim * 3, gate_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
        )

    def _compute_gate(self, curr_bev, hist_bev):
        gate_input = torch.cat([curr_bev, hist_bev, curr_bev - hist_bev], dim=1)
        return torch.sigmoid(self.temporal_gate(gate_input))

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev, z_rest, z_hat, mean_vq_loss, last_info = self._run_intra_quantization(
            curr_bev, temp, rescale_logits, return_logits, is_voxel)

        for i in range(self.recover_time):
            hist_i = sampled_bev[:, i]
            gate_i = self._compute_gate(curr_bev, hist_i)
            z_rest_i = z_rest + gate_i * hist_i
            z_q_i, _, _ = self.forward_quantizer(z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_hat = z_hat + self.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, last_info
