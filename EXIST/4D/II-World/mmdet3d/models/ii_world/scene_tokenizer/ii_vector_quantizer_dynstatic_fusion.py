import torch
import torch.nn as nn

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_fusion_base import HistoryFusionVectorQuantizerBase


@BACKBONES.register_module()
class IntraInterVectorQuantizerDynStaticGateFusion(HistoryFusionVectorQuantizerBase):
    def __init__(self, gate_hidden=None, dynamic_gate_scale=0.25, **kwargs):
        super().__init__(**kwargs)
        gate_hidden = gate_hidden or self.e_dim
        self.dynamic_gate_scale = dynamic_gate_scale
        self.motion_head = nn.Sequential(
            nn.Conv2d(self.e_dim, gate_hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.static_gate = nn.Sequential(
            nn.Conv2d(self.e_dim * 3, gate_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.dynamic_gate = nn.Sequential(
            nn.Conv2d(self.e_dim * 3, gate_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
        )

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev, z_rest, z_hat, mean_vq_loss, last_info = self._run_intra_quantization(
            curr_bev, temp, rescale_logits, return_logits, is_voxel)

        for i in range(self.recover_time):
            hist_i = sampled_bev[:, i]
            diff_i = torch.abs(curr_bev - hist_i)
            gate_input = torch.cat([curr_bev, hist_i, diff_i], dim=1)
            motion_prob = torch.sigmoid(self.motion_head(diff_i))
            static_gate = torch.sigmoid(self.static_gate(gate_input))
            dynamic_gate = self.dynamic_gate_scale * torch.sigmoid(self.dynamic_gate(gate_input))
            fusion_gate = (1.0 - motion_prob) * static_gate + motion_prob * dynamic_gate
            z_rest_i = z_rest + fusion_gate * hist_i
            z_q_i, _, _ = self.forward_quantizer(z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_hat = z_hat + self.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, last_info
