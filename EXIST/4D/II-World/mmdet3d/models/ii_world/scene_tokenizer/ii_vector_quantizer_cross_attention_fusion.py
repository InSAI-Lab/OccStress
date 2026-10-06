import torch
import torch.nn as nn

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_fusion_base import HistoryFusionVectorQuantizerBase


@BACKBONES.register_module()
class IntraInterVectorQuantizerCrossAttentionFusion(HistoryFusionVectorQuantizerBase):
    def __init__(self, num_heads=4, attn_dropout=0.0, refine_hidden=None, **kwargs):
        super().__init__(**kwargs)
        refine_hidden = refine_hidden or self.e_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.e_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ctx_proj = nn.Sequential(
            nn.Conv2d(self.e_dim, refine_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_hidden, self.e_dim, 1),
        )
        self.ctx_gate = nn.Sequential(
            nn.Conv2d(self.e_dim * 3, refine_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_hidden, 1, 1),
        )
        self.ctx_out = nn.Conv2d(self.e_dim, self.e_dim, 1)

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev, z_rest, z_hat, mean_vq_loss, last_info = self._run_intra_quantization(
            curr_bev, temp, rescale_logits, return_logits, is_voxel)
        batch_size, channels, height, width = z_hat.shape

        curr_tokens = z_hat.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hist_tokens = sampled_bev.permute(0, 1, 3, 4, 2).reshape(batch_size, sampled_bev.shape[1] * height * width, channels)
        hist_ctx, _ = self.cross_attn(curr_tokens, hist_tokens, hist_tokens, need_weights=False)
        hist_ctx = hist_ctx.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        ctx_proj = self.ctx_proj(hist_ctx)
        ctx_gate_input = torch.cat([z_hat, ctx_proj, z_hat - ctx_proj], dim=1)
        ctx_gate = torch.sigmoid(self.ctx_gate(ctx_gate_input))
        refine_q, _, _ = self.forward_quantizer(
            z_rest + ctx_proj, temp, rescale_logits, return_logits, is_voxel)
        z_hat = z_hat + ctx_gate * self.ctx_out(refine_q)

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, last_info
