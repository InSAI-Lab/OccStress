import torch
import torch.nn as nn

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_fusion_base import HistoryFusionVectorQuantizerBase


@BACKBONES.register_module()
class IntraInterVectorQuantizerSymmetricRecencyFusion(HistoryFusionVectorQuantizerBase):
    """Symmetric reliability-aware history fusion with recency bias.

    The current frame and all aligned history frames are treated as peer evidence
    sources. A shared scorer predicts a per-frame reliability score, while a
    learnable recency prior biases the fusion toward more recent frames. The
    weighted consensus is converted into a correction context and injected into
    the temporal recovery branch.
    """

    def __init__(self,
                 gate_hidden=None,
                 recency_decay=0.35,
                 use_learnable_recency=True,
                 **kwargs):
        super().__init__(**kwargs)
        gate_hidden = gate_hidden or self.e_dim
        self.use_learnable_recency = use_learnable_recency

        initial_recency = -recency_decay * torch.arange(self.recover_time + 1, dtype=torch.float32)
        if use_learnable_recency:
            self.recency_logits = nn.Parameter(initial_recency)
        else:
            self.register_buffer("recency_logits", initial_recency)

        self.score_head = nn.Sequential(
            nn.Conv2d(self.e_dim * 2, gate_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1),
        )

        self.context_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.e_dim, gate_hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(gate_hidden, self.e_dim, 1),
            )
            for _ in range(self.recover_time)
        ])

        # Start close to a recency-only baseline. The scorer then learns how to
        # override that prior when a non-current frame looks more reliable.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev, z_rest, z_hat, mean_vq_loss, last_info = self._run_intra_quantization(
            curr_bev, temp, rescale_logits, return_logits, is_voxel)

        frames = torch.cat([curr_bev.unsqueeze(1), sampled_bev], dim=1)  # [B, T+1, C, H, W]
        frame_mean = frames.mean(dim=1, keepdim=True)
        frame_dev = torch.abs(frames - frame_mean)

        b, t_all, c, h, w = frames.shape
        score_input = torch.cat([frames, frame_dev], dim=2).reshape(b * t_all, c * 2, h, w)
        score_logits = self.score_head(score_input).reshape(b, t_all, 1, h, w)
        recency_logits = self.recency_logits.view(1, t_all, 1, 1, 1).to(frames.dtype).to(frames.device)
        weights = torch.softmax(score_logits + recency_logits, dim=1)

        fused_frames = (weights * frames).sum(dim=1)
        consensus_delta = fused_frames - curr_bev

        for i in range(self.recover_time):
            context_i = self.context_proj[i](consensus_delta)
            z_rest_i = z_rest + context_i
            z_q_i, _, _ = self.forward_quantizer(z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_hat = z_hat + self.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        z_q = self.post_quant_conv(z_hat)
        fusion_info = dict(
            vq_info=last_info,
            fusion_weights=weights,
            score_logits=score_logits,
        )
        return z_q, mean_vq_loss, fusion_info
