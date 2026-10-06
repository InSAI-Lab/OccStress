import torch
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES

from .ii_vector_quantizer_fusion_base import HistoryFusionVectorQuantizerBase


@BACKBONES.register_module()
class IntraInterVectorQuantizerHistoryConsensusFusion(HistoryFusionVectorQuantizerBase):
    """History-consensus-first fusion for OccStress robustness.

    Design goals:
    - Stay close to the baseline on clean inputs.
    - Treat current/history symmetrically at the decision stage, but only after
      building a consensus from history frames so a corrupted current frame does
      not contaminate the consensus itself.
    - Be zero-shot friendly by avoiding newly initialized conv heads.
    """

    def __init__(self,
                 recency_decay=0.35,
                 current_trust_threshold=0.50,
                 current_trust_scale=6.0,
                 min_current_weight=0.15,
                 **kwargs):
        super().__init__(**kwargs)
        self.recency_decay = recency_decay
        self.current_trust_threshold = current_trust_threshold
        self.current_trust_scale = current_trust_scale
        self.min_current_weight = min_current_weight

        history_recency = -recency_decay * torch.arange(self.recover_time, dtype=torch.float32)
        self.register_buffer('history_recency_logits', history_recency)

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev, z_rest, z_hat, mean_vq_loss, last_info = self._run_intra_quantization(
            curr_bev, temp, rescale_logits, return_logits, is_voxel)

        # 1. Build a history-only consensus so a corrupted current frame does not
        #    shift the reference itself.
        hist_mean = sampled_bev.mean(dim=1, keepdim=True)
        hist_dev = torch.abs(sampled_bev - hist_mean).mean(dim=2, keepdim=True)
        recency_logits = self.history_recency_logits.view(1, self.recover_time, 1, 1, 1)
        history_weights = torch.softmax((-hist_dev) + recency_logits.to(sampled_bev.dtype).to(sampled_bev.device), dim=1)
        hist_consensus = (history_weights * sampled_bev).sum(dim=1)

        # 2. Let current compete against the history consensus based on local
        #    cosine agreement. Clean inputs should stay close to current; when
        #    current disagrees strongly with the consistent history, consensus
        #    gets more weight.
        curr_norm = F.normalize(curr_bev, dim=1, eps=1e-6)
        hist_norm = F.normalize(hist_consensus, dim=1, eps=1e-6)
        cosine_sim = (curr_norm * hist_norm).sum(dim=1, keepdim=True)
        current_weight = torch.sigmoid(
            (cosine_sim - self.current_trust_threshold) * self.current_trust_scale)
        current_weight = current_weight * (1.0 - self.min_current_weight) + self.min_current_weight

        fused = current_weight * curr_bev + (1.0 - current_weight) * hist_consensus
        consensus_delta = fused - curr_bev

        # 3. Keep the original temporal recovery skeleton. We inject the same
        #    consensus correction at each time-recovery stage so pretrained
        #    recover_time_conv weights remain usable in a zero-shot setting.
        for i in range(self.recover_time):
            z_rest_i = z_rest + consensus_delta
            z_q_i, _, _ = self.forward_quantizer(z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_hat = z_hat + self.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, last_info
