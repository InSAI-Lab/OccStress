import torch
import torch.nn.functional as F

from .ii_vector_quantizer import IntraInterVectorQuantizer


class HistoryFusionVectorQuantizerBase(IntraInterVectorQuantizer):
    """Shared utilities for history-fusion VQ variants."""

    def _run_intra_quantization(self,
                                curr_bev,
                                temp=None,
                                rescale_logits=False,
                                return_logits=False,
                                is_voxel=False):
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
        last_info = (None, None, None)

        for i in range(self.recover_stage):
            z_rest_i = F.interpolate(z_rest, shapes[i], mode='bilinear', align_corners=False)
            z_q_i, loss_i, last_info = self.forward_quantizer(
                z_rest_i, temp, rescale_logits, return_logits, is_voxel)
            z_q_i = F.interpolate(z_q_i, (width, height), mode='bilinear', align_corners=False)
            z_hat = z_hat + self.recover_scale_conv[i](z_q_i)
            z_rest = z_rest - z_q_i
            mean_vq_loss += loss_i

        mean_vq_loss *= 1 / self.recover_stage
        return curr_bev, z_rest, z_hat, mean_vq_loss, last_info
