import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS

from .ii_tokenizer import IISceneTokenizer


@DETECTORS.register_module()
class IISceneTokenizerCurrLatentBilinearSlots(IISceneTokenizer):
    """Baseline ablation: sampled_bev comes from current latent, not history.

    Keep the baseline align_bev grid construction, bilinear grid_sample, VQ and
    decoder unchanged. The only intended change is the tensor used as the
    bilinear-sampling source:

    baseline:
      cached history latent -> bilinear warp -> sampled_bev

    this ablation:
      repeated current latent -> same bilinear warp -> sampled_bev
    """

    def align_bev(self,
                  curr_bev,
                  img_metas,
                  curr_occ=None,
                  cache_bev=None,
                  cache_occ=None,
                  cache_reliability=None):
        curr_bev = curr_bev.permute(0, 1, 3, 2).unsqueeze(2)
        if cache_bev is None:
            cache_bev = curr_bev
        else:
            cache_bev = cache_bev.permute(0, 1, 3, 2).unsqueeze(2)

        bs, c, z, h, w = curr_bev.shape
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        start_mask = torch.as_tensor(start_of_sequence, device=curr_bev.device, dtype=torch.bool)
        curr_to_prev_ego_rt = torch.stack(
            [torch.as_tensor(img_meta['curr_to_prev_ego_rt'], device=curr_bev.device) for img_meta in img_metas])
        bev_aug = torch.stack([img_meta['bda_mat'].to(curr_bev.device) for img_meta in img_metas])

        # Preserve the baseline cache/update contract so downstream debug and
        # eval utilities still see a standard temporal cache object.
        if self.history_bev is None:
            self.history_bev = cache_bev.repeat(1, self.frame_number, 1, 1, 1).clone()
            self.bev_aug = bev_aug.clone()

        if start_mask.any():
            self.history_bev[start_mask] = cache_bev[start_mask].repeat(1, self.frame_number, 1, 1, 1)
            self.bev_aug[start_mask] = bev_aug[start_mask]

        self.history_bev = self.history_bev.detach()

        # The ablation itself: use the current latent as the source tensor for
        # all slots instead of the cached history latent.
        tmp_bev = cache_bev.repeat(1, self.frame_number, 1, 1, 1).detach().clone()

        history_occ = None
        raw_history_bev = None
        sampled_reliability = None
        if self.eval_aligned_history and cache_occ is not None:
            history_occ = cache_occ.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).detach().clone()
            raw_history_bev = tmp_bev.reshape(
                bs, self.frame_number, c, z, h, w).permute(0, 1, 2, 3, 5, 4).squeeze(3).clone()

        tmp_reliability = None
        if cache_reliability is not None:
            batch_size = start_mask.shape[0]
            while cache_reliability.dim() > 3 and cache_reliability.shape[0] == 1:
                cache_reliability = cache_reliability.squeeze(0)
            if (cache_reliability.dim() == 4 and cache_reliability.shape[0] != batch_size
                    and cache_reliability.shape[1] == batch_size):
                cache_reliability = cache_reliability.transpose(0, 1)
            if cache_reliability.dim() == 4 and cache_reliability.shape[1] == 1:
                cache_reliability = cache_reliability.squeeze(1)
            if cache_reliability.dim() != 3 or cache_reliability.shape[0] != batch_size:
                cache_reliability = cache_reliability.reshape(
                    batch_size, cache_reliability.shape[-2], cache_reliability.shape[-1])
            if cache_reliability.shape[-2:] != (h, w):
                cache_reliability = F.interpolate(
                    cache_reliability.unsqueeze(1).to(curr_bev.dtype),
                    size=(h, w),
                    mode='nearest',
                ).squeeze(1)
            tmp_reliability = cache_reliability.unsqueeze(1).repeat(1, self.frame_number, 1, 1).unsqueeze(2)

        grid = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid, self.dx, self.bx)

        rt_flow = (torch.inverse(feat2bev) @ self.bev_aug @ curr_to_prev_ego_rt @ torch.inverse(bev_aug) @ feat2bev)
        grid = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid

        normalize_factor = torch.tensor(
            [w - 1.0, h - 1.0, 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        grid[..., 2] = 0.0

        sampled_bev = F.grid_sample(
            tmp_bev,
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear')
        if tmp_reliability is not None:
            sampled_reliability = F.grid_sample(
                tmp_reliability.to(curr_bev.dtype),
                grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
                align_corners=True,
                mode='nearest',
            ).squeeze(2)

        bev_cat = torch.cat([cache_bev, sampled_bev], dim=1)
        self.history_bev = bev_cat[:, :-self.vq_channel, ...].detach().clone()
        if history_occ is not None:
            self.history_occ = torch.cat([cache_occ.unsqueeze(1), history_occ[:, :-1]], dim=1).detach().clone()
        if sampled_reliability is not None:
            self.history_reliability = torch.cat(
                [cache_reliability.unsqueeze(1), sampled_reliability[:, :-1]], dim=1).detach().clone()

        sampled_bev = sampled_bev.reshape(
            bs, self.frame_number, c, z, h, w).permute(0, 1, 2, 3, 5, 4).squeeze(3)

        if sampled_reliability is None:
            return sampled_bev.clone(), history_occ, raw_history_bev, None
        return sampled_bev.clone(), history_occ, raw_history_bev, sampled_reliability.clone()
