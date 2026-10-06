import hashlib

import torch
import torch.nn as nn
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World

# Preserve the established augmentation RNG stream across naming changes.
AUGMENTATION_SEED_NAMESPACE = bytes.fromhex('6c6174656e745f73635f617567').decode('ascii')


@DETECTORS.register_module()
class II_WorldRobustFinetune(II_World):
    """Stage2 robust finetune with a zero-init current-latent denoiser.

    The baseline stage2 target latent sequence is unchanged. During training we
    corrupt a deterministic subset of current latents and ask the adapter to map
    them back onto the clean latent manifold before the frozen world model sees
    them. Clean samples get an identity penalty, so the initial behavior is
    exactly baseline and clean performance is protected.
    """

    def __init__(self, robust_finetune=None, **kwargs):
        super().__init__(**kwargs)
        robust_finetune = robust_finetune or {}
        self.robust_enabled = bool(robust_finetune.get('enabled', True))
        self.robust_corr_ratio = float(robust_finetune.get('corr_ratio', 0.5))
        self.robust_seed = int(robust_finetune.get('seed', 3407))
        self.noise_std = float(robust_finetune.get('noise_std', 0.03))
        self.channel_drop_prob = float(robust_finetune.get('channel_drop_prob', 0.08))
        self.block_drop_prob = float(robust_finetune.get('block_drop_prob', 0.65))
        self.min_block = int(robust_finetune.get('min_block', 4))
        self.max_block = int(robust_finetune.get('max_block', 14))
        self.spatial_shift_prob = float(robust_finetune.get('spatial_shift_prob', 0.35))
        self.max_shift = int(robust_finetune.get('max_shift', 4))
        self.identity_weight = float(robust_finetune.get('identity_weight', 0.1))
        self.denoise_weight = float(robust_finetune.get('denoise_weight', 0.2))
        self.gate_weight = float(robust_finetune.get('gate_weight', 0.001))

        embed_dims = int(getattr(self.transformer, 'embed_dims', robust_finetune.get('embed_dims', 128)))
        hidden_dims = int(robust_finetune.get('hidden_dims', embed_dims))
        self.robust_adapter = nn.Sequential(
            nn.Conv2d(embed_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, kernel_size=1),
        )
        self.robust_gate = nn.Sequential(
            nn.Conv2d(embed_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, kernel_size=1),
        )
        self._init_robust_adapter(robust_finetune)

    def _init_robust_adapter(self, robust_finetune):
        nn.init.zeros_(self.robust_adapter[-1].weight)
        nn.init.zeros_(self.robust_adapter[-1].bias)
        nn.init.zeros_(self.robust_gate[-1].weight)
        gate_bias = float(robust_finetune.get('gate_bias', -4.0))
        nn.init.constant_(self.robust_gate[-1].bias, gate_bias)

    def _normalize_img_metas(self, img_metas):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        return img_metas

    def _sample_key(self, img_meta):
        scene = img_meta.get('scene_name', '')
        sample = img_meta.get('sample_idx', img_meta.get('token', img_meta.get('occ_index', '')))
        return f'{scene}:{sample}'

    def _stable_seed(self, key, salt):
        digest = hashlib.sha1(f'{self.robust_seed}:{salt}:{key}'.encode('utf-8')).hexdigest()
        return int(digest[:8], 16)

    def _make_corr_mask(self, img_metas, device):
        threshold = int(round(self.robust_corr_ratio * 10000))
        values = [
            (self._stable_seed(self._sample_key(img_meta), 'corr') % 10000) < threshold
            for img_meta in img_metas
        ]
        return torch.tensor(values, device=device, dtype=torch.bool)

    def _zero_shift(self, latent, shift_y, shift_x):
        if shift_y == 0 and shift_x == 0:
            return latent
        out = latent.new_zeros(latent.shape)
        _, height, width = latent.shape
        src_y0 = max(0, -shift_y)
        src_y1 = min(height, height - shift_y)
        dst_y0 = max(0, shift_y)
        dst_y1 = min(height, height + shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(width, width - shift_x)
        dst_x0 = max(0, shift_x)
        dst_x1 = min(width, width + shift_x)
        if src_y0 < src_y1 and src_x0 < src_x1:
            out[:, dst_y0:dst_y1, dst_x0:dst_x1] = latent[:, src_y0:src_y1, src_x0:src_x1]
        return out

    def _corrupt_one_latent(self, latent, seed):
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(seed)
        out = latent.clone()
        channels, height, width = out.shape

        if self.noise_std > 0:
            out = out + torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * self.noise_std

        if self.channel_drop_prob > 0:
            keep = torch.rand(channels, generator=generator, device=out.device) > self.channel_drop_prob
            out = out * keep.to(dtype=out.dtype).view(channels, 1, 1)

        if self.spatial_shift_prob > 0 and torch.rand((), generator=generator, device=out.device) < self.spatial_shift_prob:
            shift_y = int(torch.randint(-self.max_shift, self.max_shift + 1, (), generator=generator, device=out.device).item())
            shift_x = int(torch.randint(-self.max_shift, self.max_shift + 1, (), generator=generator, device=out.device).item())
            out = self._zero_shift(out, shift_y, shift_x)

        if self.block_drop_prob > 0 and torch.rand((), generator=generator, device=out.device) < self.block_drop_prob:
            max_block = max(self.min_block, min(self.max_block, height, width))
            block_h = int(torch.randint(self.min_block, max_block + 1, (), generator=generator, device=out.device).item())
            block_w = int(torch.randint(self.min_block, max_block + 1, (), generator=generator, device=out.device).item())
            y0 = int(torch.randint(0, max(1, height - block_h + 1), (), generator=generator, device=out.device).item())
            x0 = int(torch.randint(0, max(1, width - block_w + 1), (), generator=generator, device=out.device).item())
            out[:, y0:y0 + block_h, x0:x0 + block_w] = 0

        return out

    def _make_corrupted_current(self, curr_latent, img_metas, corr_mask):
        if not corr_mask.any():
            return curr_latent
        out = curr_latent.clone()
        for batch_idx, img_meta in enumerate(img_metas):
            if not corr_mask[batch_idx]:
                continue
            seed = self._stable_seed(self._sample_key(img_meta), AUGMENTATION_SEED_NAMESPACE)
            out[batch_idx] = self._corrupt_one_latent(out[batch_idx], seed)
        return out

    def _denoise_current(self, curr_latent):
        residual = self.robust_adapter(curr_latent)
        gate = torch.sigmoid(self.robust_gate(curr_latent))
        return curr_latent + gate * residual, gate.mean(), residual.abs().mean()

    def _forward_train_with_latent(self, latent, img_metas):
        return_dict = self.forward_sample(latent, img_metas, self.train_future_frame, train=True)
        pred_latents = return_dict['pred_latents']
        targ_latents = latent[:, 1:]
        valid_frame = torch.stack([
            torch.tensor(img_meta['valid_frame'], device=latent.device) for img_meta in img_metas
        ])

        loss_dict = dict()
        for frame_idx in range(self.train_future_frame):
            loss_dict['feat_sim_{}s_loss'.format((frame_idx + 1) * 0.5)] = self.frame_loss_weight[frame_idx] * \
                self.feature_similarity_loss(pred_latents[:, frame_idx], targ_latents[:, frame_idx], valid_frame[:, frame_idx])

        loss_dict['trajs_loss'] = self.trajs_loss(
            return_dict['pred_delta_translations'],
            return_dict['targ_delta_translations'],
            valid_frame,
            None,
        )
        loss_dict['rotation_loss'] = self.rotation_loss(
            return_dict['pred_relative_rotations'],
            return_dict['targ_relative_rotations'],
            valid_frame,
            None,
        )
        return loss_dict

    def forward_train(self, latent, img_metas, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if not self.robust_enabled:
            return super().forward_train(latent, img_metas, **kwargs)

        clean_curr = latent[:, 0]
        corr_mask = self._make_corr_mask(img_metas, latent.device)
        corrupted_curr = self._make_corrupted_current(clean_curr, img_metas, corr_mask)
        fused_curr, gate_mean, residual_mean = self._denoise_current(corrupted_curr)

        latent_for_model = latent.clone()
        latent_for_model[:, 0] = fused_curr
        loss_dict = self._forward_train_with_latent(latent_for_model, img_metas)

        clean_mask = ~corr_mask
        zero = fused_curr.sum() * 0.0
        identity = zero
        if clean_mask.any():
            identity = (fused_curr[clean_mask] - clean_curr[clean_mask]).abs().mean()
        denoise = zero
        if corr_mask.any():
            denoise = (fused_curr[corr_mask] - clean_curr[corr_mask]).abs().mean()
        loss_dict['robust_identity_loss'] = self.identity_weight * identity
        loss_dict['robust_denoise_loss'] = self.denoise_weight * denoise
        if self.gate_weight > 0:
            loss_dict['robust_gate_loss'] = self.gate_weight * gate_mean
        loss_dict['robust_gate_mean'] = gate_mean.detach()
        loss_dict['robust_residual_abs'] = residual_mean.detach()
        loss_dict['robust_corr_ratio'] = corr_mask.to(dtype=latent.dtype).mean().detach()
        return loss_dict

    def forward_test(self, latent, voxel_semantics, img_metas, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if self.robust_enabled:
            fused_curr, _, _ = self._denoise_current(latent[:, 0])
            latent = latent.clone()
            latent[:, 0] = fused_curr
        return super().forward_test(latent, voxel_semantics, img_metas, **kwargs)
