import hashlib

import torch
import torch.nn as nn
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World


@DETECTORS.register_module()
class II_WorldVoteFusion(II_World):
    """Stage2 world model with an optional vote-latent current prior.

    The base target latent sequence is left unchanged. The vote latent is used
    only as an input-side residual prior for the current frame, so disabling the
    gate recovers the original no-history-token stage2 behavior.
    """

    def __init__(self, vote_fusion=None, **kwargs):
        super().__init__(**kwargs)
        vote_fusion = vote_fusion or {}
        self.vote_fusion_enabled = bool(vote_fusion.get('enabled', True))
        self.vote_corr_ratio = float(vote_fusion.get('corr_ratio', 0.5))
        self.vote_identity_weight = float(vote_fusion.get('identity_weight', 0.02))
        self.vote_gate_weight = float(vote_fusion.get('gate_weight', 0.001))
        self.vote_seed = int(vote_fusion.get('seed', 3407))
        self.vote_noise_std = float(vote_fusion.get('noise_std', 0.02))
        self.vote_channel_drop_prob = float(vote_fusion.get('channel_drop_prob', 0.08))
        self.vote_block_drop_prob = float(vote_fusion.get('block_drop_prob', 0.65))
        self.vote_min_block = int(vote_fusion.get('min_block', 4))
        self.vote_max_block = int(vote_fusion.get('max_block', 14))

        embed_dims = int(getattr(self.transformer, 'embed_dims', vote_fusion.get('embed_dims', 128)))
        hidden_dims = int(vote_fusion.get('hidden_dims', embed_dims))
        in_dims = embed_dims * 3 + 1
        self.vote_adapter = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, kernel_size=1),
        )
        self.vote_gate = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, kernel_size=1),
        )
        self._init_vote_fusion(vote_fusion)

    def _init_vote_fusion(self, vote_fusion):
        nn.init.zeros_(self.vote_adapter[-1].weight)
        nn.init.zeros_(self.vote_adapter[-1].bias)
        nn.init.zeros_(self.vote_gate[-1].weight)
        gate_bias = float(vote_fusion.get('gate_bias', -4.0))
        nn.init.constant_(self.vote_gate[-1].bias, gate_bias)

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
        digest = hashlib.sha1(f'{self.vote_seed}:{salt}:{key}'.encode('utf-8')).hexdigest()
        return int(digest[:8], 16)

    def _make_corr_mask(self, img_metas, device):
        values = []
        threshold = int(round(self.vote_corr_ratio * 10000))
        for img_meta in img_metas:
            values.append((self._stable_seed(self._sample_key(img_meta), 'corr') % 10000) < threshold)
        return torch.tensor(values, device=device, dtype=torch.bool)

    def _corrupt_one_latent(self, latent, seed):
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(seed)
        out = latent.clone()
        channels, height, width = out.shape

        if self.vote_noise_std > 0:
            out = out + torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * self.vote_noise_std

        if self.vote_channel_drop_prob > 0:
            channel_keep = torch.rand(channels, generator=generator, device=out.device) > self.vote_channel_drop_prob
            out = out * channel_keep.to(dtype=out.dtype).view(channels, 1, 1)

        if torch.rand((), generator=generator, device=out.device) < self.vote_block_drop_prob:
            max_block = max(self.vote_min_block, min(self.vote_max_block, height, width))
            block_h = int(torch.randint(self.vote_min_block, max_block + 1, (), generator=generator, device=out.device).item())
            block_w = int(torch.randint(self.vote_min_block, max_block + 1, (), generator=generator, device=out.device).item())
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
            seed = self._stable_seed(self._sample_key(img_meta), 'latent_aug')
            out[batch_idx] = self._corrupt_one_latent(out[batch_idx], seed)
        return out

    def _normalize_vote_inputs(self, vote_latent, vote_confidence, curr_latent):
        if vote_latent is None:
            return None, None
        if vote_latent.dim() == 5 and vote_latent.shape[1] == 1:
            vote_latent = vote_latent[:, 0]
        vote_latent = vote_latent.to(device=curr_latent.device, dtype=curr_latent.dtype)
        if vote_confidence is None:
            vote_confidence = torch.ones(
                curr_latent.shape[0], 1, curr_latent.shape[-2], curr_latent.shape[-1],
                device=curr_latent.device,
                dtype=curr_latent.dtype,
            )
        else:
            if vote_confidence.dim() == 3:
                vote_confidence = vote_confidence.unsqueeze(1)
            if vote_confidence.dim() == 5 and vote_confidence.shape[1] == 1:
                vote_confidence = vote_confidence[:, 0]
            vote_confidence = vote_confidence.to(device=curr_latent.device, dtype=curr_latent.dtype)
        return vote_latent, vote_confidence

    def _fuse_current(self, curr_latent, vote_latent, vote_confidence):
        vote_latent, vote_confidence = self._normalize_vote_inputs(vote_latent, vote_confidence, curr_latent)
        if vote_latent is None or not self.vote_fusion_enabled:
            zero = curr_latent.new_zeros(())
            return curr_latent, zero, zero
        fusion_input = torch.cat([curr_latent, vote_latent, vote_latent - curr_latent, vote_confidence], dim=1)
        residual = self.vote_adapter(fusion_input)
        gate = torch.sigmoid(self.vote_gate(fusion_input))
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

    def forward_train(self, latent, img_metas, vote_latent=None, vote_confidence=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if vote_latent is None or not self.vote_fusion_enabled:
            return super().forward_train(latent, img_metas, **kwargs)

        corr_mask = self._make_corr_mask(img_metas, latent.device)
        curr_input = self._make_corrupted_current(latent[:, 0], img_metas, corr_mask)
        fused_curr, gate_mean, residual_mean = self._fuse_current(curr_input, vote_latent, vote_confidence)

        latent_for_model = latent.clone()
        latent_for_model[:, 0] = fused_curr
        loss_dict = self._forward_train_with_latent(latent_for_model, img_metas)

        clean_mask = ~corr_mask
        identity = fused_curr.sum() * 0.0
        if clean_mask.any():
            identity = (fused_curr[clean_mask] - latent[:, 0][clean_mask]).abs().mean()
        loss_dict['vote_identity_loss'] = self.vote_identity_weight * identity
        if self.vote_gate_weight > 0:
            loss_dict['vote_gate_loss'] = self.vote_gate_weight * gate_mean
        loss_dict['vote_gate_mean'] = gate_mean.detach()
        loss_dict['vote_residual_abs'] = residual_mean.detach()
        loss_dict['vote_corr_ratio'] = corr_mask.to(dtype=latent.dtype).mean().detach()
        return loss_dict

    def forward_test(self, latent, voxel_semantics, img_metas, vote_latent=None, vote_confidence=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if vote_latent is not None and self.vote_fusion_enabled:
            fused_curr, _, _ = self._fuse_current(latent[:, 0], vote_latent, vote_confidence)
            latent = latent.clone()
            latent[:, 0] = fused_curr
        return super().forward_test(latent, voxel_semantics, img_metas, **kwargs)
