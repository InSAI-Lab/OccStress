import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World, tic


@DETECTORS.register_module()
class II_WorldHistorySelectiveGate(II_World):
    """Frozen-stage2 reliability-aware current/history correction.

    The baseline stage2 path repeats the current latent as all history memory.
    This module keeps that behavior exactly at initialization by setting both
    learnable fusion strengths to zero, then trains only the small gate/correction
    modules while the world model backbone is frozen.
    """

    def __init__(self, history_gate=None, **kwargs):
        super().__init__(**kwargs)
        history_gate = history_gate or {}
        self.history_gate_enabled = bool(history_gate.get('enabled', True))
        self.history_frame_number_gate = int(history_gate.get('history_frame_number', 4))
        self.use_flow_summary = bool(history_gate.get('use_flow_summary', False))
        self.flow_summary_dim = int(history_gate.get('flow_summary_dim', 5 if self.use_flow_summary else 0))

        self.current_corr_ratio = float(history_gate.get('current_corr_ratio', 0.5))
        self.hist_corr_ratio = float(history_gate.get('history_corr_ratio', history_gate.get('corr_ratio', 0.5)))
        self.hist_slot_corr_prob = float(history_gate.get('slot_corr_prob', 0.5))
        self.hist_seed = int(history_gate.get('seed', 3407))
        self.corruption_mode = str(history_gate.get('corruption_mode', 'synthetic'))
        self.force_history_strength_zero = bool(history_gate.get('force_history_strength_zero', False))
        self.noise_std = float(history_gate.get('noise_std', 0.03))
        self.channel_drop_prob = float(history_gate.get('channel_drop_prob', 0.08))
        self.block_drop_prob = float(history_gate.get('block_drop_prob', 0.65))
        self.min_block = int(history_gate.get('min_block', 4))
        self.max_block = int(history_gate.get('max_block', 14))
        self.spatial_shift_prob = float(history_gate.get('spatial_shift_prob', 0.35))
        self.max_shift = int(history_gate.get('max_shift', 4))
        self.occstress_style_modes = history_gate.get(
            'occstress_style_modes', ('dropout', 'hole', 'semantic', 'misalignment'))
        if isinstance(self.occstress_style_modes, str):
            self.occstress_style_modes = tuple(item.strip() for item in self.occstress_style_modes.split(',') if item.strip())
        else:
            self.occstress_style_modes = tuple(self.occstress_style_modes)
        self.occstress_noise_std = float(history_gate.get('occstress_noise_std', 0.05))
        self.occstress_dropout_channel_drop_prob = float(history_gate.get('occstress_dropout_channel_drop_prob', 0.35))
        self.occstress_hole_min_block = int(history_gate.get('occstress_hole_min_block', 6))
        self.occstress_hole_max_block = int(history_gate.get('occstress_hole_max_block', 18))
        self.occstress_hole_max_blocks = int(history_gate.get('occstress_hole_max_blocks', 2))
        self.occstress_misalignment_prob = float(history_gate.get('occstress_misalignment_prob', 1.0))
        self.occstress_max_shift = int(history_gate.get('occstress_max_shift', 6))
        self.occstress_semantic_channel_mix_prob = float(history_gate.get('occstress_semantic_channel_mix_prob', 0.5))
        self.occstress_semantic_noise_std = float(history_gate.get('occstress_semantic_noise_std', 0.06))

        self.current_identity_weight = float(history_gate.get('current_identity_weight', 0.05))
        self.current_denoise_weight = float(history_gate.get('current_denoise_weight', 0.1))
        self.current_gate_supervision_weight = float(history_gate.get('current_gate_supervision_weight', 0.02))
        self.history_gate_supervision_weight = float(history_gate.get('history_gate_supervision_weight', 0.02))
        self.strength_reg_weight = float(history_gate.get('strength_reg_weight', 0.001))
        self.clean_consistency_weight = float(history_gate.get('clean_consistency_weight', 0.0))
        self.history_gate_clean_target = float(history_gate.get('history_gate_clean_target', 0.85))
        self.history_gate_corrupt_target = float(history_gate.get('history_gate_corrupt_target', 0.0))
        self.current_gate_clean_target = float(history_gate.get('current_gate_clean_target', 0.0))
        self.current_gate_corrupt_target = float(history_gate.get('current_gate_corrupt_target', 0.85))

        embed_dims = int(getattr(self.transformer, 'embed_dims', history_gate.get('embed_dims', 128)))
        hidden_dims = int(history_gate.get('hidden_dims', embed_dims))
        meta_dims = 1 + (self.flow_summary_dim if self.use_flow_summary else 0)
        history_in_dims = embed_dims * 7 + meta_dims
        current_in_dims = embed_dims * 6 + meta_dims

        self.history_adapter = nn.Sequential(
            nn.Conv2d(history_in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, kernel_size=1),
        )
        self.history_reliability_gate = nn.Sequential(
            nn.Conv2d(history_in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, kernel_size=1),
        )
        self.current_adapter = nn.Sequential(
            nn.Conv2d(current_in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, kernel_size=1),
        )
        self.current_reliability_gate = nn.Sequential(
            nn.Conv2d(current_in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, kernel_size=1),
        )
        self.history_strength = nn.Parameter(torch.tensor(float(history_gate.get('history_strength_init', 0.0))))
        self.current_strength = nn.Parameter(torch.tensor(float(history_gate.get('current_strength_init', 0.0))))
        self._last_history_gate_stats = None
        self._init_selective_gate(history_gate)

    def _init_selective_gate(self, history_gate):
        for module in [self.history_adapter, self.current_adapter]:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        nn.init.zeros_(self.history_reliability_gate[-1].weight)
        nn.init.zeros_(self.current_reliability_gate[-1].weight)
        nn.init.constant_(self.history_reliability_gate[-1].bias, float(history_gate.get('history_gate_bias', 0.0)))
        nn.init.constant_(self.current_reliability_gate[-1].bias, float(history_gate.get('current_gate_bias', -4.0)))

    def _normalize_img_metas(self, img_metas):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        return img_metas

    def _to_model_tensor(self, value, device, dtype):
        if value is None:
            return None
        if isinstance(value, DataContainer):
            value = value.data
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=dtype)

    def _sample_key(self, img_meta):
        scene = img_meta.get('scene_name', '')
        sample = img_meta.get('sample_idx', img_meta.get('token', img_meta.get('occ_index', '')))
        return f'{scene}:{sample}'

    def _stable_seed(self, key, salt):
        digest = hashlib.sha1(f'{self.hist_seed}:{salt}:{key}'.encode('utf-8')).hexdigest()
        return int(digest[:8], 16)

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
        if self.corruption_mode == 'occstress_style':
            return self._corrupt_one_latent_occstress_style(latent, seed)

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

    def _corrupt_one_latent_occstress_style(self, latent, seed):
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(seed)
        out = latent.clone()
        channels, height, width = out.shape
        modes = self.occstress_style_modes or ('dropout',)
        mode = modes[seed % len(modes)]

        if mode == 'dropout':
            if self.occstress_dropout_channel_drop_prob > 0:
                keep = torch.rand(channels, generator=generator, device=out.device) > self.occstress_dropout_channel_drop_prob
                out = out * keep.to(dtype=out.dtype).view(channels, 1, 1)
            if self.occstress_noise_std > 0:
                out = out + torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * self.occstress_noise_std
            return out

        if mode == 'hole':
            max_block = max(self.occstress_hole_min_block, min(self.occstress_hole_max_block, height, width))
            num_blocks = int(torch.randint(
                1, max(1, self.occstress_hole_max_blocks) + 1, (), generator=generator, device=out.device).item())
            for _ in range(num_blocks):
                block_h = int(torch.randint(
                    self.occstress_hole_min_block, max_block + 1, (), generator=generator, device=out.device).item())
                block_w = int(torch.randint(
                    self.occstress_hole_min_block, max_block + 1, (), generator=generator, device=out.device).item())
                y0 = int(torch.randint(0, max(1, height - block_h + 1), (), generator=generator, device=out.device).item())
                x0 = int(torch.randint(0, max(1, width - block_w + 1), (), generator=generator, device=out.device).item())
                out[:, y0:y0 + block_h, x0:x0 + block_w] = 0
            return out

        if mode == 'misalignment':
            if self.occstress_misalignment_prob > 0 and torch.rand((), generator=generator, device=out.device) < self.occstress_misalignment_prob:
                shift_y = int(torch.randint(-self.occstress_max_shift, self.occstress_max_shift + 1, (), generator=generator, device=out.device).item())
                shift_x = int(torch.randint(-self.occstress_max_shift, self.occstress_max_shift + 1, (), generator=generator, device=out.device).item())
                if shift_y == 0 and shift_x == 0 and self.occstress_max_shift > 0:
                    shift_x = self.occstress_max_shift
                out = self._zero_shift(out, shift_y, shift_x)
            return out

        if mode == 'semantic':
            if channels > 1 and self.occstress_semantic_channel_mix_prob > 0:
                shift = int(torch.randint(1, channels, (), generator=generator, device=out.device).item())
                mixed = torch.roll(out, shifts=shift, dims=0)
                use_mixed = torch.rand(channels, generator=generator, device=out.device) < self.occstress_semantic_channel_mix_prob
                out = torch.where(use_mixed.to(dtype=torch.bool).view(channels, 1, 1), mixed, out)
            if self.occstress_semantic_noise_std > 0:
                out = out + torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * self.occstress_semantic_noise_std
            return out

        if self.occstress_noise_std > 0:
            out = out + torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype) * self.occstress_noise_std
        return out

    def _make_current_corr_mask(self, img_metas, device):
        threshold = int(round(self.current_corr_ratio * 10000))
        values = [
            (self._stable_seed(self._sample_key(img_meta), 'current_corr') % 10000) < threshold
            for img_meta in img_metas
        ]
        return torch.tensor(values, device=device, dtype=torch.bool)

    def _make_corrupted_current(self, curr_latent, img_metas, corr_mask):
        if not corr_mask.any():
            return curr_latent
        out = curr_latent.clone()
        for batch_idx, img_meta in enumerate(img_metas):
            if not corr_mask[batch_idx]:
                continue
            seed = self._stable_seed(self._sample_key(img_meta), 'current_aug')
            out[batch_idx] = self._corrupt_one_latent(out[batch_idx], seed)
        return out

    def _make_corrupted_history(self, history_latent, img_metas):
        batch_size, num_history = history_latent.shape[:2]
        out = history_latent.clone()
        corr_slot_mask = torch.zeros(batch_size, num_history, device=history_latent.device, dtype=torch.bool)
        sample_threshold = int(round(self.hist_corr_ratio * 10000))
        slot_threshold = int(round(self.hist_slot_corr_prob * 10000))

        for batch_idx, img_meta in enumerate(img_metas):
            key = self._sample_key(img_meta)
            sample_corrupt = (self._stable_seed(key, 'hist_sample') % 10000) < sample_threshold
            if not sample_corrupt:
                continue
            any_slot = False
            for slot_idx in range(num_history):
                slot_corrupt = (self._stable_seed(key, f'hist_slot_{slot_idx}') % 10000) < slot_threshold
                if not slot_corrupt:
                    continue
                seed = self._stable_seed(key, f'hist_aug_{slot_idx}')
                out[batch_idx, slot_idx] = self._corrupt_one_latent(out[batch_idx, slot_idx], seed)
                corr_slot_mask[batch_idx, slot_idx] = True
                any_slot = True
            if not any_slot:
                slot_idx = self._stable_seed(key, 'hist_force_slot') % num_history
                seed = self._stable_seed(key, f'hist_aug_force_{slot_idx}')
                out[batch_idx, slot_idx] = self._corrupt_one_latent(out[batch_idx, slot_idx], seed)
                corr_slot_mask[batch_idx, slot_idx] = True
        return out, corr_slot_mask

    def _history_meta_features(self, curr_latent, num_history, height, width, history_flow_summary=None):
        batch_size = curr_latent.shape[0]
        age = torch.linspace(
            1.0, 1.0 / max(num_history, 1), num_history,
            device=curr_latent.device,
            dtype=curr_latent.dtype,
        ).view(1, num_history, 1, 1, 1).expand(batch_size, -1, -1, height, width)
        meta_features = [age]
        if self.use_flow_summary:
            if history_flow_summary is None:
                history_flow_summary = curr_latent.new_zeros(batch_size, num_history, self.flow_summary_dim)
            else:
                history_flow_summary = history_flow_summary[..., :self.flow_summary_dim]
                if history_flow_summary.shape[-1] < self.flow_summary_dim:
                    pad = curr_latent.new_zeros(batch_size, num_history, self.flow_summary_dim - history_flow_summary.shape[-1])
                    history_flow_summary = torch.cat([history_flow_summary, pad], dim=-1)
            flow_meta = history_flow_summary.view(batch_size, num_history, self.flow_summary_dim, 1, 1)
            meta_features.append(flow_meta.expand(-1, -1, -1, height, width))
        return torch.cat(meta_features, dim=2)

    def _build_history_inputs(self, curr_latent, history_latent, history_flow_summary=None):
        batch_size, num_history, channels, height, width = history_latent.shape
        curr = curr_latent.unsqueeze(1).expand(-1, num_history, -1, -1, -1)
        history_mean = history_latent.mean(dim=1, keepdim=True).expand_as(history_latent)
        history_std = history_latent.std(dim=1, keepdim=True, unbiased=False).expand_as(history_latent)
        meta = self._history_meta_features(curr_latent, num_history, height, width, history_flow_summary)
        features = [
            curr,
            history_latent,
            history_latent - curr,
            (history_latent - curr).abs(),
            history_mean,
            (history_latent - history_mean).abs(),
            history_std,
            meta,
        ]
        return torch.cat(features, dim=2).reshape(batch_size * num_history, -1, height, width)

    def _build_current_inputs(self, curr_latent, history_latent, consensus, history_flow_summary=None):
        batch_size, num_history, channels, height, width = history_latent.shape
        history_mean = history_latent.mean(dim=1)
        history_std = history_latent.std(dim=1, unbiased=False)
        if self.use_flow_summary:
            meta = self._history_meta_features(curr_latent, num_history, height, width, history_flow_summary).mean(dim=1)
        else:
            meta = self._history_meta_features(curr_latent, num_history, height, width, None).mean(dim=1)
        features = [
            curr_latent,
            consensus,
            consensus - curr_latent,
            (consensus - curr_latent).abs(),
            history_mean,
            history_std,
            meta,
        ]
        return torch.cat(features, dim=1)

    def _selective_fuse(self, curr_latent, history_latent, history_flow_summary=None):
        batch_size, num_history, channels, height, width = history_latent.shape
        history_input = self._build_history_inputs(curr_latent, history_latent, history_flow_summary)
        history_residual = self.history_adapter(history_input).reshape(batch_size, num_history, channels, height, width)
        history_gate = torch.sigmoid(self.history_reliability_gate(history_input)).reshape(
            batch_size, num_history, 1, height, width)
        history_candidate = history_latent + history_residual

        weight_sum = history_gate.sum(dim=1).clamp_min(1e-4)
        consensus = (history_candidate * history_gate).sum(dim=1) / weight_sum
        current_input = self._build_current_inputs(curr_latent, history_latent, consensus, history_flow_summary)
        current_residual = self.current_adapter(current_input)
        current_gate = torch.sigmoid(self.current_reliability_gate(current_input))
        curr_robust = curr_latent + self.current_strength * current_gate * (
            consensus - curr_latent + current_residual)

        curr_anchor = curr_robust.unsqueeze(1).expand_as(history_candidate)
        history_strength_effective = self.history_strength
        if self.force_history_strength_zero:
            history_strength_effective = self.history_strength * 0.0
        fused_history = curr_anchor + history_strength_effective * history_gate * (history_candidate - curr_anchor)

        self._last_history_gate_stats = dict(
            history_gate=history_gate,
            current_gate=current_gate,
            curr_robust=curr_robust,
            history_residual_abs=history_residual.abs().mean(),
            current_residual_abs=current_residual.abs().mean(),
            current_delta_abs=(curr_robust - curr_latent).abs().mean(),
            history_delta_abs=(fused_history - curr_anchor).abs().mean(),
            history_strength_effective=history_strength_effective,
        )
        return curr_robust, fused_history

    def init_state(self, trans_infos, latent, history_latent=None,
                   history_flow_summary=None, gate_enabled=True):
        history_info, curr_info = super().init_state(trans_infos, latent)
        if not gate_enabled or not self.history_gate_enabled or history_latent is None:
            self._last_history_gate_stats = None
            return history_info, curr_info

        device, dtype = latent.device, latent.dtype
        history_latent = self._to_model_tensor(history_latent, device, dtype)
        if history_latent.dim() == 6 and history_latent.shape[1] == 1:
            history_latent = history_latent[:, 0]
        if history_latent.shape[1] != self.history_frame_number_gate:
            history_latent = history_latent[:, -self.history_frame_number_gate:]
        history_flow_summary = self._to_model_tensor(history_flow_summary, device, dtype)

        curr_latent = latent[:, 0]
        curr_robust, fused_history = self._selective_fuse(curr_latent, history_latent, history_flow_summary)
        curr_info['curr_latent'] = curr_robust
        history_info['history_token'] = torch.cat([fused_history, curr_robust.unsqueeze(1)], dim=1)
        return history_info, curr_info

    def _capture_observe_state(self):
        return (
            None if self.observe_relative_rotation is None else self.observe_relative_rotation.clone(),
            None if self.observe_delta_translation is None else self.observe_delta_translation.clone(),
            None if self.observe_ego_lcf_feat is None else self.observe_ego_lcf_feat.clone(),
        )

    def _restore_observe_state(self, state):
        self.observe_relative_rotation, self.observe_delta_translation, self.observe_ego_lcf_feat = state

    def forward_sample(self, latent, img_metas, predict_future_frame, train=True,
                       history_latent=None, history_flow_summary=None,
                       gate_enabled=True, update_observe_state=True, **kwargs):
        trans_infos = self.load_transformation_info(img_metas, latent)
        self.process_observe_info(trans_infos, latent, start_update=True)
        history_info, curr_info = self.init_state(
            trans_infos,
            latent,
            history_latent=history_latent,
            history_flow_summary=history_flow_summary,
            gate_enabled=gate_enabled,
        )

        pred_latents = []
        pred_relative_rotations, pred_delta_translations = [], []
        for frame_idx in range(predict_future_frame):
            use_gt_rate = torch.rand(size=(latent.shape[0],), device=latent.device) < self.sample_rate
            plan_query = self.pose_encoder.forward_encoder(history_info)
            pred_trans_info = self.transformer(
                curr_info=curr_info,
                history_info=history_info,
                plan_queries=plan_query,
            )
            pred_trans_info = self.pose_encoder.get_ego_feat(
                pred_trans_info=pred_trans_info,
                curr_info=curr_info,
                start_of_sequence=trans_infos['start_of_sequence'],
            )
            if frame_idx != predict_future_frame - 1:
                curr_info = self.update_curr_info(curr_info, trans_infos, pred_trans_info, use_gt_rate, frame_idx, train)
                history_info = self.update_history_info(history_info, curr_info)
            pred_latents.append(pred_trans_info['pred_latent'])
            pred_delta_translations.append(pred_trans_info['pred_delta_translation'])
            pred_relative_rotations.append(pred_trans_info['pred_relative_rotation'])

        if update_observe_state:
            self.process_observe_info(trans_infos, latent, start_update=False)
        return dict(
            pred_latents=torch.stack(pred_latents, dim=1),
            pred_delta_translations=torch.stack(pred_delta_translations, dim=1),
            pred_relative_rotations=torch.stack(pred_relative_rotations, dim=1),
            targ_delta_translations=trans_infos['ego_to_global_delta_translation'],
            targ_relative_rotations=trans_infos['ego_to_global_relative_rotation'],
        )

    def _world_loss(self, return_dict, latent, img_metas):
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
        return loss_dict, valid_frame

    def _prediction_consistency_loss(self, pred_latents, target_latents, valid_frame):
        loss = pred_latents.sum() * 0.0
        for frame_idx in range(self.train_future_frame):
            valid = valid_frame[:, frame_idx]
            if valid.any():
                loss = loss + self.frame_loss_weight[frame_idx] * \
                    (pred_latents[valid, frame_idx] - target_latents[valid, frame_idx]).abs().mean()
        return loss

    def _history_gate_supervision_loss(self, gate, corr_slot_mask):
        if gate is None:
            return corr_slot_mask.sum() * 0.0
        target = torch.full(gate.shape[:2], self.history_gate_clean_target, device=gate.device, dtype=gate.dtype)
        target = torch.where(
            corr_slot_mask,
            torch.full_like(target, self.history_gate_corrupt_target),
            target,
        )
        target = target[:, :, None, None, None].expand_as(gate)
        return F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), target)

    def _current_gate_supervision_loss(self, gate, corr_mask):
        if gate is None:
            return corr_mask.sum() * 0.0
        target = torch.full((gate.shape[0],), self.current_gate_clean_target, device=gate.device, dtype=gate.dtype)
        target = torch.where(
            corr_mask,
            torch.full_like(target, self.current_gate_corrupt_target),
            target,
        )
        target = target[:, None, None, None].expand_as(gate)
        return F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), target)

    def _clean_consistency_forward(self, latent, img_metas, history_latent, history_flow_summary):
        state = self._capture_observe_state()
        clean_gated = self.forward_sample(
            latent,
            img_metas,
            self.train_future_frame,
            train=False,
            history_latent=history_latent,
            history_flow_summary=history_flow_summary,
            gate_enabled=True,
            update_observe_state=False,
        )
        self._restore_observe_state(state)
        with torch.no_grad():
            baseline = self.forward_sample(
                latent,
                img_metas,
                self.train_future_frame,
                train=False,
                history_latent=None,
                history_flow_summary=None,
                gate_enabled=False,
                update_observe_state=False,
            )
        self._restore_observe_state(state)
        return clean_gated, baseline

    def forward_train(self, latent, img_metas, history_latent=None, history_flow_summary=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if not self.history_gate_enabled or history_latent is None:
            return super().forward_train(latent, img_metas, **kwargs)

        history_latent = self._to_model_tensor(history_latent, latent.device, latent.dtype)
        history_flow_summary = self._to_model_tensor(history_flow_summary, latent.device, latent.dtype)
        current_corr_mask = self._make_current_corr_mask(img_metas, latent.device)
        corrupted_current = self._make_corrupted_current(latent[:, 0], img_metas, current_corr_mask)
        corrupted_history, history_corr_slot_mask = self._make_corrupted_history(history_latent, img_metas)

        consistency_loss = None
        if self.clean_consistency_weight > 0:
            clean_gated, clean_baseline = self._clean_consistency_forward(
                latent, img_metas, history_latent, history_flow_summary)
            valid_frame = torch.stack([
                torch.tensor(img_meta['valid_frame'], device=latent.device) for img_meta in img_metas
            ])
            consistency_loss = self._prediction_consistency_loss(
                clean_gated['pred_latents'], clean_baseline['pred_latents'].detach(), valid_frame)

        latent_for_model = latent.clone()
        latent_for_model[:, 0] = corrupted_current
        return_dict = self.forward_sample(
            latent_for_model,
            img_metas,
            self.train_future_frame,
            train=True,
            history_latent=corrupted_history,
            history_flow_summary=history_flow_summary,
        )
        stats = self._last_history_gate_stats
        loss_dict, _ = self._world_loss(return_dict, latent, img_metas)

        if stats is not None:
            curr_robust = stats['curr_robust']
            zero = curr_robust.sum() * 0.0
            clean_mask = ~current_corr_mask
            identity = zero
            if clean_mask.any():
                identity = (curr_robust[clean_mask] - latent[:, 0][clean_mask]).abs().mean()
            denoise = zero
            if current_corr_mask.any():
                denoise = (curr_robust[current_corr_mask] - latent[:, 0][current_corr_mask]).abs().mean()
            loss_dict['select_current_identity_loss'] = self.current_identity_weight * identity
            loss_dict['select_current_denoise_loss'] = self.current_denoise_weight * denoise
            loss_dict['select_current_gate_sup_loss'] = self.current_gate_supervision_weight * \
                self._current_gate_supervision_loss(stats['current_gate'], current_corr_mask)
            loss_dict['select_history_gate_sup_loss'] = self.history_gate_supervision_weight * \
                self._history_gate_supervision_loss(stats['history_gate'], history_corr_slot_mask)
            if self.strength_reg_weight > 0:
                loss_dict['select_strength_reg_loss'] = self.strength_reg_weight * (
                    self.current_strength.abs() + self.history_strength.abs())
            if consistency_loss is not None:
                loss_dict['select_clean_consistency_loss'] = self.clean_consistency_weight * consistency_loss
            loss_dict['select_current_gate_mean'] = stats['current_gate'].mean().detach()
            loss_dict['select_history_gate_mean'] = stats['history_gate'].mean().detach()
            loss_dict['select_current_corr_ratio'] = current_corr_mask.to(dtype=latent.dtype).mean().detach()
            loss_dict['select_history_corr_ratio'] = history_corr_slot_mask.to(dtype=latent.dtype).mean().detach()
            loss_dict['select_current_strength'] = self.current_strength.detach()
            loss_dict['select_history_strength'] = self.history_strength.detach()
            loss_dict['select_history_strength_effective'] = stats['history_strength_effective'].detach()
            loss_dict['select_current_delta_abs'] = stats['current_delta_abs'].detach()
            loss_dict['select_history_delta_abs'] = stats['history_delta_abs'].detach()
            loss_dict['select_current_residual_abs'] = stats['current_residual_abs'].detach()
            loss_dict['select_history_residual_abs'] = stats['history_residual_abs'].detach()
        return loss_dict

    def forward_test(self, latent, voxel_semantics, img_metas, history_latent=None,
                     history_flow_summary=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        start_time = tic()
        sample_dict = self.forward_sample(
            latent,
            img_metas,
            self.test_future_frame,
            train=False,
            history_latent=history_latent,
            history_flow_summary=history_flow_summary,
        )

        return_dict = dict()
        sample_idx = img_metas[0]['sample_idx']
        if self.task_mode == 'generate':
            targ_future_voxel_semantics = voxel_semantics[:, self.test_previous_frame + 1:]
            targ_curr_voxel_semantics = voxel_semantics[:, self.test_previous_frame:self.test_previous_frame + 1]
            pred_curr_voxel_semantics = self.obtain_scene_from_token(latent[:, 0])
            pred_curr_voxel_semantics = pred_curr_voxel_semantics.softmax(-1).argmax(-1)
            if self.dataset_type != 'waymo':
                return_dict['pred_curr_semantics'] = pred_curr_voxel_semantics.cpu().numpy().astype('uint8')
                return_dict['targ_curr_semantics'] = targ_curr_voxel_semantics.cpu().numpy().astype('uint8')

            pred_latents = sample_dict['pred_latents']
            pred_voxel_semantics = self.obtain_scene_from_token(pred_latents)
            pred_voxel_semantics = pred_voxel_semantics.softmax(-1).argmax(-1)
            bs = pred_voxel_semantics.shape[0]
            end_time = tic()

            if self.dataset_type == 'waymo':
                if self.eval_metric == 'forecasting_miou':
                    pred_voxel_semantics = pred_voxel_semantics[:, [1, 3, 5]]
                    targ_future_voxel_semantics = targ_future_voxel_semantics[:, [1, 3, 5]]
                elif self.eval_metric == 'miou':
                    pred_voxel_semantics = pred_voxel_semantics[:, [self.eval_time]]
                    targ_future_voxel_semantics = targ_future_voxel_semantics[:, [self.eval_time]]

            if self.eval_metric == 'forecasting_miou':
                return_dict['pred_futu_semantics'] = pred_voxel_semantics.cpu().numpy().astype('uint8')
                return_dict['targ_futu_semantics'] = targ_future_voxel_semantics.cpu().numpy().astype('uint8')
            elif self.eval_metric == 'miou':
                return_dict['semantics'] = pred_voxel_semantics.cpu().numpy().astype('uint8')
                return_dict['targ_semantics'] = targ_future_voxel_semantics.cpu().numpy()

        return_dict['occ_path'] = [img_meta['occ_path'] for img_meta in img_metas]
        return_dict['occ_index'] = [img_meta['occ_index'] for img_meta in img_metas]
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return_dict['sample_idx'] = sample_idx
        return_dict['time'] = (end_time - start_time) / self.test_future_frame / bs
        return [return_dict]
