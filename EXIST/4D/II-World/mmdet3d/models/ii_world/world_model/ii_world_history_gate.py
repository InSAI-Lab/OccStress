import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World, tic


@DETECTORS.register_module()
class II_WorldHistoryGate(II_World):
    """Stage2-only reliability-aware history fusion.

    The locked baseline stage1 tokenizer can still export the regular stage2
    tokens. This module optionally reads explicit previous-frame tokens and
    replaces the baseline current-repeat memory with a zero-init gated residual
    path. At initialization the memory is exactly the baseline repeated current
    token, so finetuning starts from the locked stage2 behavior.
    """

    def __init__(self, history_gate=None, **kwargs):
        super().__init__(**kwargs)
        history_gate = history_gate or {}
        self.history_gate_enabled = bool(history_gate.get('enabled', True))
        self.history_frame_number_gate = int(history_gate.get('history_frame_number', 4))
        self.use_flow_summary = bool(history_gate.get('use_flow_summary', False))
        self.flow_summary_dim = int(history_gate.get('flow_summary_dim', 5 if self.use_flow_summary else 0))

        self.hist_corr_ratio = float(history_gate.get('corr_ratio', 0.5))
        self.hist_slot_corr_prob = float(history_gate.get('slot_corr_prob', 0.5))
        self.hist_seed = int(history_gate.get('seed', 3407))
        self.noise_std = float(history_gate.get('noise_std', 0.03))
        self.channel_drop_prob = float(history_gate.get('channel_drop_prob', 0.08))
        self.block_drop_prob = float(history_gate.get('block_drop_prob', 0.65))
        self.min_block = int(history_gate.get('min_block', 4))
        self.max_block = int(history_gate.get('max_block', 14))
        self.spatial_shift_prob = float(history_gate.get('spatial_shift_prob', 0.35))
        self.max_shift = int(history_gate.get('max_shift', 4))

        self.consistency_weight = float(history_gate.get('consistency_weight', 0.05))
        self.gate_supervision_weight = float(history_gate.get('gate_supervision_weight', 0.02))
        self.gate_clean_target = float(history_gate.get('gate_clean_target', 0.7))
        self.gate_corrupt_target = float(history_gate.get('gate_corrupt_target', 0.0))

        embed_dims = int(getattr(self.transformer, 'embed_dims', history_gate.get('embed_dims', 128)))
        hidden_dims = int(history_gate.get('hidden_dims', embed_dims))
        meta_dims = 1 + (self.flow_summary_dim if self.use_flow_summary else 0)
        in_dims = embed_dims * 4 + meta_dims

        self.history_adapter = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, embed_dims, kernel_size=1),
        )
        self.history_reliability_gate = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims, 1, kernel_size=1),
        )
        self._last_history_gate_stats = None
        self._init_history_gate(history_gate)

    def _init_history_gate(self, history_gate):
        nn.init.zeros_(self.history_adapter[-1].weight)
        nn.init.zeros_(self.history_adapter[-1].bias)
        nn.init.zeros_(self.history_reliability_gate[-1].weight)
        gate_bias = float(history_gate.get('gate_bias', -4.0))
        nn.init.constant_(self.history_reliability_gate[-1].bias, gate_bias)

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

    def _build_gate_inputs(self, curr_latent, history_latent, history_flow_summary=None):
        batch_size, num_history, channels, height, width = history_latent.shape
        curr = curr_latent.unsqueeze(1).expand(-1, num_history, -1, -1, -1)
        features = [curr, history_latent, history_latent - curr, (history_latent - curr).abs()]

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
            flow_meta = flow_meta.expand(-1, -1, -1, height, width)
            meta_features.append(flow_meta)

        gate_input = torch.cat(features + meta_features, dim=2)
        gate_input = gate_input.reshape(batch_size * num_history, -1, height, width)
        return gate_input

    def _fuse_history(self, curr_latent, history_latent, history_flow_summary=None):
        gate_input = self._build_gate_inputs(curr_latent, history_latent, history_flow_summary)
        batch_size, num_history, channels, height, width = history_latent.shape
        residual = self.history_adapter(gate_input).reshape(batch_size, num_history, channels, height, width)
        gate = torch.sigmoid(self.history_reliability_gate(gate_input)).reshape(batch_size, num_history, 1, height, width)
        curr = curr_latent.unsqueeze(1).expand_as(history_latent)
        candidate = history_latent + residual
        fused_history = curr + gate * (candidate - curr)
        self._last_history_gate_stats = dict(
            gate=gate,
            residual_abs=residual.abs().mean(),
            fused_delta_abs=(fused_history - curr).abs().mean(),
        )
        return fused_history

    def init_state(self, trans_infos, latent, history_latent=None, history_flow_summary=None):
        history_info, curr_info = super().init_state(trans_infos, latent)
        if not self.history_gate_enabled or history_latent is None:
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
        fused_history = self._fuse_history(curr_latent, history_latent, history_flow_summary)
        history_info['history_token'] = torch.cat([fused_history, curr_latent.unsqueeze(1)], dim=1)
        return history_info, curr_info

    def forward_sample(self, latent, img_metas, predict_future_frame, train=True,
                       history_latent=None, history_flow_summary=None, **kwargs):
        trans_infos = self.load_transformation_info(img_metas, latent)
        self.process_observe_info(trans_infos, latent, start_update=True)
        history_info, curr_info = self.init_state(
            trans_infos,
            latent,
            history_latent=history_latent,
            history_flow_summary=history_flow_summary,
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

    def _gate_supervision_loss(self, gate, corr_slot_mask):
        if gate is None:
            return corr_slot_mask.sum() * 0.0
        target = torch.full(
            gate.shape[:2],
            self.gate_clean_target,
            device=gate.device,
            dtype=gate.dtype,
        )
        target = torch.where(
            corr_slot_mask,
            torch.full_like(target, self.gate_corrupt_target),
            target,
        )
        target = target[:, :, None, None, None].expand_as(gate)
        return F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), target)

    def forward_train(self, latent, img_metas, history_latent=None, history_flow_summary=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if not self.history_gate_enabled or history_latent is None:
            return super().forward_train(latent, img_metas, **kwargs)

        history_latent = self._to_model_tensor(history_latent, latent.device, latent.dtype)
        history_flow_summary = self._to_model_tensor(history_flow_summary, latent.device, latent.dtype)
        corrupted_history, corr_slot_mask = self._make_corrupted_history(history_latent, img_metas)

        return_dict = self.forward_sample(
            latent,
            img_metas,
            self.train_future_frame,
            train=True,
            history_latent=corrupted_history,
            history_flow_summary=history_flow_summary,
        )
        student_stats = self._last_history_gate_stats
        loss_dict, valid_frame = self._world_loss(return_dict, latent, img_metas)

        if self.consistency_weight > 0:
            with torch.no_grad():
                clean_return = self.forward_sample(
                    latent,
                    img_metas,
                    self.train_future_frame,
                    train=True,
                    history_latent=history_latent,
                    history_flow_summary=history_flow_summary,
                )
            loss_dict['history_consistency_loss'] = self.consistency_weight * self._prediction_consistency_loss(
                return_dict['pred_latents'],
                clean_return['pred_latents'].detach(),
                valid_frame,
            )

        if self.gate_supervision_weight > 0 and student_stats is not None:
            loss_dict['history_gate_sup_loss'] = self.gate_supervision_weight * self._gate_supervision_loss(
                student_stats['gate'],
                corr_slot_mask,
            )
            loss_dict['history_gate_mean'] = student_stats['gate'].mean().detach()
            loss_dict['history_gate_corrupt_ratio'] = corr_slot_mask.to(dtype=latent.dtype).mean().detach()
            loss_dict['history_residual_abs'] = student_stats['residual_abs'].detach()
            loss_dict['history_fused_delta_abs'] = student_stats['fused_delta_abs'].detach()
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
