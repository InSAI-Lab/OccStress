import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_prior_slot import IISceneTokenizerCurrBevKalmanPriorSlot


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanUnifiedMotionBank(IISceneTokenizerCurrBevKalmanPriorSlot):
    """Unified-motion 4-slot history bank with latent-only Kalman analysis.

    Design:
    - keep baseline encoder / VQ / decoder frozen
    - stop mixing baseline pose-aligned sampled_bev with a separate full-flow prior
    - rebuild all 4 history slots from one motion source:
      `static ego warp + dynamic residual flow`
    - keep Kalman only as a latent analysis / auxiliary-learning path
    - final readout stays baseline-like:
      `vq(curr_bev, sampled_bev_motion) -> decoder`
    """

    def __init__(self,
                 unified_motion_use_dynamic_gate=True,
                 unified_motion_dynamic_transport='backward',
                 kalman_posterior_aux_loss_weight=0.25,
                 kalman_posterior_dynamic_boost=1.0,
                 kalman_bad_threshold=0.5,
                 kalman_debug_decode_state=False,
                 **kwargs):
        super().__init__(
            kalman_posterior_aux_loss_weight=kalman_posterior_aux_loss_weight,
            kalman_debug_decode_state=kalman_debug_decode_state,
            **kwargs,
        )
        self.unified_motion_use_dynamic_gate = unified_motion_use_dynamic_gate
        self.unified_motion_dynamic_transport = unified_motion_dynamic_transport
        self.kalman_posterior_dynamic_boost = kalman_posterior_dynamic_boost
        self.kalman_bad_threshold = kalman_bad_threshold
        self.kalman_debug_decode_state = kalman_debug_decode_state

    def _normalize_flow_sequence_tensor(self, tensor, batch_size, has_vec):
        tensor = self.normalize_batched_tensor(tensor, batch_size)
        expected_dim = 6 if has_vec else 5
        if tensor.dim() == expected_dim - 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == expected_dim and tensor.shape[0] != batch_size and tensor.shape[1] == batch_size:
            tensor = tensor.transpose(0, 1)
        return tensor

    def _normalize_clean_sequence(self, voxel_semantics, voxel_semantics_clean, batch_size):
        if voxel_semantics_clean is None:
            return voxel_semantics
        return self.normalize_batched_tensor(voxel_semantics_clean, batch_size)

    def _derive_bad_mask_latent(self, curr_input, curr_target, current_reliability, target_hw):
        if current_reliability is not None:
            if current_reliability.dim() == 4:
                current_reliability = current_reliability[:, -1]
            reliability_latent = F.adaptive_avg_pool2d(current_reliability.unsqueeze(1), target_hw)
            return (reliability_latent < self.kalman_bad_threshold).float()

        bad_voxel = (curr_input != curr_target).float().amax(dim=-1)
        if bad_voxel.dim() == 4 and bad_voxel.shape[1] == 1:
            bad_voxel = bad_voxel[:, 0]
        bad_latent = F.adaptive_max_pool2d(bad_voxel.unsqueeze(1), target_hw)
        return (bad_latent > 0).float()

    def _masked_latent_loss(self, pred, target, latent_mask):
        if latent_mask is None:
            return pred.sum() * 0.0
        valid = latent_mask.bool()
        if not valid.any().item():
            return pred.sum() * 0.0
        loss_map = F.smooth_l1_loss(pred, target, reduction='none')
        weight = latent_mask.to(loss_map.dtype)
        denom = (weight.sum() * pred.shape[1]).clamp_min(1.0)
        return (loss_map * weight).sum() / denom

    def _aggregate_decomposed_flow_to_latent(self,
                                             static_flow,
                                             static_valid,
                                             dynamic_residual_flow,
                                             dynamic_residual_valid,
                                             dynamic_mask,
                                             target_hw):
        static_flow_latent, static_valid_ratio, _ = self._aggregate_flow_to_latent(
            static_flow, static_valid, dynamic_mask, target_hw)
        dynamic_residual_latent, dynamic_residual_valid_ratio, dynamic_ratio = self._aggregate_flow_to_latent(
            dynamic_residual_flow, dynamic_residual_valid, dynamic_mask, target_hw)
        dynamic_gate = dynamic_ratio * dynamic_residual_valid_ratio if self.unified_motion_use_dynamic_gate else dynamic_ratio
        return dict(
            static_flow=static_flow_latent,
            static_valid_ratio=static_valid_ratio,
            dynamic_residual_flow=dynamic_residual_latent,
            dynamic_residual_valid_ratio=dynamic_residual_valid_ratio,
            dynamic_ratio=dynamic_ratio,
            dynamic_gate=dynamic_gate.clamp(0.0, 1.0),
        )

    def _aggregate_dynamic_residual_to_latent(self,
                                              dynamic_residual_flow,
                                              dynamic_residual_valid,
                                              dynamic_mask,
                                              target_hw):
        if dynamic_residual_flow.dim() != 5:
            raise ValueError(
                f'Expected dynamic_residual_flow [B,H,W,Z,3], got {tuple(dynamic_residual_flow.shape)}')
        if dynamic_residual_valid.dim() != 4 or dynamic_mask.dim() != 4:
            raise ValueError(
                f'Expected dynamic_residual_valid/dynamic_mask [B,H,W,Z], got '
                f'{tuple(dynamic_residual_valid.shape)} and {tuple(dynamic_mask.shape)}')

        batch, occ_h, occ_w, occ_z, _ = dynamic_residual_flow.shape
        latent_h, latent_w = target_hw
        if occ_h % latent_h != 0 or occ_w % latent_w != 0:
            raise ValueError(
                f'Occupancy grid {(occ_h, occ_w)} is not divisible by latent grid {(latent_h, latent_w)}')
        scale_h = occ_h // latent_h
        scale_w = occ_w // latent_w

        dynamic_valid = (dynamic_residual_valid.float() * dynamic_mask.float()).unsqueeze(-1)
        flow_xy = (dynamic_residual_flow[..., :2] * dynamic_valid).reshape(
            batch, latent_h, scale_h, latent_w, scale_w, occ_z, 2)
        dynamic_valid = dynamic_valid.reshape(batch, latent_h, scale_h, latent_w, scale_w, occ_z, 1)

        flow_sum = flow_xy.sum(dim=(2, 4, 5))
        dynamic_count = dynamic_valid.sum(dim=(2, 4, 5)).clamp_min(1.0)
        flow_latent = flow_sum / dynamic_count
        flow_latent[..., 0] = flow_latent[..., 0] / float(scale_h)
        flow_latent[..., 1] = flow_latent[..., 1] / float(scale_w)

        dynamic_valid_ratio = dynamic_valid.mean(dim=(2, 4, 5)).squeeze(-1)
        dynamic_ratio = dynamic_mask.float().reshape(
            batch, latent_h, scale_h, latent_w, scale_w, occ_z).mean(dim=(2, 4, 5))
        return flow_latent.permute(0, 3, 1, 2), dynamic_valid_ratio.unsqueeze(1), dynamic_ratio.unsqueeze(1)

    def _to_align_bev_layout(self, feature):
        # Baseline align_bev warps features in [B, C, H, W] layout, while
        # forward_encoder returns BEV latent as [B, C, W, H]. Keep the same
        # contract here before using grid_sample-based warping.
        return feature.permute(0, 1, 3, 2).contiguous()

    def _from_align_bev_layout(self, feature):
        return feature.permute(0, 1, 3, 2).contiguous()

    def _warp_raw_bev_feature(self, feature, backward_flow):
        feature_hw = self._to_align_bev_layout(feature)
        warped_hw = self._warp_feature(feature_hw, backward_flow)
        return self._from_align_bev_layout(warped_hw)

    def _scatter_feature_hw(self, feature_hw, forward_flow, source_mask):
        batch_size, channels, height, width = feature_hw.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=feature_hw.device, dtype=forward_flow.dtype),
            torch.arange(width, device=feature_hw.device, dtype=forward_flow.dtype),
            indexing='ij',
        )
        yy_long = yy.long()
        xx_long = xx.long()
        dst_y = torch.round(yy.unsqueeze(0) + forward_flow[:, 0]).long()
        dst_x = torch.round(xx.unsqueeze(0) + forward_flow[:, 1]).long()
        in_bounds = (
            (dst_y >= 0) & (dst_y < height) &
            (dst_x >= 0) & (dst_x < width)
        )
        if source_mask is None:
            src_valid = in_bounds
        else:
            src_valid = in_bounds & (source_mask[:, 0] > 0.5)

        out = feature_hw.new_zeros(feature_hw.shape)
        out_mask = feature_hw.new_zeros((batch_size, 1, height, width))
        for batch_idx in range(batch_size):
            valid = src_valid[batch_idx]
            if not valid.any().item():
                continue
            src_y = yy_long[valid]
            src_x = xx_long[valid]
            dst_y_b = dst_y[batch_idx][valid]
            dst_x_b = dst_x[batch_idx][valid]
            out[batch_idx, :, dst_y_b, dst_x_b] = feature_hw[batch_idx, :, src_y, src_x]
            out_mask[batch_idx, 0, dst_y_b, dst_x_b] = 1.0
        return out, out_mask

    def _scatter_raw_bev_feature(self, feature, forward_flow, source_mask):
        feature_hw = self._to_align_bev_layout(feature)
        scatter_hw, scatter_mask = self._scatter_feature_hw(feature_hw, forward_flow, source_mask)
        return self._from_align_bev_layout(scatter_hw), self._from_align_bev_layout(scatter_mask)

    def _apply_unified_motion(self, feature, motion_step):
        static_flow = motion_step['static_flow']
        dynamic_total_flow = static_flow + motion_step['dynamic_residual_flow']
        if self.unified_motion_dynamic_transport == 'scatter':
            source_mask = motion_step.get('dynamic_source_mask', None)
            static_source = feature if source_mask is None else feature * (1.0 - source_mask)
            static_warp = self._warp_raw_bev_feature(static_source, static_flow)
            dynamic_warp, scatter_mask = self._scatter_raw_bev_feature(
                feature if source_mask is None else feature * source_mask,
                motion_step['forward_total_flow'],
                source_mask,
            )
            aligned = torch.where(scatter_mask.expand_as(static_warp) > 0.5, dynamic_warp, static_warp)
            total_flow = dynamic_total_flow
        else:
            static_warp = self._warp_raw_bev_feature(feature, static_flow)
            dynamic_warp = self._warp_raw_bev_feature(feature, dynamic_total_flow)
            if self.unified_motion_use_dynamic_gate:
                gate = motion_step['dynamic_gate']
                aligned = (1.0 - gate) * static_warp + gate * dynamic_warp
                total_flow = static_flow + gate * motion_step['dynamic_residual_flow']
            else:
                aligned = dynamic_warp
                total_flow = dynamic_total_flow
        return aligned, static_warp, dynamic_warp, total_flow

    def _build_motion_step(self,
                           static_flow_step,
                           static_valid_step,
                           dynamic_residual_flow_step,
                           dynamic_residual_valid_step,
                           dynamic_mask_step,
                           target_hw,
                           dtype):
        motion_step = self._aggregate_decomposed_flow_to_latent(
            static_flow_step.to(dtype),
            static_valid_step,
            dynamic_residual_flow_step.to(dtype),
            dynamic_residual_valid_step,
            dynamic_mask_step,
            target_hw,
        )
        return motion_step

    def _predict_kalman_state_unified(self, prev_state_bev, prev_state_cov, motion_step):
        prior_mean, static_mean, dynamic_mean, total_flow = self._apply_unified_motion(prev_state_bev, motion_step)
        static_cov = self._warp_raw_bev_feature(prev_state_cov, motion_step['static_flow'])
        dynamic_cov = self._warp_raw_bev_feature(
            prev_state_cov,
            motion_step['static_flow'] + motion_step['dynamic_residual_flow'])
        if self.unified_motion_use_dynamic_gate:
            dynamic_weight = motion_step['dynamic_gate']
            prior_cov = (1.0 - dynamic_weight) * static_cov + dynamic_weight * dynamic_cov
        else:
            dynamic_weight = motion_step['dynamic_residual_valid_ratio']
            prior_cov = dynamic_cov
        process_noise = self._build_process_noise(
            total_flow,
            motion_step['static_valid_ratio'],
            dynamic_weight,
        )
        prior_cov = prior_cov.clamp_min(self.kalman_min_cov) + process_noise
        return prior_mean, prior_cov, process_noise, static_mean, dynamic_mean, total_flow

    def _build_unified_history_bank(self, bev_seq, motion_steps):
        seq_len = len(bev_seq)
        slots = []
        slot_debug = []
        for src_idx in range(seq_len - 1):
            aligned = bev_seq[src_idx]
            step_records = []
            for step_idx in range(src_idx + 1, seq_len):
                aligned, static_warp, dynamic_warp, total_flow = self._apply_unified_motion(
                    aligned,
                    motion_steps[step_idx - 1],
                )
                step_records.append(dict(
                    target_step=step_idx,
                    static_warp=static_warp,
                    dynamic_warp=dynamic_warp,
                    aligned=aligned,
                    total_flow=total_flow,
                    static_flow=motion_steps[step_idx - 1]['static_flow'],
                    dynamic_residual_flow=motion_steps[step_idx - 1]['dynamic_residual_flow'],
                    static_valid_ratio=motion_steps[step_idx - 1]['static_valid_ratio'],
                    dynamic_residual_valid_ratio=motion_steps[step_idx - 1]['dynamic_residual_valid_ratio'],
                    dynamic_ratio=motion_steps[step_idx - 1]['dynamic_ratio'],
                    dynamic_gate=motion_steps[step_idx - 1]['dynamic_gate'],
                ))
            slots.append(aligned)
            slot_debug.append(step_records)
        # Match baseline sampled_bev ordering: slot0 is the newest history (t-1),
        # and later slots move farther back in time.
        slots = list(reversed(slots))
        slot_debug = list(reversed(slot_debug))
        sampled_bev_motion = torch.stack(slots, dim=1)
        return sampled_bev_motion, slot_debug

    def _forward_unified_motion(self,
                                voxel_semantics,
                                voxel_semantics_clean,
                                img_metas,
                                oracle_static_flow_seq,
                                oracle_static_flow_valid_seq,
                                oracle_dynamic_residual_flow_seq,
                                oracle_dynamic_residual_flow_valid_seq,
                                oracle_dynamic_mask_seq,
                                frame_reliability_map=None):
        batch_size, seq_len = voxel_semantics.shape[:2]
        if seq_len < 2:
            raise ValueError(f'Unified-motion route expects at least 2 frames, got {seq_len}')

        curr_target = voxel_semantics_clean[:, -1:]
        _, _, occ_h, occ_w, occ_z = curr_target.shape

        bev_seq = []
        shape_seq = []
        for frame_idx in range(seq_len):
            bev_k, shapes_k = self.forward_encoder(voxel_semantics[:, frame_idx:frame_idx + 1])
            bev_seq.append(bev_k)
            shape_seq.append(shapes_k)
        curr_bev = bev_seq[-1]
        curr_shapes = shape_seq[-1]
        clean_curr_bev, _ = self.forward_encoder(curr_target)

        motion_steps = []
        for step_idx in range(seq_len - 1):
            motion_steps.append(self._build_motion_step(
                oracle_static_flow_seq[:, step_idx],
                oracle_static_flow_valid_seq[:, step_idx],
                oracle_dynamic_residual_flow_seq[:, step_idx],
                oracle_dynamic_residual_flow_valid_seq[:, step_idx],
                oracle_dynamic_mask_seq[:, step_idx],
                curr_bev.shape[-2:],
                curr_bev.dtype,
            ))

        sampled_bev_motion, slot_debug = self._build_unified_history_bank(bev_seq, motion_steps)
        z_sampled, embed_loss, _ = self.vq(curr_bev, sampled_bev_motion, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        reliability = frame_reliability_map
        posterior_mean = bev_seq[0]
        posterior_cov = self._initial_covariance(
            batch_size, posterior_mean.shape[-2], posterior_mean.shape[-1],
            posterior_mean.dtype, posterior_mean.device)
        step_debug = []
        last_prior_mean = posterior_mean
        last_prior_cov = posterior_cov
        last_gain_group = posterior_cov.new_zeros(posterior_cov.shape)
        last_motion_step = motion_steps[0]

        for step_idx in range(1, seq_len):
            motion_step = motion_steps[step_idx - 1]
            reliability_latent = None
            if reliability is not None:
                reliability_latent = self._pool_current_map(
                    reliability[:, step_idx:step_idx + 1],
                    posterior_mean.shape[-2:],
                )
            prior_mean, prior_cov, process_noise, static_prior, dynamic_prior, total_flow = self._predict_kalman_state_unified(
                posterior_mean, posterior_cov, motion_step)
            posterior_mean, posterior_cov, gain_group, observation_noise = self._update_kalman_state(
                prior_mean, prior_cov, bev_seq[step_idx], reliability_latent=reliability_latent)
            last_prior_mean = prior_mean
            last_prior_cov = prior_cov
            last_gain_group = gain_group
            last_motion_step = motion_step
            step_debug.append(dict(
                observation_bev=bev_seq[step_idx],
                reliability_latent=reliability_latent,
                process_noise=process_noise,
                observation_noise=observation_noise,
                prior_mean=prior_mean,
                posterior_mean=posterior_mean,
                gain_group=gain_group,
                static_prior=static_prior,
                dynamic_prior=dynamic_prior,
                total_flow=total_flow,
                static_flow=motion_step['static_flow'],
                dynamic_residual_flow=motion_step['dynamic_residual_flow'],
                static_valid_ratio=motion_step['static_valid_ratio'],
                dynamic_residual_valid_ratio=motion_step['dynamic_residual_valid_ratio'],
                dynamic_ratio=motion_step['dynamic_ratio'],
                dynamic_gate=motion_step['dynamic_gate'],
            ))

        debug = dict(
            bev_seq=bev_seq,
            shape_seq=shape_seq,
            curr_bev=curr_bev,
            curr_shapes=curr_shapes,
            clean_curr_bev=clean_curr_bev,
            sampled_bev_motion=sampled_bev_motion,
            slot_debug=slot_debug,
            motion_steps=motion_steps,
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            last_prior_mean=last_prior_mean,
            last_prior_cov=last_prior_cov,
            last_gain_group=last_gain_group,
            last_motion_step=last_motion_step,
            step_debug=step_debug,
        )
        return logits, embed_loss, curr_target, debug

    def forward_train(self,
                      voxel_semantics,
                      img_metas,
                      oracle_static_flow_seq,
                      oracle_static_flow_valid_seq,
                      oracle_dynamic_residual_flow_seq,
                      oracle_dynamic_residual_flow_valid_seq,
                      oracle_dynamic_mask_seq,
                      frame_reliability_map=None,
                      voxel_semantics_clean=None,
                      **kwargs):
        batch_size = len(img_metas)
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        static_flow_seq = self._normalize_flow_sequence_tensor(oracle_static_flow_seq, batch_size, has_vec=True)
        static_valid_seq = self._normalize_flow_sequence_tensor(oracle_static_flow_valid_seq, batch_size, has_vec=False)
        dynamic_residual_flow_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_seq, batch_size, has_vec=True)
        dynamic_residual_valid_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        reliability = self.normalize_batched_tensor(frame_reliability_map, batch_size) if frame_reliability_map is not None else None

        logits, embed_loss, curr_target, debug = self._forward_unified_motion(
            seq,
            clean_seq,
            img_metas,
            static_flow_seq,
            static_valid_seq,
            dynamic_residual_flow_seq,
            dynamic_residual_valid_seq,
            dynamic_seq,
            frame_reliability_map=reliability,
        )

        losses = dict()
        losses.update(self.reconstruct_loss(logits, curr_target))
        losses['embed_loss'] = self.embed_loss_weight * embed_loss

        if self.kalman_posterior_aux_loss_weight > 0:
            current_reliability = reliability[:, -1] if reliability is not None else None
            bad_mask = self._derive_bad_mask_latent(
                seq[:, -1],
                clean_seq[:, -1],
                current_reliability,
                debug['posterior_mean'].shape[-2:],
            )
            dynamic_mask = (debug['last_motion_step']['dynamic_gate'] > 0.0).float()
            posterior_mask = torch.clamp(
                bad_mask * (1.0 + self.kalman_posterior_dynamic_boost * dynamic_mask),
                min=0.0,
                max=1.0,
            )
            posterior_aux = self._masked_latent_loss(
                debug['posterior_mean'],
                debug['clean_curr_bev'],
                posterior_mask,
            )
            losses['kalman_posterior_aux_loss'] = posterior_aux * self.kalman_posterior_aux_loss_weight

        losses['kalman_q_mean'] = debug['step_debug'][-1]['process_noise'].mean().detach()
        losses['kalman_r_mean'] = debug['step_debug'][-1]['observation_noise'].mean().detach()
        losses['kalman_gain_mean'] = debug['last_gain_group'].mean().detach()
        losses['motion_dynamic_gate_mean'] = debug['last_motion_step']['dynamic_gate'].mean().detach()
        return losses

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_static_flow_seq,
                     oracle_static_flow_valid_seq,
                     oracle_dynamic_residual_flow_seq,
                     oracle_dynamic_residual_flow_valid_seq,
                     oracle_dynamic_mask_seq,
                     frame_reliability_map=None,
                     voxel_semantics_clean=None,
                     **kwargs):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        batch_size = len(img_metas)
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        static_flow_seq = self._normalize_flow_sequence_tensor(oracle_static_flow_seq, batch_size, has_vec=True)
        static_valid_seq = self._normalize_flow_sequence_tensor(oracle_static_flow_valid_seq, batch_size, has_vec=False)
        dynamic_residual_flow_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_seq, batch_size, has_vec=True)
        dynamic_residual_valid_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        reliability = self.normalize_batched_tensor(frame_reliability_map, batch_size) if frame_reliability_map is not None else None

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        static_flow_seq = self._to_model_device(static_flow_seq, dtype=torch.float32)
        static_valid_seq = self._to_model_device(static_valid_seq)
        dynamic_residual_flow_seq = self._to_model_device(dynamic_residual_flow_seq, dtype=torch.float32)
        dynamic_residual_valid_seq = self._to_model_device(dynamic_residual_valid_seq)
        dynamic_seq = self._to_model_device(dynamic_seq)
        reliability = self._to_model_device(reliability, dtype=torch.float32)

        logits, _, curr_target, debug = self._forward_unified_motion(
            seq,
            clean_seq,
            img_metas,
            static_flow_seq,
            static_valid_seq,
            dynamic_residual_flow_seq,
            dynamic_residual_valid_seq,
            dynamic_seq,
            frame_reliability_map=reliability,
        )

        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        result = [dict(
            semantics=pred,
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=seq[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
            time=0.0,
        )]
        return result

    @torch.no_grad()
    def collect_unified_motion_debug(self,
                                     voxel_semantics,
                                     voxel_semantics_clean,
                                     img_metas,
                                     oracle_static_flow_seq,
                                     oracle_static_flow_valid_seq,
                                     oracle_dynamic_residual_flow_seq,
                                     oracle_dynamic_residual_flow_valid_seq,
                                     oracle_dynamic_mask_seq,
                                     frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        static_flow_seq = self._normalize_flow_sequence_tensor(oracle_static_flow_seq, batch_size, has_vec=True)
        static_valid_seq = self._normalize_flow_sequence_tensor(oracle_static_flow_valid_seq, batch_size, has_vec=False)
        dynamic_residual_flow_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_seq, batch_size, has_vec=True)
        dynamic_residual_valid_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        reliability = self.normalize_batched_tensor(frame_reliability_map, batch_size) if frame_reliability_map is not None else None

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        static_flow_seq = self._to_model_device(static_flow_seq, dtype=torch.float32)
        static_valid_seq = self._to_model_device(static_valid_seq)
        dynamic_residual_flow_seq = self._to_model_device(dynamic_residual_flow_seq, dtype=torch.float32)
        dynamic_residual_valid_seq = self._to_model_device(dynamic_residual_valid_seq)
        dynamic_seq = self._to_model_device(dynamic_seq)
        reliability = self._to_model_device(reliability, dtype=torch.float32)

        logits, _, curr_target, debug = self._forward_unified_motion(
            seq,
            clean_seq,
            img_metas,
            static_flow_seq,
            static_valid_seq,
            dynamic_residual_flow_seq,
            dynamic_residual_valid_seq,
            dynamic_seq,
            frame_reliability_map=reliability,
        )

        input_shape = (batch_size, 1) + tuple(curr_target.shape[-3:])

        def _decode_pred(bev_feature, shapes):
            decode_out = self._decode_bev_feature(bev_feature, shapes, input_shape)
            if isinstance(decode_out, dict):
                pred = decode_out['y_pred']
            else:
                pred = decode_out
            if pred.shape[-1] > 1:
                pred = pred.softmax(dim=-1).argmax(dim=-1)
            return pred.detach().cpu().numpy()

        def _decode_slot_bank(slot_bank, shapes):
            preds = []
            for slot_idx in range(slot_bank.shape[1]):
                preds.append(_decode_pred(slot_bank[:, slot_idx], shapes))
            return np.stack(preds, axis=1)

        step_decode_debug = []
        for step_idx, step in enumerate(debug['step_debug'], start=1):
            step_decode_debug.append(dict(
                observation_decode_pred=_decode_pred(step['observation_bev'], debug['shape_seq'][step_idx]),
                prior_decode_pred=_decode_pred(step['prior_mean'], debug['shape_seq'][step_idx]),
                posterior_decode_pred=_decode_pred(step['posterior_mean'], debug['shape_seq'][step_idx]),
                static_prior_decode_pred=_decode_pred(step['static_prior'], debug['shape_seq'][step_idx]),
                dynamic_prior_decode_pred=_decode_pred(step['dynamic_prior'], debug['shape_seq'][step_idx]),
            ))

        clean_curr_bev = debug['clean_curr_bev']
        latent_metrics = dict(
            obs_l1=float(F.l1_loss(debug['curr_bev'], clean_curr_bev).item()),
            prior_l1=float(F.l1_loss(debug['last_prior_mean'], clean_curr_bev).item()),
            post_l1=float(F.l1_loss(debug['posterior_mean'], clean_curr_bev).item()),
            obs_cos=float(F.cosine_similarity(debug['curr_bev'].flatten(1), clean_curr_bev.flatten(1), dim=1).mean().item()),
            prior_cos=float(F.cosine_similarity(debug['last_prior_mean'].flatten(1), clean_curr_bev.flatten(1), dim=1).mean().item()),
            post_cos=float(F.cosine_similarity(debug['posterior_mean'].flatten(1), clean_curr_bev.flatten(1), dim=1).mean().item()),
        )

        return dict(
            voxel_semantics_seq=seq.detach().cpu().numpy().astype(np.uint8),
            voxel_semantics_clean_seq=clean_seq.detach().cpu().numpy().astype(np.uint8),
            curr_input=seq[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            unified_motion_pred=logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            init_decode_pred=_decode_pred(debug['bev_seq'][0], debug['shape_seq'][0]).astype(np.uint8),
            curr_decode_pred=_decode_pred(debug['curr_bev'], debug['curr_shapes']).astype(np.uint8),
            last_prior_decode_pred=_decode_pred(debug['last_prior_mean'], debug['curr_shapes']).astype(np.uint8),
            posterior_decode_pred=_decode_pred(debug['posterior_mean'], debug['curr_shapes']).astype(np.uint8),
            sampled_bev_motion_decode_pred=_decode_slot_bank(debug['sampled_bev_motion'], debug['curr_shapes']).astype(np.uint8),
            sampled_bev_motion=debug['sampled_bev_motion'].detach().cpu().numpy(),
            curr_bev=debug['curr_bev'].detach().cpu().numpy(),
            clean_curr_bev=clean_curr_bev.detach().cpu().numpy(),
            last_prior_mean=debug['last_prior_mean'].detach().cpu().numpy(),
            posterior_mean=debug['posterior_mean'].detach().cpu().numpy(),
            last_gain_group=debug['last_gain_group'].detach().cpu().numpy(),
            latent_metrics=latent_metrics,
            slot_debug=[
                [
                    dict(
                        target_step=record['target_step'],
                        static_warp=record['static_warp'].detach().cpu().numpy(),
                        dynamic_warp=record['dynamic_warp'].detach().cpu().numpy(),
                        aligned=record['aligned'].detach().cpu().numpy(),
                        total_flow=record['total_flow'].detach().cpu().numpy(),
                        static_flow=record['static_flow'].detach().cpu().numpy(),
                        dynamic_residual_flow=record['dynamic_residual_flow'].detach().cpu().numpy(),
                        static_valid_ratio=record['static_valid_ratio'].detach().cpu().numpy(),
                        dynamic_residual_valid_ratio=record['dynamic_residual_valid_ratio'].detach().cpu().numpy(),
                        dynamic_ratio=record['dynamic_ratio'].detach().cpu().numpy(),
                        dynamic_gate=record['dynamic_gate'].detach().cpu().numpy(),
                    )
                    for record in slot_records
                ]
                for slot_records in debug['slot_debug']
            ],
            step_debug=[
                dict(
                    observation_bev=step['observation_bev'].detach().cpu().numpy(),
                    prior_mean=step['prior_mean'].detach().cpu().numpy(),
                    posterior_mean=step['posterior_mean'].detach().cpu().numpy(),
                    static_prior=step['static_prior'].detach().cpu().numpy(),
                    dynamic_prior=step['dynamic_prior'].detach().cpu().numpy(),
                    total_flow=step['total_flow'].detach().cpu().numpy(),
                    static_flow=step['static_flow'].detach().cpu().numpy(),
                    dynamic_residual_flow=step['dynamic_residual_flow'].detach().cpu().numpy(),
                    static_valid_ratio=step['static_valid_ratio'].detach().cpu().numpy(),
                    dynamic_residual_valid_ratio=step['dynamic_residual_valid_ratio'].detach().cpu().numpy(),
                    dynamic_ratio=step['dynamic_ratio'].detach().cpu().numpy(),
                    dynamic_gate=step['dynamic_gate'].detach().cpu().numpy(),
                    process_noise=step['process_noise'].detach().cpu().numpy(),
                    observation_noise=step['observation_noise'].detach().cpu().numpy(),
                    gain_group=step['gain_group'].detach().cpu().numpy(),
                )
                for step in debug['step_debug']
            ],
            step_decode_debug=step_decode_debug,
            oracle_static_flow_seq=static_flow_seq.detach().cpu().numpy(),
            oracle_static_flow_valid_seq=static_valid_seq.detach().cpu().numpy(),
            oracle_dynamic_residual_flow_seq=dynamic_residual_flow_seq.detach().cpu().numpy(),
            oracle_dynamic_residual_flow_valid_seq=dynamic_residual_valid_seq.detach().cpu().numpy(),
            oracle_dynamic_mask_seq=dynamic_seq.detach().cpu().numpy(),
        )

    def train(self, mode=True):
        super().train(mode)
        self.kalman_state_bev = None
        self.kalman_state_cov = None
        return self
