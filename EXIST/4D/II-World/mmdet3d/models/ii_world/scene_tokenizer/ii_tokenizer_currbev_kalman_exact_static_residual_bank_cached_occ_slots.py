import time

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_exact_static_residual_bank_cached import (
    IISceneTokenizerCurrBevKalmanExactStaticResidualBankCached,
)


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanExactStaticResidualBankCachedOccSlots(
        IISceneTokenizerCurrBevKalmanExactStaticResidualBankCached):
    """Cached history bank built in occupancy space, then encoded into sampled_bev slots.

    Main idea:
    - keep the current branch exactly baseline-like: `curr_occ -> encoder -> curr_bev`
    - replace latent-space history transport with occ-space transport:
      `history_occ -> aligned_hist_occ -> encoder -> sampled_bev`
    - keep the parent class' explicit-sequence / Kalman route intact
    """

    DYNAMIC_OCC_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)

    def __init__(self,
                 occ_free_cls=17,
                 occ_slot_use_clean_cache=False,
                 profile_runtime=False,
                 profile_runtime_interval=10,
                 **kwargs):
        self.occ_free_cls = occ_free_cls
        self.occ_slot_use_clean_cache = occ_slot_use_clean_cache
        self.profile_runtime = profile_runtime
        self.profile_runtime_interval = max(1, int(profile_runtime_interval))
        super().__init__(**kwargs)
        self.motion_history_occ = None
        self.motion_history_occ_valid = None
        self._occ_grid_cache = {}
        self._profile_step = 0
        dynamic_lut = torch.zeros(256, dtype=torch.bool)
        for cls_id in self.DYNAMIC_OCC_CLASS_IDS:
            dynamic_lut[cls_id] = True
        self.register_buffer('dynamic_occ_lut', dynamic_lut, persistent=False)

    def _clear_motion_history(self):
        super()._clear_motion_history()
        self.motion_history_occ = None
        self.motion_history_occ_valid = None

    def _dynamic_occ_mask(self, occ):
        if occ.dim() == 5 and occ.shape[1] == 1:
            occ = occ[:, 0]
        occ_long = occ.long().clamp_(0, self.dynamic_occ_lut.numel() - 1)
        return self.dynamic_occ_lut[occ_long]

    def _get_occ_grid(self, occ_h, occ_w, occ_z, device, dtype):
        key = (str(device), str(dtype), occ_h, occ_w, occ_z)
        cached = self._occ_grid_cache.get(key)
        if cached is None:
            cached = torch.meshgrid(
                torch.arange(occ_h, device=device, dtype=dtype),
                torch.arange(occ_w, device=device, dtype=dtype),
                torch.arange(occ_z, device=device, dtype=dtype),
                indexing='ij',
            )
            self._occ_grid_cache[key] = cached
        return cached

    def _profile_stamp(self, enabled):
        if not enabled:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=torch.cuda.current_device())
        return time.perf_counter()

    def _profile_elapsed_ms(self, start, enabled):
        if not enabled or start is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=torch.cuda.current_device())
        return (time.perf_counter() - start) * 1000.0

    def _next_profile_flag(self):
        if not self.profile_runtime:
            return False
        self._profile_step += 1
        return (self._profile_step % self.profile_runtime_interval) == 0

    def _zero_occ_motion(self, batch_size, occ_shape, device):
        occ_h, occ_w, occ_z = occ_shape
        zero_flow = torch.zeros((batch_size, occ_h, occ_w, occ_z, 3), device=device, dtype=torch.float32)
        zero_valid = torch.zeros((batch_size, occ_h, occ_w, occ_z), device=device, dtype=torch.float32)
        return dict(
            static_flow=zero_flow,
            static_valid=zero_valid,
            static_flow_forward=zero_flow.clone(),
            static_valid_forward=zero_valid.clone(),
            dynamic_residual_flow=zero_flow.clone(),
            dynamic_residual_valid=zero_valid.clone(),
            dynamic_residual_flow_forward=zero_flow.clone(),
            dynamic_residual_valid_forward=zero_valid.clone(),
        )

    def _normalize_occ_motion_inputs(self,
                                     batch_size,
                                     seq,
                                     oracle_static_flow,
                                     oracle_static_flow_valid,
                                     oracle_static_flow_forward,
                                     oracle_static_flow_valid_forward,
                                     oracle_dynamic_residual_flow,
                                     oracle_dynamic_residual_flow_valid,
                                     oracle_dynamic_residual_flow_forward,
                                     oracle_dynamic_residual_flow_valid_forward):
        if oracle_dynamic_residual_flow is None or oracle_static_flow is None:
            return self._zero_occ_motion(batch_size, tuple(seq.shape[-3:]), seq.device)

        def _norm(x, dtype):
            return self._to_model_device(self.normalize_batched_tensor(x, batch_size), dtype=dtype)

        static_flow = _norm(oracle_static_flow, torch.float32)
        static_valid = _norm(oracle_static_flow_valid, torch.float32)
        static_flow_forward = _norm(oracle_static_flow_forward, torch.float32)
        static_valid_forward = _norm(oracle_static_flow_valid_forward, torch.float32)
        dynamic_flow = _norm(oracle_dynamic_residual_flow, torch.float32)
        dynamic_valid = _norm(oracle_dynamic_residual_flow_valid, torch.float32)
        dynamic_flow_forward = _norm(oracle_dynamic_residual_flow_forward, torch.float32)
        dynamic_valid_forward = _norm(oracle_dynamic_residual_flow_valid_forward, torch.float32)
        return dict(
            static_flow=static_flow,
            static_valid=static_valid,
            static_flow_forward=static_flow_forward,
            static_valid_forward=static_valid_forward,
            dynamic_residual_flow=dynamic_flow,
            dynamic_residual_valid=dynamic_valid,
            dynamic_residual_flow_forward=dynamic_flow_forward,
            dynamic_residual_valid_forward=dynamic_valid_forward,
        )

    def _warp_occ_nearest(self, prev_occ, backward_flow, valid_mask, source_known_mask=None):
        batch_size, occ_h, occ_w, occ_z = prev_occ.shape
        device = prev_occ.device
        dtype = backward_flow.dtype

        yy, xx, zz = self._get_occ_grid(occ_h, occ_w, occ_z, device, dtype)
        src_y = torch.round(yy.unsqueeze(0) + backward_flow[..., 0]).long()
        src_x = torch.round(xx.unsqueeze(0) + backward_flow[..., 1]).long()
        src_z = torch.round(zz.unsqueeze(0) + backward_flow[..., 2]).long()

        in_bounds = (
            (src_y >= 0) & (src_y < occ_h) &
            (src_x >= 0) & (src_x < occ_w) &
            (src_z >= 0) & (src_z < occ_z)
        )
        sample_valid = in_bounds & (valid_mask > 0.5)

        src_y = src_y.clamp(0, occ_h - 1)
        src_x = src_x.clamp(0, occ_w - 1)
        src_z = src_z.clamp(0, occ_z - 1)
        flat_idx = (src_y * (occ_w * occ_z) + src_x * occ_z + src_z).view(batch_size, -1)
        prev_flat = prev_occ.view(batch_size, -1)
        sampled = torch.gather(prev_flat, 1, flat_idx)
        if source_known_mask is not None:
            source_known_flat = source_known_mask.view(batch_size, -1)
            gathered_known = torch.gather(source_known_flat, 1, flat_idx).view(batch_size, occ_h, occ_w, occ_z)
            sample_valid = sample_valid & gathered_known
        out = torch.full_like(prev_flat, fill_value=self.occ_free_cls)
        valid_flat = sample_valid.view(batch_size, -1)
        out[valid_flat] = sampled[valid_flat]
        return out.view(batch_size, occ_h, occ_w, occ_z), sample_valid

    def _scatter_occ_forward(self, prev_occ, prev_dyn, prev_valid, forward_flow, valid_mask):
        batch_size, occ_h, occ_w, occ_z = prev_occ.shape
        device = prev_occ.device
        dtype = forward_flow.dtype
        yy, xx, zz = self._get_occ_grid(occ_h, occ_w, occ_z, device, dtype)
        dst_y = torch.round(yy.unsqueeze(0) + forward_flow[..., 0]).long()
        dst_x = torch.round(xx.unsqueeze(0) + forward_flow[..., 1]).long()
        dst_z = torch.round(zz.unsqueeze(0) + forward_flow[..., 2]).long()
        in_bounds = (
            (dst_y >= 0) & (dst_y < occ_h) &
            (dst_x >= 0) & (dst_x < occ_w) &
            (dst_z >= 0) & (dst_z < occ_z)
        )
        src_valid = prev_dyn & prev_valid & (valid_mask > 0.5) & (prev_occ != self.occ_free_cls) & in_bounds
        out = torch.full_like(prev_occ, fill_value=self.occ_free_cls)
        out_dyn = torch.zeros_like(prev_dyn, dtype=torch.bool)
        for batch_idx in range(batch_size):
            valid = src_valid[batch_idx]
            if not valid.any().item():
                continue
            out[batch_idx, dst_y[batch_idx][valid], dst_x[batch_idx][valid], dst_z[batch_idx][valid]] = prev_occ[batch_idx][valid]
            out_dyn[batch_idx, dst_y[batch_idx][valid], dst_x[batch_idx][valid], dst_z[batch_idx][valid]] = True
        return out, out_dyn

    def _pool_occ_valid_to_latent_mask(self, sampled_occ_valid, out_hw):
        batch_size, num_slots, occ_h, occ_w, occ_z = sampled_occ_valid.shape
        out_h, out_w = out_hw
        valid = sampled_occ_valid.to(dtype=torch.float32)
        valid = valid.permute(0, 1, 4, 2, 3).reshape(batch_size * num_slots, 1, occ_z, occ_h, occ_w)
        kernel_h = max(1, occ_h // out_h)
        kernel_w = max(1, occ_w // out_w)
        pooled = F.avg_pool3d(
            valid,
            kernel_size=(occ_z, kernel_h, kernel_w),
            stride=(occ_z, kernel_h, kernel_w),
        ).squeeze(2)
        pooled = (pooled > 0.0).to(dtype=self.class_embeds.weight.dtype)
        return pooled.reshape(batch_size, num_slots, 1, out_h, out_w)

    def _apply_occ_motion(self, prev_occ, prev_valid, motion_step, collect_debug=False):
        prev_dyn = self._dynamic_occ_mask(prev_occ) & prev_valid
        static_flow = motion_step['static_flow']
        static_valid = motion_step['static_valid']
        if self.unified_motion_dynamic_transport == 'scatter':
            static_source = prev_occ.masked_fill(prev_dyn, self.occ_free_cls)
            static_branch, static_valid_mask = self._warp_occ_nearest(
                static_source,
                static_flow,
                static_valid,
                source_known_mask=prev_valid & (~prev_dyn),
            )
            forward_total_flow = motion_step['static_flow_forward'] + motion_step['dynamic_residual_flow_forward']
            forward_total_valid = motion_step['static_valid_forward'] * motion_step['dynamic_residual_valid_forward']
            dynamic_occ, dynamic_mask = self._scatter_occ_forward(
                prev_occ,
                prev_dyn,
                prev_valid,
                forward_total_flow,
                forward_total_valid,
            )
            aligned = torch.where(dynamic_mask, dynamic_occ, static_branch)
            aligned_valid = dynamic_mask | static_valid_mask
            if not collect_debug:
                return aligned, aligned_valid, None, None, None, None, None
            dynamic_branch = dynamic_occ
        else:
            total_flow = static_flow + motion_step['dynamic_residual_flow']
            total_valid = static_valid * motion_step['dynamic_residual_valid']
            static_branch, static_valid_mask = self._warp_occ_nearest(prev_occ, static_flow, static_valid)
            aligned, aligned_valid = self._warp_occ_nearest(prev_occ, total_flow, total_valid)
            if not collect_debug:
                return aligned, aligned_valid, None, None, None, None, None
            dynamic_branch = aligned
            dynamic_mask = self._dynamic_occ_mask(aligned) & aligned_valid
        return aligned, aligned_valid, static_branch, dynamic_branch, dynamic_mask, static_valid_mask, dynamic_mask

    def _align_cached_history_occ_with_motion(self, curr_occ, motion_step, img_metas, collect_debug=False):
        batch_size, _, occ_h, occ_w, occ_z = curr_occ.shape
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        start_mask = torch.as_tensor(start_of_sequence, device=curr_occ.device, dtype=torch.bool)
        curr_occ_noc = curr_occ[:, 0].to(torch.uint8)
        curr_occ_valid = torch.ones_like(curr_occ_noc, dtype=torch.bool)

        need_reinit = (
            self.motion_history_occ is None or
            self.motion_history_occ_valid is None or
            self.motion_history_occ.shape != (batch_size, self.frame_number, occ_h, occ_w, occ_z) or
            self.motion_history_occ_valid.shape != (batch_size, self.frame_number, occ_h, occ_w, occ_z)
        )
        if need_reinit:
            self.motion_history_occ = curr_occ_noc.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).detach().clone()
            self.motion_history_occ_valid = curr_occ_valid.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).detach().clone()

        if start_mask.any():
            self.motion_history_occ[start_mask] = curr_occ_noc[start_mask].unsqueeze(1).repeat(
                1, self.frame_number, 1, 1, 1)
            self.motion_history_occ_valid[start_mask] = curr_occ_valid[start_mask].unsqueeze(1).repeat(
                1, self.frame_number, 1, 1, 1)

        aligned_slots = []
        aligned_valid_slots = []
        slot_debug = []
        for slot_idx in range(self.frame_number):
            slot_occ = self.motion_history_occ[:, slot_idx]
            slot_valid = self.motion_history_occ_valid[:, slot_idx]
            aligned_slot, aligned_valid_slot, static_slot, dynamic_slot, dynamic_mask_slot, static_valid_slot, dynamic_valid_slot = self._apply_occ_motion(
                slot_occ, slot_valid, motion_step, collect_debug=collect_debug)
            if start_mask.any():
                aligned_slot[start_mask] = curr_occ_noc[start_mask]
                aligned_valid_slot[start_mask] = True
                if collect_debug:
                    static_slot[start_mask] = curr_occ_noc[start_mask]
                    dynamic_slot[start_mask] = curr_occ_noc[start_mask]
                    dynamic_mask_slot[start_mask] = False
                    static_valid_slot[start_mask] = True
                    dynamic_valid_slot[start_mask] = False
            aligned_slots.append(aligned_slot)
            aligned_valid_slots.append(aligned_valid_slot)
            if collect_debug:
                slot_debug.append([
                    dict(
                        target_step=-1,
                        static_occ=static_slot,
                        dynamic_occ=dynamic_slot,
                        aligned_occ=aligned_slot,
                        aligned_valid=aligned_valid_slot,
                        static_valid=static_valid_slot,
                        dynamic_valid=dynamic_valid_slot,
                        dynamic_mask=dynamic_mask_slot,
                    )
                ])

        aligned_occ = torch.stack(aligned_slots, dim=1)
        aligned_valid = torch.stack(aligned_valid_slots, dim=1)
        self.motion_history_occ = torch.cat(
            [curr_occ_noc.unsqueeze(1), aligned_occ[:, :-1]], dim=1).detach().clone()
        self.motion_history_occ_valid = torch.cat(
            [curr_occ_valid.unsqueeze(1), aligned_valid[:, :-1]], dim=1).detach().clone()

        if not collect_debug:
            return aligned_occ, aligned_valid, None
        return aligned_occ, aligned_valid, slot_debug

    def _encode_occ_slot_bank(self, sampled_occ_motion, sampled_occ_valid=None):
        batch_size, num_slots, occ_h, occ_w, occ_z = sampled_occ_motion.shape
        flat_occ = sampled_occ_motion.reshape(batch_size * num_slots, 1, occ_h, occ_w, occ_z).long()
        slot_bev, _ = self.forward_encoder(flat_occ)
        slot_bev = slot_bev.reshape(batch_size, num_slots, *slot_bev.shape[1:])
        if sampled_occ_valid is None:
            return slot_bev
        latent_valid = self._pool_occ_valid_to_latent_mask(sampled_occ_valid, slot_bev.shape[-2:])
        return slot_bev * latent_valid

    def _forward_cached_current(self,
                                voxel_semantics,
                                voxel_semantics_clean,
                                img_metas,
                                oracle_static_flow,
                                oracle_static_flow_valid,
                                oracle_static_flow_forward,
                                oracle_static_flow_valid_forward,
                                oracle_dynamic_residual_flow,
                                oracle_dynamic_residual_flow_valid,
                                oracle_dynamic_residual_flow_forward=None,
                                oracle_dynamic_residual_flow_valid_forward=None,
                                oracle_dynamic_mask_source=None,
                                collect_debug=False,
                                profile_runtime=False):
        if voxel_semantics.dim() == 4:
            voxel_semantics = voxel_semantics.unsqueeze(1)
        if voxel_semantics_clean.dim() == 4:
            voxel_semantics_clean = voxel_semantics_clean.unsqueeze(1)

        batch_size = voxel_semantics.shape[0]
        curr_input = voxel_semantics[:, -1:]
        curr_target = voxel_semantics_clean[:, -1:]
        _, _, occ_h, occ_w, occ_z = curr_target.shape

        total_start = self._profile_stamp(profile_runtime)
        stamp = self._profile_stamp(profile_runtime)
        curr_bev, curr_shapes = self.forward_encoder(curr_input)
        curr_encode_ms = self._profile_elapsed_ms(stamp, profile_runtime)

        motion_step = dict(
            static_flow=oracle_static_flow.to(torch.float32),
            static_valid=oracle_static_flow_valid.to(torch.float32),
            static_flow_forward=oracle_static_flow_forward.to(torch.float32),
            static_valid_forward=oracle_static_flow_valid_forward.to(torch.float32),
            dynamic_residual_flow=oracle_dynamic_residual_flow.to(torch.float32),
            dynamic_residual_valid=oracle_dynamic_residual_flow_valid.to(torch.float32),
            dynamic_residual_flow_forward=oracle_dynamic_residual_flow_forward.to(torch.float32),
            dynamic_residual_valid_forward=oracle_dynamic_residual_flow_valid_forward.to(torch.float32),
        )

        cache_occ = curr_target if self.occ_slot_use_clean_cache else curr_input
        stamp = self._profile_stamp(profile_runtime)
        sampled_occ_motion, sampled_occ_valid, occ_slot_debug = self._align_cached_history_occ_with_motion(
            cache_occ, motion_step, img_metas, collect_debug=collect_debug)
        occ_align_ms = self._profile_elapsed_ms(stamp, profile_runtime)
        stamp = self._profile_stamp(profile_runtime)
        if collect_debug:
            sampled_bev_motion = self._encode_occ_slot_bank(sampled_occ_motion, sampled_occ_valid)
        else:
            with torch.no_grad():
                sampled_bev_motion = self._encode_occ_slot_bank(sampled_occ_motion, sampled_occ_valid)
        slot_encode_ms = self._profile_elapsed_ms(stamp, profile_runtime)
        if not collect_debug:
            del curr_input
            del sampled_occ_motion
            del sampled_occ_valid
            del occ_slot_debug
            del cache_occ
            del motion_step
            del oracle_static_flow
            del oracle_static_flow_valid
            del oracle_static_flow_forward
            del oracle_static_flow_valid_forward
            del oracle_dynamic_residual_flow
            del oracle_dynamic_residual_flow_valid
            del oracle_dynamic_residual_flow_forward
            del oracle_dynamic_residual_flow_valid_forward

        stamp = self._profile_stamp(profile_runtime)
        z_sampled, embed_loss, _ = self.vq(curr_bev, sampled_bev_motion, is_voxel=False)
        vq_ms = self._profile_elapsed_ms(stamp, profile_runtime)
        stamp = self._profile_stamp(profile_runtime)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))
        decoder_ms = self._profile_elapsed_ms(stamp, profile_runtime)
        total_model_ms = self._profile_elapsed_ms(total_start, profile_runtime)

        debug = None
        if collect_debug:
            clean_curr_bev, _ = self.forward_encoder(curr_target)
            debug = dict(
                curr_bev=curr_bev,
                curr_shapes=curr_shapes,
                clean_curr_bev=clean_curr_bev,
                sampled_occ_motion=sampled_occ_motion,
                sampled_occ_valid=sampled_occ_valid,
                sampled_bev_motion=sampled_bev_motion,
                occ_slot_debug=occ_slot_debug,
                motion_step=motion_step,
            )
        profile_stats = None
        if profile_runtime:
            profile_stats = dict(
                curr_encode_ms=curr_encode_ms,
                occ_align_ms=occ_align_ms,
                slot_encode_ms=slot_encode_ms,
                vq_ms=vq_ms,
                decoder_ms=decoder_ms,
                model_total_ms=total_model_ms,
            )
        return logits, embed_loss, curr_target, debug, profile_stats

    def forward_train(self,
                      voxel_semantics,
                      img_metas,
                      oracle_static_flow=None,
                      oracle_static_flow_valid=None,
                      oracle_static_flow_forward=None,
                      oracle_static_flow_valid_forward=None,
                      oracle_dynamic_residual_flow=None,
                      oracle_dynamic_residual_flow_valid=None,
                      oracle_dynamic_mask=None,
                      oracle_dynamic_mask_source=None,
                      oracle_dynamic_residual_flow_forward=None,
                      oracle_dynamic_residual_flow_valid_forward=None,
                      frame_reliability_map=None,
                      voxel_semantics_clean=None,
                      oracle_static_flow_seq=None,
                      oracle_static_flow_valid_seq=None,
                      oracle_dynamic_residual_flow_seq=None,
                      oracle_dynamic_residual_flow_valid_seq=None,
                      oracle_dynamic_mask_seq=None,
                      oracle_dynamic_source_mask_seq=None,
                      oracle_dynamic_residual_flow_forward_seq=None,
                      oracle_dynamic_residual_flow_valid_forward_seq=None,
                      **kwargs):
        batch_size = len(img_metas)
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        if seq.dim() == 4:
            seq = seq.unsqueeze(1)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)

        if oracle_dynamic_residual_flow_seq is not None and seq.shape[1] > 1:
            return super().forward_train(
                voxel_semantics=voxel_semantics,
                img_metas=img_metas,
                oracle_static_flow_seq=oracle_static_flow_seq,
                oracle_static_flow_valid_seq=oracle_static_flow_valid_seq,
                oracle_dynamic_residual_flow_seq=oracle_dynamic_residual_flow_seq,
                oracle_dynamic_residual_flow_valid_seq=oracle_dynamic_residual_flow_valid_seq,
                oracle_dynamic_residual_flow_forward_seq=oracle_dynamic_residual_flow_forward_seq,
                oracle_dynamic_residual_flow_valid_forward_seq=oracle_dynamic_residual_flow_valid_forward_seq,
                oracle_dynamic_mask_seq=oracle_dynamic_mask_seq,
                oracle_dynamic_source_mask_seq=oracle_dynamic_source_mask_seq,
                frame_reliability_map=frame_reliability_map,
                voxel_semantics_clean=voxel_semantics_clean,
                **kwargs,
            )

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        motion = self._normalize_occ_motion_inputs(
            batch_size,
            seq,
            oracle_static_flow,
            oracle_static_flow_valid,
            oracle_static_flow_forward,
            oracle_static_flow_valid_forward,
            oracle_dynamic_residual_flow,
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_residual_flow_forward,
            oracle_dynamic_residual_flow_valid_forward,
        )
        profile_runtime = self._next_profile_flag()
        logits, embed_loss, curr_target, _, profile_stats = self._forward_cached_current(
            seq,
            clean_seq,
            img_metas,
            motion['static_flow'],
            motion['static_valid'],
            motion['static_flow_forward'],
            motion['static_valid_forward'],
            motion['dynamic_residual_flow'],
            motion['dynamic_residual_valid'],
            motion['dynamic_residual_flow_forward'],
            motion['dynamic_residual_valid_forward'],
            oracle_dynamic_mask_source,
            profile_runtime=profile_runtime,
        )
        losses = dict()
        losses.update(self.reconstruct_loss(logits, curr_target))
        losses['embed_loss'] = self.embed_loss_weight * embed_loss
        if profile_stats is not None:
            for key, value in profile_stats.items():
                losses[f'time_{key}'] = logits.new_tensor(value)
        return losses

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_static_flow=None,
                     oracle_static_flow_valid=None,
                     oracle_static_flow_forward=None,
                     oracle_static_flow_valid_forward=None,
                     oracle_dynamic_residual_flow=None,
                     oracle_dynamic_residual_flow_valid=None,
                     oracle_dynamic_mask=None,
                     oracle_dynamic_mask_source=None,
                     oracle_dynamic_residual_flow_forward=None,
                     oracle_dynamic_residual_flow_valid_forward=None,
                     frame_reliability_map=None,
                     voxel_semantics_clean=None,
                     oracle_static_flow_seq=None,
                     oracle_static_flow_valid_seq=None,
                     oracle_dynamic_residual_flow_seq=None,
                     oracle_dynamic_residual_flow_valid_seq=None,
                     oracle_dynamic_mask_seq=None,
                     oracle_dynamic_source_mask_seq=None,
                     oracle_dynamic_residual_flow_forward_seq=None,
                     oracle_dynamic_residual_flow_valid_forward_seq=None,
                     **kwargs):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        batch_size = len(img_metas)
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        if seq.dim() == 4:
            seq = seq.unsqueeze(1)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)

        if oracle_dynamic_residual_flow_seq is not None and seq.shape[1] > 1:
            return super().forward_test(
                voxel_semantics=voxel_semantics,
                img_metas=img_metas,
                oracle_static_flow_seq=oracle_static_flow_seq,
                oracle_static_flow_valid_seq=oracle_static_flow_valid_seq,
                oracle_dynamic_residual_flow_seq=oracle_dynamic_residual_flow_seq,
                oracle_dynamic_residual_flow_valid_seq=oracle_dynamic_residual_flow_valid_seq,
                oracle_dynamic_residual_flow_forward_seq=oracle_dynamic_residual_flow_forward_seq,
                oracle_dynamic_residual_flow_valid_forward_seq=oracle_dynamic_residual_flow_valid_forward_seq,
                oracle_dynamic_mask_seq=oracle_dynamic_mask_seq,
                oracle_dynamic_source_mask_seq=oracle_dynamic_source_mask_seq,
                frame_reliability_map=frame_reliability_map,
                voxel_semantics_clean=voxel_semantics_clean,
                **kwargs,
            )

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        motion = self._normalize_occ_motion_inputs(
            batch_size,
            seq,
            oracle_static_flow,
            oracle_static_flow_valid,
            oracle_static_flow_forward,
            oracle_static_flow_valid_forward,
            oracle_dynamic_residual_flow,
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_residual_flow_forward,
            oracle_dynamic_residual_flow_valid_forward,
        )
        logits, _, curr_target, _, _ = self._forward_cached_current(
            seq,
            clean_seq,
            img_metas,
            motion['static_flow'],
            motion['static_valid'],
            motion['static_flow_forward'],
            motion['static_valid_forward'],
            motion['dynamic_residual_flow'],
            motion['dynamic_residual_valid'],
            motion['dynamic_residual_flow_forward'],
            motion['dynamic_residual_valid_forward'],
            oracle_dynamic_mask_source,
        )
        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        return [dict(
            semantics=pred,
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=seq[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
            time=0.0,
        )]

    @torch.no_grad()
    def collect_occ_slot_cached_debug(self,
                                      voxel_semantics,
                                      voxel_semantics_clean,
                                      img_metas,
                                      oracle_static_flow,
                                      oracle_static_flow_valid,
                                      oracle_static_flow_forward,
                                      oracle_static_flow_valid_forward,
                                      oracle_dynamic_residual_flow,
                                      oracle_dynamic_residual_flow_valid,
                                      oracle_dynamic_residual_flow_forward,
                                      oracle_dynamic_residual_flow_valid_forward):
        batch_size = voxel_semantics.shape[0]
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        if seq.dim() == 4:
            seq = seq.unsqueeze(1)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        motion = self._normalize_occ_motion_inputs(
            batch_size,
            seq,
            oracle_static_flow,
            oracle_static_flow_valid,
            oracle_static_flow_forward,
            oracle_static_flow_valid_forward,
            oracle_dynamic_residual_flow,
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_residual_flow_forward,
            oracle_dynamic_residual_flow_valid_forward,
        )
        self._clear_motion_history()
        logits, _, curr_target, debug, _ = self._forward_cached_current(
            seq,
            clean_seq,
            img_metas,
            motion['static_flow'],
            motion['static_valid'],
            motion['static_flow_forward'],
            motion['static_valid_forward'],
            motion['dynamic_residual_flow'],
            motion['dynamic_residual_valid'],
            motion['dynamic_residual_flow_forward'],
            motion['dynamic_residual_valid_forward'],
            None,
            collect_debug=True,
        )

        batch_size = seq.shape[0]
        input_shape = (batch_size, 1) + tuple(clean_seq[:, -1:].shape[-3:])

        def _decode_pred(bev_feature, shapes):
            decode_out = self._decode_bev_feature(bev_feature, shapes, input_shape)
            pred = decode_out['y_pred'] if isinstance(decode_out, dict) else decode_out
            if pred.shape[-1] > 1:
                pred = pred.softmax(dim=-1).argmax(dim=-1)
            return pred.detach().cpu().numpy().astype(np.uint8)

        def _decode_slot_bank(slot_bank, shapes):
            preds = []
            for slot_idx in range(slot_bank.shape[1]):
                preds.append(_decode_pred(slot_bank[:, slot_idx], shapes))
            return np.stack(preds, axis=1)

        curr_bev = debug['curr_bev']
        curr_shapes = debug['curr_shapes']
        sampled_bev_motion = debug['sampled_bev_motion']
        final_logits = self.forward_decoder(self.vq(curr_bev, sampled_bev_motion, is_voxel=False)[0], curr_shapes, input_shape)
        final_pred = final_logits.softmax(-1).argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)
        curr_decode_pred = _decode_pred(curr_bev, curr_shapes)

        return dict(
            voxel_semantics_seq=seq.detach().cpu().numpy().astype(np.uint8),
            voxel_semantics_clean_seq=clean_seq.detach().cpu().numpy().astype(np.uint8),
            curr_input=seq[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            curr_decode_pred=curr_decode_pred,
            final_pred=final_pred,
            sampled_occ_motion=debug['sampled_occ_motion'].detach().cpu().numpy().astype(np.uint8),
            sampled_occ_valid=debug['sampled_occ_valid'].detach().cpu().numpy(),
            sampled_occ_motion_dynamic_mask=self._dynamic_occ_mask(debug['sampled_occ_motion']).detach().cpu().numpy(),
            sampled_bev_motion=sampled_bev_motion.detach().cpu().numpy(),
            sampled_bev_motion_decode_pred=_decode_slot_bank(sampled_bev_motion, curr_shapes),
            occ_slot_debug=[
                [
                    dict(
                        target_step=record['target_step'],
                        static_occ=record['static_occ'].detach().cpu().numpy().astype(np.uint8),
                        dynamic_occ=record['dynamic_occ'].detach().cpu().numpy().astype(np.uint8),
                        aligned_occ=record['aligned_occ'].detach().cpu().numpy().astype(np.uint8),
                        aligned_valid=record['aligned_valid'].detach().cpu().numpy(),
                        static_valid=record['static_valid'].detach().cpu().numpy(),
                        dynamic_valid=record['dynamic_valid'].detach().cpu().numpy(),
                        dynamic_mask=record['dynamic_mask'].detach().cpu().numpy(),
                    )
                    for record in slot_records
                ]
                for slot_records in debug['occ_slot_debug']
            ],
        )

    @torch.no_grad()
    def collect_occ_slot_cached_sequence_debug(self,
                                               voxel_semantics,
                                               voxel_semantics_clean,
                                               img_metas,
                                               oracle_static_flow_seq,
                                               oracle_static_flow_valid_seq,
                                               oracle_static_flow_forward_seq,
                                               oracle_static_flow_valid_forward_seq,
                                               oracle_dynamic_residual_flow_seq,
                                               oracle_dynamic_residual_flow_valid_seq,
                                               oracle_dynamic_residual_flow_forward_seq,
                                               oracle_dynamic_residual_flow_valid_forward_seq):
        batch_size = voxel_semantics.shape[0]
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)

        static_flow_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_static_flow_seq, batch_size, has_vec=True), dtype=torch.float32)
        static_valid_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_static_flow_valid_seq, batch_size, has_vec=False), dtype=torch.float32)
        static_flow_forward_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_static_flow_forward_seq, batch_size, has_vec=True), dtype=torch.float32)
        static_valid_forward_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_static_flow_valid_forward_seq, batch_size, has_vec=False), dtype=torch.float32)
        dynamic_flow_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_seq, batch_size, has_vec=True), dtype=torch.float32)
        dynamic_valid_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_seq, batch_size, has_vec=False), dtype=torch.float32)
        dynamic_flow_forward_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_forward_seq, batch_size, has_vec=True), dtype=torch.float32)
        dynamic_valid_forward_seq = self._to_model_device(
            self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_forward_seq, batch_size, has_vec=False), dtype=torch.float32)

        self._clear_motion_history()
        curr_bev = None
        curr_shapes = None
        curr_target = clean_seq[:, -1:]
        sampled_occ_motion = None
        sampled_occ_valid = None
        sampled_bev_motion = None
        occ_slot_debug = None

        for step_idx in range(seq.shape[1]):
            curr_frame = seq[:, step_idx:step_idx + 1]
            curr_bev, curr_shapes = self.forward_encoder(curr_frame)
            step_metas = []
            for meta in img_metas:
                step_metas.append(dict(
                    start_of_sequence=(step_idx == 0),
                    scene_name=meta.get('scene_name', ''),
                    sample_idx=meta.get('sample_idx', ''),
                    index=meta.get('index', -1),
                ))

            cache_occ = clean_seq[:, step_idx:step_idx + 1] if self.occ_slot_use_clean_cache else curr_frame
            if step_idx == 0:
                curr_occ_noc = cache_occ[:, 0]
                self.motion_history_occ = curr_occ_noc.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).detach().clone()
                self.motion_history_occ_valid = torch.ones_like(self.motion_history_occ, dtype=torch.bool)
                sampled_occ_motion = self.motion_history_occ.clone()
                sampled_occ_valid = self.motion_history_occ_valid.clone()
                occ_slot_debug = [
                    [dict(
                        target_step=0,
                        static_occ=self.motion_history_occ[:, slot_idx],
                        dynamic_occ=self.motion_history_occ[:, slot_idx],
                        aligned_occ=self.motion_history_occ[:, slot_idx],
                        aligned_valid=sampled_occ_valid[:, slot_idx],
                        static_valid=sampled_occ_valid[:, slot_idx],
                        dynamic_valid=self._dynamic_occ_mask(self.motion_history_occ[:, slot_idx]),
                        dynamic_mask=self._dynamic_occ_mask(self.motion_history_occ[:, slot_idx]),
                    )]
                    for slot_idx in range(self.frame_number)
                ]
                sampled_bev_motion = self._encode_occ_slot_bank(sampled_occ_motion, sampled_occ_valid)
                continue

            motion_step = dict(
                static_flow=static_flow_seq[:, step_idx - 1],
                static_valid=static_valid_seq[:, step_idx - 1],
                static_flow_forward=static_flow_forward_seq[:, step_idx - 1],
                static_valid_forward=static_valid_forward_seq[:, step_idx - 1],
                dynamic_residual_flow=dynamic_flow_seq[:, step_idx - 1],
                dynamic_residual_valid=dynamic_valid_seq[:, step_idx - 1],
                dynamic_residual_flow_forward=dynamic_flow_forward_seq[:, step_idx - 1],
                dynamic_residual_valid_forward=dynamic_valid_forward_seq[:, step_idx - 1],
            )
            sampled_occ_motion, sampled_occ_valid, occ_slot_debug = self._align_cached_history_occ_with_motion(
                cache_occ, motion_step, step_metas, collect_debug=True)
            sampled_bev_motion = self._encode_occ_slot_bank(sampled_occ_motion, sampled_occ_valid)

        clean_curr_bev, _ = self.forward_encoder(curr_target)
        input_shape = (batch_size, 1) + tuple(curr_target.shape[-3:])

        def _decode_pred(bev_feature, shapes):
            decode_out = self._decode_bev_feature(bev_feature, shapes, input_shape)
            pred = decode_out['y_pred'] if isinstance(decode_out, dict) else decode_out
            if pred.shape[-1] > 1:
                pred = pred.softmax(dim=-1).argmax(dim=-1)
            return pred.detach().cpu().numpy().astype(np.uint8)

        def _decode_slot_bank(slot_bank, shapes):
            preds = []
            for slot_idx in range(slot_bank.shape[1]):
                preds.append(_decode_pred(slot_bank[:, slot_idx], shapes))
            return np.stack(preds, axis=1)

        final_logits = self.forward_decoder(self.vq(curr_bev, sampled_bev_motion, is_voxel=False)[0], curr_shapes, input_shape)
        final_pred = final_logits.softmax(-1).argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)
        curr_decode_pred = _decode_pred(curr_bev, curr_shapes)

        return dict(
            voxel_semantics_seq=seq.detach().cpu().numpy().astype(np.uint8),
            voxel_semantics_clean_seq=clean_seq.detach().cpu().numpy().astype(np.uint8),
            curr_input=seq[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            curr_decode_pred=curr_decode_pred,
            final_pred=final_pred,
            sampled_occ_motion=sampled_occ_motion.detach().cpu().numpy().astype(np.uint8),
            sampled_occ_valid=sampled_occ_valid.detach().cpu().numpy(),
            sampled_occ_motion_dynamic_mask=self._dynamic_occ_mask(sampled_occ_motion).detach().cpu().numpy(),
            sampled_bev_motion=sampled_bev_motion.detach().cpu().numpy(),
            sampled_bev_motion_decode_pred=_decode_slot_bank(sampled_bev_motion, curr_shapes),
            curr_bev=curr_bev.detach().cpu().numpy(),
            clean_curr_bev=clean_curr_bev.detach().cpu().numpy(),
            occ_slot_debug=[
                [
                    dict(
                        target_step=record['target_step'],
                        static_occ=record['static_occ'].detach().cpu().numpy().astype(np.uint8),
                        dynamic_occ=record['dynamic_occ'].detach().cpu().numpy().astype(np.uint8),
                        aligned_occ=record['aligned_occ'].detach().cpu().numpy().astype(np.uint8),
                        aligned_valid=record['aligned_valid'].detach().cpu().numpy(),
                        static_valid=record['static_valid'].detach().cpu().numpy(),
                        dynamic_valid=record['dynamic_valid'].detach().cpu().numpy(),
                        dynamic_mask=record['dynamic_mask'].detach().cpu().numpy(),
                    )
                    for record in slot_records
                ]
                for slot_records in occ_slot_debug
            ],
        )
