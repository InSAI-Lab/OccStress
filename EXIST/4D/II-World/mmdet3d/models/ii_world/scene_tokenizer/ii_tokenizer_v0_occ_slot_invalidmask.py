"""Clean v0 tokenizer variant with occupancy-space history slots.

This class intentionally inherits the locked baseline tokenizer and only
replaces how ``sampled_bev`` history slots are produced. The current branch,
VQ, decoder, and losses stay identical to ``IISceneTokenizerV0Original``.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_v0_original import IISceneTokenizerV0Original


@DETECTORS.register_module()
class IISceneTokenizerV0OccSlotInvalidMask(IISceneTokenizerV0Original):
    """Build history slots in occupancy space, then encode them as BEV memory.

    The cached occupancy slots are detached history, matching the baseline
    history-BEV behavior. Invalid regions created by occ-space motion are not
    zeroed before the encoder; instead, their encoded latent cells are masked
    after encoding to avoid pre-encode free-space pollution.
    """

    DYNAMIC_OCC_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)

    def __init__(self,
                 occ_free_cls=17,
                 occ_slot_use_clean_cache=False,
                 occ_slot_apply_latent_valid=True,
                 occ_slot_latent_valid_threshold=0.0,
                 unified_motion_dynamic_transport='scatter',
                 profile_runtime=False,
                 profile_runtime_interval=10,
                 **kwargs):
        self.occ_free_cls = int(occ_free_cls)
        self.occ_slot_use_clean_cache = occ_slot_use_clean_cache
        self.occ_slot_apply_latent_valid = occ_slot_apply_latent_valid
        self.occ_slot_latent_valid_threshold = float(occ_slot_latent_valid_threshold)
        self.unified_motion_dynamic_transport = unified_motion_dynamic_transport
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

    def train(self, mode=True):
        super().train(mode)
        self._clear_motion_history()
        return self

    def _clear_motion_history(self):
        self.history_bev = None
        self.bev_aug = None
        self.motion_history_occ = None
        self.motion_history_occ_valid = None

    def _unwrap_datacontainer(self, data):
        if isinstance(data, DataContainer):
            data = data.data
        if isinstance(data, (list, tuple)) and len(data) == 1:
            return self._unwrap_datacontainer(data[0])
        return data

    def _normalize_img_metas(self, img_metas):
        img_metas = self._unwrap_datacontainer(img_metas)
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        return img_metas

    def normalize_batched_tensor(self, tensor, batch_size):
        tensor = self._unwrap_datacontainer(tensor)
        while isinstance(tensor, (list, tuple)) and len(tensor) == 1:
            tensor = tensor[0]
        while tensor.dim() >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == batch_size:
            tensor = tensor.squeeze(0)
        return tensor

    def _to_model_device(self, data, dtype=None):
        device = self.class_embeds.weight.device
        if isinstance(data, torch.Tensor):
            data = data.to(device=device)
            if dtype is not None:
                data = data.to(dtype=dtype)
            return data
        if isinstance(data, (list, tuple)):
            return type(data)(self._to_model_device(item, dtype=dtype) for item in data)
        return data

    def _normalize_clean_sequence(self, voxel_semantics, voxel_semantics_clean, batch_size):
        if voxel_semantics_clean is None:
            return voxel_semantics
        clean = self.normalize_batched_tensor(voxel_semantics_clean, batch_size)
        if clean.dim() == 4:
            clean = clean.unsqueeze(1)
        return clean

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
        pooled = (pooled > self.occ_slot_latent_valid_threshold).to(dtype=self.class_embeds.weight.dtype)
        return pooled.reshape(batch_size, num_slots, 1, out_h, out_w)

    def _apply_occ_motion(self, prev_occ, prev_valid, motion_step):
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
            return aligned, aligned_valid

        total_flow = static_flow + motion_step['dynamic_residual_flow']
        total_valid = static_valid * motion_step['dynamic_residual_valid']
        return self._warp_occ_nearest(prev_occ, total_flow, total_valid)

    def _align_cached_history_occ_with_motion(self, curr_occ, motion_step, img_metas):
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
        for slot_idx in range(self.frame_number):
            slot_occ = self.motion_history_occ[:, slot_idx]
            slot_valid = self.motion_history_occ_valid[:, slot_idx]
            aligned_slot, aligned_valid_slot = self._apply_occ_motion(slot_occ, slot_valid, motion_step)
            if start_mask.any():
                aligned_slot[start_mask] = curr_occ_noc[start_mask]
                aligned_valid_slot[start_mask] = True
            aligned_slots.append(aligned_slot)
            aligned_valid_slots.append(aligned_valid_slot)

        aligned_occ = torch.stack(aligned_slots, dim=1)
        aligned_valid = torch.stack(aligned_valid_slots, dim=1)
        self.motion_history_occ = torch.cat(
            [curr_occ_noc.unsqueeze(1), aligned_occ[:, :-1]], dim=1).detach().clone()
        self.motion_history_occ_valid = torch.cat(
            [curr_occ_valid.unsqueeze(1), aligned_valid[:, :-1]], dim=1).detach().clone()
        return aligned_occ, aligned_valid

    def _encode_occ_slot_bank(self, sampled_occ_motion, sampled_occ_valid=None):
        batch_size, num_slots, occ_h, occ_w, occ_z = sampled_occ_motion.shape
        flat_occ = sampled_occ_motion.reshape(batch_size * num_slots, 1, occ_h, occ_w, occ_z).long()
        slot_bev, _ = self.forward_encoder(flat_occ)
        slot_bev = slot_bev.reshape(batch_size, num_slots, *slot_bev.shape[1:])
        if sampled_occ_valid is None or not self.occ_slot_apply_latent_valid:
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
                                oracle_dynamic_residual_flow_forward,
                                oracle_dynamic_residual_flow_valid_forward,
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
        sampled_occ_motion, sampled_occ_valid = self._align_cached_history_occ_with_motion(
            cache_occ, motion_step, img_metas)
        occ_align_ms = self._profile_elapsed_ms(stamp, profile_runtime)

        stamp = self._profile_stamp(profile_runtime)
        with torch.no_grad():
            sampled_bev_motion = self._encode_occ_slot_bank(sampled_occ_motion, sampled_occ_valid)
        slot_encode_ms = self._profile_elapsed_ms(stamp, profile_runtime)

        stamp = self._profile_stamp(profile_runtime)
        z_sampled, embed_loss, _ = self.vq(curr_bev, sampled_bev_motion, is_voxel=False)
        vq_ms = self._profile_elapsed_ms(stamp, profile_runtime)
        stamp = self._profile_stamp(profile_runtime)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))
        decoder_ms = self._profile_elapsed_ms(stamp, profile_runtime)
        total_model_ms = self._profile_elapsed_ms(total_start, profile_runtime)

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
        return logits, embed_loss, curr_target, profile_stats

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
                      voxel_semantics_clean=None,
                      **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        batch_size = len(img_metas)
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
        logits, embed_loss, curr_target, profile_stats = self._forward_cached_current(
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
            profile_runtime=self._next_profile_flag(),
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
                     voxel_semantics_clean=None,
                     **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        batch_size = len(img_metas)
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

        start_time = time.time()
        logits, _, curr_target, _ = self._forward_cached_current(
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
            profile_runtime=False,
        )
        end_time = time.time()
        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        return [dict(
            semantics=pred,
            targ_semantics=curr_target.cpu().numpy().astype(np.uint8),
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=seq[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
            time=end_time - start_time,
        )]
