import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_exact_static_residual_bank import (
    IISceneTokenizerCurrBevKalmanExactStaticResidualBank,
)


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanExactStaticResidualBankCached(
        IISceneTokenizerCurrBevKalmanExactStaticResidualBank):
    """Baseline-style cached history bank with exact-static + dynamic residual slot warp.

    Design goals:
    - keep the original explicit-sequence / Kalman route intact in the parent class
    - provide a lighter cache-based route for fair tokenizer training
    - reuse the fixed slot-warp logic:
      W/H -> H/W -> warp -> W/H
    - only require the current-step motion (`t-1 -> t`) during normal train/test
    """

    def __init__(self, **kwargs):
        self.freeze_kalman_modules = kwargs.pop('freeze_kalman_modules', False)
        super().__init__(**kwargs)
        self.motion_history_bev = None
        if self.freeze_kalman_modules:
            self._freeze_kalman_parameters()

    def _freeze_kalman_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith('kalman_'):
                param.requires_grad = False

    def _clear_motion_history(self):
        self.motion_history_bev = None

    def train(self, mode=True):
        super().train(mode)
        self._clear_motion_history()
        return self

    def _extract_current_step_rt(self, img_metas, device, dtype):
        stepwise = []
        for meta in img_metas:
            curr_rt = meta.get('curr_to_prev_ego_rt', None)
            if curr_rt is None:
                curr_rt = np.eye(4, dtype=np.float32)
            elif torch.is_tensor(curr_rt):
                curr_rt = curr_rt.detach().cpu().numpy()
            stepwise.append(np.asarray(curr_rt, dtype=np.float32))
        return torch.as_tensor(np.stack(stepwise, axis=0), device=device, dtype=dtype)

    def _prepare_single_step_motion(self,
                                    curr_bev,
                                    img_metas,
                                    oracle_dynamic_residual_flow,
                                    oracle_dynamic_residual_flow_valid,
                                    oracle_dynamic_mask,
                                    oracle_dynamic_residual_flow_forward=None,
                                    oracle_dynamic_residual_flow_valid_forward=None,
                                    oracle_dynamic_mask_source=None):
        if oracle_dynamic_residual_flow_forward is None:
            oracle_dynamic_residual_flow_forward = oracle_dynamic_residual_flow
        if oracle_dynamic_residual_flow_valid_forward is None:
            oracle_dynamic_residual_flow_valid_forward = oracle_dynamic_residual_flow_valid
        if oracle_dynamic_mask_source is None:
            oracle_dynamic_mask_source = oracle_dynamic_mask
        step_rt = self._extract_current_step_rt(img_metas, curr_bev.device, curr_bev.dtype)
        bev_aug = self._stack_bev_aug(img_metas, curr_bev.device, curr_bev.dtype)
        return self._build_motion_step(
            step_rt,
            bev_aug,
            oracle_dynamic_residual_flow.to(curr_bev.dtype),
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_residual_flow_forward.to(curr_bev.dtype),
            oracle_dynamic_residual_flow_valid_forward,
            oracle_dynamic_mask,
            oracle_dynamic_mask_source,
            curr_bev.shape[-2:],
            curr_bev.dtype,
            curr_bev.device,
        )

    def _expand_motion_step_for_slots(self, motion_step, num_slots):
        expanded = {}
        for key, value in motion_step.items():
            if torch.is_tensor(value) and value.shape[0] > 0:
                expanded[key] = value.repeat_interleave(num_slots, dim=0)
            else:
                expanded[key] = value
        return expanded

    def _align_cached_history_with_motion(self, curr_bev, motion_step, img_metas):
        batch_size, channels, width, height = curr_bev.shape
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        start_mask = torch.as_tensor(start_of_sequence, device=curr_bev.device, dtype=torch.bool)

        need_reinit = (
            self.motion_history_bev is None or
            self.motion_history_bev.shape != (batch_size, self.frame_number, channels, width, height)
        )
        if need_reinit:
            self.motion_history_bev = curr_bev.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).detach().clone()

        if start_mask.any():
            self.motion_history_bev[start_mask] = curr_bev[start_mask].unsqueeze(1).repeat(
                1, self.frame_number, 1, 1, 1)

        tmp_history = self.motion_history_bev.detach()
        flat_history = tmp_history.reshape(batch_size * self.frame_number, channels, width, height)
        expanded_motion = self._expand_motion_step_for_slots(motion_step, self.frame_number)
        aligned_flat, static_flat, dynamic_flat, total_flow_flat = self._apply_unified_motion(
            flat_history, expanded_motion)

        sampled_bev_motion = aligned_flat.reshape(batch_size, self.frame_number, channels, width, height)
        static_bank = static_flat.reshape(batch_size, self.frame_number, channels, width, height)
        dynamic_bank = dynamic_flat.reshape(batch_size, self.frame_number, channels, width, height)
        total_flow_bank = total_flow_flat.reshape(batch_size, self.frame_number, total_flow_flat.shape[1], width, height)

        if start_mask.any():
            repeated_curr = curr_bev[start_mask].unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1)
            sampled_bev_motion[start_mask] = repeated_curr
            static_bank[start_mask] = repeated_curr
            dynamic_bank[start_mask] = repeated_curr
            total_flow_bank[start_mask] = 0

        self.motion_history_bev = torch.cat(
            [curr_bev.unsqueeze(1), sampled_bev_motion[:, :-1]], dim=1).detach().clone()

        slot_debug = []
        for slot_idx in range(self.frame_number):
            slot_debug.append([
                dict(
                    target_step=-1,
                    static_warp=static_bank[:, slot_idx],
                    dynamic_warp=dynamic_bank[:, slot_idx],
                    aligned=sampled_bev_motion[:, slot_idx],
                    total_flow=total_flow_bank[:, slot_idx],
                    static_flow=motion_step['static_flow'],
                    dynamic_residual_flow=motion_step['dynamic_residual_flow'],
                    static_valid_ratio=motion_step['static_valid_ratio'],
                    dynamic_residual_valid_ratio=motion_step['dynamic_residual_valid_ratio'],
                    dynamic_ratio=motion_step['dynamic_ratio'],
                    dynamic_gate=motion_step['dynamic_gate'],
                )
            ])

        return sampled_bev_motion, slot_debug

    def _forward_cached_current(self,
                                voxel_semantics,
                                voxel_semantics_clean,
                                img_metas,
                                oracle_dynamic_residual_flow,
                                oracle_dynamic_residual_flow_valid,
                                oracle_dynamic_mask,
                                oracle_dynamic_residual_flow_forward=None,
                                oracle_dynamic_residual_flow_valid_forward=None,
                                oracle_dynamic_mask_source=None):
        batch_size = voxel_semantics.shape[0]
        curr_input = voxel_semantics[:, -1:]
        curr_target = voxel_semantics_clean[:, -1:]
        _, _, occ_h, occ_w, occ_z = curr_target.shape

        curr_bev, curr_shapes = self.forward_encoder(curr_input)
        clean_curr_bev, _ = self.forward_encoder(curr_target)

        motion_step = self._prepare_single_step_motion(
            curr_bev,
            img_metas,
            oracle_dynamic_residual_flow,
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_mask,
            oracle_dynamic_residual_flow_forward,
            oracle_dynamic_residual_flow_valid_forward,
            oracle_dynamic_mask_source,
        )
        sampled_bev_motion, slot_debug = self._align_cached_history_with_motion(
            curr_bev, motion_step, img_metas)

        z_sampled, embed_loss, _ = self.vq(curr_bev, sampled_bev_motion, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        debug = dict(
            curr_bev=curr_bev,
            curr_shapes=curr_shapes,
            clean_curr_bev=clean_curr_bev,
            sampled_bev_motion=sampled_bev_motion,
            slot_debug=slot_debug,
            motion_step=motion_step,
        )
        return logits, embed_loss, curr_target, debug

    def _zero_current_step_motion(self, batch_size, occ_shape, device, dtype):
        occ_h, occ_w, occ_z = occ_shape
        zero_flow = torch.zeros((batch_size, occ_h, occ_w, occ_z, 3), device=device, dtype=dtype)
        zero_valid = torch.zeros((batch_size, occ_h, occ_w, occ_z), device=device, dtype=dtype)
        zero_mask = torch.zeros((batch_size, occ_h, occ_w, occ_z), device=device, dtype=dtype)
        return zero_flow, zero_valid, zero_mask, zero_flow.clone(), zero_valid.clone(), zero_mask.clone()

    def forward_train(self,
                      voxel_semantics,
                      img_metas,
                      oracle_static_flow=None,
                      oracle_static_flow_valid=None,
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

        # Keep the original explicit-sequence / Kalman route available.
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
        if oracle_dynamic_residual_flow is None:
            zero_flow, zero_valid, zero_mask, zero_flow_forward, zero_valid_forward, zero_source_mask = self._zero_current_step_motion(
                batch_size, tuple(seq.shape[-3:]), seq.device, torch.float32)
            oracle_dynamic_residual_flow = zero_flow
            oracle_dynamic_residual_flow_valid = zero_valid
            oracle_dynamic_mask = zero_mask
            oracle_dynamic_residual_flow_forward = zero_flow_forward
            oracle_dynamic_residual_flow_valid_forward = zero_valid_forward
            oracle_dynamic_mask_source = zero_source_mask
        else:
            oracle_dynamic_residual_flow = self._to_model_device(
                self.normalize_batched_tensor(oracle_dynamic_residual_flow, batch_size), dtype=torch.float32)
            oracle_dynamic_residual_flow_valid = self._to_model_device(
                self.normalize_batched_tensor(oracle_dynamic_residual_flow_valid, batch_size), dtype=torch.float32)
            oracle_dynamic_mask = self._to_model_device(
                self.normalize_batched_tensor(oracle_dynamic_mask, batch_size), dtype=torch.float32)
            if oracle_dynamic_residual_flow_forward is None:
                oracle_dynamic_residual_flow_forward = oracle_dynamic_residual_flow
            else:
                oracle_dynamic_residual_flow_forward = self._to_model_device(
                    self.normalize_batched_tensor(oracle_dynamic_residual_flow_forward, batch_size), dtype=torch.float32)
            if oracle_dynamic_residual_flow_valid_forward is None:
                oracle_dynamic_residual_flow_valid_forward = oracle_dynamic_residual_flow_valid
            else:
                oracle_dynamic_residual_flow_valid_forward = self._to_model_device(
                    self.normalize_batched_tensor(oracle_dynamic_residual_flow_valid_forward, batch_size), dtype=torch.float32)
            if oracle_dynamic_mask_source is None:
                oracle_dynamic_mask_source = oracle_dynamic_mask
            else:
                oracle_dynamic_mask_source = self._to_model_device(
                    self.normalize_batched_tensor(oracle_dynamic_mask_source, batch_size), dtype=torch.float32)

        logits, embed_loss, curr_target, _ = self._forward_cached_current(
            seq,
            clean_seq,
            img_metas,
            oracle_dynamic_residual_flow,
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_mask,
            oracle_dynamic_residual_flow_forward,
            oracle_dynamic_residual_flow_valid_forward,
            oracle_dynamic_mask_source,
        )

        losses = dict()
        losses.update(self.reconstruct_loss(logits, curr_target))
        losses['embed_loss'] = self.embed_loss_weight * embed_loss
        return losses

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_static_flow=None,
                     oracle_static_flow_valid=None,
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
        if oracle_dynamic_residual_flow is None:
            zero_flow, zero_valid, zero_mask, zero_flow_forward, zero_valid_forward, zero_source_mask = self._zero_current_step_motion(
                batch_size, tuple(seq.shape[-3:]), seq.device, torch.float32)
            oracle_dynamic_residual_flow = zero_flow
            oracle_dynamic_residual_flow_valid = zero_valid
            oracle_dynamic_mask = zero_mask
            oracle_dynamic_residual_flow_forward = zero_flow_forward
            oracle_dynamic_residual_flow_valid_forward = zero_valid_forward
            oracle_dynamic_mask_source = zero_source_mask
        else:
            oracle_dynamic_residual_flow = self._to_model_device(
                self.normalize_batched_tensor(oracle_dynamic_residual_flow, batch_size), dtype=torch.float32)
            oracle_dynamic_residual_flow_valid = self._to_model_device(
                self.normalize_batched_tensor(oracle_dynamic_residual_flow_valid, batch_size), dtype=torch.float32)
            oracle_dynamic_mask = self._to_model_device(
                self.normalize_batched_tensor(oracle_dynamic_mask, batch_size), dtype=torch.float32)
            if oracle_dynamic_residual_flow_forward is None:
                oracle_dynamic_residual_flow_forward = oracle_dynamic_residual_flow
            else:
                oracle_dynamic_residual_flow_forward = self._to_model_device(
                    self.normalize_batched_tensor(oracle_dynamic_residual_flow_forward, batch_size), dtype=torch.float32)
            if oracle_dynamic_residual_flow_valid_forward is None:
                oracle_dynamic_residual_flow_valid_forward = oracle_dynamic_residual_flow_valid
            else:
                oracle_dynamic_residual_flow_valid_forward = self._to_model_device(
                    self.normalize_batched_tensor(oracle_dynamic_residual_flow_valid_forward, batch_size), dtype=torch.float32)
            if oracle_dynamic_mask_source is None:
                oracle_dynamic_mask_source = oracle_dynamic_mask
            else:
                oracle_dynamic_mask_source = self._to_model_device(
                    self.normalize_batched_tensor(oracle_dynamic_mask_source, batch_size), dtype=torch.float32)

        logits, _, curr_target, _ = self._forward_cached_current(
            seq,
            clean_seq,
            img_metas,
            oracle_dynamic_residual_flow,
            oracle_dynamic_residual_flow_valid,
            oracle_dynamic_mask,
            oracle_dynamic_residual_flow_forward,
            oracle_dynamic_residual_flow_valid_forward,
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
    def collect_unified_motion_debug(self,
                                     voxel_semantics,
                                     voxel_semantics_clean,
                                     img_metas,
                                     oracle_static_flow_seq,
                                     oracle_static_flow_valid_seq,
                                     oracle_dynamic_residual_flow_seq,
                                     oracle_dynamic_residual_flow_valid_seq,
                                     oracle_dynamic_mask_seq,
                                     oracle_dynamic_source_mask_seq=None,
                                     oracle_dynamic_residual_flow_forward_seq=None,
                                     oracle_dynamic_residual_flow_valid_forward_seq=None,
                                     frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        dynamic_residual_flow_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_seq, batch_size, has_vec=True)
        dynamic_residual_valid_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_seq, batch_size, has_vec=False)
        if oracle_dynamic_residual_flow_forward_seq is None:
            oracle_dynamic_residual_flow_forward_seq = oracle_dynamic_residual_flow_seq
        if oracle_dynamic_residual_flow_valid_forward_seq is None:
            oracle_dynamic_residual_flow_valid_forward_seq = oracle_dynamic_residual_flow_valid_seq
        if oracle_dynamic_source_mask_seq is None:
            oracle_dynamic_source_mask_seq = oracle_dynamic_mask_seq
        dynamic_residual_flow_forward_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_forward_seq, batch_size, has_vec=True)
        dynamic_residual_valid_forward_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_residual_flow_valid_forward_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        dynamic_source_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_source_mask_seq, batch_size, has_vec=False)

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        dynamic_residual_flow_seq = self._to_model_device(dynamic_residual_flow_seq, dtype=torch.float32)
        dynamic_residual_valid_seq = self._to_model_device(dynamic_residual_valid_seq, dtype=torch.float32)
        dynamic_residual_flow_forward_seq = self._to_model_device(dynamic_residual_flow_forward_seq, dtype=torch.float32)
        dynamic_residual_valid_forward_seq = self._to_model_device(dynamic_residual_valid_forward_seq, dtype=torch.float32)
        dynamic_seq = self._to_model_device(dynamic_seq, dtype=torch.float32)
        dynamic_source_seq = self._to_model_device(dynamic_source_seq, dtype=torch.float32)

        self._clear_motion_history()

        stepwise_rts = self._extract_stepwise_ego_rts(img_metas, seq.shape[1], seq.device, torch.float32)
        sampled_bev_motion = None
        curr_bev = None
        curr_shapes = None
        slot_debug = None

        for step_idx in range(seq.shape[1]):
            curr_frame = seq[:, step_idx:step_idx + 1]
            curr_bev, curr_shapes = self.forward_encoder(curr_frame)

            step_metas = []
            for b, meta in enumerate(img_metas):
                bda_mat = meta.get('bda_mat')
                if not torch.is_tensor(bda_mat):
                    bda_mat = torch.as_tensor(bda_mat, dtype=curr_bev.dtype)
                step_metas.append(dict(
                    start_of_sequence=(step_idx == 0),
                    curr_to_prev_ego_rt=stepwise_rts[b, step_idx].detach().cpu().numpy(),
                    bda_mat=bda_mat,
                    scene_name=meta.get('scene_name', ''),
                    sample_idx=meta.get('sample_idx', ''),
                    index=meta.get('index', -1),
                ))

            if step_idx == 0:
                self.motion_history_bev = curr_bev.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).detach().clone()
                sampled_bev_motion = self.motion_history_bev.clone()
                slot_debug = [
                    [dict(
                        target_step=0,
                        static_warp=self.motion_history_bev[:, slot_idx],
                        dynamic_warp=self.motion_history_bev[:, slot_idx],
                        aligned=self.motion_history_bev[:, slot_idx],
                        total_flow=torch.zeros(
                            curr_bev.shape[0], 2, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                        static_flow=torch.zeros(
                            curr_bev.shape[0], 2, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                        dynamic_residual_flow=torch.zeros(
                            curr_bev.shape[0], 2, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                        static_valid_ratio=torch.ones(
                            curr_bev.shape[0], 1, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                        dynamic_residual_valid_ratio=torch.zeros(
                            curr_bev.shape[0], 1, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                        dynamic_ratio=torch.zeros(
                            curr_bev.shape[0], 1, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                        dynamic_gate=torch.zeros(
                            curr_bev.shape[0], 1, curr_bev.shape[-2], curr_bev.shape[-1],
                            device=curr_bev.device, dtype=curr_bev.dtype),
                    )]
                    for slot_idx in range(self.frame_number)
                ]
                continue

            motion_step = self._prepare_single_step_motion(
                curr_bev,
                step_metas,
                dynamic_residual_flow_seq[:, step_idx - 1],
                dynamic_residual_valid_seq[:, step_idx - 1],
                dynamic_seq[:, step_idx - 1],
                dynamic_residual_flow_forward_seq[:, step_idx - 1],
                dynamic_residual_valid_forward_seq[:, step_idx - 1],
                dynamic_source_seq[:, step_idx - 1],
            )
            sampled_bev_motion, slot_debug = self._align_cached_history_with_motion(
                curr_bev, motion_step, step_metas)

        clean_curr_bev, _ = self.forward_encoder(clean_seq[:, -1:])
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

        z_sampled, _, _ = self.vq(curr_bev, sampled_bev_motion, is_voxel=False)
        final_logits = self.forward_decoder(z_sampled, curr_shapes, input_shape)
        final_pred = final_logits.softmax(-1).argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)
        curr_decode_pred = _decode_pred(curr_bev, curr_shapes)

        return dict(
            voxel_semantics_seq=seq.detach().cpu().numpy().astype(np.uint8),
            voxel_semantics_clean_seq=clean_seq.detach().cpu().numpy().astype(np.uint8),
            curr_input=seq[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_target=clean_seq[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_decode_pred=curr_decode_pred,
            final_pred=final_pred,
            sampled_bev_motion=sampled_bev_motion.detach().cpu().numpy(),
            sampled_bev_motion_decode_pred=_decode_slot_bank(sampled_bev_motion, curr_shapes),
            curr_bev=curr_bev.detach().cpu().numpy(),
            clean_curr_bev=clean_curr_bev.detach().cpu().numpy(),
            oracle_dynamic_residual_flow_forward_seq=dynamic_residual_flow_forward_seq.detach().cpu().numpy(),
            oracle_dynamic_residual_flow_valid_forward_seq=dynamic_residual_valid_forward_seq.detach().cpu().numpy(),
            oracle_dynamic_source_mask_seq=dynamic_source_seq.detach().cpu().numpy(),
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
                for slot_records in slot_debug
            ],
        )
