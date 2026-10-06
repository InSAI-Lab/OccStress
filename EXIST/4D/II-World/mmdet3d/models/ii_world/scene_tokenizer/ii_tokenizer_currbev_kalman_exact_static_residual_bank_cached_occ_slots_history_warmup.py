import torch
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_exact_static_residual_bank_cached_occ_slots import (
    IISceneTokenizerCurrBevKalmanExactStaticResidualBankCachedOccSlots,
)


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanExactStaticResidualBankCachedOccSlotsHistoryWarmup(
        IISceneTokenizerCurrBevKalmanExactStaticResidualBankCachedOccSlots):
    """Occ-slot history warmup without changing feature amplitude.

    Warmup strategy:
    - early: always use zero history
    - middle: batch-wise stochastic switch between zero history and occ-slot history
    - late: always use occ-slot history
    """

    def __init__(self,
                 history_warmup_enable=True,
                 history_zero_until=2000,
                 history_full_until=8000,
                 history_switch_mode='batch',
                 history_switch_seed=0,
                 **kwargs):
        self.history_warmup_enable = history_warmup_enable
        self.history_zero_until = int(history_zero_until)
        self.history_full_until = int(history_full_until)
        self.history_switch_mode = history_switch_mode
        self.history_switch_seed = int(history_switch_seed)
        self.history_warmup_iter = 0
        self._history_warmup_last_prob = 1.0
        self._history_warmup_last_use_real = 1.0
        super().__init__(**kwargs)
        if self.history_switch_mode != 'batch':
            raise ValueError(
                f'Unsupported history_switch_mode={self.history_switch_mode!r}')

    def set_history_warmup_iter(self, curr_iter):
        self.history_warmup_iter = int(curr_iter)

    def _set_history_warmup_stats(self, prob, use_real):
        self._history_warmup_last_prob = float(prob)
        self._history_warmup_last_use_real = float(use_real)

    def _history_occ_probability(self):
        if not self.history_warmup_enable:
            return 1.0
        if self.history_warmup_iter < self.history_zero_until:
            return 0.0
        if self.history_warmup_iter >= self.history_full_until:
            return 1.0
        ramp = max(1, self.history_full_until - self.history_zero_until)
        return float(self.history_warmup_iter - self.history_zero_until) / float(ramp)

    def _history_occ_use_real(self, prob):
        if prob <= 0.0:
            return False
        if prob >= 1.0:
            return True
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.history_switch_seed + self.history_warmup_iter)
        return bool(torch.rand((), generator=generator).item() < prob)

    def _apply_history_warmup(self, sampled_bev_motion, collect_debug=False):
        if (not self.training) or collect_debug or (not self.history_warmup_enable):
            self._set_history_warmup_stats(1.0, 1.0)
            return sampled_bev_motion
        prob = self._history_occ_probability()
        use_real = self._history_occ_use_real(prob)
        self._set_history_warmup_stats(prob, 1.0 if use_real else 0.0)
        if use_real:
            return sampled_bev_motion
        return torch.zeros_like(sampled_bev_motion)

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
            sampled_bev_motion = self._encode_occ_slot_bank(
                sampled_occ_motion, sampled_occ_valid)
        else:
            with torch.no_grad():
                sampled_bev_motion = self._encode_occ_slot_bank(
                    sampled_occ_motion, sampled_occ_valid)
        slot_encode_ms = self._profile_elapsed_ms(stamp, profile_runtime)

        sampled_bev_for_vq = self._apply_history_warmup(
            sampled_bev_motion, collect_debug=collect_debug)

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
        z_sampled, embed_loss, _ = self.vq(
            curr_bev, sampled_bev_for_vq, is_voxel=False)
        vq_ms = self._profile_elapsed_ms(stamp, profile_runtime)

        stamp = self._profile_stamp(profile_runtime)
        logits = self.forward_decoder(
            z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))
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
                sampled_bev_for_vq=sampled_bev_for_vq,
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

    def forward_train(self, *args, **kwargs):
        self._set_history_warmup_stats(1.0, 1.0)
        losses = super().forward_train(*args, **kwargs)
        metric_device = self.class_embeds.weight
        losses['history_occ_prob'] = metric_device.new_tensor(
            self._history_warmup_last_prob)
        losses['history_occ_use_real'] = metric_device.new_tensor(
            self._history_warmup_last_use_real)
        return losses
