import torch

from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_exact_static_residual_bank import IISceneTokenizerCurrBevKalmanExactStaticResidualBank


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanExactStaticDynamicSumBank(
    IISceneTokenizerCurrBevKalmanExactStaticResidualBank
):
    """Use one summed motion field on latent slots.

    Motion rule:
      total_flow = static_flow + dynamic_gate * dynamic_residual_flow

    The history slot is then produced by a single backward warp with that total
    flow, instead of blending two separately warped features.
    """

    def _apply_unified_motion(self, feature, motion_step):
        static_flow = motion_step['static_flow']
        dynamic_total_flow = static_flow + motion_step['dynamic_residual_flow']
        scaled_dynamic_flow = motion_step['dynamic_gate'] * motion_step['dynamic_residual_flow']
        total_flow = static_flow + scaled_dynamic_flow

        static_warp = self._warp_raw_bev_feature(feature, static_flow)
        dynamic_warp = self._warp_raw_bev_feature(feature, dynamic_total_flow)
        aligned = self._warp_raw_bev_feature(feature, total_flow)
        return aligned, static_warp, dynamic_warp, total_flow

    def _predict_kalman_state_unified(self, prev_state_bev, prev_state_cov, motion_step):
        prior_mean, static_mean, dynamic_mean, total_flow = self._apply_unified_motion(
            prev_state_bev, motion_step)
        prior_cov = self._warp_raw_bev_feature(prev_state_cov, total_flow)
        process_noise = self._build_process_noise(
            total_flow,
            motion_step['static_valid_ratio'],
            motion_step['dynamic_gate'],
        )
        prior_cov = prior_cov.clamp_min(self.kalman_min_cov) + process_noise
        return prior_mean, prior_cov, process_noise, static_mean, dynamic_mean, total_flow
