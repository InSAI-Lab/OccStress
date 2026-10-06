import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_oracle_history_bridge import IISceneTokenizerOracleHistoryBridge


def _softplus_inv(value):
    value = float(value)
    return math.log(math.expm1(value))


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanPriorSlot(IISceneTokenizerOracleHistoryBridge):
    """State-space tokenizer that runs Kalman-style recursion in curr_bev space.

    Design goal:
    - keep the baseline readout path intact:
      `curr_bev -> align_bev -> vq(curr_bev, sampled_bev) -> decoder`
    - move the temporal fusion into an explicit state-space recursion in
      `curr_bev` space:
      `x_{t-1}^+, P_{t-1}^+ -> x_t^-, P_t^- -> x_t^+, P_t^+`
    - inject only the **predicted prior** `x_t^-` into the newest history slot,
      so the baseline multi-scale VQ still performs the final fusion/readout.
    """

    def __init__(self,
                 kalman_slot=0,
                 kalman_hidden=None,
                 kalman_cov_groups=8,
                 kalman_init_cov=0.10,
                 kalman_min_cov=1e-4,
                 kalman_posterior_aux_loss_weight=0.0,
                 kalman_use_clean_cache=True,
                 kalman_use_flow_mag=True,
                 kalman_use_valid_ratio=True,
                 kalman_use_dynamic_ratio=True,
                 kalman_use_reliability=True,
                 kalman_debug_decode_state=False,
                 **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, 'oracle_bridge_alpha_head'):
            del self.oracle_bridge_alpha_head

        hidden = kalman_hidden or self.vq_channel
        cov_groups = int(kalman_cov_groups)
        if cov_groups <= 0:
            raise ValueError(f'kalman_cov_groups must be positive, got {cov_groups}')

        self.kalman_slot = kalman_slot
        self.kalman_cov_groups = cov_groups
        self.kalman_min_cov = kalman_min_cov
        self.kalman_posterior_aux_loss_weight = kalman_posterior_aux_loss_weight
        self.kalman_use_clean_cache = kalman_use_clean_cache
        self.kalman_use_flow_mag = kalman_use_flow_mag
        self.kalman_use_valid_ratio = kalman_use_valid_ratio
        self.kalman_use_dynamic_ratio = kalman_use_dynamic_ratio
        self.kalman_use_reliability = kalman_use_reliability
        self.kalman_debug_decode_state = kalman_debug_decode_state

        q_in_channels = 0
        q_in_channels += 1 if kalman_use_flow_mag else 0
        q_in_channels += 1 if kalman_use_valid_ratio else 0
        q_in_channels += 1 if kalman_use_dynamic_ratio else 0
        if q_in_channels == 0:
            q_in_channels = 1

        r_in_channels = self.vq_channel
        r_in_channels += 1 if kalman_use_reliability else 0

        self.kalman_q_head = nn.Sequential(
            nn.Conv2d(q_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, cov_groups, 1),
        )
        self.kalman_r_head = nn.Sequential(
            nn.Conv2d(r_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, cov_groups, 1),
        )
        self.kalman_log_p0 = nn.Parameter(
            torch.full((1, cov_groups, 1, 1), _softplus_inv(kalman_init_cov), dtype=torch.float32)
        )

        self.kalman_state_bev = None
        self.kalman_state_cov = None

    def _expand_group_map(self, group_map, channels):
        group_channels = group_map.shape[1]
        if group_channels == channels:
            return group_map
        group_idx = torch.div(
            torch.arange(channels, device=group_map.device) * group_channels,
            channels,
            rounding_mode='floor')
        return group_map[:, group_idx]

    def _initial_covariance(self, batch_size, height, width, dtype, device):
        p0 = F.softplus(self.kalman_log_p0) + self.kalman_min_cov
        return p0.to(device=device, dtype=dtype).expand(batch_size, -1, height, width).clone()

    def _prepare_kalman_previous_state(self, prev_bev, img_metas):
        batch_size, _, height, width = prev_bev.shape
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        start_mask = torch.as_tensor(start_of_sequence, device=prev_bev.device, dtype=torch.bool)
        init_cov = self._initial_covariance(batch_size, height, width, prev_bev.dtype, prev_bev.device)

        need_reinit = (
            self.kalman_state_bev is None or
            self.kalman_state_cov is None or
            self.kalman_state_bev.shape != prev_bev.shape or
            self.kalman_state_cov.shape != init_cov.shape
        )
        if need_reinit:
            self.kalman_state_bev = prev_bev.detach().clone()
            self.kalman_state_cov = init_cov.detach().clone()

        prev_state_bev = self.kalman_state_bev.detach().clone()
        prev_state_cov = self.kalman_state_cov.detach().clone()
        if need_reinit:
            prev_state_bev = prev_bev.clone()
            prev_state_cov = init_cov.clone()
        if start_mask.any():
            prev_state_bev[start_mask] = prev_bev[start_mask]
            prev_state_cov[start_mask] = init_cov[start_mask]
        return prev_state_bev, prev_state_cov

    def _build_process_noise(self, flow_latent, valid_ratio, dynamic_ratio):
        inputs = []
        if self.kalman_use_flow_mag:
            inputs.append(flow_latent.norm(dim=1, keepdim=True))
        if self.kalman_use_valid_ratio:
            inputs.append(1.0 - valid_ratio)
        if self.kalman_use_dynamic_ratio:
            inputs.append(dynamic_ratio)
        if not inputs:
            inputs.append(flow_latent.new_zeros(flow_latent.shape[0], 1, flow_latent.shape[2], flow_latent.shape[3]))
        return F.softplus(self.kalman_q_head(torch.cat(inputs, dim=1))) + self.kalman_min_cov

    def _build_observation_noise(self, curr_bev, reliability_latent=None):
        inputs = [curr_bev]
        if self.kalman_use_reliability:
            if reliability_latent is None:
                reliability_latent = curr_bev.new_ones(curr_bev.shape[0], 1, curr_bev.shape[2], curr_bev.shape[3])
            inputs.append(1.0 - reliability_latent)
        return F.softplus(self.kalman_r_head(torch.cat(inputs, dim=1))) + self.kalman_min_cov

    def _predict_kalman_state(self, prev_state_bev, prev_state_cov, flow_latent, valid_ratio, dynamic_ratio):
        prior_mean = self._warp_feature(prev_state_bev, flow_latent)
        prior_cov = self._warp_feature(prev_state_cov, flow_latent)
        process_noise = self._build_process_noise(flow_latent, valid_ratio, dynamic_ratio)
        prior_cov = prior_cov.clamp_min(self.kalman_min_cov) + process_noise
        return prior_mean, prior_cov, process_noise

    def _update_kalman_state(self, prior_mean, prior_cov, curr_bev, reliability_latent=None):
        observation_noise = self._build_observation_noise(curr_bev, reliability_latent)
        gain_group = prior_cov / (prior_cov + observation_noise + 1e-6)
        gain = self._expand_group_map(gain_group, curr_bev.shape[1]).to(curr_bev.dtype)
        posterior_mean = prior_mean + gain * (curr_bev - prior_mean)
        posterior_cov = (1.0 - gain_group).pow(2) * prior_cov + gain_group.pow(2) * observation_noise
        posterior_cov = posterior_cov.clamp_min(self.kalman_min_cov)
        return posterior_mean, posterior_cov, gain_group, observation_noise

    def _inject_kalman_prior(self, sampled_bev, prior_mean):
        sampled_bev_kalman = sampled_bev.clone()
        sampled_bev_kalman[:, self.kalman_slot] = prior_mean
        return sampled_bev_kalman

    def _decode_bev_feature(self, bev_feature, shapes, input_shape):
        latent = self._quantize_latent_for_decode(bev_feature)
        return self.forward_decoder(latent, shapes, input_shape)

    def _forward_kalman_pair(self, voxel_semantics, voxel_semantics_clean, img_metas,
                             oracle_flow, oracle_flow_valid, oracle_dynamic_mask,
                             frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        prev_clean, curr_input, curr_target = self._split_pairwise_inputs(
            voxel_semantics, voxel_semantics_clean)

        _, _, occ_h, occ_w, occ_z = curr_target.shape
        curr_bev, curr_shapes = self.forward_encoder(curr_input)
        prev_bev, _ = self.forward_encoder(prev_clean)
        clean_curr_bev = None
        if self.kalman_use_clean_cache:
            clean_curr_bev, _ = self.forward_encoder(curr_target)

        sampled_bev, _, _, sampled_reliability = self.align_bev(
            curr_bev,
            img_metas,
            cache_bev=clean_curr_bev if clean_curr_bev is not None else None,
            cache_reliability=frame_reliability_map[:, -1] if frame_reliability_map is not None else None,
        )

        flow_latent, valid_ratio, dynamic_ratio = self._aggregate_flow_to_latent(
            oracle_flow.to(curr_bev.dtype),
            oracle_flow_valid,
            oracle_dynamic_mask,
            curr_bev.shape[-2:],
        )
        reliability_latent = self._pool_current_map(
            frame_reliability_map if frame_reliability_map is not None else None,
            curr_bev.shape[-2:])

        prev_state_bev, prev_state_cov = self._prepare_kalman_previous_state(prev_bev, img_metas)
        prior_mean, prior_cov, process_noise = self._predict_kalman_state(
            prev_state_bev, prev_state_cov, flow_latent, valid_ratio, dynamic_ratio)
        posterior_mean, posterior_cov, gain_group, observation_noise = self._update_kalman_state(
            prior_mean, prior_cov, curr_bev, reliability_latent=reliability_latent)

        sampled_bev_kalman = self._inject_kalman_prior(sampled_bev, prior_mean)
        z_sampled, loss, info = self.vq(curr_bev, sampled_bev_kalman, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        self.kalman_state_bev = posterior_mean.detach().clone()
        self.kalman_state_cov = posterior_cov.detach().clone()

        debug = dict(
            curr_bev=curr_bev,
            curr_shapes=curr_shapes,
            prev_bev=prev_bev,
            sampled_bev=sampled_bev,
            sampled_bev_kalman=sampled_bev_kalman,
            prior_mean=prior_mean,
            posterior_mean=posterior_mean,
            prior_cov=prior_cov,
            posterior_cov=posterior_cov,
            gain_group=gain_group,
            process_noise=process_noise,
            observation_noise=observation_noise,
            flow_latent=flow_latent,
            valid_ratio=valid_ratio,
            dynamic_ratio=dynamic_ratio,
            reliability_latent=reliability_latent,
            sampled_reliability=sampled_reliability,
        )
        return logits, loss, curr_target, debug

    @torch.no_grad()
    def collect_kalman_debug(self,
                             voxel_semantics,
                             voxel_semantics_clean,
                             img_metas,
                             oracle_flow,
                             oracle_flow_valid,
                             oracle_dynamic_mask,
                             frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        voxel_semantics = self._normalize_sequence_tensor(voxel_semantics, batch_size)
        voxel_semantics_clean = self._normalize_sequence_tensor(voxel_semantics_clean, batch_size)
        if frame_reliability_map is not None:
            frame_reliability_map = self.normalize_frame_reliability_map(frame_reliability_map, batch_size)

        voxel_semantics = self._to_model_device(voxel_semantics)
        voxel_semantics_clean = self._to_model_device(voxel_semantics_clean)
        oracle_flow = self._to_model_device(oracle_flow, dtype=torch.float32)
        oracle_flow_valid = self._to_model_device(oracle_flow_valid)
        oracle_dynamic_mask = self._to_model_device(oracle_dynamic_mask)
        frame_reliability_map = self._to_model_device(frame_reliability_map, dtype=torch.float32)

        prev_clean, curr_input, curr_target = self._split_pairwise_inputs(
            voxel_semantics, voxel_semantics_clean)
        logits, _, _, debug = self._forward_kalman_pair(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
            frame_reliability_map=frame_reliability_map,
        )

        baseline_logits = self.forward_decoder(
            self.vq(debug['curr_bev'], debug['sampled_bev'], is_voxel=False)[0],
            debug['curr_shapes'],
            (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4]),
        )

        input_shape = (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4])
        prior_logits = self._decode_bev_feature(debug['prior_mean'], debug['curr_shapes'], input_shape)
        posterior_logits = self._decode_bev_feature(debug['posterior_mean'], debug['curr_shapes'], input_shape)
        slot_logits = self._decode_bev_feature(
            debug['sampled_bev_kalman'][:, self.kalman_slot],
            debug['curr_shapes'],
            input_shape,
        )

        return dict(
            prev_clean=prev_clean.detach().cpu().numpy().astype(np.uint8),
            curr_input=curr_input.detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            kalman_pred=logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            baseline_pred=baseline_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            prior_pred=prior_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            posterior_pred=posterior_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            prior_slot_pred=slot_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            gain_mean=debug['gain_group'].mean(dim=1).detach().cpu().numpy(),
            prior_cov_mean=debug['prior_cov'].mean(dim=1).detach().cpu().numpy(),
            posterior_cov_mean=debug['posterior_cov'].mean(dim=1).detach().cpu().numpy(),
            q_mean=debug['process_noise'].mean(dim=1).detach().cpu().numpy(),
            r_mean=debug['observation_noise'].mean(dim=1).detach().cpu().numpy(),
            flow_latent=debug['flow_latent'].detach().cpu().numpy(),
            valid_ratio=debug['valid_ratio'].detach().cpu().numpy(),
            dynamic_ratio=debug['dynamic_ratio'].detach().cpu().numpy(),
            reliability_latent=(
                None if debug['reliability_latent'] is None
                else debug['reliability_latent'].detach().cpu().numpy()
            ),
            oracle_flow=oracle_flow.detach().cpu().numpy(),
            oracle_flow_valid=oracle_flow_valid.detach().cpu().numpy(),
            oracle_dynamic_mask=oracle_dynamic_mask.detach().cpu().numpy(),
            baseline_history_bev=debug['sampled_bev'][:, self.kalman_slot].detach().cpu().numpy(),
            prior_bev=debug['prior_mean'].detach().cpu().numpy(),
            posterior_bev=debug['posterior_mean'].detach().cpu().numpy(),
        )

    def forward_train(self,
                      voxel_semantics,
                      img_metas,
                      oracle_flow,
                      oracle_flow_valid,
                      oracle_dynamic_mask,
                      frame_reliability_map=None,
                      **kwargs):
        batch_size = len(img_metas)
        voxel_semantics = self._normalize_sequence_tensor(voxel_semantics, batch_size)
        voxel_semantics_clean = kwargs.get('voxel_semantics_clean', voxel_semantics)
        voxel_semantics_clean = self._normalize_sequence_tensor(voxel_semantics_clean, batch_size)
        if frame_reliability_map is not None:
            frame_reliability_map = self.normalize_frame_reliability_map(frame_reliability_map, batch_size)

        voxel_semantics = self._to_model_device(voxel_semantics)
        voxel_semantics_clean = self._to_model_device(voxel_semantics_clean)
        oracle_flow = self._to_model_device(oracle_flow, dtype=torch.float32)
        oracle_flow_valid = self._to_model_device(oracle_flow_valid)
        oracle_dynamic_mask = self._to_model_device(oracle_dynamic_mask)
        frame_reliability_map = self._to_model_device(frame_reliability_map, dtype=torch.float32)

        logits, loss, curr_target, debug = self._forward_kalman_pair(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
            frame_reliability_map=frame_reliability_map,
        )

        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, curr_target))
        loss_dict['embed_loss'] = self.embed_loss_weight * loss
        if self.kalman_posterior_aux_loss_weight > 0:
            posterior_logits = self._decode_bev_feature(
                debug['posterior_mean'],
                debug['curr_shapes'],
                (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4]),
            )
            posterior_aux = self.reconstruct_loss(posterior_logits, curr_target)['recon_loss']
            loss_dict['kalman_posterior_aux_recon_loss'] = (
                self.kalman_posterior_aux_loss_weight * posterior_aux
            )
        loss_dict['kalman_q_mean'] = debug['process_noise'].mean() * 0.0 + debug['process_noise'].mean().detach()
        loss_dict['kalman_r_mean'] = debug['observation_noise'].mean() * 0.0 + debug['observation_noise'].mean().detach()
        loss_dict['kalman_gain_mean'] = debug['gain_group'].mean() * 0.0 + debug['gain_group'].mean().detach()
        return loss_dict

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_flow,
                     oracle_flow_valid,
                     oracle_dynamic_mask,
                     frame_reliability_map=None,
                     **kwargs):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        batch_size = len(img_metas)
        voxel_semantics = self._normalize_sequence_tensor(voxel_semantics, batch_size)
        voxel_semantics_clean = kwargs.get('voxel_semantics_clean', voxel_semantics)
        voxel_semantics_clean = self._normalize_sequence_tensor(voxel_semantics_clean, batch_size)
        if frame_reliability_map is not None:
            frame_reliability_map = self.normalize_frame_reliability_map(frame_reliability_map, batch_size)

        voxel_semantics = self._to_model_device(voxel_semantics)
        voxel_semantics_clean = self._to_model_device(voxel_semantics_clean)
        oracle_flow = self._to_model_device(oracle_flow, dtype=torch.float32)
        oracle_flow_valid = self._to_model_device(oracle_flow_valid)
        oracle_dynamic_mask = self._to_model_device(oracle_dynamic_mask)
        frame_reliability_map = self._to_model_device(frame_reliability_map, dtype=torch.float32)

        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        wall_start = None
        if start_time is not None:
            start_time.record()
        else:
            wall_start = torch.tensor(0.0)

        logits, _, curr_target, _ = self._forward_kalman_pair(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
            frame_reliability_map=frame_reliability_map,
        )

        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 1000.0
        else:
            elapsed = 0.0

        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        output_dict = dict(
            semantics=pred,
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=voxel_semantics[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
            time=elapsed,
        )
        return [output_dict]
