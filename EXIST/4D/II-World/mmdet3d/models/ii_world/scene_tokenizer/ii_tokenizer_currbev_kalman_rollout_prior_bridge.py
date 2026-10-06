import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_prior_slot import IISceneTokenizerCurrBevKalmanPriorSlot


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanRolloutPriorBridge(IISceneTokenizerCurrBevKalmanPriorSlot):
    """Causal t-4 -> t Kalman rollout that only provides a dynamic prior.

    Design:
    - keep baseline encoder / full temporal+spatial VQ / decoder frozen
    - run Kalman recursion in curr_bev space across the full history window
    - use the final prior x_t^- only as a motion/structure prior
    - adapt and inject that prior into the newest history slot, while the
      final semantic readout is still the baseline:
        `vq(curr_bev, sampled_bev_bridge) -> decoder`
    """

    def __init__(self,
                 kalman_prior_hidden=None,
                 kalman_prior_use_gain=True,
                 kalman_prior_use_reliability=True,
                 kalman_prior_use_dynamic_ratio=True,
                 kalman_prior_use_valid_ratio=True,
                 kalman_inject_slot=0,
                 kalman_gate_init_bias=-3.0,
                 kalman_posterior_aux_loss_weight=0.25,
                 kalman_posterior_dynamic_boost=1.0,
                 kalman_bad_threshold=0.5,
                 kalman_debug_decode_state=False,
                 **kwargs):
        super().__init__(
            kalman_slot=kalman_inject_slot,
            kalman_posterior_aux_loss_weight=kalman_posterior_aux_loss_weight,
            kalman_debug_decode_state=kalman_debug_decode_state,
            **kwargs,
        )
        hidden = kalman_prior_hidden or self.vq_channel * 2
        self.kalman_prior_use_gain = kalman_prior_use_gain
        self.kalman_prior_use_reliability = kalman_prior_use_reliability
        self.kalman_prior_use_dynamic_ratio = kalman_prior_use_dynamic_ratio
        self.kalman_prior_use_valid_ratio = kalman_prior_use_valid_ratio
        self.kalman_posterior_dynamic_boost = kalman_posterior_dynamic_boost
        self.kalman_bad_threshold = kalman_bad_threshold
        self.kalman_debug_decode_state = kalman_debug_decode_state

        adapter_in_channels = self.vq_channel * 3
        adapter_in_channels += 1 if kalman_prior_use_gain else 0
        adapter_in_channels += 1 if kalman_prior_use_reliability else 0
        adapter_in_channels += 1 if kalman_prior_use_dynamic_ratio else 0
        adapter_in_channels += 1 if kalman_prior_use_valid_ratio else 0

        self.kalman_prior_adapter = nn.Sequential(
            nn.Conv2d(adapter_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.vq_channel, 1),
        )
        self.kalman_inject_head = nn.Sequential(
            nn.Conv2d(adapter_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

        nn.init.zeros_(self.kalman_prior_adapter[-1].weight)
        nn.init.zeros_(self.kalman_prior_adapter[-1].bias)
        nn.init.zeros_(self.kalman_inject_head[-1].weight)
        nn.init.constant_(self.kalman_inject_head[-1].bias, kalman_gate_init_bias)

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

    def _build_prior_bridge_feature(self,
                                    prior_mean,
                                    history_slot,
                                    curr_bev,
                                    gain_group,
                                    reliability_latent=None,
                                    dynamic_ratio=None,
                                    valid_ratio=None):
        inputs = [
            prior_mean,
            history_slot,
            (prior_mean - history_slot).abs(),
        ]
        if self.kalman_prior_use_gain:
            inputs.append(gain_group.mean(dim=1, keepdim=True))
        if self.kalman_prior_use_reliability:
            if reliability_latent is None:
                reliability_latent = prior_mean.new_ones(
                    prior_mean.shape[0], 1, prior_mean.shape[2], prior_mean.shape[3])
            inputs.append(1.0 - reliability_latent)
        if self.kalman_prior_use_dynamic_ratio:
            if dynamic_ratio is None:
                dynamic_ratio = prior_mean.new_zeros(
                    prior_mean.shape[0], 1, prior_mean.shape[2], prior_mean.shape[3])
            inputs.append(dynamic_ratio)
        if self.kalman_prior_use_valid_ratio:
            if valid_ratio is None:
                valid_ratio = prior_mean.new_ones(
                    prior_mean.shape[0], 1, prior_mean.shape[2], prior_mean.shape[3])
            inputs.append(valid_ratio)
        return torch.cat(inputs, dim=1)

    def _bridge_kalman_prior(self,
                             sampled_bev,
                             prior_mean,
                             gain_group,
                             reliability_latent=None,
                             dynamic_ratio=None,
                             valid_ratio=None):
        slot = self.kalman_slot
        history_slot = sampled_bev[:, slot]
        bridge_feature = self._build_prior_bridge_feature(
            prior_mean,
            history_slot,
            history_slot,
            gain_group,
            reliability_latent=reliability_latent,
            dynamic_ratio=dynamic_ratio,
            valid_ratio=valid_ratio,
        )
        prior_delta = self.kalman_prior_adapter(bridge_feature)
        prior_adapted = prior_mean + prior_delta
        alpha = torch.sigmoid(self.kalman_inject_head(bridge_feature))
        if dynamic_ratio is not None:
            alpha = alpha * dynamic_ratio
        if valid_ratio is not None:
            alpha = alpha * valid_ratio
        sampled_bev_bridge = sampled_bev.clone()
        sampled_bev_bridge[:, slot] = (1.0 - alpha) * history_slot + alpha * prior_adapted
        return sampled_bev_bridge, prior_adapted, alpha

    def _forward_rollout(self,
                         voxel_semantics,
                         voxel_semantics_clean,
                         img_metas,
                         oracle_flow_seq,
                         oracle_flow_valid_seq,
                         oracle_dynamic_mask_seq,
                         frame_reliability_map=None):
        batch_size, seq_len = voxel_semantics.shape[:2]
        if seq_len < 2:
            raise ValueError(f'Kalman rollout expects at least 2 frames, got {seq_len}')

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

        current_reliability = None
        if frame_reliability_map is not None:
            current_reliability = frame_reliability_map[:, -1] if frame_reliability_map.dim() == 4 else frame_reliability_map

        sampled_bev, _, _, sampled_reliability = self.align_bev(
            curr_bev,
            img_metas,
            cache_bev=clean_curr_bev if self.kalman_use_clean_cache else None,
            cache_reliability=current_reliability if current_reliability is not None else None,
        )

        posterior_mean = bev_seq[0]
        posterior_cov = self._initial_covariance(
            batch_size, posterior_mean.shape[-2], posterior_mean.shape[-1],
            posterior_mean.dtype, posterior_mean.device)

        step_debug = []
        last_prior_mean = posterior_mean
        last_prior_cov = posterior_cov
        last_gain_group = posterior_cov.new_zeros(posterior_cov.shape)
        last_valid_ratio = posterior_cov.new_ones(batch_size, 1, posterior_cov.shape[-2], posterior_cov.shape[-1])
        last_dynamic_ratio = posterior_cov.new_zeros(batch_size, 1, posterior_cov.shape[-2], posterior_cov.shape[-1])
        last_reliability_latent = self._pool_current_map(current_reliability, posterior_mean.shape[-2:])

        for step_idx in range(1, seq_len):
            flow_latent, valid_ratio, dynamic_ratio = self._aggregate_flow_to_latent(
                oracle_flow_seq[:, step_idx - 1].to(curr_bev.dtype),
                oracle_flow_valid_seq[:, step_idx - 1],
                oracle_dynamic_mask_seq[:, step_idx - 1],
                posterior_mean.shape[-2:],
            )
            reliability_latent = None
            if frame_reliability_map is not None:
                reliability_latent = self._pool_current_map(
                    frame_reliability_map[:, step_idx:step_idx + 1],
                    posterior_mean.shape[-2:])

            prior_mean, prior_cov, process_noise = self._predict_kalman_state(
                posterior_mean, posterior_cov, flow_latent, valid_ratio, dynamic_ratio)
            posterior_mean, posterior_cov, gain_group, observation_noise = self._update_kalman_state(
                prior_mean, prior_cov, bev_seq[step_idx], reliability_latent=reliability_latent)

            last_prior_mean = prior_mean
            last_prior_cov = prior_cov
            last_gain_group = gain_group
            last_valid_ratio = valid_ratio
            last_dynamic_ratio = dynamic_ratio
            last_reliability_latent = reliability_latent
            step_debug.append(dict(
                observation_bev=bev_seq[step_idx],
                flow_latent=flow_latent,
                valid_ratio=valid_ratio,
                dynamic_ratio=dynamic_ratio,
                reliability_latent=reliability_latent,
                process_noise=process_noise,
                observation_noise=observation_noise,
                prior_mean=prior_mean,
                posterior_mean=posterior_mean,
                gain_group=gain_group,
            ))

        sampled_bev_bridge, prior_adapted, inject_alpha = self._bridge_kalman_prior(
            sampled_bev,
            last_prior_mean,
            last_gain_group,
            reliability_latent=last_reliability_latent,
            dynamic_ratio=last_dynamic_ratio,
            valid_ratio=last_valid_ratio,
        )
        z_sampled, embed_loss, _ = self.vq(curr_bev, sampled_bev_bridge, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        debug = dict(
            bev_seq=bev_seq,
            shape_seq=shape_seq,
            curr_bev=curr_bev,
            clean_curr_bev=clean_curr_bev,
            curr_shapes=curr_shapes,
            sampled_bev=sampled_bev,
            sampled_bev_bridge=sampled_bev_bridge,
            last_prior_mean=last_prior_mean,
            last_prior_cov=last_prior_cov,
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            last_gain_group=last_gain_group,
            last_valid_ratio=last_valid_ratio,
            last_dynamic_ratio=last_dynamic_ratio,
            last_reliability_latent=last_reliability_latent,
            prior_adapted=prior_adapted,
            inject_alpha=inject_alpha,
            step_debug=step_debug,
        )
        return logits, embed_loss, curr_target, debug

    def forward_train(self,
                      voxel_semantics,
                      img_metas,
                      oracle_flow_seq,
                      oracle_flow_valid_seq,
                      oracle_dynamic_mask_seq,
                      frame_reliability_map=None,
                      voxel_semantics_clean=None,
                      **kwargs):
        batch_size = len(img_metas)
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        flow_seq = self._normalize_flow_sequence_tensor(oracle_flow_seq, batch_size, has_vec=True)
        flow_valid_seq = self._normalize_flow_sequence_tensor(oracle_flow_valid_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        reliability = self.normalize_batched_tensor(frame_reliability_map, batch_size) if frame_reliability_map is not None else None

        logits, embed_loss, curr_target, debug = self._forward_rollout(
            seq,
            clean_seq,
            img_metas,
            flow_seq,
            flow_valid_seq,
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
            dynamic_mask = (debug['last_dynamic_ratio'] > 0.0).float()
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
        losses['kalman_inject_alpha_mean'] = debug['inject_alpha'].mean().detach()
        return losses

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_flow_seq,
                     oracle_flow_valid_seq,
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
        flow_seq = self._normalize_flow_sequence_tensor(oracle_flow_seq, batch_size, has_vec=True)
        flow_valid_seq = self._normalize_flow_sequence_tensor(oracle_flow_valid_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        reliability = self.normalize_batched_tensor(frame_reliability_map, batch_size) if frame_reliability_map is not None else None

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        flow_seq = self._to_model_device(flow_seq, dtype=torch.float32)
        flow_valid_seq = self._to_model_device(flow_valid_seq)
        dynamic_seq = self._to_model_device(dynamic_seq)
        reliability = self._to_model_device(reliability, dtype=torch.float32)

        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_event is not None:
            start_event.record()

        logits, _, curr_target, debug = self._forward_rollout(
            seq,
            clean_seq,
            img_metas,
            flow_seq,
            flow_valid_seq,
            dynamic_seq,
            frame_reliability_map=reliability,
        )
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed = 0.0

        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        result = [dict(
            semantics=pred,
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=seq[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
            time=elapsed,
        )]
        if self.kalman_debug_decode_state:
            input_shape = (batch_size, 1) + tuple(curr_target.shape[-3:])
            step_decode_debug = []
            for step in debug['step_debug']:
                step_decode_debug.append(dict(
                    observation_decode_pred=self._decode_bev_feature(
                        step['observation_bev'],
                        debug['curr_shapes'],
                        input_shape,
                    )['y_pred'].detach().cpu().numpy(),
                    prior_decode_pred=self._decode_bev_feature(
                        step['prior_mean'],
                        debug['curr_shapes'],
                        input_shape,
                    )['y_pred'].detach().cpu().numpy(),
                    posterior_decode_pred=self._decode_bev_feature(
                        step['posterior_mean'],
                        debug['curr_shapes'],
                        input_shape,
                    )['y_pred'].detach().cpu().numpy(),
                ))
            result[0]['kalman_debug'] = dict(
                sampled_bev_bridge=debug['sampled_bev_bridge'].detach().cpu().numpy(),
                sampled_bev=debug['sampled_bev'].detach().cpu().numpy(),
                last_prior_mean=debug['last_prior_mean'].detach().cpu().numpy(),
                posterior_mean=debug['posterior_mean'].detach().cpu().numpy(),
                prior_adapted=debug['prior_adapted'].detach().cpu().numpy(),
                inject_alpha=debug['inject_alpha'].detach().cpu().numpy(),
                last_gain_group=debug['last_gain_group'].detach().cpu().numpy(),
                last_valid_ratio=debug['last_valid_ratio'].detach().cpu().numpy(),
                last_dynamic_ratio=debug['last_dynamic_ratio'].detach().cpu().numpy(),
                last_reliability_latent=None if debug['last_reliability_latent'] is None else debug['last_reliability_latent'].detach().cpu().numpy(),
                curr_bev=debug['curr_bev'].detach().cpu().numpy(),
                clean_curr_bev=debug['clean_curr_bev'].detach().cpu().numpy(),
                step_debug=[
                    dict(
                        observation_bev=step['observation_bev'].detach().cpu().numpy(),
                        flow_latent=step['flow_latent'].detach().cpu().numpy(),
                        valid_ratio=step['valid_ratio'].detach().cpu().numpy(),
                        dynamic_ratio=step['dynamic_ratio'].detach().cpu().numpy(),
                        reliability_latent=None if step['reliability_latent'] is None else step['reliability_latent'].detach().cpu().numpy(),
                        process_noise=step['process_noise'].detach().cpu().numpy(),
                        observation_noise=step['observation_noise'].detach().cpu().numpy(),
                        prior_mean=step['prior_mean'].detach().cpu().numpy(),
                        posterior_mean=step['posterior_mean'].detach().cpu().numpy(),
                        gain_group=step['gain_group'].detach().cpu().numpy(),
                    )
                    for step in debug['step_debug']
                ],
                step_decode_debug=step_decode_debug,
            )
        return result

    @torch.no_grad()
    def collect_prior_bridge_debug(self,
                                   voxel_semantics,
                                   voxel_semantics_clean,
                                   img_metas,
                                   oracle_flow_seq,
                                   oracle_flow_valid_seq,
                                   oracle_dynamic_mask_seq,
                                   frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        seq = self.normalize_batched_tensor(voxel_semantics, batch_size)
        clean_seq = self._normalize_clean_sequence(seq, voxel_semantics_clean, batch_size)
        flow_seq = self._normalize_flow_sequence_tensor(oracle_flow_seq, batch_size, has_vec=True)
        flow_valid_seq = self._normalize_flow_sequence_tensor(oracle_flow_valid_seq, batch_size, has_vec=False)
        dynamic_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)
        reliability = self.normalize_batched_tensor(frame_reliability_map, batch_size) if frame_reliability_map is not None else None

        seq = self._to_model_device(seq)
        clean_seq = self._to_model_device(clean_seq)
        flow_seq = self._to_model_device(flow_seq, dtype=torch.float32)
        flow_valid_seq = self._to_model_device(flow_valid_seq)
        dynamic_seq = self._to_model_device(dynamic_seq)
        reliability = self._to_model_device(reliability, dtype=torch.float32)

        logits, _, curr_target, debug = self._forward_rollout(
            seq,
            clean_seq,
            img_metas,
            flow_seq,
            flow_valid_seq,
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

        init_decode_pred = _decode_pred(debug['bev_seq'][0], debug['shape_seq'][0])
        curr_decode_pred = _decode_pred(debug['curr_bev'], debug['curr_shapes'])
        last_prior_decode_pred = _decode_pred(debug['last_prior_mean'], debug['curr_shapes'])
        posterior_decode_pred = _decode_pred(debug['posterior_mean'], debug['curr_shapes'])
        prior_adapted_decode_pred = _decode_pred(debug['prior_adapted'], debug['curr_shapes'])
        baseline_history_decode_pred = _decode_pred(debug['sampled_bev'][:, self.kalman_slot], debug['curr_shapes'])
        bridge_history_decode_pred = _decode_pred(debug['sampled_bev_bridge'][:, self.kalman_slot], debug['curr_shapes'])
        sampled_bev_decode_pred = _decode_slot_bank(debug['sampled_bev'], debug['curr_shapes'])
        sampled_bev_bridge_decode_pred = _decode_slot_bank(debug['sampled_bev_bridge'], debug['curr_shapes'])

        step_decode_debug = []
        for step_idx, step in enumerate(debug['step_debug'], start=1):
            step_decode_debug.append(dict(
                observation_decode_pred=_decode_pred(step['observation_bev'], debug['shape_seq'][step_idx]),
                prior_decode_pred=_decode_pred(step['prior_mean'], debug['shape_seq'][step_idx]),
                posterior_decode_pred=_decode_pred(step['posterior_mean'], debug['shape_seq'][step_idx]),
            ))

        return dict(
            voxel_semantics_seq=seq.detach().cpu().numpy().astype(np.uint8),
            voxel_semantics_clean_seq=clean_seq.detach().cpu().numpy().astype(np.uint8),
            curr_input=seq[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            curr_bev=debug['curr_bev'].detach().cpu().numpy(),
            clean_curr_bev=debug['clean_curr_bev'].detach().cpu().numpy(),
            prior_bridge_pred=logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            init_decode_pred=init_decode_pred.astype(np.uint8),
            curr_decode_pred=curr_decode_pred.astype(np.uint8),
            last_prior_decode_pred=last_prior_decode_pred.astype(np.uint8),
            posterior_decode_pred=posterior_decode_pred.astype(np.uint8),
            prior_adapted_decode_pred=prior_adapted_decode_pred.astype(np.uint8),
            baseline_history_decode_pred=baseline_history_decode_pred.astype(np.uint8),
            bridge_history_decode_pred=bridge_history_decode_pred.astype(np.uint8),
            sampled_bev_decode_pred=sampled_bev_decode_pred.astype(np.uint8),
            sampled_bev_bridge_decode_pred=sampled_bev_bridge_decode_pred.astype(np.uint8),
            last_prior_mean=debug['last_prior_mean'].detach().cpu().numpy(),
            posterior_mean=debug['posterior_mean'].detach().cpu().numpy(),
            prior_adapted=debug['prior_adapted'].detach().cpu().numpy(),
            inject_alpha=debug['inject_alpha'].detach().cpu().numpy(),
            last_gain_group=debug['last_gain_group'].detach().cpu().numpy(),
            last_valid_ratio=debug['last_valid_ratio'].detach().cpu().numpy(),
            last_dynamic_ratio=debug['last_dynamic_ratio'].detach().cpu().numpy(),
            last_reliability_latent=(
                None if debug['last_reliability_latent'] is None
                else debug['last_reliability_latent'].detach().cpu().numpy()
            ),
            step_debug=[
                dict(
                    observation_bev=step['observation_bev'].detach().cpu().numpy(),
                    flow_latent=step['flow_latent'].detach().cpu().numpy(),
                    valid_ratio=step['valid_ratio'].detach().cpu().numpy(),
                    dynamic_ratio=step['dynamic_ratio'].detach().cpu().numpy(),
                    reliability_latent=(
                        None if step['reliability_latent'] is None
                        else step['reliability_latent'].detach().cpu().numpy()
                    ),
                    process_noise=step['process_noise'].detach().cpu().numpy(),
                    observation_noise=step['observation_noise'].detach().cpu().numpy(),
                    prior_mean=step['prior_mean'].detach().cpu().numpy(),
                    posterior_mean=step['posterior_mean'].detach().cpu().numpy(),
                    gain_group=step['gain_group'].detach().cpu().numpy(),
                )
                for step in debug['step_debug']
            ],
            step_decode_debug=step_decode_debug,
            oracle_flow_seq=flow_seq.detach().cpu().numpy(),
            oracle_flow_valid_seq=flow_valid_seq.detach().cpu().numpy(),
            oracle_dynamic_mask_seq=dynamic_seq.detach().cpu().numpy(),
        )

    def train(self, mode=True):
        super().train(mode)
        self.kalman_state_bev = None
        self.kalman_state_cov = None
        return self
