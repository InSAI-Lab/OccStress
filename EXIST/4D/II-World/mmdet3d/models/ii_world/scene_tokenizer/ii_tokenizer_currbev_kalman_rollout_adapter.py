import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_prior_slot import IISceneTokenizerCurrBevKalmanPriorSlot


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanRolloutAdapter(IISceneTokenizerCurrBevKalmanPriorSlot):
    """Sequential t-4 -> t Kalman rollout in curr_bev space.

    Design:
    - keep encoder / spatial VQ / decoder from the baseline
    - move temporal fusion into a causal Kalman rollout over the full history
      window `[t-4, ..., t]`
    - read out the final posterior `x_t^+` through a tiny adapter before the
      baseline spatial-only VQ/decoder
    """

    def __init__(self,
                 kalman_readout_hidden=None,
                 kalman_readout_use_gain=True,
                 kalman_readout_use_reliability=True,
                 kalman_readout_use_dynamic_ratio=True,
                 kalman_readout_use_valid_ratio=True,
                 kalman_latent_repair_loss_weight=1.0,
                 kalman_latent_keep_loss_weight=0.25,
                 kalman_dynamic_boost=1.0,
                 kalman_bad_threshold=0.5,
                 kalman_debug_decode_state=False,
                 **kwargs):
        super().__init__(**kwargs)

        hidden = kalman_readout_hidden or self.vq_channel * 2
        self.kalman_readout_use_gain = kalman_readout_use_gain
        self.kalman_readout_use_reliability = kalman_readout_use_reliability
        self.kalman_readout_use_dynamic_ratio = kalman_readout_use_dynamic_ratio
        self.kalman_readout_use_valid_ratio = kalman_readout_use_valid_ratio
        self.kalman_latent_repair_loss_weight = kalman_latent_repair_loss_weight
        self.kalman_latent_keep_loss_weight = kalman_latent_keep_loss_weight
        self.kalman_dynamic_boost = kalman_dynamic_boost
        self.kalman_bad_threshold = kalman_bad_threshold
        self.kalman_debug_decode_state = kalman_debug_decode_state

        adapter_in_channels = self.vq_channel * 3
        adapter_in_channels += 1 if kalman_readout_use_gain else 0
        adapter_in_channels += 1 if kalman_readout_use_reliability else 0
        adapter_in_channels += 1 if kalman_readout_use_dynamic_ratio else 0
        adapter_in_channels += 1 if kalman_readout_use_valid_ratio else 0

        self.kalman_readout_adapter = nn.Sequential(
            nn.Conv2d(adapter_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.vq_channel, 1),
        )
        nn.init.zeros_(self.kalman_readout_adapter[-1].weight)
        nn.init.zeros_(self.kalman_readout_adapter[-1].bias)

    def _normalize_flow_sequence_tensor(self, tensor, batch_size, has_vec):
        tensor = self.normalize_batched_tensor(tensor, batch_size)
        expected_dim = 6 if has_vec else 5
        if tensor.dim() == expected_dim - 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == expected_dim and tensor.shape[0] != batch_size and tensor.shape[1] == batch_size:
            tensor = tensor.transpose(0, 1)
        return tensor

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

    def _build_readout_feature(self,
                               posterior_mean,
                               curr_bev,
                               gain_group,
                               reliability_latent=None,
                               dynamic_ratio=None,
                               valid_ratio=None):
        inputs = [
            posterior_mean,
            curr_bev,
            (posterior_mean - curr_bev).abs(),
        ]
        if self.kalman_readout_use_gain:
            inputs.append(gain_group.mean(dim=1, keepdim=True))
        if self.kalman_readout_use_reliability:
            if reliability_latent is None:
                reliability_latent = posterior_mean.new_ones(
                    posterior_mean.shape[0], 1, posterior_mean.shape[2], posterior_mean.shape[3])
            inputs.append(1.0 - reliability_latent)
        if self.kalman_readout_use_dynamic_ratio:
            if dynamic_ratio is None:
                dynamic_ratio = posterior_mean.new_zeros(
                    posterior_mean.shape[0], 1, posterior_mean.shape[2], posterior_mean.shape[3])
            inputs.append(dynamic_ratio)
        if self.kalman_readout_use_valid_ratio:
            if valid_ratio is None:
                valid_ratio = posterior_mean.new_ones(
                    posterior_mean.shape[0], 1, posterior_mean.shape[2], posterior_mean.shape[3])
            inputs.append(valid_ratio)
        return torch.cat(inputs, dim=1)

    def _decode_bev_feature(self, bev_feature, shapes, input_shape):
        temporal_context = bev_feature.new_zeros((
            bev_feature.shape[0],
            self.vq.recover_time,
            bev_feature.shape[1],
            bev_feature.shape[2],
            bev_feature.shape[3],
        ))
        latent_q, _, _ = self.vq(bev_feature, temporal_context, is_voxel=False)
        return self.forward_decoder(latent_q, shapes, input_shape)

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

        readout_feature = self._build_readout_feature(
            posterior_mean,
            curr_bev,
            last_gain_group,
            reliability_latent=last_reliability_latent,
            dynamic_ratio=last_dynamic_ratio,
            valid_ratio=last_valid_ratio,
        )
        readout_delta = self.kalman_readout_adapter(readout_feature)
        bev_for_vq = posterior_mean + readout_delta

        temporal_context = bev_for_vq.new_zeros((
            bev_for_vq.shape[0],
            self.vq.recover_time,
            bev_for_vq.shape[1],
            bev_for_vq.shape[2],
            bev_for_vq.shape[3],
        ))
        z_sampled, embed_loss, _ = self.vq(bev_for_vq, temporal_context, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        debug = dict(
            bev_seq=bev_seq,
            shape_seq=shape_seq,
            curr_bev=curr_bev,
            clean_curr_bev=clean_curr_bev,
            curr_shapes=curr_shapes,
            last_prior_mean=last_prior_mean,
            last_prior_cov=last_prior_cov,
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            last_gain_group=last_gain_group,
            last_valid_ratio=last_valid_ratio,
            last_dynamic_ratio=last_dynamic_ratio,
            last_reliability_latent=last_reliability_latent,
            bev_for_vq=bev_for_vq,
            readout_delta=readout_delta,
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
                      **kwargs):
        batch_size = len(img_metas)
        voxel_semantics = self._normalize_sequence_tensor(voxel_semantics, batch_size)
        voxel_semantics_clean = kwargs.get('voxel_semantics_clean', voxel_semantics)
        voxel_semantics_clean = self._normalize_sequence_tensor(voxel_semantics_clean, batch_size)
        if frame_reliability_map is not None:
            frame_reliability_map = self.normalize_frame_reliability_map(frame_reliability_map, batch_size)

        oracle_flow_seq = self._normalize_flow_sequence_tensor(oracle_flow_seq, batch_size, has_vec=True)
        oracle_flow_valid_seq = self._normalize_flow_sequence_tensor(oracle_flow_valid_seq, batch_size, has_vec=False)
        oracle_dynamic_mask_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)

        voxel_semantics = self._to_model_device(voxel_semantics)
        voxel_semantics_clean = self._to_model_device(voxel_semantics_clean)
        oracle_flow_seq = self._to_model_device(oracle_flow_seq, dtype=torch.float32)
        oracle_flow_valid_seq = self._to_model_device(oracle_flow_valid_seq)
        oracle_dynamic_mask_seq = self._to_model_device(oracle_dynamic_mask_seq)
        frame_reliability_map = self._to_model_device(frame_reliability_map, dtype=torch.float32)

        logits, embed_loss, curr_target, debug = self._forward_rollout(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow_seq,
            oracle_flow_valid_seq,
            oracle_dynamic_mask_seq,
            frame_reliability_map=frame_reliability_map,
        )

        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, curr_target))
        loss_dict['embed_loss'] = self.embed_loss_weight * embed_loss

        current_reliability = None
        if frame_reliability_map is not None:
            current_reliability = frame_reliability_map[:, -1] if frame_reliability_map.dim() == 4 else frame_reliability_map
        bad_mask_latent = self._derive_bad_mask_latent(
            voxel_semantics[:, -1:],
            curr_target,
            current_reliability,
            debug['posterior_mean'].shape[-2:])
        good_mask_latent = 1.0 - bad_mask_latent

        if self.kalman_latent_repair_loss_weight > 0:
            latent_repair = self._masked_latent_loss(debug['bev_for_vq'], debug['clean_curr_bev'], bad_mask_latent)
            loss_dict['kalman_latent_repair_loss'] = self.kalman_latent_repair_loss_weight * latent_repair

        if self.kalman_dynamic_boost > 0:
            dynamic_mask_latent = (debug['last_dynamic_ratio'] > 0).float()
            latent_dynamic = self._masked_latent_loss(
                debug['bev_for_vq'],
                debug['clean_curr_bev'],
                bad_mask_latent * dynamic_mask_latent,
            )
            loss_dict['kalman_dynamic_latent_loss'] = self.kalman_dynamic_boost * latent_dynamic

        if self.kalman_latent_keep_loss_weight > 0:
            latent_keep = self._masked_latent_loss(debug['bev_for_vq'], debug['curr_bev'], good_mask_latent)
            loss_dict['kalman_latent_keep_loss'] = self.kalman_latent_keep_loss_weight * latent_keep

        loss_dict['kalman_q_mean'] = debug['step_debug'][-1]['process_noise'].mean().detach()
        loss_dict['kalman_r_mean'] = debug['step_debug'][-1]['observation_noise'].mean().detach()
        loss_dict['kalman_gain_mean'] = debug['last_gain_group'].mean().detach()
        loss_dict['kalman_readout_delta_norm'] = debug['readout_delta'].pow(2).sum(dim=1).sqrt().mean().detach()
        return loss_dict

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_flow_seq,
                     oracle_flow_valid_seq,
                     oracle_dynamic_mask_seq,
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

        oracle_flow_seq = self._normalize_flow_sequence_tensor(oracle_flow_seq, batch_size, has_vec=True)
        oracle_flow_valid_seq = self._normalize_flow_sequence_tensor(oracle_flow_valid_seq, batch_size, has_vec=False)
        oracle_dynamic_mask_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)

        voxel_semantics = self._to_model_device(voxel_semantics)
        voxel_semantics_clean = self._to_model_device(voxel_semantics_clean)
        oracle_flow_seq = self._to_model_device(oracle_flow_seq, dtype=torch.float32)
        oracle_flow_valid_seq = self._to_model_device(oracle_flow_valid_seq)
        oracle_dynamic_mask_seq = self._to_model_device(oracle_dynamic_mask_seq)
        frame_reliability_map = self._to_model_device(frame_reliability_map, dtype=torch.float32)

        logits, _, curr_target, _ = self._forward_rollout(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow_seq,
            oracle_flow_valid_seq,
            oracle_dynamic_mask_seq,
            frame_reliability_map=frame_reliability_map,
        )
        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        output_dict = dict(
            semantics=pred,
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=voxel_semantics[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
        )
        return [output_dict]

    @torch.no_grad()
    def collect_rollout_debug(self,
                              voxel_semantics,
                              voxel_semantics_clean,
                              img_metas,
                              oracle_flow_seq,
                              oracle_flow_valid_seq,
                              oracle_dynamic_mask_seq,
                              frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        voxel_semantics = self._normalize_sequence_tensor(voxel_semantics, batch_size)
        voxel_semantics_clean = self._normalize_sequence_tensor(voxel_semantics_clean, batch_size)
        if frame_reliability_map is not None:
            frame_reliability_map = self.normalize_frame_reliability_map(frame_reliability_map, batch_size)

        oracle_flow_seq = self._normalize_flow_sequence_tensor(oracle_flow_seq, batch_size, has_vec=True)
        oracle_flow_valid_seq = self._normalize_flow_sequence_tensor(oracle_flow_valid_seq, batch_size, has_vec=False)
        oracle_dynamic_mask_seq = self._normalize_flow_sequence_tensor(oracle_dynamic_mask_seq, batch_size, has_vec=False)

        voxel_semantics = self._to_model_device(voxel_semantics)
        voxel_semantics_clean = self._to_model_device(voxel_semantics_clean)
        oracle_flow_seq = self._to_model_device(oracle_flow_seq, dtype=torch.float32)
        oracle_flow_valid_seq = self._to_model_device(oracle_flow_valid_seq)
        oracle_dynamic_mask_seq = self._to_model_device(oracle_dynamic_mask_seq)
        frame_reliability_map = self._to_model_device(frame_reliability_map, dtype=torch.float32)

        logits, _, curr_target, debug = self._forward_rollout(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow_seq,
            oracle_flow_valid_seq,
            oracle_dynamic_mask_seq,
            frame_reliability_map=frame_reliability_map,
        )

        curr_decode_logits = None
        prior_decode_logits = None
        posterior_decode_logits = None
        init_decode_logits = None
        step_decode_debug = []
        if self.kalman_debug_decode_state:
            input_shape = (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4])
            init_decode_logits = self._decode_bev_feature(debug['bev_seq'][0], debug['shape_seq'][0], input_shape)
            curr_decode_logits = self._decode_bev_feature(debug['curr_bev'], debug['curr_shapes'], input_shape)
            prior_decode_logits = self._decode_bev_feature(debug['last_prior_mean'], debug['curr_shapes'], input_shape)
            posterior_decode_logits = self._decode_bev_feature(debug['bev_for_vq'], debug['curr_shapes'], input_shape)
            for step_idx, step in enumerate(debug['step_debug'], start=1):
                obs_decode_logits = self._decode_bev_feature(debug['bev_seq'][step_idx], debug['shape_seq'][step_idx], input_shape)
                step_prior_decode_logits = self._decode_bev_feature(step['prior_mean'], debug['shape_seq'][step_idx], input_shape)
                step_post_decode_logits = self._decode_bev_feature(step['posterior_mean'], debug['shape_seq'][step_idx], input_shape)
                step_decode_debug.append(dict(
                    observation_decode_pred=obs_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
                    prior_decode_pred=step_prior_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
                    posterior_decode_pred=step_post_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
                ))

        return dict(
            voxel_semantics_seq=voxel_semantics.detach().cpu().numpy().astype(np.uint8),
            voxel_semantics_clean_seq=voxel_semantics_clean.detach().cpu().numpy().astype(np.uint8),
            curr_input=voxel_semantics[:, -1:].detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            rollout_pred=logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            init_decode_pred=(
                None if init_decode_logits is None
                else init_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            curr_decode_pred=(
                None if curr_decode_logits is None
                else curr_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            prior_decode_pred=(
                None if prior_decode_logits is None
                else prior_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            posterior_decode_pred=(
                None if posterior_decode_logits is None
                else posterior_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            bev_for_vq=debug['bev_for_vq'].detach().cpu().numpy(),
            posterior_mean=debug['posterior_mean'].detach().cpu().numpy(),
            last_prior_mean=debug['last_prior_mean'].detach().cpu().numpy(),
            last_gain_group=debug['last_gain_group'].detach().cpu().numpy(),
            last_valid_ratio=debug['last_valid_ratio'].detach().cpu().numpy(),
            last_dynamic_ratio=debug['last_dynamic_ratio'].detach().cpu().numpy(),
            last_reliability_latent=(
                None if debug['last_reliability_latent'] is None
                else debug['last_reliability_latent'].detach().cpu().numpy()
            ),
            step_debug=[
                dict(
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
            oracle_flow_seq=oracle_flow_seq.detach().cpu().numpy(),
            oracle_flow_valid_seq=oracle_flow_valid_seq.detach().cpu().numpy(),
            oracle_dynamic_mask_seq=oracle_dynamic_mask_seq.detach().cpu().numpy(),
        )
