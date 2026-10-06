import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_oracle_history_bridge import IISceneTokenizerOracleHistoryBridge


@DETECTORS.register_module()
class IISceneTokenizerCurrBevRecoveryMLP(IISceneTokenizerOracleHistoryBridge):
    """Frozen-backbone curr_bev recovery mixer.

    This design keeps the baseline VQ/decoder path unchanged:
      recovered_curr_bev + sampled_bev -> baseline VQ -> decoder

    The only new learnable component is a recovery mixer operating in
    `curr_bev` space. It receives:
      - corrupted current BEV feature
      - aligned 4-slot history BEV feature stack
      - optional aux maps (flow magnitude / valid ratio / dynamic ratio /
        reliability)

    and predicts a residual correction:
      recovered_curr_bev = curr_bev + delta_bev

    Because encoder is frozen, training can directly supervise against the
    clean latent teacher:
      clean_curr_bev = Encoder(curr_clean)
    """

    def __init__(self,
                 recovery_hidden=None,
                 recovery_use_clean_cache=True,
                 recovery_use_flow_mag=True,
                 recovery_use_valid_ratio=True,
                 recovery_use_dynamic_ratio=True,
                 recovery_use_reliability=True,
                 recovery_latent_repair_loss_weight=1.0,
                 recovery_latent_keep_loss_weight=0.25,
                 recovery_dynamic_boost=0.0,
                 recovery_bad_threshold=0.5,
                 recovery_debug_decode_state=False,
                 **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, 'oracle_bridge_alpha_head'):
            del self.oracle_bridge_alpha_head

        hidden = recovery_hidden or self.vq_channel * 2
        aux_channels = 0
        aux_channels += 1 if recovery_use_flow_mag else 0
        aux_channels += 1 if recovery_use_valid_ratio else 0
        aux_channels += 1 if recovery_use_dynamic_ratio else 0
        aux_channels += 1 if recovery_use_reliability else 0
        in_channels = self.vq_channel * (1 + self.frame_number) + aux_channels

        self.recovery_use_clean_cache = recovery_use_clean_cache
        self.recovery_use_flow_mag = recovery_use_flow_mag
        self.recovery_use_valid_ratio = recovery_use_valid_ratio
        self.recovery_use_dynamic_ratio = recovery_use_dynamic_ratio
        self.recovery_use_reliability = recovery_use_reliability
        self.recovery_latent_repair_loss_weight = recovery_latent_repair_loss_weight
        self.recovery_latent_keep_loss_weight = recovery_latent_keep_loss_weight
        self.recovery_dynamic_boost = recovery_dynamic_boost
        self.recovery_bad_threshold = recovery_bad_threshold
        self.recovery_debug_decode_state = recovery_debug_decode_state

        self.recovery_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.vq_channel, 1),
        )
        nn.init.zeros_(self.recovery_mlp[-1].weight)
        nn.init.zeros_(self.recovery_mlp[-1].bias)

    def _flatten_history(self, sampled_bev):
        batch_size, num_hist, channels, height, width = sampled_bev.shape
        return sampled_bev.reshape(batch_size, num_hist * channels, height, width)

    def _build_recovery_input(self,
                              curr_bev,
                              sampled_bev,
                              flow_latent,
                              valid_ratio,
                              dynamic_ratio,
                              reliability_latent=None):
        inputs = [curr_bev, self._flatten_history(sampled_bev)]
        if self.recovery_use_flow_mag:
            inputs.append(flow_latent.norm(dim=1, keepdim=True))
        if self.recovery_use_valid_ratio:
            inputs.append(valid_ratio)
        if self.recovery_use_dynamic_ratio:
            inputs.append(dynamic_ratio)
        if self.recovery_use_reliability:
            if reliability_latent is None:
                reliability_latent = curr_bev.new_ones(curr_bev.shape[0], 1, curr_bev.shape[2], curr_bev.shape[3])
            inputs.append(1.0 - reliability_latent)
        return torch.cat(inputs, dim=1)

    def _recover_curr_bev(self,
                          curr_bev,
                          sampled_bev,
                          flow_latent,
                          valid_ratio,
                          dynamic_ratio,
                          reliability_latent=None):
        recovery_input = self._build_recovery_input(
            curr_bev,
            sampled_bev,
            flow_latent,
            valid_ratio,
            dynamic_ratio,
            reliability_latent=reliability_latent,
        )
        delta_bev = self.recovery_mlp(recovery_input)
        recovered_curr_bev = curr_bev + delta_bev
        return recovered_curr_bev, delta_bev

    def _derive_bad_mask_latent(self, curr_input, curr_target, frame_reliability_map, target_hw):
        if frame_reliability_map is not None:
            reliability_latent = self._pool_current_map(frame_reliability_map, target_hw)
            return (reliability_latent < self.recovery_bad_threshold).float()

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

    def _forward_recovery_pair(self, voxel_semantics, voxel_semantics_clean, img_metas,
                               oracle_flow, oracle_flow_valid, oracle_dynamic_mask,
                               frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        prev_clean, curr_input, curr_target = self._split_pairwise_inputs(
            voxel_semantics, voxel_semantics_clean)

        _, _, occ_h, occ_w, occ_z = curr_target.shape
        curr_bev, curr_shapes = self.forward_encoder(curr_input)
        prev_bev, _ = self.forward_encoder(prev_clean)
        clean_curr_bev, _ = self.forward_encoder(curr_target)

        sampled_bev, _, _, sampled_reliability = self.align_bev(
            curr_bev,
            img_metas,
            cache_bev=clean_curr_bev if self.recovery_use_clean_cache else curr_bev,
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
            curr_bev.shape[-2:],
        )

        recovered_curr_bev, delta_bev = self._recover_curr_bev(
            curr_bev,
            sampled_bev,
            flow_latent,
            valid_ratio,
            dynamic_ratio,
            reliability_latent=reliability_latent,
        )

        z_sampled, loss, _ = self.vq(recovered_curr_bev, sampled_bev, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        debug = dict(
            curr_bev=curr_bev,
            curr_shapes=curr_shapes,
            prev_bev=prev_bev,
            clean_curr_bev=clean_curr_bev,
            sampled_bev=sampled_bev,
            recovered_curr_bev=recovered_curr_bev,
            delta_bev=delta_bev,
            flow_latent=flow_latent,
            valid_ratio=valid_ratio,
            dynamic_ratio=dynamic_ratio,
            reliability_latent=reliability_latent,
            sampled_reliability=sampled_reliability,
        )
        return logits, loss, curr_target, debug

    @torch.no_grad()
    def collect_recovery_debug(self,
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
        logits, _, _, debug = self._forward_recovery_pair(
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

        recovered_decode_logits = None
        curr_decode_logits = None
        clean_decode_logits = None
        if self.recovery_debug_decode_state:
            input_shape = (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4])
            recovered_decode_logits = self._decode_bev_feature(debug['recovered_curr_bev'], debug['curr_shapes'], input_shape)
            curr_decode_logits = self._decode_bev_feature(debug['curr_bev'], debug['curr_shapes'], input_shape)
            clean_decode_logits = self._decode_bev_feature(debug['clean_curr_bev'], debug['curr_shapes'], input_shape)

        return dict(
            prev_clean=prev_clean.detach().cpu().numpy().astype(np.uint8),
            curr_input=curr_input.detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            recovery_pred=logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            baseline_pred=baseline_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            curr_decode_pred=(
                None if curr_decode_logits is None
                else curr_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            clean_decode_pred=(
                None if clean_decode_logits is None
                else clean_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            recovered_decode_pred=(
                None if recovered_decode_logits is None
                else recovered_decode_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            flow_latent=debug['flow_latent'].detach().cpu().numpy(),
            valid_ratio=debug['valid_ratio'].detach().cpu().numpy(),
            dynamic_ratio=debug['dynamic_ratio'].detach().cpu().numpy(),
            reliability_latent=(
                None if debug['reliability_latent'] is None
                else debug['reliability_latent'].detach().cpu().numpy()
            ),
            delta_norm=debug['delta_bev'].norm(dim=1).detach().cpu().numpy(),
            oracle_flow=oracle_flow.detach().cpu().numpy(),
            oracle_flow_valid=oracle_flow_valid.detach().cpu().numpy(),
            oracle_dynamic_mask=oracle_dynamic_mask.detach().cpu().numpy(),
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

        logits, loss, curr_target, debug = self._forward_recovery_pair(
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

        bad_mask_latent = self._derive_bad_mask_latent(
            curr_input=voxel_semantics[:, -1:],
            curr_target=curr_target,
            frame_reliability_map=frame_reliability_map,
            target_hw=debug['curr_bev'].shape[-2:],
        )
        if self.recovery_latent_repair_loss_weight > 0:
            latent_repair = self._masked_latent_loss(
                debug['recovered_curr_bev'],
                debug['clean_curr_bev'],
                bad_mask_latent,
            )
            loss_dict['recovery_latent_repair_loss'] = (
                self.recovery_latent_repair_loss_weight * latent_repair
            )

            if self.recovery_dynamic_boost > 0:
                dynamic_mask_latent = (debug['dynamic_ratio'] > 0).float()
                bad_dynamic_mask = bad_mask_latent * dynamic_mask_latent
                dynamic_repair = self._masked_latent_loss(
                    debug['recovered_curr_bev'],
                    debug['clean_curr_bev'],
                    bad_dynamic_mask,
                )
                loss_dict['recovery_dynamic_latent_loss'] = (
                    self.recovery_dynamic_boost * dynamic_repair
                )

        if self.recovery_latent_keep_loss_weight > 0:
            good_mask_latent = 1.0 - bad_mask_latent
            latent_keep = self._masked_latent_loss(
                debug['recovered_curr_bev'],
                debug['curr_bev'].detach(),
                good_mask_latent,
            )
            loss_dict['recovery_latent_keep_loss'] = (
                self.recovery_latent_keep_loss_weight * latent_keep
            )

        loss_dict['recovery_delta_norm'] = debug['delta_bev'].norm(dim=1).mean() * 0.0 + debug['delta_bev'].norm(dim=1).mean().detach()
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

        logits, _, curr_target, _ = self._forward_recovery_pair(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
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
