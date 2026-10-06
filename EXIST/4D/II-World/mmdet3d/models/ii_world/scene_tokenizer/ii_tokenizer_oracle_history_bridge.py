import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer import IISceneTokenizer


@DETECTORS.register_module()
class IISceneTokenizerOracleHistoryBridge(IISceneTokenizer):
    """Oracle-history bridge that preserves the baseline VQ/decoder path.

    The key idea is to keep the original tokenizer main path unchanged:
    `curr_bev -> align_bev -> vq(curr_bev, sampled_bev) -> decoder`.
    Oracle information is injected only by replacing/blending the newest
    history slot inside ``sampled_bev`` using a GT-flow-warped previous clean
    BEV feature.
    """

    def __init__(self,
                 oracle_bridge_hidden=None,
                 oracle_bridge_slot=0,
                 oracle_bridge_use_clean_cache=True,
                 oracle_bridge_use_flow_mag=True,
                 oracle_bridge_use_valid_ratio=True,
                 oracle_bridge_use_dynamic_ratio=True,
                 oracle_bridge_use_reliability=True,
                 oracle_bridge_repair_loss_weight=0.0,
                 oracle_bridge_keep_baseline_loss_weight=0.0,
                 oracle_bridge_bad_threshold=0.5,
                 oracle_bridge_dynamic_repair_boost=0.0,
                 oracle_bridge_debug_decode_oracle=False,
                 **kwargs):
        super().__init__(**kwargs)
        hidden = oracle_bridge_hidden or self.vq_channel
        aux_channels = 0
        aux_channels += 1 if oracle_bridge_use_flow_mag else 0
        aux_channels += 1 if oracle_bridge_use_valid_ratio else 0
        aux_channels += 1 if oracle_bridge_use_dynamic_ratio else 0
        aux_channels += 1 if oracle_bridge_use_reliability else 0
        gate_in_channels = self.vq_channel * 3 + aux_channels

        self.oracle_bridge_slot = oracle_bridge_slot
        self.oracle_bridge_use_clean_cache = oracle_bridge_use_clean_cache
        self.oracle_bridge_use_flow_mag = oracle_bridge_use_flow_mag
        self.oracle_bridge_use_valid_ratio = oracle_bridge_use_valid_ratio
        self.oracle_bridge_use_dynamic_ratio = oracle_bridge_use_dynamic_ratio
        self.oracle_bridge_use_reliability = oracle_bridge_use_reliability
        self.oracle_bridge_repair_loss_weight = oracle_bridge_repair_loss_weight
        self.oracle_bridge_keep_baseline_loss_weight = oracle_bridge_keep_baseline_loss_weight
        self.oracle_bridge_bad_threshold = oracle_bridge_bad_threshold
        self.oracle_bridge_dynamic_repair_boost = oracle_bridge_dynamic_repair_boost
        self.oracle_bridge_debug_decode_oracle = oracle_bridge_debug_decode_oracle

        self.oracle_bridge_alpha_head = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.oracle_bridge_alpha_head[-1].weight)
        nn.init.constant_(self.oracle_bridge_alpha_head[-1].bias, -3.0)

    def _normalize_sequence_tensor(self, tensor, batch_size):
        tensor = self.normalize_batched_tensor(tensor, batch_size)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)
        return tensor

    def _model_device(self):
        return self.class_embeds.weight.device

    def _to_model_device(self, tensor, dtype=None):
        if tensor is None:
            return None
        device = self._model_device()
        if dtype is None:
            return tensor.to(device)
        return tensor.to(device=device, dtype=dtype)

    def _split_pairwise_inputs(self, voxel_semantics, voxel_semantics_clean):
        if voxel_semantics.shape[1] < 2:
            raise ValueError(
                f'IISceneTokenizerOracleHistoryBridge expects at least 2 frames '
                f'(prev + current), got shape {tuple(voxel_semantics.shape)}')
        prev_clean = voxel_semantics_clean[:, -2:-1]
        curr_input = voxel_semantics[:, -1:]
        curr_target = voxel_semantics_clean[:, -1:]
        return prev_clean, curr_input, curr_target

    def _pool_current_map(self, current_map, target_hw):
        if current_map is None:
            return None
        if current_map.dim() == 4:
            current_map = current_map[:, -1]
        if current_map.dim() != 3:
            raise ValueError(f'Expected reliability map [B,T,H,W] or [B,H,W], got {tuple(current_map.shape)}')
        return F.adaptive_avg_pool2d(current_map.unsqueeze(1), target_hw)

    def _aggregate_flow_to_latent(self, flow, valid, dynamic_mask, target_hw):
        if flow.dim() != 5:
            raise ValueError(f'Expected oracle_flow [B,H,W,Z,3], got {tuple(flow.shape)}')
        if valid.dim() != 4 or dynamic_mask.dim() != 4:
            raise ValueError(
                f'Expected oracle_flow_valid/oracle_dynamic_mask [B,H,W,Z], '
                f'got {tuple(valid.shape)} and {tuple(dynamic_mask.shape)}')

        batch, occ_h, occ_w, occ_z, _ = flow.shape
        latent_h, latent_w = target_hw
        if occ_h % latent_h != 0 or occ_w % latent_w != 0:
            raise ValueError(
                f'Occupancy grid {(occ_h, occ_w)} is not divisible by latent grid {(latent_h, latent_w)}')
        scale_h = occ_h // latent_h
        scale_w = occ_w // latent_w

        flow_xy = flow[..., :2]
        valid_f = valid.float().unsqueeze(-1)
        dynamic_f = dynamic_mask.float().unsqueeze(-1)

        flow_xy = (flow_xy * valid_f).reshape(batch, latent_h, scale_h, latent_w, scale_w, occ_z, 2)
        valid_f = valid_f.reshape(batch, latent_h, scale_h, latent_w, scale_w, occ_z, 1)
        dynamic_f = dynamic_f.reshape(batch, latent_h, scale_h, latent_w, scale_w, occ_z, 1)

        flow_sum = flow_xy.sum(dim=(2, 4, 5))
        valid_sum = valid_f.sum(dim=(2, 4, 5)).clamp_min(1.0)
        flow_latent = flow_sum / valid_sum
        # Flow GT is stored in occupancy-voxel displacement units on the
        # original 200x200 grid. After pooling to the 50x50 BEV latent grid,
        # convert the offsets into latent-cell units before using them in
        # grid_sample-based warping.
        flow_latent[..., 0] = flow_latent[..., 0] / float(scale_h)
        flow_latent[..., 1] = flow_latent[..., 1] / float(scale_w)
        valid_ratio = valid_f.mean(dim=(2, 4, 5)).squeeze(-1)
        dynamic_ratio = dynamic_f.mean(dim=(2, 4, 5)).squeeze(-1)

        flow_latent = flow_latent.permute(0, 3, 1, 2).contiguous()
        valid_ratio = valid_ratio.unsqueeze(1).contiguous()
        dynamic_ratio = dynamic_ratio.unsqueeze(1).contiguous()
        return flow_latent, valid_ratio, dynamic_ratio

    def _warp_feature(self, feature, backward_flow):
        batch, _, height, width = feature.shape
        device = feature.device
        dtype = feature.dtype
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij')

        src_y = y_coords.unsqueeze(0) + backward_flow[:, 0]
        src_x = x_coords.unsqueeze(0) + backward_flow[:, 1]

        if width > 1:
            norm_x = (src_x / (width - 1.0)) * 2.0 - 1.0
        else:
            norm_x = torch.zeros_like(src_x)
        if height > 1:
            norm_y = (src_y / (height - 1.0)) * 2.0 - 1.0
        else:
            norm_y = torch.zeros_like(src_y)

        grid = torch.stack([norm_x, norm_y], dim=-1)
        return F.grid_sample(
            feature,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

    def _build_bridge_alpha(self, baseline_history, oracle_history, flow_latent,
                            valid_ratio, dynamic_ratio, reliability_latent=None):
        aux_inputs = [baseline_history, oracle_history, (oracle_history - baseline_history).abs()]
        if self.oracle_bridge_use_flow_mag:
            aux_inputs.append(flow_latent.norm(dim=1, keepdim=True))
        if self.oracle_bridge_use_valid_ratio:
            aux_inputs.append(valid_ratio)
        if self.oracle_bridge_use_dynamic_ratio:
            aux_inputs.append(dynamic_ratio)
        if self.oracle_bridge_use_reliability:
            if reliability_latent is None:
                reliability_latent = valid_ratio.new_ones(valid_ratio.shape)
            aux_inputs.append(1.0 - reliability_latent)

        alpha = torch.sigmoid(self.oracle_bridge_alpha_head(torch.cat(aux_inputs, dim=1)))
        return alpha * valid_ratio

    def _inject_oracle_history(self, sampled_bev, oracle_history_bev, alpha):
        sampled_bev_bridge = sampled_bev.clone()
        slot = self.oracle_bridge_slot
        sampled_bev_bridge[:, slot] = (1.0 - alpha) * sampled_bev[:, slot] + alpha * oracle_history_bev
        return sampled_bev_bridge

    def _masked_reconstruct_loss(self, logits, target, voxel_mask):
        if voxel_mask is None:
            return logits.sum() * 0.0
        valid_count = int(voxel_mask.bool().sum().item())
        if valid_count <= 1:
            return logits.sum() * 0.0
        masked_target = target.clone()
        masked_target = masked_target.masked_fill(~voxel_mask.bool(), 255)
        return self.reconstruct_loss(logits, masked_target)['recon_loss']

    def _forward_bridge_pair(self, voxel_semantics, voxel_semantics_clean, img_metas,
                             oracle_flow, oracle_flow_valid, oracle_dynamic_mask,
                             frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        prev_clean, curr_input, curr_target = self._split_pairwise_inputs(
            voxel_semantics, voxel_semantics_clean)

        _, _, occ_h, occ_w, occ_z = curr_target.shape
        curr_bev, curr_shapes = self.forward_encoder(curr_input)
        prev_bev, _ = self.forward_encoder(prev_clean)
        clean_curr_bev = None
        if self.oracle_bridge_use_clean_cache:
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
        oracle_history_bev = self._warp_feature(prev_bev, flow_latent)
        alpha = self._build_bridge_alpha(
            sampled_bev[:, self.oracle_bridge_slot],
            oracle_history_bev,
            flow_latent,
            valid_ratio,
            dynamic_ratio,
            reliability_latent=reliability_latent,
        )
        sampled_bev_bridge = self._inject_oracle_history(sampled_bev, oracle_history_bev, alpha)

        z_sampled, loss, info = self.vq(curr_bev, sampled_bev_bridge, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))
        debug = dict(
            curr_bev=curr_bev,
            curr_shapes=curr_shapes,
            prev_bev=prev_bev,
            sampled_bev=sampled_bev,
            sampled_bev_bridge=sampled_bev_bridge,
            oracle_history_bev=oracle_history_bev,
            alpha=alpha,
            flow_latent=flow_latent,
            valid_ratio=valid_ratio,
            dynamic_ratio=dynamic_ratio,
            reliability_latent=reliability_latent,
            sampled_reliability=sampled_reliability,
        )
        return logits, loss, curr_target, debug

    @torch.no_grad()
    def collect_bridge_debug(self,
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
        logits, _, _, debug = self._forward_bridge_pair(
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
            self.forward_encoder(curr_input)[1],
            (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4]),
        )

        def _decode_bev_feature(bev_feature, shapes, input_shape):
            latent = self._quantize_latent_for_decode(bev_feature)
            return self.forward_decoder(latent, shapes, input_shape)

        curr_bev_shapes = self.forward_encoder(curr_input)[1]
        input_shape = (batch_size, 1, curr_target.shape[2], curr_target.shape[3], curr_target.shape[4])
        oracle_history_logits = None
        baseline_history_logits = None
        bridge_history_logits = None
        if self.oracle_bridge_debug_decode_oracle:
            oracle_history_logits = _decode_bev_feature(debug['oracle_history_bev'], curr_bev_shapes, input_shape)
            baseline_history_logits = _decode_bev_feature(
                debug['sampled_bev'][:, self.oracle_bridge_slot], curr_bev_shapes, input_shape)
            bridge_history_logits = _decode_bev_feature(
                debug['sampled_bev_bridge'][:, self.oracle_bridge_slot], curr_bev_shapes, input_shape)

        return dict(
            prev_clean=prev_clean.detach().cpu().numpy().astype(np.uint8),
            curr_input=curr_input.detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            bridge_pred=logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            baseline_pred=baseline_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8),
            oracle_history_pred=(
                None if oracle_history_logits is None
                else oracle_history_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            baseline_history_pred=(
                None if baseline_history_logits is None
                else baseline_history_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            bridge_history_pred=(
                None if bridge_history_logits is None
                else bridge_history_logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)
            ),
            alpha=debug['alpha'].detach().cpu().numpy(),
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
            baseline_history_bev=debug['sampled_bev'][:, self.oracle_bridge_slot].detach().cpu().numpy(),
            oracle_history_bev=debug['oracle_history_bev'].detach().cpu().numpy(),
            bridge_history_bev=debug['sampled_bev_bridge'][:, self.oracle_bridge_slot].detach().cpu().numpy(),
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

        logits, loss, curr_target, debug = self._forward_bridge_pair(
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

        if (self.oracle_bridge_repair_loss_weight > 0 or
                self.oracle_bridge_keep_baseline_loss_weight > 0):
            current_reliability = None
            if frame_reliability_map is not None:
                current_reliability = frame_reliability_map[:, -1] if frame_reliability_map.dim() == 4 else frame_reliability_map

            if current_reliability is not None:
                bad_mask = (current_reliability < self.oracle_bridge_bad_threshold).unsqueeze(1).unsqueeze(-1)
                bad_mask = bad_mask.expand_as(curr_target)
                good_mask = ~bad_mask

                if self.oracle_bridge_repair_loss_weight > 0:
                    repair_loss = self._masked_reconstruct_loss(logits, curr_target, bad_mask)
                    loss_dict['repair_recon_loss'] = self.oracle_bridge_repair_loss_weight * repair_loss

                    if self.oracle_bridge_dynamic_repair_boost > 0:
                        dynamic_mask = oracle_dynamic_mask.bool().unsqueeze(1).expand_as(curr_target)
                        bad_dynamic_mask = bad_mask & dynamic_mask
                        bad_dynamic_loss = self._masked_reconstruct_loss(logits, curr_target, bad_dynamic_mask)
                        loss_dict['repair_dynamic_loss'] = (
                            self.oracle_bridge_dynamic_repair_boost * bad_dynamic_loss
                        )

                if self.oracle_bridge_keep_baseline_loss_weight > 0:
                    with torch.no_grad():
                        baseline_latent = self.vq(debug['curr_bev'], debug['sampled_bev'], is_voxel=False)[0]
                        baseline_logits = self.forward_decoder(
                            baseline_latent,
                            debug['curr_shapes'],
                            curr_target.shape,
                        )
                        baseline_target = baseline_logits.softmax(-1).argmax(-1)
                    keep_loss = self._masked_reconstruct_loss(logits, baseline_target, good_mask)
                    loss_dict['keep_baseline_loss'] = self.oracle_bridge_keep_baseline_loss_weight * keep_loss
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

        start_time = time.time()
        logits, _, curr_target, _ = self._forward_bridge_pair(
            voxel_semantics,
            voxel_semantics_clean,
            img_metas,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
            frame_reliability_map=frame_reliability_map,
        )
        end_time = time.time()

        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        output_dict = dict(
            semantics=pred,
            target=curr_target.cpu().numpy().astype(np.uint8),
            input_curr_semantics=voxel_semantics[:, -1].cpu().numpy().astype(np.uint8),
            index=[img_meta['index'] for img_meta in img_metas],
            time=end_time - start_time,
        )
        return [output_dict]
