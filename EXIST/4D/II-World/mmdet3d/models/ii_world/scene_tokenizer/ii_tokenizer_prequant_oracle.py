# Modified from IISceneTokenizer for pairwise pre-quant oracle fusion.
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer import IISceneTokenizer


@DETECTORS.register_module()
class IISceneTokenizerPreQuantOracle(IISceneTokenizer):
    """Pairwise oracle-flow tokenizer with pre-quant latent fusion.

    The model keeps the recurrent state in the continuous pre-quant latent
    space. For the first oracle stage we do not roll out a recurrent cache; we
    only fuse one clean previous frame with one current frame. Oracle flow is
    supplied as current-grid backward flow, which is used to sample the
    previous pre-quant latent into the current frame.
    """

    def __init__(self,
                 oracle_gate_hidden=None,
                 oracle_prior_loss_weight=0.0,
                 oracle_quantize_readout=True,
                 oracle_use_flow_mag=True,
                 oracle_use_valid_ratio=True,
                 oracle_use_dynamic_ratio=True,
                 oracle_use_reliability=True,
                 **kwargs):
        super().__init__(**kwargs)
        hidden = oracle_gate_hidden or self.vq.e_dim
        aux_channels = 0
        aux_channels += 1 if oracle_use_flow_mag else 0
        aux_channels += 1 if oracle_use_valid_ratio else 0
        aux_channels += 1 if oracle_use_dynamic_ratio else 0
        aux_channels += 1 if oracle_use_reliability else 0
        gate_in_channels = self.vq.e_dim * 3 + aux_channels

        self.oracle_prior_loss_weight = oracle_prior_loss_weight
        self.oracle_quantize_readout = oracle_quantize_readout
        self.oracle_use_flow_mag = oracle_use_flow_mag
        self.oracle_use_valid_ratio = oracle_use_valid_ratio
        self.oracle_use_dynamic_ratio = oracle_use_dynamic_ratio
        self.oracle_use_reliability = oracle_use_reliability

        self.oracle_update_head = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.vq.e_dim, 1),
        )
        nn.init.zeros_(self.oracle_update_head[-1].weight)
        nn.init.constant_(self.oracle_update_head[-1].bias, -1.0)

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
                f'IISceneTokenizerPreQuantOracle expects at least 2 frames '
                f'(prev + current), got shape {tuple(voxel_semantics.shape)}')
        prev_clean = voxel_semantics_clean[:, -2:-1]
        curr_input = voxel_semantics[:, -1:]
        curr_target = voxel_semantics_clean[:, -1:]
        return prev_clean, curr_input, curr_target

    def _encode_prequant(self, voxel_semantics):
        latent, shapes = self.forward_encoder(voxel_semantics)
        return self.vq.quant_conv(latent), shapes

    def _pool_current_map(self, current_map, target_hw):
        if current_map is None:
            return None
        if current_map.dim() == 4:
            current_map = current_map[:, -1]
        if current_map.dim() != 3:
            raise ValueError(f'Expected reliability map [B,T,H,W] or [B,H,W], got {tuple(current_map.shape)}')
        current_map = current_map.unsqueeze(1)
        return F.adaptive_avg_pool2d(current_map, target_hw)

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
                f'Occupancy grid {(occ_h, occ_w)} is not divisible by latent grid '
                f'{(latent_h, latent_w)}')
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
        valid_ratio = valid_f.mean(dim=(2, 4, 5)).squeeze(-1)
        dynamic_ratio = dynamic_f.mean(dim=(2, 4, 5)).squeeze(-1)

        flow_latent = flow_latent.permute(0, 3, 1, 2).contiguous()
        valid_ratio = valid_ratio.unsqueeze(1).contiguous()
        dynamic_ratio = dynamic_ratio.unsqueeze(1).contiguous()
        return flow_latent, valid_ratio, dynamic_ratio

    def _warp_previous_to_current(self, prev_prequant, backward_flow, valid_ratio):
        batch, _, height, width = prev_prequant.shape
        device = prev_prequant.device
        dtype = prev_prequant.dtype
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
        warped = F.grid_sample(
            prev_prequant,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return warped * valid_ratio

    def _fuse_prequant(self, prior_prequant, curr_prequant, flow_latent, valid_ratio,
                       dynamic_ratio, reliability_latent=None):
        innovation = curr_prequant - prior_prequant
        gate_inputs = [prior_prequant, curr_prequant, innovation.abs()]
        if self.oracle_use_flow_mag:
            gate_inputs.append(flow_latent.norm(dim=1, keepdim=True))
        if self.oracle_use_valid_ratio:
            gate_inputs.append(valid_ratio)
        if self.oracle_use_dynamic_ratio:
            gate_inputs.append(dynamic_ratio)
        if self.oracle_use_reliability:
            if reliability_latent is None:
                reliability_latent = valid_ratio.new_ones(valid_ratio.shape)
            gate_inputs.append(1.0 - reliability_latent)

        gate = torch.sigmoid(self.oracle_update_head(torch.cat(gate_inputs, dim=1)))
        posterior_prequant = prior_prequant + gate * innovation
        return posterior_prequant, gate

    def _decode_prequant(self, prequant_latent, shapes, input_shape):
        if self.oracle_quantize_readout:
            quantized, embed_loss, _ = self.vq.forward_quantizer(prequant_latent, is_voxel=False)
            latent_for_decoder = self.vq.post_quant_conv(quantized)
        else:
            embed_loss = prequant_latent.sum() * 0.0
            latent_for_decoder = self.vq.post_quant_conv(prequant_latent)
        logits = self.forward_decoder(latent_for_decoder, shapes, input_shape)
        return logits, embed_loss

    def _forward_oracle_pair(self, voxel_semantics, voxel_semantics_clean, oracle_flow,
                             oracle_flow_valid, oracle_dynamic_mask, frame_reliability_map=None):
        batch_size = voxel_semantics.shape[0]
        prev_clean, curr_input, curr_target = self._split_pairwise_inputs(
            voxel_semantics, voxel_semantics_clean)

        _, _, occ_h, occ_w, occ_z = curr_target.shape
        prev_prequant, _ = self._encode_prequant(prev_clean)
        curr_prequant, curr_shapes = self._encode_prequant(curr_input)

        flow_latent, valid_ratio, dynamic_ratio = self._aggregate_flow_to_latent(
            oracle_flow.to(curr_prequant.device).to(curr_prequant.dtype),
            oracle_flow_valid.to(curr_prequant.device),
            oracle_dynamic_mask.to(curr_prequant.device),
            curr_prequant.shape[-2:],
        )
        reliability_latent = self._pool_current_map(
            frame_reliability_map.to(curr_prequant.device).to(curr_prequant.dtype)
            if frame_reliability_map is not None else None,
            curr_prequant.shape[-2:])

        prior_prequant = self._warp_previous_to_current(prev_prequant, flow_latent, valid_ratio)
        posterior_prequant, gate = self._fuse_prequant(
            prior_prequant,
            curr_prequant,
            flow_latent,
            valid_ratio,
            dynamic_ratio,
            reliability_latent=reliability_latent,
        )

        logits, embed_loss = self._decode_prequant(
            posterior_prequant,
            curr_shapes,
            (batch_size, 1, occ_h, occ_w, occ_z),
        )
        prior_logits = None
        if self.oracle_prior_loss_weight > 0:
            prior_logits, _ = self._decode_prequant(
                prior_prequant,
                curr_shapes,
                (batch_size, 1, occ_h, occ_w, occ_z),
            )
        debug = dict(
            prev_prequant=prev_prequant,
            curr_prequant=curr_prequant,
            gate=gate,
            prior_prequant=prior_prequant,
            posterior_prequant=posterior_prequant,
            flow_latent=flow_latent,
            valid_ratio=valid_ratio,
            dynamic_ratio=dynamic_ratio,
            reliability_latent=reliability_latent,
            prior_logits=prior_logits,
        )
        return logits, embed_loss, curr_target, debug

    @torch.no_grad()
    def collect_oracle_debug(self,
                             voxel_semantics,
                             voxel_semantics_clean,
                             oracle_flow,
                             oracle_flow_valid,
                             oracle_dynamic_mask,
                             frame_reliability_map=None):
        """Run the oracle pair path and expose decoded intermediate states.

        This helper is intended for offline visualization/debugging. It keeps
        the recurrent/path logic identical to training/test but additionally
        decodes the previous/current/prior/posterior pre-quant states so they
        can be visualized in occupancy space.
        """
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

        _, _, occ_h, occ_w, occ_z = curr_target.shape
        prev_prequant, prev_shapes = self._encode_prequant(prev_clean)
        curr_prequant, curr_shapes = self._encode_prequant(curr_input)
        flow_latent, valid_ratio, dynamic_ratio = self._aggregate_flow_to_latent(
            oracle_flow.to(curr_prequant.dtype),
            oracle_flow_valid,
            oracle_dynamic_mask,
            curr_prequant.shape[-2:],
        )
        reliability_latent = self._pool_current_map(
            frame_reliability_map if frame_reliability_map is not None else None,
            curr_prequant.shape[-2:])

        prior_prequant = self._warp_previous_to_current(prev_prequant, flow_latent, valid_ratio)
        posterior_prequant, gate = self._fuse_prequant(
            prior_prequant,
            curr_prequant,
            flow_latent,
            valid_ratio,
            dynamic_ratio,
            reliability_latent=reliability_latent,
        )

        prev_logits, _ = self._decode_prequant(
            prev_prequant,
            prev_shapes,
            (batch_size, 1, occ_h, occ_w, occ_z),
        )
        curr_logits, _ = self._decode_prequant(
            curr_prequant,
            curr_shapes,
            (batch_size, 1, occ_h, occ_w, occ_z),
        )
        prior_logits, _ = self._decode_prequant(
            prior_prequant,
            curr_shapes,
            (batch_size, 1, occ_h, occ_w, occ_z),
        )
        posterior_logits, _ = self._decode_prequant(
            posterior_prequant,
            curr_shapes,
            (batch_size, 1, occ_h, occ_w, occ_z),
        )

        def _to_pred(logits):
            return logits.softmax(-1).argmax(-1).detach().cpu().numpy().astype(np.uint8)

        return dict(
            prev_clean=prev_clean.detach().cpu().numpy().astype(np.uint8),
            curr_input=curr_input.detach().cpu().numpy().astype(np.uint8),
            curr_target=curr_target.detach().cpu().numpy().astype(np.uint8),
            prev_decode=_to_pred(prev_logits),
            curr_decode=_to_pred(curr_logits),
            prior_decode=_to_pred(prior_logits),
            posterior_decode=_to_pred(posterior_logits),
            gate=gate.detach().cpu().numpy(),
            flow_latent=flow_latent.detach().cpu().numpy(),
            valid_ratio=valid_ratio.detach().cpu().numpy(),
            dynamic_ratio=dynamic_ratio.detach().cpu().numpy(),
            reliability_latent=(None if reliability_latent is None else reliability_latent.detach().cpu().numpy()),
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

        logits, embed_loss, curr_target, debug = self._forward_oracle_pair(
            voxel_semantics,
            voxel_semantics_clean,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
            frame_reliability_map=frame_reliability_map,
        )

        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, curr_target))
        loss_dict['embed_loss'] = self.embed_loss_weight * embed_loss
        if self.oracle_prior_loss_weight > 0 and debug['prior_logits'] is not None:
            prior_loss = self.reconstruct_loss(debug['prior_logits'], curr_target)['recon_loss']
            loss_dict['prior_recon_loss'] = self.oracle_prior_loss_weight * prior_loss
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
        logits, _, curr_target, debug = self._forward_oracle_pair(
            voxel_semantics,
            voxel_semantics_clean,
            oracle_flow,
            oracle_flow_valid,
            oracle_dynamic_mask,
            frame_reliability_map=frame_reliability_map,
        )
        end_time = time.time()

        pred = logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)
        output_dict = dict(
            semantics=pred,
            input_curr_semantics=voxel_semantics[:, -1].cpu().numpy().astype(np.uint8),
            target_curr_semantics=curr_target[:, 0].cpu().numpy().astype(np.uint8),
            oracle_gate_mean=float(debug['gate'].mean().detach().cpu()),
            oracle_valid_ratio=float(debug['valid_ratio'].mean().detach().cpu()),
            oracle_dynamic_ratio=float(debug['dynamic_ratio'].mean().detach().cpu()),
            index=[img_meta['index'] for img_meta in img_metas],
            time=end_time - start_time,
        )
        return [output_dict]
