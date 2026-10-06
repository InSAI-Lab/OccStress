import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World, tic


class ResidualLatentAdapter(nn.Module):
    """Identity-initialized residual adapter for BEV latent tensors."""

    def __init__(self, channels, hidden_channels=None, residual_scale=1.0):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3,
                      padding=1, groups=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        if x.dim() == 5:
            batch, frames, channels, height, width = x.shape
            x_flat = x.reshape(batch * frames, channels, height, width)
            y_flat = x_flat + self.residual_scale * self.net(x_flat)
            return y_flat.reshape(batch, frames, channels, height, width)

        return x + self.residual_scale * self.net(x)


@DETECTORS.register_module()
class II_WorldLatentAdapter(II_World):
    """Stage2 world model with prediction-space latent adapters.

    ``adapter_in`` maps frozen tokenizer reconstruction latents into a
    prediction-friendly space. ``adapter_out`` maps predicted latents back to
    the tokenizer decoder space for evaluation.
    """

    def __init__(self,
                 adapter_channels=128,
                 adapter_hidden_channels=None,
                 adapter_residual_scale=1.0,
                 adapter_cycle_loss_weight=0.1,
                 adapter_cycle_current_weight=1.0,
                 adapter_target_detach=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.adapter_in = ResidualLatentAdapter(
            adapter_channels,
            hidden_channels=adapter_hidden_channels,
            residual_scale=adapter_residual_scale)
        self.adapter_out = ResidualLatentAdapter(
            adapter_channels,
            hidden_channels=adapter_hidden_channels,
            residual_scale=adapter_residual_scale)
        self.adapter_cycle_loss_weight = adapter_cycle_loss_weight
        self.adapter_cycle_current_weight = adapter_cycle_current_weight
        self.adapter_target_detach = adapter_target_detach

    def to_prediction_space(self, latent):
        return self.adapter_in(latent)

    def to_reconstruction_space(self, latent):
        return self.adapter_out(latent)

    def adapter_cycle_loss(self, pred_space_latent, rec_space_latent,
                           valid_frame):
        recovered = self.to_reconstruction_space(pred_space_latent)
        cycle_loss = self.adapter_cycle_current_weight * F.mse_loss(
            recovered[:, 0], rec_space_latent[:, 0])

        for frame_idx in range(self.train_future_frame):
            cycle_loss = cycle_loss + self.frame_loss_weight[frame_idx] * \
                self.feature_similarity_loss(
                    recovered[:, frame_idx + 1],
                    rec_space_latent[:, frame_idx + 1],
                    valid_frame[:, frame_idx])

        return self.adapter_cycle_loss_weight * cycle_loss

    def forward_train(self, latent, img_metas, **kwargs):
        pred_space_latent = self.to_prediction_space(latent)

        return_dict = self.forward_sample(
            pred_space_latent, img_metas, self.train_future_frame, train=True)
        pred_latents = return_dict['pred_latents']
        targ_latents = pred_space_latent[:, 1:]
        if self.adapter_target_detach:
            targ_latents = targ_latents.detach()

        pred_delta_translations = return_dict['pred_delta_translations']
        targ_delta_translations = return_dict['targ_delta_translations']
        pred_relative_rotations = return_dict['pred_relative_rotations']
        targ_relative_rotations = return_dict['targ_relative_rotations']

        valid_frame = torch.stack(
            [torch.tensor(img_meta['valid_frame'], device=latent.device)
             for img_meta in img_metas])

        loss_dict = dict()
        for frame_idx in range(self.train_future_frame):
            loss_dict['feat_sim_{}s_loss'.format((frame_idx + 1) * 0.5)] = \
                self.frame_loss_weight[frame_idx] * self.feature_similarity_loss(
                    pred_latents[:, frame_idx],
                    targ_latents[:, frame_idx],
                    valid_frame[:, frame_idx])

        loss_dict['adapter_cycle_loss'] = self.adapter_cycle_loss(
            pred_space_latent, latent, valid_frame)
        loss_dict['trajs_loss'] = self.trajs_loss(
            pred_delta_translations, targ_delta_translations, valid_frame, None)
        loss_dict['rotation_loss'] = self.rotation_loss(
            pred_relative_rotations, targ_relative_rotations, valid_frame, None)

        return loss_dict

    def forward_test(self, latent, voxel_semantics, img_metas, **kwargs):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and \
                isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        start_time = tic()
        pred_space_latent = self.to_prediction_space(latent)
        sample_dict = self.forward_sample(
            pred_space_latent, img_metas, self.test_future_frame, train=False)

        return_dict = dict()
        sample_idx = img_metas[0]['sample_idx']

        if self.task_mode == 'generate':
            targ_future_voxel_semantics = \
                voxel_semantics[:, self.test_previous_frame + 1:]
            targ_curr_voxel_semantics = \
                voxel_semantics[:, self.test_previous_frame:self.test_previous_frame + 1]

            curr_rec_latent = self.to_reconstruction_space(pred_space_latent[:, 0])
            pred_curr_voxel_semantics = self.obtain_scene_from_token(
                curr_rec_latent)
            pred_curr_voxel_semantics = \
                pred_curr_voxel_semantics.softmax(-1).argmax(-1)
            if self.dataset_type != 'waymo':
                return_dict['pred_curr_semantics'] = \
                    pred_curr_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_curr_semantics'] = \
                    targ_curr_voxel_semantics.cpu().numpy().astype(np.uint8)

            pred_latents = self.to_reconstruction_space(
                sample_dict['pred_latents'])
            pred_voxel_semantics = self.obtain_scene_from_token(pred_latents)
            pred_voxel_semantics = pred_voxel_semantics.softmax(-1).argmax(-1)
            batch_size = pred_voxel_semantics.shape[0]
            end_time = tic()

            if self.dataset_type == 'waymo':
                if self.eval_metric == 'forecasting_miou':
                    pred_voxel_semantics = pred_voxel_semantics[:, [1, 3, 5]]
                    targ_future_voxel_semantics = \
                        targ_future_voxel_semantics[:, [1, 3, 5]]
                elif self.eval_metric == 'miou':
                    pred_voxel_semantics = \
                        pred_voxel_semantics[:, [self.eval_time]]
                    targ_future_voxel_semantics = \
                        targ_future_voxel_semantics[:, [self.eval_time]]

            if self.eval_metric == 'forecasting_miou':
                return_dict['pred_futu_semantics'] = \
                    pred_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_futu_semantics'] = \
                    targ_future_voxel_semantics.cpu().numpy().astype(np.uint8)
            elif self.eval_metric == 'miou':
                return_dict['semantics'] = \
                    pred_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_semantics'] = targ_future_voxel_semantics.cpu().numpy

        return_dict['occ_path'] = [img_meta['occ_path'] for img_meta in img_metas]
        return_dict['occ_index'] = [img_meta['occ_index'] for img_meta in img_metas]
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return_dict['sample_idx'] = sample_idx
        return_dict['time'] = (end_time - start_time) / self.test_future_frame / batch_size

        return [return_dict]
