import torch
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World


@DETECTORS.register_module()
class II_WorldCurrentScaleAug(II_World):
    """Stage2 current-latent scale augmentation.

    OccStress dropout/semantic tokenizer outputs show a sample-level latent magnitude
    shift while clean current/history magnitudes are tightly matched. This module
    keeps the baseline architecture unchanged and only augments the current
    latent during training so the world model becomes less sensitive to that
    shift. Evaluation is baseline-identical unless ``apply_at_test`` is enabled.
    """

    def __init__(self, scale_aug=None, **kwargs):
        super().__init__(**kwargs)
        scale_aug = scale_aug or {}
        self.scale_aug_enabled = bool(scale_aug.get('enabled', True))
        self.scale_aug_prob = float(scale_aug.get('prob', 0.5))
        self.scale_aug_up_prob = float(scale_aug.get('up_prob', 0.5))
        self.scale_aug_down_range = tuple(scale_aug.get('down_range', (0.90, 0.96)))
        self.scale_aug_up_range = tuple(scale_aug.get('up_range', (1.03, 1.08)))
        self.scale_aug_apply_at_test = bool(scale_aug.get('apply_at_test', False))
        self._last_scale_aug_stats = None

    def _normalize_img_metas(self, img_metas):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        return img_metas

    def _sample_scales(self, latent):
        batch_size = latent.shape[0]
        device, dtype = latent.device, latent.dtype
        apply_mask = torch.rand(batch_size, device=device) < self.scale_aug_prob
        up_mask = torch.rand(batch_size, device=device) < self.scale_aug_up_prob
        down_min, down_max = self.scale_aug_down_range
        up_min, up_max = self.scale_aug_up_range
        down_scale = torch.empty(batch_size, device=device, dtype=dtype).uniform_(down_min, down_max)
        up_scale = torch.empty(batch_size, device=device, dtype=dtype).uniform_(up_min, up_max)
        aug_scale = torch.where(up_mask, up_scale, down_scale)
        scale = torch.where(apply_mask, aug_scale, torch.ones_like(aug_scale))
        return scale.view(batch_size, 1, 1, 1), apply_mask

    def _augment_current(self, latent):
        if not self.scale_aug_enabled:
            self._last_scale_aug_stats = None
            return latent
        scale, apply_mask = self._sample_scales(latent)
        augmented = latent.clone()
        augmented[:, 0] = augmented[:, 0] * scale
        self._last_scale_aug_stats = dict(
            scale_mean=scale.mean().detach(),
            apply_ratio=apply_mask.to(dtype=latent.dtype).mean().detach(),
            delta_abs=(augmented[:, 0] - latent[:, 0]).abs().mean().detach(),
        )
        return augmented

    def forward_train(self, latent, img_metas, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        latent_for_model = self._augment_current(latent)
        loss_dict = super().forward_train(latent_for_model, img_metas, **kwargs)
        stats = self._last_scale_aug_stats
        if stats is not None:
            loss_dict['scale_aug_scale_mean'] = stats['scale_mean']
            loss_dict['scale_aug_apply_ratio'] = stats['apply_ratio']
            loss_dict['scale_aug_delta_abs'] = stats['delta_abs']
        return loss_dict

    def forward_test(self, latent, voxel_semantics, img_metas, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        if self.scale_aug_apply_at_test:
            latent = self._augment_current(latent)
        return super().forward_test(latent, voxel_semantics, img_metas, **kwargs)
