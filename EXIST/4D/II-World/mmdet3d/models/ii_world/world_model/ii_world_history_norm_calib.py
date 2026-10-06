import torch
from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_world import II_World


@DETECTORS.register_module()
class II_WorldHistoryNormCalib(II_World):
    """Conservative test-time current-token norm calibration from clean history.

    Some OccStress protocols shift the stage1 current latent mostly by global feature
    magnitude. Clean current/history token magnitudes are tightly matched, so this
    module rescales the current token only when the history/current norm ratio is
    outside a deadband. With the default deadband the clean path is nearly
    baseline-identical.
    """

    def __init__(self, norm_calib=None, **kwargs):
        super().__init__(**kwargs)
        norm_calib = norm_calib or {}
        self.norm_calib_enabled = bool(norm_calib.get('enabled', True))
        self.norm_calib_history_index = int(norm_calib.get('history_index', -1))
        self.norm_calib_strength = float(norm_calib.get('strength', 1.0))
        self.norm_calib_deadband = float(norm_calib.get('deadband', 0.03))
        self.norm_calib_min_scale = float(norm_calib.get('min_scale', 0.90))
        self.norm_calib_max_scale = float(norm_calib.get('max_scale', 1.15))
        self.norm_calib_up_mode = str(norm_calib.get('up_mode', 'scale'))
        self.norm_calib_up_blend_strength = float(norm_calib.get('up_blend_strength', 0.0))
        self.norm_calib_down_mode = str(norm_calib.get('down_mode', 'scale'))
        self.norm_calib_down_blend_strength = float(norm_calib.get('down_blend_strength', 0.05))
        self._last_norm_calib_stats = None

    def _normalize_img_metas(self, img_metas):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        return img_metas

    def _to_model_tensor(self, value, device, dtype):
        if value is None:
            return None
        if isinstance(value, DataContainer):
            value = value.data
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=dtype)

    def _calibrate_current(self, latent, history_latent):
        if not self.norm_calib_enabled or history_latent is None:
            self._last_norm_calib_stats = None
            return latent

        history_latent = self._to_model_tensor(history_latent, latent.device, latent.dtype)
        if history_latent.dim() == 6 and history_latent.shape[1] == 1:
            history_latent = history_latent[:, 0]
        if history_latent.dim() != 5 or history_latent.shape[1] == 0:
            self._last_norm_calib_stats = None
            return latent

        ref_latent = history_latent[:, self.norm_calib_history_index]
        curr = latent[:, 0]
        reduce_dims = tuple(range(1, curr.dim()))
        curr_abs = curr.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        ref_abs = ref_latent.abs().mean(dim=reduce_dims, keepdim=True)
        ratio = ref_abs / curr_abs
        raw_scale = ratio.clamp(self.norm_calib_min_scale, self.norm_calib_max_scale)
        up_mask = raw_scale > 1.0 + self.norm_calib_deadband
        down_mask = raw_scale < 1.0 - self.norm_calib_deadband
        apply_mask = up_mask | down_mask
        scale_delta = self.norm_calib_strength * (raw_scale - 1.0)
        if self.norm_calib_down_mode in ('none', 'blend'):
            scale_delta = torch.where(down_mask, torch.zeros_like(scale_delta), scale_delta)
        scale = torch.where(apply_mask, 1.0 + scale_delta, torch.ones_like(raw_scale))

        calibrated = latent.clone()
        calibrated[:, 0] = curr * scale
        if self.norm_calib_up_mode == 'blend':
            blend = self.norm_calib_up_blend_strength
            blend_mask = up_mask.to(dtype=latent.dtype)
            calibrated[:, 0] = calibrated[:, 0] + blend * blend_mask * (ref_latent - calibrated[:, 0])
        if self.norm_calib_down_mode == 'blend':
            blend = self.norm_calib_down_blend_strength
            blend_mask = down_mask.to(dtype=latent.dtype)
            calibrated[:, 0] = calibrated[:, 0] + blend * blend_mask * (ref_latent - calibrated[:, 0])
        self._last_norm_calib_stats = dict(
            raw_scale_mean=raw_scale.mean().detach(),
            scale_mean=scale.mean().detach(),
            apply_ratio=apply_mask.to(dtype=latent.dtype).mean().detach(),
            up_ratio=up_mask.to(dtype=latent.dtype).mean().detach(),
            down_ratio=down_mask.to(dtype=latent.dtype).mean().detach(),
            delta_abs=(calibrated[:, 0] - curr).abs().mean().detach(),
        )
        return calibrated

    def forward_sample(self, latent, img_metas, predict_future_frame, train=True, history_latent=None, **kwargs):
        latent = self._calibrate_current(latent, history_latent)
        return super().forward_sample(latent, img_metas, predict_future_frame, train=train, **kwargs)

    def forward_train(self, latent, img_metas, history_latent=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        loss_dict = super().forward_train(
            self._calibrate_current(latent, history_latent),
            img_metas,
            **kwargs,
        )
        stats = self._last_norm_calib_stats
        if stats is not None:
            loss_dict['norm_calib_raw_scale_mean'] = stats['raw_scale_mean']
            loss_dict['norm_calib_scale_mean'] = stats['scale_mean']
            loss_dict['norm_calib_apply_ratio'] = stats['apply_ratio']
            loss_dict['norm_calib_up_ratio'] = stats['up_ratio']
            loss_dict['norm_calib_down_ratio'] = stats['down_ratio']
            loss_dict['norm_calib_delta_abs'] = stats['delta_abs']
        return loss_dict

    def forward_test(self, latent, voxel_semantics, img_metas, history_latent=None, **kwargs):
        img_metas = self._normalize_img_metas(img_metas)
        latent = self._calibrate_current(latent, history_latent)
        return super().forward_test(latent, voxel_semantics, img_metas, **kwargs)
