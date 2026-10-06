import os.path
import time

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS

from .ii_tokenizer_v0_original import IISceneTokenizerV0Original


@DETECTORS.register_module()
class IISceneTokenizerV0LatentPerturbation(IISceneTokenizerV0Original):
    """Baseline tokenizer with a latent perturbation auxiliary decoder loss.

    The clean path is identical to the v0 baseline. A second decoder branch sees
    a small Gaussian perturbation of the decoder input latent, which encourages
    the decoder to be less brittle to stage2-like latent errors.
    """

    def __init__(self,
                 decoder_perturb_noise_schedule=((0, 0.0), (2000, 0.0), (8000, 0.03), (16000, 0.05)),
                 decoder_perturb_recon_weight=0.25,
                 decoder_perturb_kl_weight=0.05,
                 decoder_perturb_detach_base=True,
                 decoder_perturb_prob=1.0,
                 save_root_override=None,
                 export_only=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.decoder_perturb_noise_schedule = sorted(
            [(int(step), float(sigma)) for step, sigma in decoder_perturb_noise_schedule],
            key=lambda item: item[0])
        self.decoder_perturb_recon_weight = float(decoder_perturb_recon_weight)
        self.decoder_perturb_kl_weight = float(decoder_perturb_kl_weight)
        self.decoder_perturb_detach_base = bool(decoder_perturb_detach_base)
        self.decoder_perturb_prob = float(decoder_perturb_prob)
        self.export_only = bool(export_only)
        if save_root_override is not None:
            self.save_root = save_root_override
            mmcv.mkdir_or_exist(self.save_root)
        self.register_buffer('decoder_perturb_step', torch.zeros((), dtype=torch.long), persistent=True)

    def _current_noise_sigma(self):
        schedule = self.decoder_perturb_noise_schedule
        if not schedule:
            return 0.0

        step = int(self.decoder_perturb_step.item())
        if step <= schedule[0][0]:
            return schedule[0][1]

        for (left_step, left_sigma), (right_step, right_sigma) in zip(schedule[:-1], schedule[1:]):
            if step <= right_step:
                span = max(right_step - left_step, 1)
                alpha = (step - left_step) / span
                return left_sigma + alpha * (right_sigma - left_sigma)

        return schedule[-1][1]

    @staticmethod
    def _logit_kl_consistency_loss(clean_logits, noisy_logits, target):
        clean_log_prob = F.log_softmax(clean_logits.detach(), dim=-1)
        clean_prob = clean_log_prob.exp()
        noisy_log_prob = F.log_softmax(noisy_logits, dim=-1)
        per_voxel_kl = (clean_prob * (clean_log_prob - noisy_log_prob)).sum(dim=-1)
        valid = (target != 255).to(per_voxel_kl.dtype)
        return (per_voxel_kl * valid).sum() / valid.sum().clamp_min(1.0)

    def _should_apply_perturb(self, z_sampled, sigma):
        if sigma <= 0.0:
            return False
        if self.decoder_perturb_prob >= 1.0:
            return True
        if self.decoder_perturb_prob <= 0.0:
            return False
        return bool(torch.rand((), device=z_sampled.device) < self.decoder_perturb_prob)

    def forward_train(self, voxel_semantics, img_metas, **kwargs):
        bs, t, w, h, d = voxel_semantics.shape

        curr_bev, shapes = self.forward_encoder(voxel_semantics)
        sampled_bev = self.align_bev(curr_bev, img_metas)
        z_sampled, loss, info = self.vq(curr_bev, sampled_bev, is_voxel=False)

        decoder_shapes = list(shapes)
        logits = self.forward_decoder(z_sampled, list(decoder_shapes), (bs, 1, w, h, d))

        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, voxel_semantics))
        loss_dict['embed_loss'] = self.embed_loss_weight * loss

        sigma = self._current_noise_sigma()
        branch_on = self._should_apply_perturb(z_sampled, sigma)
        if branch_on and (self.decoder_perturb_recon_weight > 0.0 or self.decoder_perturb_kl_weight > 0.0):
            z_base = z_sampled.detach() if self.decoder_perturb_detach_base else z_sampled
            z_noisy = z_base + torch.randn_like(z_base) * sigma
            logits_noisy = self.forward_decoder(z_noisy, list(decoder_shapes), (bs, 1, w, h, d))

            if self.decoder_perturb_recon_weight > 0.0:
                noisy_recon = self.reconstruct_loss(logits_noisy, voxel_semantics)['recon_loss']
                loss_dict['perturb_recon_loss'] = self.decoder_perturb_recon_weight * noisy_recon

            if self.decoder_perturb_kl_weight > 0.0:
                kl_loss = self._logit_kl_consistency_loss(logits, logits_noisy, voxel_semantics)
                loss_dict['perturb_kl_loss'] = self.decoder_perturb_kl_weight * kl_loss

        loss_dict['perturb_sigma'] = z_sampled.new_tensor(sigma)
        loss_dict['perturb_branch_on'] = z_sampled.new_tensor(float(branch_on))
        self.decoder_perturb_step.add_(1)
        return loss_dict

    def forward_test(self, voxel_semantics, img_metas, **kwargs):
        if not self.export_only:
            return super().forward_test(voxel_semantics, img_metas, **kwargs)

        bs, t, w, h, d = voxel_semantics.shape
        start_time = time.time()
        curr_bev, _ = self.forward_encoder(voxel_semantics)
        sampled_bev = self.align_bev(curr_bev, img_metas)
        z_sampled, _, _ = self.vq(curr_bev, sampled_bev, is_voxel=False)
        end_time = time.time()

        if self.save_results:
            save_tokens = z_sampled.detach().cpu().numpy()
            for save_token, img_meta in zip(save_tokens, img_metas):
                if self.results_type != 'waymo':
                    scene_name = str(img_meta['scene_name'])
                    sample_idx = str(img_meta['sample_idx'])
                    out_dir = os.path.join(self.save_root, 'token_4f', scene_name)
                    mmcv.mkdir_or_exist(out_dir)
                    np.savez(os.path.join(out_dir, f'{sample_idx}.npz'), token=save_token)
                else:
                    scene_name = str(img_meta['scene_name']).zfill(3)
                    occ_path_idx = img_meta['occ_path'].split('/')[-1].split('.')[0]
                    out_dir = os.path.join(self.save_root, 'token_4f', scene_name)
                    mmcv.mkdir_or_exist(out_dir)
                    np.savez(os.path.join(out_dir, f'{occ_path_idx}.npz'), token=save_token)

        return [dict(
            index=[img_meta['index'] for img_meta in img_metas],
            time=end_time - start_time,
        )]
