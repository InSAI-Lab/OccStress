import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS

from .ii_tokenizer_v0_original import IISceneTokenizerV0Original


@DETECTORS.register_module()
class IISceneTokenizerV0DecoderPerturbation(IISceneTokenizerV0Original):
    """Decoder-only finetune with local latent perturbation consistency.

    The encoder, VQ, post-quantized latent space, and class embeddings stay
    fixed. Only the decoder learns to tolerate small perturbations around the
    baseline decoder input latent.
    """

    def __init__(self,
                 decoder_perturb_noise_schedule=((0, 0.02), (1000, 0.05), (3000, 0.08)),
                 decoder_perturb_recon_weight=0.5,
                 decoder_perturb_kl_weight=0.1,
                 freeze_encoder=True,
                 freeze_vq=True,
                 freeze_class_embeds=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.decoder_perturb_noise_schedule = sorted(
            [(int(step), float(sigma)) for step, sigma in decoder_perturb_noise_schedule],
            key=lambda item: item[0])
        self.decoder_perturb_recon_weight = float(decoder_perturb_recon_weight)
        self.decoder_perturb_kl_weight = float(decoder_perturb_kl_weight)
        self.freeze_encoder = freeze_encoder
        self.freeze_vq = freeze_vq
        self.freeze_class_embeds = freeze_class_embeds
        self.decoder_perturb_step = 0

        if self.freeze_encoder:
            self._freeze_module(self.encoder)
        if self.freeze_vq:
            self._freeze_module(self.vq)
        if self.freeze_class_embeds:
            self._freeze_module(self.class_embeds)

    @staticmethod
    def _freeze_module(module):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        if self.freeze_vq:
            self.vq.eval()
        if self.freeze_class_embeds:
            self.class_embeds.eval()
        return self

    def _current_noise_sigma(self):
        schedule = self.decoder_perturb_noise_schedule
        if not schedule:
            return 0.0

        step = self.decoder_perturb_step
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

    def forward_train(self, voxel_semantics, img_metas, **kwargs):
        bs, t, w, h, d = voxel_semantics.shape

        # Frozen baseline tokenizer path produces the same decoder input latent.
        with torch.no_grad():
            curr_bev, shapes = self.forward_encoder(voxel_semantics)
            sampled_bev = self.align_bev(curr_bev, img_metas)
            z_sampled, _, _ = self.vq(curr_bev, sampled_bev, is_voxel=False)

        z_clean = z_sampled.detach()
        logits = self.forward_decoder(z_clean, list(shapes), (bs, 1, w, h, d))

        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, voxel_semantics))
        loss_dict['embed_loss'] = logits.sum() * 0.0

        sigma = self._current_noise_sigma()
        if sigma > 0.0 and (self.decoder_perturb_recon_weight > 0.0 or self.decoder_perturb_kl_weight > 0.0):
            z_noisy = z_clean + torch.randn_like(z_clean) * sigma
            logits_noisy = self.forward_decoder(z_noisy, list(shapes), (bs, 1, w, h, d))

            if self.decoder_perturb_recon_weight > 0.0:
                noisy_recon = self.reconstruct_loss(logits_noisy, voxel_semantics)['recon_loss']
                loss_dict['perturb_recon_loss'] = self.decoder_perturb_recon_weight * noisy_recon

            if self.decoder_perturb_kl_weight > 0.0:
                kl_loss = self._logit_kl_consistency_loss(logits, logits_noisy, voxel_semantics)
                loss_dict['perturb_kl_loss'] = self.decoder_perturb_kl_weight * kl_loss

        loss_dict['perturb_sigma'] = logits.detach().new_tensor(sigma)
        self.decoder_perturb_step += 1
        return loss_dict
