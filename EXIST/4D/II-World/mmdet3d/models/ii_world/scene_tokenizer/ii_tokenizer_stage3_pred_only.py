import torch

from mmdet.models import DETECTORS

from .ii_tokenizer import IISceneTokenizer


@DETECTORS.register_module()
class IISceneTokenizerStage3PredOnly(IISceneTokenizer):
    """Decoder-only stage3 calibration on frozen stage2 predicted latents.

    This class keeps the tokenizer checkpoint format compatible with the
    baseline tokenizer. Only ``decoder`` remains trainable; encoder, VQ and
    class embeddings are frozen and kept for checkpoint/eval compatibility.
    """

    def __init__(self,
                 target_frame_index=1,
                 target_valid_index=0,
                 freeze_encoder=True,
                 freeze_vq=True,
                 freeze_class_embeds=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_frame_index = int(target_frame_index)
        self.target_valid_index = int(target_valid_index)
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_vq = bool(freeze_vq)
        self.freeze_class_embeds = bool(freeze_class_embeds)

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

    @staticmethod
    def _decoder_shapes(device):
        return [
            torch.tensor((200, 200), device=device),
            torch.tensor((100, 100), device=device),
        ]

    def forward_train(self, pred_latent, voxel_semantics, valid_frame=None, img_metas=None, **kwargs):
        # pred_latent: [B, C, H, W], cached from frozen stage2 forward_sample.
        if pred_latent.dim() == 5:
            if pred_latent.shape[1] != 1:
                raise ValueError(f'Expected pred_latent frame dim 1, got {tuple(pred_latent.shape)}')
            pred_latent = pred_latent[:, 0]

        if pred_latent.dim() != 4:
            raise ValueError(f'Expected pred_latent [B,C,H,W], got {tuple(pred_latent.shape)}')

        target = voxel_semantics[:, self.target_frame_index:self.target_frame_index + 1]
        if target.shape[1] != 1:
            raise ValueError(
                f'target_frame_index={self.target_frame_index} is invalid for '
                f'voxel_semantics shape {tuple(voxel_semantics.shape)}')

        if valid_frame is not None:
            valid = valid_frame[:, self.target_valid_index].to(device=target.device).bool()
            if valid.sum() == 0:
                zero = next(self.decoder.parameters()).sum() * 0.0
                return dict(recon_loss=zero, embed_loss=zero)
            pred_latent = pred_latent[valid]
            target = target[valid]

        bs, _, occ_h, occ_w, occ_d = target.shape
        pred_latent = pred_latent.to(device=target.device, dtype=torch.float32)
        logits = self.forward_decoder(
            pred_latent,
            self._decoder_shapes(target.device),
            (bs, 1, occ_h, occ_w, occ_d),
        )

        loss_dict = self.reconstruct_loss(logits, target)
        loss_dict['embed_loss'] = logits.sum() * 0.0
        loss_dict['pred_latent_abs_mean'] = pred_latent.detach().abs().mean()
        return loss_dict
