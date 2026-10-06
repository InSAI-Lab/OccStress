import torch
from mmdet.models import DETECTORS

from .ii_tokenizer_currlatent_bilinear_lowpass_slots_v0 import (
    IISceneTokenizerCurrLatentBilinearLowpassSlotsV0,
)


@DETECTORS.register_module()
class IISceneTokenizerCurrLatentBilinearLowpassQuantConvSlotsV0(
        IISceneTokenizerCurrLatentBilinearLowpassSlotsV0):
    """Current-latent low-pass slots projected into VQ quant space.

    The parent current-latent ablation replaces baseline history slots with
    smoothed copies of encoder-space ``curr_bev``. This variant first projects
    the current BEV through ``vq.quant_conv`` so that the inter-scene slots live
    in the same space as ``z_rest`` inside the VQ module.
    """

    def align_bev(self, curr_bev, img_metas):
        del img_metas
        if self.currlatent_detach_slots:
            with torch.no_grad():
                slot_source = self.vq.quant_conv(curr_bev)
        else:
            slot_source = self.vq.quant_conv(curr_bev)

        slots = [
            self._lowpass_repeated(slot_source, num_passes)
            for num_passes in self.currlatent_lowpass_passes
        ]
        return torch.stack(slots, dim=1).clone()
