import torch.nn.functional as F
from mmdet.models.builder import LOSSES

from .focal_loss import CustomFocalLoss, py_focal_loss_with_prob, py_sigmoid_focal_loss


@LOSSES.register_module()
class CustomFocalLossTorch(CustomFocalLoss):
    """CustomFocalLoss variant that avoids the MMCV CUDA focal-loss extension.

    Some CUDA environments may not provide the required MMCV focal-loss kernel. This class keeps the same focal-loss formula and
    reduction as CustomFocalLoss, but always uses the PyTorch implementation.
    """

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                ignore_index=255,
                reduction_override=None):
        bsz, height, width, depth = target.shape

        c = self.c[None, :, :, None].repeat(bsz, 1, 1, depth).reshape(-1)
        visible_mask = (target != ignore_index).reshape(-1).nonzero().squeeze(-1)
        if weight is not None:
            weight_mask = weight[None, :] * c[visible_mask, None]
        else:
            weight_mask = c[visible_mask, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if not self.use_sigmoid:
            raise NotImplementedError

        if self.activated:
            calculate_loss_func = py_focal_loss_with_prob
            loss_target = target.long()
        else:
            loss_target = F.one_hot(target, num_classes=num_classes + 1)
            loss_target = loss_target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss

        loss_cls = self.loss_weight * calculate_loss_func(
            pred,
            loss_target,
            weight_mask,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
