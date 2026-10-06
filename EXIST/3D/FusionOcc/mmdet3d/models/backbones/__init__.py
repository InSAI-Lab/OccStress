# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import ResNet
from .resnet import CustomResNet, CustomResNet3D
from .swin import SwinTransformer


__all__ = [
    'ResNet', 'CustomResNet', 'CustomResNet3D', 'SwinTransformer',
]
