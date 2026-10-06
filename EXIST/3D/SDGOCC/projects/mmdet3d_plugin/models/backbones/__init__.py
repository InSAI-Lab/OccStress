from mmdet.models.backbones import ResNet
from .resnet import CustomResNet
from .swin import SwinTransformer
from .spvcnn import SPVCNN

__all__ = ['ResNet', 'CustomResNet', 'SwinTransformer','SPVCNN']
