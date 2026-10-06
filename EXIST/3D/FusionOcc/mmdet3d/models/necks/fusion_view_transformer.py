# Copyright (c) Zhejiang Lab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from mmcv.runner import force_fp32
from mmdet.models.backbones.resnet import BasicBlock

from ..builder import NECKS
from .view_transformer import LSSViewTransformerBEVDepth, Mlp, SELayer, ASPP


class DepthSegNet(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 depth_channels,
                 feature_channels,
                 seg_num_classes,
                 aspp_mid_channels=-1):
        super(DepthSegNet, self).__init__()
        self.depth_channels = depth_channels
        self.feature_channels = feature_channels
        self.seg_feature = self.feature_channels // 2
        self.context_feature = self.feature_channels - self.seg_feature
        self.seg_num_classes = seg_num_classes
        self.reduce_conv_depth = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.reduce_conv_seg = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.reduce_conv_context = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.bn = nn.BatchNorm1d(27)
        self.depth_se = SELayer(mid_channels)
        depth_conv_input_channels = mid_channels
        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels, downsample=None),
                           # BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if aspp_mid_channels < 0:
            aspp_mid_channels = mid_channels
        depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.depth_channels = depth_channels
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        self.context_conv = nn.Conv2d(
            mid_channels, self.context_feature, kernel_size=3, stride=1, padding=1)
        self.seg_mlp = Mlp(27, mid_channels, mid_channels)
        self.seg_se = SELayer(mid_channels)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(mid_channels, self.seg_feature, kernel_size=3, stride=1, padding=1),
            BasicBlock(self.seg_feature, self.seg_feature)
        )
        self.seg_out = nn.Conv2d(self.seg_feature, seg_num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x_c = self.reduce_conv_seg(x)
        x_d = self.reduce_conv_depth(x)
        x_cx = self.reduce_conv_context(x)
        seg_se = self.seg_mlp(mlp_input)[..., None, None]
        seg = self.seg_se(x_c, seg_se)
        seg_feature = self.seg_conv(seg)
        seg_out = self.seg_out(seg_feature)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x_cx, context_se)
        context_feature = self.context_conv(context)
        feature = torch.cat([seg_feature, context_feature], dim=1)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x_d, depth_se)
        depth = self.depth_conv(depth)
        return depth, feature, seg_out


class CrossModalFusion(nn.Module):
    def __init__(self, mid_c, alpha=1):
        super(CrossModalFusion, self).__init__()
        self.alpha = alpha
        self.channel_mlp_c = nn.Sequential(
            nn.Linear(mid_c, mid_c),
            nn.Sigmoid(),
        )
        self.channel_mlp_d = nn.Sequential(
            nn.Linear(mid_c, mid_c),
            nn.Sigmoid(),
        )
        self.spatial_c = nn.Sequential(
            nn.Conv2d(1, mid_c // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.spatial_d = nn.Sequential(
            nn.Conv2d(1, mid_c // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(
                mid_c * 2, mid_c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_c * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, fc, fd):
        fc_ = F.adaptive_avg_pool2d(fc, (1, 1))
        fd_ = F.adaptive_avg_pool2d(fd, (1, 1))
        B, C, _, _ = fd_.shape
        w_c = self.channel_mlp_c(fc_.reshape(B, C)).reshape(B, C, 1, 1)
        w_d = self.channel_mlp_d(fd_.reshape(B, C)).reshape(B, C, 1, 1)
        fc2d = w_d * fc
        fd2c = w_c * fd
        f_fuse = self.fuse_conv(torch.concat([fc2d, fd2c], dim=1))
        f_c, f_d = f_fuse[:, :C, :, :], f_fuse[:, C:, :, :]
        f_c = torch.mean(f_c, dim=1, keepdim=True)
        f_d = torch.mean(f_d, dim=1, keepdim=True)
        zc = self.spatial_c(f_c)
        zd = self.spatial_d(f_d)
        fc_c2d = self.alpha * zd * fc + fc
        fc_d2c = self.alpha * zc * fd + fd
        return fc_c2d, fc_d2c


@NECKS.register_module()
class CrossModalLSS(LSSViewTransformerBEVDepth):
    def __init__(self, feature_channels=None, seg_num_classes=None,
                 depth_channels=88,
                 seg_down_sample=16,
                 mid_channels=None,
                 is_train=True,
                 **kwargs):
        super(CrossModalLSS, self).__init__(**kwargs)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)
        self.seg_down_sample = seg_down_sample
        depthnet_cfg = kwargs["depthnet_cfg"]
        self.mid_channels = mid_channels
        self.depth_channels = depth_channels
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(
                self.depth_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.img_reduce_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cross_model_fusion = CrossModalFusion(mid_c=self.mid_channels)
        self.further_fuse = BasicBlock(self.mid_channels * 2, self.mid_channels * 2)
        self.feature_channels = feature_channels
        self.seg_num_classes = seg_num_classes
        self.depth_net = None
        self.depth_seg_net = DepthSegNet(self.mid_channels * 2, self.mid_channels,
                                         self.D, self.feature_channels, self.seg_num_classes,
                                         **depthnet_cfg)
        self.is_train = is_train

    @force_fp32()
    def seg_loss(self, seg_pred, seg_label, mask_empty=True):
        seg_label = seg_label[:, :, ::self.seg_down_sample, ::self.seg_down_sample]
        vis_seg_label = seg_label.clone().detach()
        seg_label = seg_label.reshape(-1)
        seg_pred = seg_pred.permute(0, 2, 3, 1).contiguous()
        vis_seg_pred = seg_pred
        seg_pred = seg_pred.view(-1, self.seg_num_classes)
        if mask_empty:
            mask = seg_label != 17
            seg_pred = seg_pred[mask]
            seg_label = seg_label[mask]
        seg_loss = torch.nn.functional.cross_entropy(input=seg_pred, target=seg_label)
        return seg_loss, vis_seg_pred, vis_seg_label

    @force_fp32()
    def depth_loss(self, depth_pred, depth_label):
        depth_label, vis_depth_label = self.get_downsampled_gt_depth(depth_label)
        depth_pred = depth_pred.permute(0, 2, 3, 1).contiguous()
        vis_depth_pred = depth_pred
        depth_pred = depth_pred.view(-1, self.D)
        fg_mask = torch.max(depth_label, dim=1).values > 0.0
        depth_label = depth_label[fg_mask]
        depth_pred = depth_pred[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_pred,
                depth_label,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return depth_loss, vis_depth_pred, vis_depth_label

    def get_loss(self, depth_label, depth_pred, seg_label, seg_pred):
        seg_loss, vis_seg_pred, vis_seg_label = self.seg_loss(seg_pred, seg_label)
        depth_loss, vis_depth_pred, vis_depth_label = self.depth_loss(depth_pred, depth_label)
        return depth_loss, seg_loss, vis_depth_pred, vis_depth_label, vis_seg_pred, vis_seg_pred

    def forward(self, input, sparse_depth):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        img_input = x.view(B * N, C, H, W)

        depth_label, _ = self.get_downsampled_gt_depth(sparse_depth)
        depth_input = depth_label.clone().detach()
        if self.is_train:
            mask = torch.randint(0, 100, size=[depth_input.shape[0]]) > 50
            depth_input[mask] = depth_input[mask] * 0
        depth_input = depth_input.reshape(B * N, H, W, -1).permute(0, 3, 1, 2)

        f_c = self.img_reduce_conv(img_input)
        f_d = self.depth_encoder(depth_input)
        fc_c2d, fc_d2c = self.cross_model_fusion(f_c, f_d)
        x = self.further_fuse(torch.cat([fc_c2d, fc_d2c], dim=1))
        depth, seg_feature, seg_out = self.depth_seg_net(x, mlp_input)

        depth = depth.softmax(dim=1)
        img_3d_feat, depth = self.view_transform(input, depth, seg_feature)
        return img_3d_feat, depth, seg_out
