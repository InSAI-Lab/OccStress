# Copyright (c) Zhejiang Lab. All rights reserved.
import torch
import numpy as np
from torch import nn
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss

from ...models.backbones.lidar_encoder import CustomSparseEncoder
from .bevdet import BEVDepth4D


@DETECTORS.register_module()
class FusionDepthSeg(BEVDepth4D):
    def __init__(self, **kwargs):
        super(FusionDepthSeg, self).__init__(**kwargs)

    def prepare_img_3d_feat(self, img, sensor2keyego, ego2global, intrin,
                            post_rot, post_tran, bda, mlp_input, input_depth=None):
        x, _ = self.image_encoder(img, stereo=False)
        img_3d_feat, depth, seg = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], input_depth)
        if self.pre_process:
            img_3d_feat = self.pre_process_net(img_3d_feat)[0]
        return img_3d_feat, depth, seg

    def extract_img_3d_feat(self,
                            img_inputs,
                            input_depth):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, _ = \
            self.prepare_inputs(img_inputs, stereo=False)
        """Extract features of images."""
        img_3d_feat_list = []
        depth_key_frame = None
        seg_key_frame = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            curr_frame = fid == 0
            if self.align_after_view_transformation:
                sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrin,
                post_rot, post_tran, bda)
            inputs_curr = (img, sensor2keyego, ego2global, intrin,
                           post_rot, post_tran, bda, mlp_input, input_depth)
            if curr_frame:
                img_3d_feat, depth, pred_seg = self.prepare_img_3d_feat(*inputs_curr)
                seg_key_frame = pred_seg
                depth_key_frame = depth
            else:
                with torch.no_grad():
                    img_3d_feat, _, _ = self.prepare_img_3d_feat(*inputs_curr)
            img_3d_feat_list.append(img_3d_feat)
        if self.align_after_view_transformation:
            for adj_id in range(self.num_frame - 1):
                img_3d_feat_list[adj_id] = \
                    self.shift_feature(img_3d_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        img_3d_feat_feat = torch.cat(img_3d_feat_list, dim=1)
        return img_3d_feat_feat, depth_key_frame, seg_key_frame


@DETECTORS.register_module()
class FusionOCC(FusionDepthSeg):
    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
                 voxel_size=[0.05, 0.05, 0.05],
                 lidar_in_channel=5,
                 lidar_out_channel=32,
                 fuse_loss_weight=0.1,
                 occ_encoder_backbone=None,
                 occ_encoder_neck=None,
                 **kwargs):
        super(FusionOCC, self).__init__(
            img_bev_encoder_backbone=occ_encoder_backbone,
            img_bev_encoder_neck=occ_encoder_neck,
            **kwargs)
        self.voxel_size = voxel_size
        self.lidar_out_channel = lidar_out_channel
        self.lidar_in_channel = lidar_in_channel
        self.sparse_shape = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
        ]
        self.point_cloud_range = point_cloud_range
        self.lidar_encoder = CustomSparseEncoder(
            in_channels=self.lidar_in_channel,
            sparse_shape=self.sparse_shape,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            output_channels=self.lidar_out_channel,
            # block_type="basicblock"
            block_type="conv_module"
        )
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            out_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transformation = False
        self.fuse_loss_weight = fuse_loss_weight

    def occ_encoder(self, fusion_feat):
        return self.bev_encoder(fusion_feat)

    def extract_feat(self, lidar_feat, img, img_metas, input_depth=None, **kwargs):
        """Extract features from images and points."""
        fusion_feats, depth, pred_segs = self.extract_fusion_feat(
            lidar_feat, img, img_metas, input_depth=input_depth, **kwargs
        )
        pts_feats = None
        return fusion_feats, pts_feats, depth, pred_segs

    def forward_train(self,
                      points=None,
                      img_inputs=None,
                      segs=None,
                      sparse_depth=None,
                      **kwargs):
        lidar_feat, x_list, x_sparse_out = self.lidar_encoder(points)
        lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()

        input_depth = sparse_depth
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img_inputs, input_depth=input_depth)
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)

        losses = dict()
        depth_loss, seg_loss, vis_depth_pred, vis_depth_label, vis_seg_pred, vis_seg_label = \
            self.img_view_transformer.get_loss(sparse_depth, depth_key_frame, segs, seg_key_frame)
        losses['depth_loss'] = depth_loss * self.fuse_loss_weight
        losses['seg_loss'] = seg_loss * self.fuse_loss_weight

        occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img_inputs=None,
                    sparse_depth=None,
                    **kwargs):
        """Test function without augmentaiton."""

        sparse_depth = sparse_depth[0]
        lidar_feat, x_list, x_sparse_out = self.lidar_encoder(points)
        # N, C, D, H, W -> N,C,D,W,H
        lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()
        input_depth = sparse_depth
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img_inputs, input_depth=input_depth)
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)

        occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]
