# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.runner import force_fp32
# from mmdet3d_plugin.models.model_utils.nat import NATBlock
import numpy as np
import matplotlib.pyplot as plt


class BEVFusionv1(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.attention_bev = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.adapter_sem = nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=3, stride=1, padding=1)

    def forward(self, fuse_features):
        fuse_features = self.adapter_sem(fuse_features)  # cat + 3*3卷积 得到融合特征

        fuse_weight = self.attention_bev(fuse_features)  # 自适应pooling（指定大小为1） + 1*1卷积 + sigmoid  得到B，128,1,1  变成权重

        fusion_features = F.relu(fuse_features * fuse_weight)

        return fusion_features


@DETECTORS.register_module()
class BEVDetOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 is_centercrop=False,
                 **kwargs):
        super(BEVDetOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.is_centercrop = is_centercrop
        self.conv_up = nn.Conv2d(64, 256, 1, 1)
        self.conv_down = nn.Conv2d(256, 64, 1, 1)
        self.bev_fusions = BEVFusionv1(channel=256)

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone(x)  # 3*3卷积得到多层BEV特征
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import cv2
        # bev_feature_abs_sum = x[2].abs().sum(dim=1).squeeze().cpu().detach().numpy()
        # # 步骤 2: 归一化特征图到 [0, 1]
        # bev_feature_min = np.min(bev_feature_abs_sum)
        # bev_feature_max = np.max(bev_feature_abs_sum)
        # bev_feature_normalized = (bev_feature_abs_sum - bev_feature_min) / (bev_feature_max - bev_feature_min)
        # bev_feature_normalized=bev_feature_normalized*255
        # bev_norm_log=((self.log_normalize(bev_feature_abs_sum))*255.0).astype(np.uint8)
        # plt.imshow(bev_norm_log)
        # plt.figure()
        # plt.imshow(bev_feature_normalized)
        # plt.colorbar()
        # plt.show()

        x = self.img_bev_encoder_neck(x)  # 额外上采样  FPN后的BEV特征
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        if self.with_img_backbone:
            img_feats, pts_feats, depth, semantic = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
            img_feats = img_feats[0]

        if self.with_pts_backbone:
            bda_aug = img_inputs[-1]
            batch_size = len(points)
            for b in range(batch_size):
                p_b = torch.matmul(points[b][:, :3], bda_aug[b].T)
                points[b][:, :3] = p_b
            batch_idx = torch.cat([torch.full((points[i].shape[0],), i) for i in range(batch_size)])
            pts = torch.cat(points, dim=0)

            gt_point = kwargs['point_label']
            gt_point = torch.cat(gt_point, dim=0)
            data_dict = {
                'points': pts,  #
                'labels': gt_point,  #
                'batch_idx': batch_idx.long().to(pts.device),  # Batch index for each point
                'batch_size': batch_size,  # Assuming a single batch
                # 'co_coor':coor
            }
            data_dict = self.pts_backbone(data_dict)
            pts_bev_feat = [data_dict['layer_2']['bev_feat'], data_dict['layer_3']['bev_feat']]
            pts_bev_feat = self.pts_neck(pts_bev_feat)[0]  # B,C,256,256的BEV特征

            pts_bev_feat = pts_bev_feat.flip(dims=[2])

        if self.with_pts_backbone and self.with_img_backbone:
            img_feats_up = self.conv_up(img_feats)
            # fuse_features = self.bev_fusions(img_feats_up, pts_bev_feat)
            img_feats_up = img_feats_up.permute(0, 2, 3, 1)
            pts_bev_feat = pts_bev_feat.permute(0, 2, 3, 1)
            fuse_features = self.pts_backbone.bev_align(img_feats_up,pts_bev_feat)
            fuse_features = fuse_features.permute(0, 3, 1, 2)
            fuse_features = self.bev_fusions(fuse_features)
            fuse_features = self.conv_down(fuse_features)

        gt_depth = kwargs['gt_depth']  # (B, N_views, img_H, img_W)
        gt_sem = kwargs['gt_sem']  # (B, N_views, img_H, img_W)

        losses = dict()
        # losses['loss_pts'] = self.pts_backbone.criterion(data_dict)*0.1
        if self.with_img_backbone:
            loss_depth, loss_sem = self.img_view_transformer.get_depth_loss(gt_depth, depth, semantic,
                                                                            gt_sem)  # depth的监督
            losses['loss_depth'] = loss_depth
            losses['loss_sem'] = loss_sem
        losses['pts_loss'] = data_dict['loss']*0.1

        # bev_feature_abs_sum = img_feats.abs().sum(dim=1).squeeze().cpu().detach().numpy()
        # bev_norm_log=((self.log_normalize(bev_feature_abs_sum))*255.0).astype(np.uint8)
        # bev_feature_abs_sum1 = fuse_features.abs().sum(dim=1).squeeze().cpu().detach().numpy()
        # bev_norm_log1=((self.log_normalize(bev_feature_abs_sum1))*255.0).astype(np.uint8)
        # plt.imshow(bev_norm_log[0])
        # plt.figure()
        # plt.imshow(bev_norm_log1[0])
        # plt.show()

        img_feats = self.bev_encoder(fuse_features)

        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)

        loss_occ = self.forward_occ_train(img_feats, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(  # 交叉商
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)  测试时也为256*704的尺寸
        if self.with_img_backbone:
            img_feats, pts_feats, depth, sem = self.extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs)
            img_feats = img_feats[0]
        #
        if self.with_pts_backbone:
            bda_aug = img[-1].squeeze()
            batch_size = len(points)
            batch_idx = torch.cat([torch.full((points[i].shape[0],), i) for i in range(batch_size)])
            pts = torch.cat(points, dim=0)
            xyz_coords = pts[:, :3]
            rotated_xyz_coords = torch.matmul(xyz_coords, bda_aug.T)
            pts[:, :3] = rotated_xyz_coords
            # points = rotated_xyz_coords.cpu().numpy()
            # import open3d as o3d
            # # 创建 Open3D 点云对象
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)
            #
            # # 显示点云
            # o3d.visualization.draw_geometries([point_cloud])

            # pts[:,:3]=rotated_xyz_coords
            gt_point = kwargs['point_label']
            if isinstance(gt_point, list):
                gt_point = gt_point[0]
            gt_point = torch.cat(gt_point, dim=0)
            data_dict = {
                'points': pts,  #
                'labels': gt_point,  #
                'batch_idx': batch_idx.long().to(pts.device),  # Batch index for each point
                'batch_size': batch_size,  # Assuming a single batch
                # 'co_coor':coor
            }
            data_dict = self.pts_backbone(data_dict)
            pts_bev_feat = [data_dict['layer_2']['bev_feat'], data_dict['layer_3']['bev_feat']]
            pts_bev_feat = self.pts_neck(pts_bev_feat)[0]  # B,C,256,256的BEV特征
            # pts_bev_feat = self.conv_down(pts_bev_feat)
            pts_bev_feat = pts_bev_feat.flip(dims=[2])
        #
        if self.with_pts_backbone and self.with_img_backbone:
            img_feats_up = self.conv_up(img_feats)
            # fuse_features = self.bev_fusions(img_feats_up, pts_bev_feat)
            img_feats_up = img_feats_up.permute(0, 2, 3, 1)
            pts_bev_feat = pts_bev_feat.permute(0, 2, 3, 1)
            fuse_features = self.pts_backbone.bev_align(img_feats_up, pts_bev_feat)
            fuse_features = fuse_features.permute(0, 3, 1, 2)
            fuse_features = self.bev_fusions(fuse_features)
            fuse_features = self.conv_down(fuse_features)
        img_feats = self.bev_encoder(fuse_features)

        occ_bev_feature = img_feats
        if self.upsample:  # false
            occ_bev_feature = F.interpolate(img_feats, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.conv_up = nn.Conv2d(2 * 80, 256, 1, 1)
        self.conv_down = nn.Conv2d(256, 2 * 80, 1, 1)
        self.bev_fusions = BEVFusionv1(channel=256)
        # self.bev_align=NATBlock(dim=256, depth=3, depth_cross=1, num_heads=8, kernel_size=7, downsample=False)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)  多帧时，N_views×（1+相邻帧）
        if self.with_img_backbone:
            img_feats, pts_feats, depth, semantic = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
            img_feats = img_feats[0]

        if self.with_pts_backbone:
            bda_aug = img_inputs[-1]
            batch_size = len(points)
            # xyz_coords = points[0][:, :3]
            # # rotated_xyz_coords = torch.matmul(xyz_coords, bda_aug.T)
            # # 提取前三个维度 (x, y, z) 并传输回 CPU
            # points1 = xyz_coords.cpu().numpy()
            # import open3d as o3d
            # # 创建 Open3D 点云对象
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points1)
            # # 显示点云
            # o3d.visualization.draw_geometries([point_cloud])
            for b in range(batch_size):
                p_b = torch.matmul(points[b][:, :3], bda_aug[b].T)
                points[b][:, :3] = p_b
            batch_idx = torch.cat([torch.full((points[i].shape[0],), i) for i in range(batch_size)])
            pts = torch.cat(points, dim=0)

            gt_point = kwargs['point_label']
            gt_point = torch.cat(gt_point, dim=0)
            data_dict = {
                'points': pts,  #
                'labels': gt_point,  #
                'batch_idx': batch_idx.long().to(pts.device),  # Batch index for each point
                'batch_size': batch_size,  # Assuming a single batch
                # 'co_coor':coor
            }
            data_dict = self.pts_backbone(data_dict)
            pts_bev_feat = [data_dict['layer_2']['bev_feat'], data_dict['layer_3']['bev_feat']]
            pts_bev_feat = self.pts_neck(pts_bev_feat)[0]  # B,C,256,256的BEV特征

            pts_bev_feat = pts_bev_feat.flip(dims=[2])
            # bev_feature_abs_sum1 = pts_bev_feat.abs().sum(dim=1).squeeze().cpu().detach().numpy()
            # bev_norm_log1 = ((self.log_normalize(bev_feature_abs_sum1)) * 255.0).astype(np.uint8)

            # pts_bev_feat = self.pts_bev_encoder_backbone(pts_bev_feat)
            # pts_bev_feat = self.pts_bev_encoder_neck(pts_bev_feat)#BEV特征

        # bev_feature_abs_sum = img_feats[0].abs().sum(dim=1).squeeze().cpu().detach().numpy()
        # bev_norm_log=((self.log_normalize(bev_feature_abs_sum))*255.0).astype(np.uint8)
        # voxel_sem=kwargs['voxel_semantics'][0].squeeze(0)
        # voxel_label=voxel_sem.sum(dim=2).cpu().detach().numpy()
        # voxel_label_log = ((self.log_normalize(voxel_label)) * 255.0).astype(np.uint8)
        # plt.imshow(bev_norm_log[0])
        # plt.figure()
        # plt.imshow(bev_norm_log1[0])
        # plt.figure()
        # plt.imshow(bev_norm_log[0])
        # plt.figure()
        # plt.imshow(bev_norm_log1[0])
        # plt.figure()
        # plt.imshow(voxel_label_log)
        #
        # plt.show()

        if self.with_pts_backbone and self.with_img_backbone:
            img_feats_up = self.conv_up(img_feats)
            # fuse_features = self.bev_fusions(img_feats_up, pts_bev_feat)
            img_feats_up = img_feats_up.permute(0, 2, 3, 1)
            pts_bev_feat = pts_bev_feat.permute(0, 2, 3, 1)
            fuse_features = self.pts_backbone.bev_align(img_feats_up, pts_bev_feat)
            fuse_features = fuse_features.permute(0, 3, 1, 2)
            fuse_features = self.bev_fusions(fuse_features)
            fuse_features = self.conv_down(fuse_features)

        gt_depth = kwargs['gt_depth']  # (B, N_views, img_H, img_W)
        gt_sem = kwargs['gt_sem']  # (B, N_views, img_H, img_W)

        losses = dict()
        # losses['loss_pts'] = self.pts_backbone.criterion(data_dict)*0.1
        if self.with_img_backbone:
            loss_depth, loss_sem = self.img_view_transformer.get_depth_loss(gt_depth, depth, semantic,
                                                                            gt_sem)  # depth的监督
            losses['loss_depth'] = loss_depth
            losses['loss_sem'] = loss_sem
        losses['pts_loss'] = data_dict['loss']

        img_feats = self.bev_encoder(fuse_features)

        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)


        loss_occ = self.forward_occ_train(img_feats, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        if self.with_img_backbone:
            img_feats, pts_feats, depth, sem = self.extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs)
            img_feats = img_feats[0]

        if self.with_pts_backbone:
            bda_aug = img[-1].squeeze()
            batch_size = len(points)
            batch_idx = torch.cat([torch.full((points[i].shape[0],), i) for i in range(batch_size)])
            pts = torch.cat(points, dim=0)
            xyz_coords = pts[:, :3]
            rotated_xyz_coords = torch.matmul(xyz_coords, bda_aug.T)
            pts[:, :3] = rotated_xyz_coords
            # points = rotated_xyz_coords.cpu().numpy()
            # import open3d as o3d
            # # 创建 Open3D 点云对象
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)
            #
            # # 显示点云
            # o3d.visualization.draw_geometries([point_cloud])

            # pts[:,:3]=rotated_xyz_coords
            gt_point = kwargs['point_label']
            if isinstance(gt_point, list):
                gt_point = gt_point[0]
            gt_point = torch.cat(gt_point, dim=0)
            data_dict = {
                'points': pts,  #
                'labels': gt_point,  #
                'batch_idx': batch_idx.long().to(pts.device),  # Batch index for each point
                'batch_size': batch_size,  # Assuming a single batch
                # 'co_coor':coor
            }
            data_dict = self.pts_backbone(data_dict)
            pts_bev_feat = [data_dict['layer_2']['bev_feat'], data_dict['layer_3']['bev_feat']]
            pts_bev_feat = self.pts_neck(pts_bev_feat)[0]  # B,C,256,256的BEV特征
            # pts_bev_feat = self.conv_down(pts_bev_feat)
            pts_bev_feat = pts_bev_feat.flip(dims=[2])

        if self.with_pts_backbone and self.with_img_backbone:
            img_feats_up = self.conv_up(img_feats)
            # fuse_features = self.bev_fusions(img_feats_up, pts_bev_feat)
            img_feats_up = img_feats_up.permute(0, 2, 3, 1)
            pts_bev_feat = pts_bev_feat.permute(0, 2, 3, 1)
            fuse_features = self.pts_backbone.bev_align(img_feats_up, pts_bev_feat)
            fuse_features = fuse_features.permute(0, 3, 1, 2)
            fuse_features = self.bev_fusions(fuse_features)
            fuse_features = self.conv_down(fuse_features)
        img_feats = self.bev_encoder(fuse_features)

        occ_bev_feature = img_feats
        if self.upsample:  # false
            occ_bev_feature = F.interpolate(img_feats, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


@DETECTORS.register_module()
class BEVDetOCCTRT(BEVDetOCC):
    def __init__(self,
                 wocc=True,
                 wdet3d=True,
                 uni_train=True,
                 **kwargs):
        super(BEVDetOCCTRT, self).__init__(**kwargs)
        self.wocc = wocc
        self.wdet3d = wdet3d
        self.uni_train = uni_train

    def result_serialize(self, outs_det3d=None, outs_occ=None):
        outs_ = []
        if outs_det3d is not None:
            for out in outs_det3d:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
        if outs_occ is not None:
            outs_.append(outs_occ)
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
            self,
            img,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x[0])
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths,
                               int(self.img_view_transformer.grid_size[0].item()),
                               int(self.img_view_transformer.grid_size[1].item()),
                               int(self.img_view_transformer.grid_size[2].item())
                               )
        x = x.permute(0, 3, 1, 2).contiguous()
        # return [x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x]
        bev_feature = self.img_bev_encoder_backbone(x)
        occ_bev_feature = self.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc == True:
            if self.uni_train == True:
                if self.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                                    mode='bilinear', align_corners=True)
            outs_occ = self.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d == True:
            outs_det3d = self.pts_bbox_head([det_bev_feature])

        outs = self.result_serialize(outs_det3d, outs_occ)
        return outs

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)

