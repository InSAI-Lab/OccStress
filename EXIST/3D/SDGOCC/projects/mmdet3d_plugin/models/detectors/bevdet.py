# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.models import DETECTORS
from mmdet3d.models import CenterPoint
from mmdet3d.models import builder
import numpy as np


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    def __init__(self, img_backbone, img_neck, img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck,
                 pts_bbox_head=None, **kwargs):
        super(BEVDet, self).__init__(img_backbone=img_backbone, img_neck=img_neck, pts_bbox_head=pts_bbox_head,
                                     **kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        # self.pts_bev_encoder_backbone = builder.build_backbone(pts_bev_encoder_backbone)
        # self.pts_bev_encoder_neck = builder.build_neck(pts_bev_encoder_neck)

    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs) #B×6，C 4倍，8倍，16倍下采样特征   原始时为16,32倍（第3层）特征，4d时为4,16,32倍特征
        # if len(x)==3:
        #     x=x[1:]
        stereo_feat = None
        if stereo:
            stereo_feat = x[0] #4倍下采样特征
            # x = x[1:]
        if self.with_img_neck: #neck
            x = self.img_neck(x) #特征FPN融合
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat #融合后的16倍特征，4倍特征

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone(x) #3*3卷积得到多层BEV特征
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

        x = self.img_bev_encoder_neck(x) #额外上采样  FPN后的BEV特征
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        """
        Args:
            sensor2ego: (B, N_views=6, 4, 4)
            ego2global: (B, N_views=6, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
        Returns:
            mlp_input: (B, N_views, 27)
        """
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)   # (B, 3, 3) --> (B, N, 3, 3) 升维
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],     # fx
            intrin[:, :, 1, 1],     # fy
            intrin[:, :, 0, 2],     # cx
            intrin[:, :, 1, 2],     # cy
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2]
        ], dim=-1)      # (B, N_views, 15)  内参，后处理旋转，后处理平移，bda矩阵每个有效值
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1) #外参的9个有效参数
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)      # (B, N_views, 27)
        return mlp_input

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs=inputs[0]
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape#图像
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs #图像，相机到车辆，车辆到全局坐标系，内参，图像增强处理的矩阵（旋转与平移），BEV增强矩阵

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)   第一帧的车辆坐标系（关键帧）到全局坐标系
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)   全局坐标系到第一帧的车辆坐标系（关键帧）
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)  传感器到全局坐标系然后全部转换到第一帧的车辆坐标系（关键帧）
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def log_normalize(self, image):
        normalized_image = np.log1p(image)  # log1p 等同于 log(x + 1)，避免了 log(0) 的问题
        normalized_image -= normalized_image.min()
        normalized_image /= normalized_image.max()
        return normalized_image

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs) #增强后的图像，两个外参，内参，图像增强处理的矩阵（旋转与平移），BEV增强矩阵
        mlp_input= self.get_mlp_input(*img_inputs[1:]) #内参，后处理旋转，后处理平移，bda矩阵每个有效值，外参的9个有效参数
        img_inputs.append(mlp_input)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)  B，N，256,16,44（包含FPN）  16倍下采样
        x, depth,semantic = self.img_view_transformer([x] + img_inputs[1:],**kwargs) #初始bev特征（B,64,200,200）与depth特征
        # import numpy as np
        # import matplotlib.pyplot as plt
        # bev_feature_abs_sum = x.abs().sum(dim=1).squeeze().cpu().detach().numpy()
        # bev_norm_log=((self.log_normalize(bev_feature_abs_sum))*255.0).astype(np.uint8)
        # # 将结果移动到 CPU
        # # bev_feature_summed_cpu = np.ceil(bev_feature_normalized).astype(np.uint8) #uint8为向下取整
        # # heatmap = cv2.applyColorMap(bev_feature_summed_cpu, cv2.COLORMAP_HOT)
        # plt.imshow(bev_norm_log[0])
        # # plt.colorbar()
        # plt.axis('off')
        # plt.savefig('/media/aiboy/DeepLearn/SADAOCC/bev_feature.png',dpi=600, bbox_inches='tight',pad_inches = 0)
        # plt.savefig('/media/aiboy/DeepLearn/SADAOCC/bev_feature.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
        # plt.show()
        # x: (B, C, Dy, Dx)
        # depth: (B*N, D, fH, fW)
        # x = self.bev_encoder(x) #最后的BEV特征 1,256,200,200
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import cv2
        # bev_feature_abs_sum = x.abs().sum(dim=1).squeeze().cpu().detach().numpy()
        # # 步骤 2: 归一化特征图到 [0, 1]
        # bev_feature_min = np.min(bev_feature_abs_sum)
        # bev_feature_max = np.max(bev_feature_abs_sum)
        # bev_feature_normalized = (bev_feature_abs_sum - bev_feature_min) / (bev_feature_max - bev_feature_min)
        # bev_feature_normalized=bev_feature_normalized*255
        # bev_norm_log=((self.log_normalize(bev_feature_abs_sum))*255.0).astype(np.uint8)
        # plt.imshow(bev_norm_log)
        # plt.colorbar()
        # plt.show()
        return [x], depth,semantic

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, depth,semantic= self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth,semantic

    def forward_train(self,
                      points=None,
                      img_inputs=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_inputs:
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, _ = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], #单帧
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton.
        Returns:
            bbox_list: List[dict0, dict1, ...]   len = bs
            dict: {
                'pts_bbox':  dict: {
                              'boxes_3d': (N, 9)
                              'scores_3d': (N, )
                              'labels_3d': (N, )
                             }
            }
        """
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        # bbox_pts: List[dict0, dict1, ...],  len = batch_size
        # dict: {
        #   'boxes_3d': (N, 9)
        #   'scores_3d': (N, )
        #   'labels_3d': (N, )
        # }
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs