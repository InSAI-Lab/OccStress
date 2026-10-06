# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS
import matplotlib.pyplot as plt
from ...ops import bev_pool_v2
from ..model_utils import DepthNet
from mmdet.models.backbones.resnet import BasicBlock
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
import numpy as np


@NECKS.register_module(force=True)
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
    ):
        super(LSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample,grid_config['diffusion'])      # (D, fH, fW, 3)  3:(u, v, d)
        self.diffusion = grid_config['diffusion']
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z


    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])     # (min_x, min_y, min_z)
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])        # (dx, dy, dz)
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])                   # (Dx, Dy, Dz)

    def create_frustum(self, depth_cfg, input_size, downsample,diffusion_cfg):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        # H_in, W_in = input_size #512,1408
        # H_feat, W_feat = H_in // downsample, W_in // downsample #16倍下采样
        d = diffusion_cfg[-1]     # (D, fH, fW)   1到44.5间隔为0.5. 88个值。扩充到88，H，W
        self.D = 2*d+1#88
        return self.D    # (D, fH, fW, 3)  3:(u, v, d)

    def spread_and_average_values_optimized(self, depth_maps, radius,preserve_original=True):
        """
        在给定半径内，将非零深度值并行地传播到周围的零值区域，重叠区域取平均值。

        参数:
        depth_maps (torch.Tensor): 输入的深度图，形状为 (B, N, H, W)。
        radius (int): 扩散的半径。

        返回:
        torch.Tensor: 更新后的深度图。
        """
        B, N, H, W = depth_maps.shape  #深度图，seg_map为B*N，H，W
        kernel_size = 2 * radius + 1
        device = depth_maps.device
        # 创建扩散核
        Y, X = torch.meshgrid(torch.arange(kernel_size, device=device), torch.arange(kernel_size, device=device),
                              indexing='ij')
        dist = torch.sqrt((X - radius) ** 2 + (Y - radius) ** 2)
        kernel = (dist <= radius).float().unsqueeze(0).unsqueeze(0)
        # 重复核以匹配通道数
        kernel = kernel.repeat(N, 1, 1, 1)  # 重复核，使其大小为 [N, 1, kernel_size, kernel_size]
        # 保留原始值以避免被平均化覆盖
        original_depth_maps = depth_maps.clone() if preserve_original else None
        # 生成权重图，用于记录每个位置的值是由多少个源点贡献的
        weights = F.conv2d(depth_maps.sign(), kernel, padding=radius, groups=N)
        # 扩散深度值
        output = F.conv2d(depth_maps, kernel, padding=radius, groups=N)

        # 计算平均值
        updated_depth_maps = output / weights
        updated_depth_maps[weights == 0] = 0  # 处理除以零的情况

        if preserve_original:
            mask = original_depth_maps > 0
            updated_depth_maps[mask] = original_depth_maps[mask]

        return updated_depth_maps

    def generate_depth_maps(self, depth_map, d_min, d_max, D):
        """
        根据深度区间和深度桶的数量来生成新的深度图。

        参数:
        depth_map (torch.Tensor): 输入的深度图，形状为 (N, H, W)。
        d_min (float): 增加深度区间的最小值。
        d_max (float): 增加深度区间的最大值。
        D (int): 要生成的深度图数量。

        返回:
        torch.Tensor: 生成的多张深度图，形状为 (N, D+1, H, W)。
        """
        device = depth_map.device
        depth_bins = torch.arange(D, device=device)
        depth_maps = d_min + ((d_max - d_min) / (D * (D + 1))) * (
                depth_bins.unsqueeze(-1).unsqueeze(-1) * (depth_bins.unsqueeze(-1).unsqueeze(-1) + 1)).view(-1)
        depth_maps = torch.cat((depth_maps, torch.tensor(d_max, device=device).unsqueeze(0)), dim=-1)

        # 扩展深度图的维度，使其与 new_depth_maps 的维度匹配
        depth_map = depth_map.unsqueeze(1).expand(-1, D + 1, -1, -1)

        add_depth_maps = depth_map + depth_maps.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        add_depth_maps[depth_map == 0] = 0
        sub_depth_maps = depth_map[:, 1:, :, :] - depth_maps.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)[:, 1:, :, :].flip(
            dims=(1,))
        sub_depth_maps[depth_map[:, 1:, :, :] == 0] = 0
        new_depth_maps = torch.cat((sub_depth_maps, add_depth_maps), dim=1)
        return new_depth_maps
    #先下采样再传播
    def get_downsampled_depth(self, gt_depths,seg_map):
        # remove point not in depth range  移除范围点    语义点只有0和1  原始图像
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
        # self.downsample = 4
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample) #N×下采样的二次方
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths) #0取1e5
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # 取深度最小的那个值
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)  # 1到60   得到下采样的GT深度图

        gt_dep = torch.where(
            (gt_depths < self.grid_config['depth'][1] + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_dep = gt_dep.view(B, N, H // self.downsample,
                              W // self.downsample) #下采样深度图
        seg_map=seg_map.view(B, N, H // self.downsample,
                              W // self.downsample) #下采样深度图

        expanded_depth_map = gt_dep.clone() #B*N,H,W

        for category in torch.unique(seg_map):  #B*N,H,W   0到16
            mask = (seg_map == category).float()

            # 用掩码选择当前类别的深度图
            category_depth_map = gt_dep * mask

            # 扩展当前类别的深度图
            expanded_category_depth_map = self.spread_and_average_values_optimized(category_depth_map, 1)
            expanded_category_depth_map = expanded_category_depth_map.squeeze(1)
            # 只在同类别掩码内更新深度图
            expanded_depth_map[mask == 1] = torch.max(expanded_depth_map, expanded_category_depth_map)[mask == 1]

        gt_deps=expanded_depth_map.view(B * N, H // self.downsample,W // self.downsample)#半径为多大

        # gt_deps = self.spread_and_average_values_optimized(gt_dep,seg_map, 1).view(B * N, H // self.downsample,
        #                         W // self.downsample)#半径为多大
        depth_all = self.generate_depth_maps(gt_deps, self.diffusion[0], self.diffusion[1], self.diffusion[2])  # 得到范围深度图  从0线性增加到1，分成4个桶 加入参数
        depth_all = torch.where(
            (depth_all < self.grid_config['depth'][1] + 1) & (depth_all >= 0.0),
            depth_all, torch.zeros_like(depth_all))
        mask = (depth_all!=0)
        return depth_all,mask #B×N，2D+1，H//4，W//4

    #先传播再下采样
    # def get_depth_downsampled(self, gt_depths):
    #     # remove point not in depth range  移除范围点    语义点只有0和1  原始图像
    #     gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
    #     gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
    #     # self.downsample = 4
    #     B, N, H, W = gt_depths.shape
    #
    #     gt_deps = self.spread_and_average_values_optimized(gt_depths, 1)
    #
    #     gt_depths = gt_deps.view(
    #         B * N,
    #         H // self.downsample,
    #         self.downsample,
    #         W // self.downsample,
    #         self.downsample,
    #         1,
    #     )
    #     gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
    #     gt_depths = gt_depths.view(
    #         -1, self.downsample * self.downsample) #N×下采样的二次方
    #     gt_depths_tmp = torch.where(gt_depths == 0.0,
    #                                 1e5 * torch.ones_like(gt_depths),
    #                                 gt_depths) #0取1e5
    #     gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # 取深度最小的那个值
    #     gt_depths = gt_depths.view(B * N, H // self.downsample,
    #                                W // self.downsample)  # 1到60   得到下采样的GT深度图
    #
    #     gt_dep = torch.where(
    #         (gt_depths < self.grid_config['depth'][1] + 1) & (gt_depths >= 0.0),
    #         gt_depths, torch.zeros_like(gt_depths)) #挑选
    #     depth_all = self.generate_depth_maps(gt_dep, 0.0, 1.0, 4)  # 得到范围深度图  从0线性增加到1，分成4个桶
    #     depth_all = torch.where(
    #         (depth_all < self.grid_config['depth'][1] + 1) & (depth_all >= 0.0),
    #         depth_all, torch.zeros_like(depth_all))
    #
    #
    #     return depth_all #B×N，2D+1，H//4，W//4

    def creat_point(self,depth, H_in, W_in):
        B, N, D, H, W = depth.shape

        # 创建x和y坐标网格
        x = torch.linspace(0, W_in - 1, W, dtype=torch.float, device=depth.device)
        y = torch.linspace(0, H_in - 1, H, dtype=torch.float, device=depth.device)

        # 重新整形和扩展坐标网格
        x = x.view(1, 1, 1, 1, W).expand(B, N, D, H, W)
        y = y.view(1, 1, 1, H, 1).expand(B, N, D, H, W)

        # 在最后一个维度上组合原始tensor和坐标网格
        expanded_tensor = torch.stack((x,y,depth), dim=-1)

        return expanded_tensor #B×N×D×H×W×3

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda,depth,seg_map):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            sensor2ego (torch.Tensor): Transformation from camera coordinate system to
                ego coordinate system in shape (B, N_cams, 4, 4).
            ego2global (torch.Tensor): Translation from ego coordinate system to
                global coordinate system in shape (B, N_cams, 4, 4).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).
            bda (torch.Tensor): Transformation in bev. (B, 3, 3)

        Returns:
            torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)
        """
        B, N, H, W = depth.shape

        #生成多层深度图，每个像素点都存在不同的深度值  B×N，2D+1，H//4，W//4
        depth_all,mask = self.get_downsampled_depth(depth,seg_map)  # 得到范围深度图  从0线性增加到1，分成4个桶
        depth_all = depth_all.view(B,N,-1,H//self.downsample,W//self.downsample)
        mask = mask.view(B,N,-1,H//self.downsample,W//self.downsample)
        # depth_1=self.get_depth_downsampled(depth)  #先插值再下采样
        # d从2到41.5，步长为0.5(80份)    x从0到703，步长为16.3    y从0到255，步长为17   根据特征尺度如16,44在高度上构建坐标范围是2到41.5
        depth_points=self.creat_point(depth_all,H,W)#B×N，2D+1，H//4，W//4，3
        # post-transformation   反图像增强
        # (D, fH, fW, 3) - (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)  得到B，N，D，fH，fW，3
        points = depth_points - post_trans.view(B, N, 1, 1, 1, 3)#16倍网格点转换到图像增强后的图像平面  16倍网格点  给每个图像像素点加深度分布。
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))  # 将网格点乘以逆旋转矩阵

        # cam_to_ego
        # (B, N_, D, fH, fW, 3, 1)  3: (du, dv, d)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5) #将x,y乘以d，然后将d放到最后一维  去归一化
        # R_{c->e} @ K^-1
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs)) #相机到像素点内参以及外参旋转矩阵
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1) #旋转到车辆坐标系下
        # (B, N, D, fH, fW, 3) + (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)  #车辆坐标下的三维点

        # (B, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1) --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1) #增加bev数据增强的变换
        return points,mask

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        """
        Args:
            coor: (B, N, D, fH, fW, 3)
            depth: (B, N, D, fH, fW)
            feat: (B, N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
        """
        #BEV的全局排序，深度的全局排序，特征的全局排序，每个体素的起始索引，每个体素的长度（点数）
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat)     # (B, C, Dz, Dy, Dx)
            dummy = torch.cat(dummy.unbind(dim=2), 1)   # (B, C*Dz, Dy, Dx)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C) 图像特征
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]), #200,200,1
                          feat.shape[-1])       # (B, Dz, Dy, Dx, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, #深度归一化图，图像特征，深度排序，特征排序，bev全局排序，体素起始索引，体素长度
                               bev_feat_shape, interval_starts,
                               interval_lengths)    # (B, C, Dz, Dy, Dx) 1,200,200
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)     # (B, C*Dz, Dy, Dx)  沿第三纬度分解并在第一维度拼接  分解与拼接 等价与torch.squeeze(tensor, dim=2)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )
                interval_lengths: (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W #点数不变的
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)    # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)   # [0, 1, ...,B*N*fH*fW-1]
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W) # (B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()     # (B*N*D*fH*fW,)   在D方向上复制

        # convert coordinate into the voxel space   到体素坐标
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)      # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3) 体素坐标
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor) #对每个点对应的batch_id
        coor = torch.cat((coor, batch_idx), 1)      # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)  四维体素坐标

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) #将网格展开为一维   每个点的全局排名
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort() #全局排名排序
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order] #按照全局排名排序，相同排名的点属于同一个体素

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1] #相邻体素不同，说明是不同的体素
        interval_starts = torch.where(kept)[0].int() #每个体素的起始点
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts) #每个体素的长度（点数）
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:7])       # (B, N, D, fH, fW, 3)
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def get_bev(self, coor,mask, depth, image_feat):
        B, N, D, H, W, _ = coor.shape
        _, C, H, W = image_feat.shape
        num_points = B * N * D * H * W
        mask=mask.view(num_points)
        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.  得到正的体素网格：X，Y范围为1，体素Z周
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)  # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)  # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        volumn_features = (depth.unsqueeze(1) * image_feat.unsqueeze(2))  # b,n,c,d,h,w  每个点的特征
        volumn_features = volumn_features.permute(0, 2, 3, 4, 1).reshape(-1, C)

        coor,volumn_features =coor[mask],volumn_features[mask]

        # filter out points that are outside box   生成200*200*1的网格特征  在高度上只构建一个体素
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        coor, point_feat = coor[kept], volumn_features[kept]  # N*4,N*C  N的量级为100W

        # 计算全局排名
        ranks_bev = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) + \
                    coor[:, 2] * (self.grid_size[1] * self.grid_size[0]) + \
                    coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        coor = coor[order] #N*4
        # 过滤和排序特征
        point_feat = point_feat[order] #N*C

        # 生成体素的起始索引和索引长度
        vo_coor, inverse_indices = torch.unique(coor, return_inverse=True, dim=0)

        # 使用scatter_add并行累加特征
        pc_add = torch_scatter.scatter_add(point_feat, inverse_indices, dim=0,
                                                   dim_size=inverse_indices.max() + 1)

        # kept = torch.ones(len(ranks_bev), dtype=torch.bool, device=coor.device)
        # kept[1:] = ranks_bev[order][1:] != ranks_bev[order][:-1]
        # interval_starts = torch.where(kept)[0] #体素起始索引
        # interval_lengths = torch.zeros_like(interval_starts)
        # interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        # interval_lengths[-1] = len(ranks_bev) - interval_starts[-1] #体素长度
        # vo_coor=coor[interval_starts] #体素坐标
        #
        # pc_add = torch.zeros((len(interval_starts), C), device=coor.device)
        # for i, start in enumerate(interval_starts):
        #     length = interval_lengths[i]
        #     pc_add[i] = torch_scatter.scatter_add(point_feat[start:start + length],
        #                                           torch.zeros(length, dtype=torch.long, device=coor.device), dim=0,
        #                                           dim_size=1)
        bev_maps = torch.zeros(B, 1, 200,200, C,
                               device=pc_add.device)
        # 填充每个批次的 BEV 网格
        for i in range(B):
            mask = vo_coor[:, 3] == i  # 找到当前批次的体素
            bev_maps[i, 0, vo_coor[mask, 1], vo_coor[mask, 0]] = pc_add[mask]

        return torch.squeeze(bev_maps.permute(0, 4, 2, 3,1),dim=-1)

    def log_normalize(self, image):
        normalized_image = np.log1p(image)  # log1p 等同于 log(x + 1)，避免了 log(0) 的问题
        normalized_image -= normalized_image.min()
        normalized_image /= normalized_image.max()
        return normalized_image

    def view_transform_core(self, input, depth, seg_map,tran_feat,**kwargs):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)      # (B, N, C, fH, fW)
            feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
            depth = depth.view(B, N, self.D, H, W)      # (B, N, D, fH, fW)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])   # (B, Dz, Dy, Dx, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)   # (B, C, Dz, Dy, Dx)

            bev_feat = bev_feat.squeeze(2)      # (B, C, Dy, Dx)
        else:
            gt_depth = kwargs['gt_depth']
            if isinstance(gt_depth, list):
                gt_depth = gt_depth[0]
            with torch.no_grad():
                coor,mask = self.get_ego_coor(*input[1:7],gt_depth,seg_map)   # (B, N, D, fH, fW, 3)  车辆坐标系下的三维点
            torch.cuda.empty_cache()
            bev_feat_add=self.get_bev(coor,mask,depth,tran_feat)  # (B, C, Dy, Dx)
            # print('get_bev time:',time.time()-time1)
            # bev_feat = self.voxel_pooling_v2(
            #     coor, depth.view(B, N, self.D, H, W),
            #     tran_feat.view(B, N, self.out_channels, H, W))      # (B, C*Dz(=1), Dy, Dx)
            # import numpy as np
            # import matplotlib.pyplot as plt
            # bev_feature_abs_sum1 = bev_feat_add.abs().sum(dim=1).squeeze().cpu().detach().numpy()
            # bev_norm_log1 = ((self.log_normalize(bev_feature_abs_sum1)) * 255.0).astype(np.uint8)
            # plt.imshow(bev_norm_log1[0])
            # plt.show()
        return bev_feat_add, depth #初始bev特征与深度图

    def view_transform(self, input, depth,seg_map, tran_feat,**kwargs):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, C, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth,seg_map, tran_feat,**kwargs)

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x = input[0]    # (B, N, C_in, fH, fW)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N, C_in, fH, fW)

        # (B*N, C_in, fH, fW) --> (B*N, D+C, fH, fW)
        x = self.depth_net(x) #1*1卷积从256到152（88+64）
        depth_digit = x[:, :self.D, ...]    # (B*N, D, fH, fW) 取出88的深度图
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N, C, fH, fW) 纹理特征
        depth = depth_digit.softmax(dim=1) #深度图归一化
        return self.view_transform(input, depth, tran_feat)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0,loss_sem_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.loss_sem_weight = loss_sem_weight
        self.class_=17
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            **depthnet_cfg)
        self.bias=depthnet_cfg['bias']
        self.depth_net1 = nn.Conv2d(
            self.in_channels, self.class_+self.D + self.out_channels, kernel_size=1, padding=0)

        if depthnet_cfg['stereo']==True:
            depth_conv_input_channels = self.D + self.D
            downsample = nn.Conv2d(depth_conv_input_channels,
                                   self.D, 1, 1, 0)
            depth_conv_list = [BasicBlock(depth_conv_input_channels, self.D,downsample=downsample),
                               BasicBlock(self.D, self.D)]
            depth_conv_list.append(
                nn.Conv2d(
                    self.D,
                    self.D,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            self.depth_conv = nn.Sequential(*depth_conv_list)
            self.cost_volumn_net=self.depth_net.cost_volumn_net
        # if stereo:
        #     downsample = nn.Conv2d(self.in_channels+self.D,
        #                             mid_channels, 1, 1, 0)
        #     cost_volumn_net = []
        #     for stage in range(int(1)):
        #         cost_volumn_net.extend([
        #             nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
        #                       stride=1, padding=1),
        #             nn.BatchNorm2d(depth_channels)])
        #     self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
        #     self.bias = bias

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

    def get_grid(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda,depth,gt_sem):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            sensor2ego (torch.Tensor): Transformation from camera coordinate system to
                ego coordinate system in shape (B, N_cams, 4, 4).
            ego2global (torch.Tensor): Translation from ego coordinate system to
                global coordinate system in shape (B, N_cams, 4, 4).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).
            bda (torch.Tensor): Transformation in bev. (B, 3, 3)

        Returns:
            torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)
        """
        B, N, H, W = depth.shape

        #生成多层深度图，每个像素点都存在不同的深度值  B×N，2D+1，H//4，W//4
        depth_all,mask = self.get_downsampled_depth(depth,gt_sem)  # 得到范围深度图  从0线性增加到1，分成4个桶
        depth_all = depth_all.view(B,N,-1,H//self.downsample,W//self.downsample)
        mask = mask.view(B,N,-1,H//self.downsample,W//self.downsample)
        # depth_1=self.get_depth_downsampled(depth)  #先插值再下采样
        # d从2到41.5，步长为0.5(80份)    x从0到703，步长为16.3    y从0到255，步长为17   根据特征尺度如16,44在高度上构建坐标范围是2到41.5
        depth_points=self.creat_point(depth_all,H,W)#B×N，2D+1，H//4，W//4，3

        return depth_points,mask

    def gen_grid(self, metas,coor,mask, B, N, D, H, W, hi, wi):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
            B: batchsize
            N: N_views
            D: D
            H: fH_stereo
            W: fW_stereo
            hi: H_img
            wi: W_img
        Returns:
            grid: (B*N_views, D*fH_stereo, fW_stereo, 2)
        """
        # frustum = metas['frustum']      # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        # 逆图像增广:
        points = coor - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        # (u, v, d) --> (du, dv, d)
        # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)  #像素去归一化

        # cur_pixel --> curr_camera --> prev_camera   当前像素坐标转换到前一帧相机坐标系
        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        neg_mask = points[..., 2, 0] < 1e-3 #去掉深度很小的点
        # prev_camera --> prev_pixel
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        # (du, dv, d) --> (u, v)   (B, N_views, D, fH_stereo, fW_stereo, 2, 1)
        points = points[..., :2, :] / points[..., 2:3, :] #归一化

        # 图像增广   当前帧像素3D点转到到之前帧并图像增强
        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 1, 1, 2)   # (B, N_views, D, fH_stereo, fW_stereo, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0 #归一化到-1~1
        px[neg_mask] = -2
        py[neg_mask] = -2
        px[~mask] = -2
        py[~mask] = -2
        grid = torch.stack([px, py], dim=-1)    # (B, N_views, D, fH_stereo, fW_stereo, 2)
        grid = grid.view(B * N, D * H, W, 2)    # (B*N_views, D*fH_stereo, fW_stereo, 2) 网格
        return grid

    def calculate_cost_volumn(self, metas,coor,mask):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            cost_volumn: (B*N_views, D, fH_stereo, fW_stereo)
        """
        prev, curr = metas['cv_feat_list']   # (B*N_views, C_stereo, fH_stereo, fW_stereo)  四倍下采样的第二帧，第一帧图像特征
        group_size = 4
        _, c, hf, wf = curr.shape   #
        hi, wi = hf * 4, wf * 4     # H_img, W_img
        # B, N, _ = metas['post_trans'].shape
        B,N,D, H, W, _ = coor.shape #这个参数控制
        # D=9
        grid = self.gen_grid(metas,coor,mask, B, N, D, H, W, hi, wi).to(curr.dtype)   # (B*N_views, D*fH_stereo, fW_stereo, 2)  当前网格转换到之前帧的网格

        prev = prev.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        curr = curr.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size): #分组为 通道/4
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...] #按通道取出4个通道
            wrap_prev = F.grid_sample(prev_curr, grid, #12,4，H/4,W/4    12,D*fH_stereo, fW_stereo, 2
                                      align_corners=True,
                                      padding_mode='zeros')     # (B*N_views, group_size4, D*fH_stereo, fW_stereo)
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            # (B*N_views, group_size, 1, fH_stereo, fW_stereo) - (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # --> (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # https://github.com/HuangJunJie2017/BEVDet/issues/278   对特征增加D纬度然后相减    对应特征相减得到匹配成本
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)      # (B*N_views, D, fH_stereo, fW_stereo)  通道维度求和
            cost_volumn += cost_volumn_tmp  # (B*N_views, D, fH_stereo, fW_stereo)
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0  #特征为0的地方加上bias
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias

        # matching cost --> prob
        cost_volumn = - cost_volumn  #取反
        cost_volumn = cost_volumn.softmax(dim=1) #代价体积
        return cost_volumn

    def forward(self, input, stereo_metas=None, **kwargs):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
                mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]   #第一帧的信息
        # self.D = 9
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N_views, C, fH, fW) 得到第几倍的特征
        gt_depth = kwargs['gt_depth']
        if isinstance(gt_depth, list):
            gt_depth = gt_depth[0]


        x = self.depth_net(x, mlp_input, stereo_metas) # (B*6, D+C_context, fH, fW) 加入两帧特征构建的volumn  输入的当前帧图像，MLP信息以及历史帧的volumn'
        #x = self.depth_net1(x)  # (B*N_views, D+C_context, fH, fW)  88个通道
        depth = x[:, :self.D, ...]
        semantic = x[:, self.D:self.class_+self.D, ...]    # (B*N_views, D, fH, fW)  88个通道
        tran_feat = x[:, self.class_+self.D:self.class_+self.D + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)  80个通道纹理特征
        semantic = semantic.softmax(dim=1)  # (B*N_views, C, fH, fW) 通道上取置信度
        sem_map=torch.argmax(semantic,dim=1)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # bev_feature_abs_sum = sem_map[0].squeeze().cpu().detach().numpy()
        # bev_norm_log=((self.log_normalize(bev_feature_abs_sum))*255.0).astype(np.uint8)
        # plt.imshow(bev_norm_log)
        # plt.axis('off')
        # plt.show()

        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    # https://github.com/HuangJunJie2017/BEVDet/issues/278
                    coor, mask = self.get_grid(*input[1:7], gt_depth, sem_map)  # (B, N, D, fH, fW, 3)  车辆坐标系下的三维点
                    cost_volumn = self.calculate_cost_volumn(stereo_metas,coor,mask)      # (B*N_views, D, fH_stereo, fW_stereo)  将当前帧特征转换到上一帧视角下，然后计算特征差异（相减） 88通道
            cost_volumn = self.cost_volumn_net(cost_volumn)     # (B*N_views, D, fH, fW)
            depth = torch.cat([depth, cost_volumn], dim=1)      # (B*N_views, C_mid+D, fH, fW)  深度由代价体素和深度特征拼接  88通道
            depth = self.depth_conv(depth)  # (B*N_views, D, fH, fW)  88通道

        depth = depth.softmax(dim=1)  # (B*N_views, D, fH, fW) 通道上取置信度
        bev_feat, depth = self.view_transform(input, depth,sem_map, tran_feat,**kwargs)
        return bev_feat, depth, semantic

    def get_downsampled_gt_depth(self, gt_depths,gt_sem):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """
        if isinstance(gt_depths, list):
            gt_depths = gt_depths[0]
        if isinstance(gt_sem, list):
            gt_sem = gt_sem[0]
        B, N, H, W = gt_depths.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample,
                                   1)
        # (B*N_views, fH, fW, 1, downsample, downsample)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # (B*N_views*fH*fW, downsample, downsample)
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample) #N×16*16
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths) #0就赋值1e5
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  #取非0的最小值
        # (B*N_views, fH, fW)
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample) #变到特征图
        if not self.sid: #true
            # (D - (min_dist - interval_dist)) / interval_dist
            # = (D - min_dist) / interval_dist + 1
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:#深度图归一化
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        #取范围内的深度值
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))     # (B*N_views, fH, fW)

        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]   # (B*N_views*fH*fW, D)  one-hot编码

        gt_sem = gt_sem.view(B * N, H // self.downsample,self.downsample, W // self.downsample,self.downsample,1) #变到特征图
        gt_sem = gt_sem.permute(0,1,3,5,2,4).contiguous() #变到特征图
        gt_sem = gt_sem.view(-1,self.downsample*self.downsample) #N*4*4  16个值
        # gt_semmmm=gt_sem.cpu().detach().numpy()
        # mask = gt_sem != 0
        # # 使用高级索引提取非零元素
        # # filtered_elements = [gt_sem[i][mask[i]] for i in torch.arange(gt_sem.size(0), device=gt_sem.device)]
        # # # 计算每行的众数
        # # mode_vals = torch.tensor(
        # #     [torch.mode(elements).values.item() if elements.numel() > 0 else 0 for elements in filtered_elements],
        # #     device=gt_sem.device)
        #
        # 使用掩码提取非零元素并计算每行的众数
        # import time
        # time1=time.time()
        # filtered_elements = gt_sem[mask].split(mask.sum(dim=1).tolist())
        # mode_vals = torch.zeros(gt_sem.size(0), device=gt_sem.device)
        #
        #
        # for i, elements in enumerate(filtered_elements):
        #     if elements.numel() > 0:
        #         mode_vals[i] = torch.mode(elements).values
        # print('time:', time.time() - time1)

        mode_vals=torch.max(gt_sem, dim=-1).values  #取非0的最小值
        sem_label = mode_vals.view(B * N, H // self.downsample, W // self.downsample) #变到特征图
        sem_label = F.one_hot(sem_label.long(), num_classes=18).view(-1, 18) #one-hot编码

        return gt_depths.float(),sem_label.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds,semantic,gt_sem):
        """
        Args:
            depth_labels: (B, N_views, img_h, img_w)
            depth_preds: (B*N_views, D, fH, fW)
        Returns:

        """
        depth_labels,sem_labels = self.get_downsampled_gt_depth(depth_labels,gt_sem)      # (B*N_views*fH*fW, D)  下采样
        # (B*N_views, D, fH, fW) --> (B*N_views, fH, fW, D) --> (B*N_views*fH*fW, D)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        semantic_preds = semantic.permute(0, 2, 3,1).contiguous().view(-1, 17)
        # fg_sem=torch.max(sem_labels, dim=1).values > 0.0

        fg_mask = torch.max(depth_labels, dim=1).values > 0.0 #只对大于0的深度值进行计算loss
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        sem_labels = sem_labels[fg_mask]
        sem_labels = sem_labels[:,1:]
        semantic_preds = semantic_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
            pred=semantic_preds
            target=sem_labels
            alpha = 0.25
            gamma = 2
            pt=(1-pred)*target + pred*(1-target)
            focal_weight = (alpha * target + (1-alpha)*(1-target))* pt.pow(gamma)
            sem_loss = F.cross_entropy(pred, target.argmax(dim=1),reduction='none').unsqueeze(1) * focal_weight
            sem_loss = sem_loss.sum() / max(1.0, len(sem_loss))
        return self.loss_depth_weight * depth_loss, self.loss_sem_weight * sem_loss


@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):
    def __init__(self,  **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)


if __name__ == '__main__':
    #测试
    import torch
    from mmdet.models.necks import LSSViewTransformerBEVDepth
    import numpy as np
