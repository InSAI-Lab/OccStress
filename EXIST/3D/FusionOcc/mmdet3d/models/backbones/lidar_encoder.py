# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_scatter
from torch.nn import functional as F
from torch import nn as nn
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import BACKBONES

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


@BACKBONES.register_module()
class CustomSparseEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            sparse_shape,
            point_cloud_range,
            voxel_size,
            order=("conv", "norm", "act"),
            norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
            base_channels=16,
            encoder_channels=([16, 16, 32], [32, 32, 48], [48, 48, 64], [64, 64]),
            encoder_paddings=([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]),
            output_channels=32,
            block_type="conv_module",
    ):
        super(CustomSparseEncoder, self).__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, (list, tuple)) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.conv_input = make_sparse_convmodule(
            in_channels,
            self.base_channels,
            1,
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="subm_input",
            conv_type="SubMConv3d",
            order=("conv",),
        )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            1,
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="subm_output",
            conv_type="SubMConv3d",
            order=("conv",),
        )

    @torch.no_grad()
    @force_fp32()
    def scatter_voxelize(self, points):
        feats, coords, bs_list, inv_idx_list = [], [], [], []
        empty_batch_ids = []
        count = 0
        for k, pt in enumerate(points):
            if pt.numel() == 0:
                # Some nuScenes-C pointcloud corruptions can remove every point in a
                # sample. Insert a zero-feature dummy voxel to keep sparse tensor
                # shapes valid, then zero the encoded feature for this batch item
                # after sparse encoding.
                feat = torch.zeros((1, self.in_channels), device=pt.device, dtype=pt.dtype)
                uniq_coord_bs = torch.tensor([[k, 0, 0, 0]], device=pt.device, dtype=torch.int32)
                inv_idx = torch.zeros((0,), device=pt.device, dtype=torch.long)
                empty_batch_ids.append(k)
                bs_list.append([count, count + feat.shape[0]])
                count += feat.shape[0]
                feats.append(feat)
                inv_idx_list.append(inv_idx)
                coords.append(uniq_coord_bs)
                continue
            coord = torch.zeros_like(pt)[:, :3]
            eps = 0
            coord[:, 0] = (pt[:, 0] - self.point_cloud_range[0]) / (self.voxel_size[0] + eps)
            coord[:, 1] = (pt[:, 1] - self.point_cloud_range[1]) / (self.voxel_size[1] + eps)
            coord[:, 2] = (pt[:, 2] - self.point_cloud_range[2]) / (self.voxel_size[2] + eps)
            coord = torch.floor(coord).int()
            uniq_coord, inv_idx = torch.unique(coord, return_inverse=True, dim=0)
            feat = torch_scatter.scatter_mean(pt, inv_idx, dim=0)

            bs_list.append([count, count + feat.shape[0]])
            count += feat.shape[0]

            feats.append(feat)
            inv_idx_list.append(inv_idx)
            uniq_coord_bs = F.pad(uniq_coord, (1, 0), mode="constant", value=k)
            coords.append(uniq_coord_bs)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)

        return feats, coords, bs_list, inv_idx_list, empty_batch_ids

    def make_encoder_layers(
            self,
            make_block,
            norm_cfg,
            in_channels,
            block_type="conv_module",
            conv_cfg=dict(type="SubMConv3d"),
    ):
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

    @auto_fp16(apply_to=("voxel_features",))
    def encode(self, voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        # N, C, H, W, D -> N,C,D,H,W
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        return spatial_features, encode_features, out

    def forward(self, points, **kwargs):
        lidar_feats, coords, bs_list, inv_idx_list, empty_batch_ids = self.scatter_voxelize(points)
        batch_size = len(points)
        lidar_feat, x_list, x_sparse_out = self.encode(lidar_feats, coords, batch_size)
        if empty_batch_ids:
            lidar_feat[empty_batch_ids] = 0
        return lidar_feat, x_list, x_sparse_out
