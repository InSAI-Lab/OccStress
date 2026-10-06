#!/usr/bin/env python
# encoding: utf-8
import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet3d.ops import Voxelization
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse
from torch.autograd import Variable
from mmdet3d.models import DETECTORS
from ..model_utils.nat import NATBlock

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def lovasz_grad(self,gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def flatten_probas(self,probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)
        elif probas.dim() == 5:
            # 3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        # B, C, H, W = probas.size()
        # probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        valid_idx = valid.nonzero(as_tuple=False).view(-1)
        vprobas = probas[valid_idx]
        vlabels = labels[valid]
        return vprobas, vlabels

    def mean(self,l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def isnan(self,x):
        return x != x

    def lovasz_softmax_flat(self,probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0 or labels.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas.sum() * 0.
        if probas.dim() == 1:
            probas = probas.unsqueeze(0)
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == 'present' and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return self.mean(losses)

    def lovasz_softmax(self,probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if per_image:
            loss = self.mean(
                self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def forward(self, probas, labels):
        return self.lovasz_softmax(probas, labels, ignore=self.ignore)


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape): # （一维）维度 1000*（x+50）/100
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])  #体素尺寸为0.1m 体素网格坐标点
        return idx.long()# 变换为长整型

    def forward(self, data_dict):
        points = data_dict['points'][:, :4]  # xyz
        kept = (points[:, 0] > self.coors_range_xyz[0][0]) & (points[:, 0] < self.coors_range_xyz[0][1]) & \
               (points[:, 1] > self.coors_range_xyz[1][0]) & (points[:, 1] < self.coors_range_xyz[1][1]) & \
               (points[:, 2] > self.coors_range_xyz[2][0]) & (points[:, 2] < self.coors_range_xyz[2][1])
        # kept = (points[:, 0] > self.coors_range_xyz[0][0]) & (points[:, 0] < self.coors_range_xyz[0][1]) & \
        #        (points[:, 1] > self.coors_range_xyz[1][0]) & (points[:, 1] < self.coors_range_xyz[1][1]) & \
        #        (points[:, 2] > -5) & (points[:, 2] < 3)
        data_dict['points'] = points[kept]
        data_dict['labels'] = data_dict['labels'][kept]
        data_dict['batch_idx'] = data_dict['batch_idx'][kept]
        pc = data_dict['points'][:, :3]  # xyzi
        labels = data_dict['labels']

        for idx, scale in enumerate(self.scale_list):  # 2，4，8，16，1    0.2，0.4，0.8.1.6 0.1  坐标变小，坐标尺度变大（坐标刻度增大） 五个坐标系
            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0],
                                        np.ceil(self.spatial_shape[0] / scale))  # 所在索引
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))
            # 第一个维度为第几个batch
            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx, yidx, zidx], dim=-1).long()  # 全部点
            # 挑出不重复的点并排序  以及该数据对应的原始数据的索引 一个体素内重复索引点
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)  #
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            max_labels = torch_scatter.scatter_max(labels, unq_inv, dim=0)[0]
            min_labels = torch_scatter.scatter_min(labels, unq_inv, dim=0)[0]
            mask = (max_labels - min_labels) > 0.1
            max_labels[mask] = -1
            # aaa=torch.sum(mask)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,  # 全部点坐标  TO 体素坐标 体素网格坐标点(重复) N*（B索引+3坐标）
                'coors_inv': unq_inv,  # 每个点对应的体素网格索引（范围为体素网格数量）   体素坐标在点坐标下的索引（有重复）每个点属于哪一个体素坐标
                'coors': unq.type(torch.int32),  # 体素坐标
                'voxel_labels': max_labels  # 体素块内的标签
            }
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        # pc_mean1 = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0) #体素网格*3 # 对索引相同的元素进行操作，得到体素块均值，然后得到每个点的特征
        # p111=pc_mean1[inv_idx] # 体素网格*3
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx] # 维度为0的索引相同的元素进行操作，得到体素块均值，然后得到每个点的特征
        nor_pc = point[:, :3] - pc_mean  #  归一化  在网格内的点减去网格内点的均值

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]  # 100米
        intervals = (crop_range / cur_grid_size).to(point.device) # 0.1，0.1，0.1  体素的尺寸
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)  # 体素实际距离中心
        center_to_point = point[:, :3] - voxel_centers  #去中心化

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1)  # 原始点，归一化点（点减去均值），归一化中心点（点减去体素中心） 10维度特征cat
        return pc_feature

    def forward(self, data_dict):  #以点为基础（N*C）以MLP作为特征提取，体素由点到体素网格的索引得到，均值为初始体素特征并使用稀疏卷积得到特征。
        pt_fea = self.prepare_input(  # 得到体素特征。·原始点，归一化点（点减去均值），归一化中心点（点减去体素中心）特征cat  10维度
            data_dict['points'],   # 全部点
            data_dict['scale_1']['full_coors'][:, 1:],  # 取整的全部点
            data_dict['scale_1']['coors_inv']           # 点到体素的索引
        )
        pt_fea = self.PPmodel(pt_fea)  # 输入初始点云特征    两层linear 升维+不变 点特征

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)  # 特征取平均得到体素特征 体素N*C（初始特征）每个体素均值特征
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(   #输入稠密点云体素，得到稀疏点云体素
            features=features,
            indices=data_dict['scale_1']['coors'].int(),   # 每个batch中对应的体素坐标
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),  # 60，1000，1000
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict


class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
    #离散卷积tensor   体素网格N*C
    def forward(self, x):
        identity = self.layers_in(x)  # 一层1*1*1稀疏卷积+BN
        output = self.layers(x)       # 两层3*3*3稀疏卷积+BN
        #替换卷积核的特征，可以实现不同的卷积核
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1)) #  x.features = F.relu(x.features)


class point_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    @staticmethod  # 体素坐标与体素特诊  坐标除以2，特征在更大的体素网格内取平均  得到尺度为2的特征
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]   #N*1  属于的batch
        coors = coors[:, 1:] // scale #体素网格坐标点除以2,向下取整
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]  # 尺度1的体素网格对应的体素网格2的索引 (重复)
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    # 体素特征作为初始点特征；一次下采样经过几次线性层（并索引回上一层） cat 未采样经过几次线性层 得到当前尺度的点特征  转换到初始点特征，取平均得到下一层的初始特征
    def forward(self, features, data_dict):
        output, inv = self.downsample(data_dict['coors'], features)  # 将特征下采样到尺度为2的体素特征
        identity = self.layer_in(features)  #一层线性层（尺度为1的体素特征）
        output = self.PPmodel(output)[inv]  #三层线性层（尺度为2的体素特征）并索引回第一层特征
        output = torch.cat([identity, output], dim=1)  # 通道cat   两层特征融合
        # out1=output[data_dict['coors_inv']] # 点个数*C
        #点特征在尺度2上进行平均   变成体素特征  尺度2
        v_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict['coors_inv']]),  # 先将体素转换回点的特征，再经过两层线性（MLP） 换成pointMLP
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        )
        #转换到下一个尺度
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']

        return v_feat


class SPVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.p_enc = point_encoder(in_channels, out_channels, scale)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']  # 上一个尺度的点到体素格子索引
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv'] # 现在的尺寸中每个点体素格子索引
        # voxel encoder   体素特征 N*C   网格的索引（存在点的体素索引）  整体网格三维  batch_size
        v_fea = self.v_enc(data_dict['sparse_tensor'])  # 三层卷积   得到尺度为1（0.01m）的体素特征
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']

        indices = v_fea.indices
        feats = v_fea.features
        indices = indices.long().view(-1,4).to(feats.device)
        batch_indices = indices[:, 0].view(-1, 1)
        y_indices = indices[:, 2]
        x_indices = indices[:, 3]
        valid_mask = (x_indices < self.spatial_shape[1]) & (y_indices < self.spatial_shape[2])
        y_indices = y_indices[valid_mask].view(-1, 1)
        x_indices = x_indices[valid_mask].view(-1, 1)
        batch_indices = batch_indices[valid_mask].view(-1, 1)
        feats = feats[valid_mask]

        bxy_indices = torch.cat([batch_indices, x_indices,y_indices], dim=1)

        # 找到所有唯一的 batch、x、y 坐标对，并生成唯一索引
        unique_bxy, inverse_indices = torch.unique(bxy_indices, dim=0,
                                                   return_inverse=True)
        # 构建 BEV 图
        batch_size = batch_indices.max().item() + 1
        bev_shape = (batch_size, self.spatial_shape[1], self.spatial_shape[2], v_fea.features.size(1))  # (B, H, W, C)
        bev_feat = torch.zeros(bev_shape, dtype=feats.dtype, device=feats.device)

        # 计算压缩后的特征，按唯一的 batch, x, y 对聚合 z 轴上的特征
        bev_values = torch_scatter.scatter_add(feats, inverse_indices, dim=0)

        # 填充 BEV 图
        for b in range(batch_size):
            mask = unique_bxy[:, 0] == b
            bev_feat[b,unique_bxy[mask, 1],unique_bxy[mask, 2]] = bev_values[mask]

        data_dict['layer_{}'.format(self.layer_id)]['bev_feat'] = bev_feat.permute(0, 3, 1, 2)
        #将得到的体素特征转换到点特征在下采样尺度取平均
        #vv_feat=v_fea.features[coors_inv_last] #索引回原始点的个数然后在第二尺度网格内取平均
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)  # 第二尺度特征：在下采样（0.02m）网格内取平均

        # point encoder   点第二尺度特征  当前尺度的点特征
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].features+v_fea.features,  # 原始特征+一维体素特征（体素特征）
            data_dict=data_dict
        )

        # fusion and pooling   点特征+体素特征 转换成稀疏张量  下一个尺度的初始体素特征
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv,   # 相加用来融合
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )

        return p_fea[coors_inv]   # 转换成所有点特征


class criterion(nn.Module):
    def __init__(self, lambda_lovasz=1, ignore_index=0,scale_list=[2, 4, 8, 16],hide_size=256,num_classes=17,):
        super(criterion, self).__init__()
        self.lambda_lovasz = lambda_lovasz
        self.scale_list = scale_list
        self.hiden_size = hide_size
        self.num_classes = num_classes
        seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=ignore_index
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=ignore_index
        )
        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(len(self.scale_list)):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )


    def voxelize_labels(self, labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1) # 所有点的标签  第一个为标签，第二个为batch，3，4，5为点的体素坐标
        #返回每个batch中标签相同的体素坐标以及出现次数(每个网格有多少个点)   包含相同坐标而标签不同的体素坐标
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0) #以行为维度  归一化到体素标签（每个网格相同的标签）
        # 不带标签的体素坐标BXYZ
        aaa, inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)  #每个batch中体素坐标以及对应的索引  最终的体素网格坐标，存在标签不同
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]  # 根据索引从count中取最大值     取出相同体素网格内标签对应点最多的体素坐标的索引（网格中相同误分类？）
        labels = unq_lbxyz[:, 0][label_ind]  # 对每个网格加一个索引
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss+self.lambda_lovasz

    def get_volex_loss(self, data_dict,idx):
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)  # 对每个尺度体素网格进行预测 20类
        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'],
                                              data_dict['layer_{}'.format(idx)]['full_coors'])  # 体素网格标签（标签点最多对应的label）
        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full.long())  # 3D交叉熵和lovasz
        return seg_loss_3d


    def forward(self, data_dict):
        # loss_main_ce = self.ce_loss(data_dict['logits'], data_dict['labels'].long())  # 输入的是预测（没有softmax）和标签  交叉熵损失
        # loss_main_lovasz = self.lovasz_loss(F.softmax(data_dict['logits'], dim=1), data_dict['labels'].long()) #输入的是预测标签 标签平滑损失
        # loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz
        # data_dict['loss_main_ce'] = loss_main_ce
        # data_dict['loss_main_lovasz'] = loss_main_lovasz
        # data_dict['loss'] += loss_main
        for idx in range(len(self.scale_list)): #0,1,2,3
            if idx!=2:
                pass
            else:
                singlescale_loss = self.get_volex_loss(data_dict, idx)
                data_dict['loss'] += singlescale_loss

        return data_dict

@DETECTORS.register_module()
class SPVCNN(nn.Module):
    def __init__(self,
                 input_dims=3,
                 spatial_shape=[1000, 1000, 70],
                 scale_list=[2, 4, 8, 16],
                 hide_size=256,
                 num_classes=17,
                 ignore_label=0,
                 max_volume_space=[50, 50, 3],
                 min_volume_space=[-50, -50, -3],
                 lambda_lovasz=1,
                 pretrained=None,):
        super(SPVCNN, self).__init__()
        self.input_dims = input_dims
        self.hiden_size = hide_size
        self.num_classes = num_classes
        self.scale_list = scale_list
        self.ignore_label = ignore_label
        self.lambda_lovasz = lambda_lovasz
        self.num_scales = len(self.scale_list)
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(spatial_shape)
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):  # 4
            self.spv_enc.append(SPVBlock(
                in_channels=self.hiden_size,  # 64
                out_channels=self.hiden_size,  # 64
                indice_key='spv_' + str(i),
                scale=self.scale_list[i],  # 2，4，8，16
                last_scale=self.scale_list[i - 1] if i > 0 else 1,  # 上一个尺度
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist())
            )

        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )
        # loss
        self.criterion = criterion(lambda_lovasz=1, ignore_index=0,scale_list=[2, 4, 8, 16],hide_size=256,num_classes=17)
        self.bev_align=NATBlock(dim=256, depth=3, depth_cross=1, num_heads=8, kernel_size=7, downsample=False)
        if pretrained:
            self.load_pretrained(pretrained)



    def load_pretrained(self, pretrained_path):
        # Load the state_dict from the given pretrained model path
        state_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()

        for k, v in state_dict.items():
            if k in model_dict and len(v.shape) == 5 and v.nelement() > 0:
                # 检查是否需要交换第二维度和最后一维度
                if v.shape[1] == model_dict[k].shape[-1] and v.shape[-1] == model_dict[k].shape[1]:
                    v = v.permute(0, 4, 2, 3, 1)
                    # print(f"Permuted layer {k}: new shape {v.shape}")
                model_dict[k].copy_(v)
            elif k in model_dict:
                model_dict[k].copy_(v)
            # else:
                # print(f"Skipping {k} as it is not in the model.")
        print(f"Loaded pretrained model weights from {pretrained_path}")

    def log_normalize(self, image):
        normalized_image = np.log1p(image)  # log1p 等同于 log(x + 1)，避免了 log(0) 的问题
        normalized_image -= normalized_image.min()
        normalized_image /= normalized_image.max()
        return normalized_image

    def forward(self, data_dict):  # 体素化，特征提取，融合，分类
        # data_dict中点，以及对应的batch索引，标签，batch_size，标签，第一个数据的原始标签和长度，原始点云到体素点云的索引
        with torch.no_grad():  # 不需要梯度
            data_dict = self.voxelizer(data_dict)  # 体素化

        data_dict = self.voxel_3d_generator(data_dict)  # 得到初始点云特征，体素均值特征转换成初始稀疏张量  第一尺度

        enc_feats = []  # 4个尺度的点特征
        for i in range(self.num_scales):  # 4个体素转换特征转换成点特征 尺寸，  编码器
            enc_feats.append(self.spv_enc[i](data_dict))  # 得到的都是point数量N的向量
        # # for idx in range(self.num_scales):#0.1,0.2,0.4,0.8,1.6,1.6
        # #     pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        # #     N, C = pts_feat.shape
        # #     bev_maps = torch.zeros(800//2**idx, 800//2**idx, C,
        # #                            device=pts_feat.device)
        # #     voxel_xyz= data_dict['scale_{}'.format(2**idx)]['coors'][:,1:]
        # #     for i in range(voxel_xyz.shape[0]):
        # #         z, y, x = voxel_xyz[i,:]
        # #         feature = pts_feat[i,:]
        # #         bev_maps[x, y] += feature
        # for idx in range(self.num_scales):
        #     bev_maps = data_dict['layer_{}'.format(idx)]['bev_feat']
        #     bev_feat=bev_maps.abs().sum(dim=1).squeeze().cpu().detach().numpy()
        #     bev_norm_log1 = ((self.log_normalize(bev_feat)) * 255.0).astype(np.uint8)
        #     import matplotlib.pyplot as plt
        #     bev_map=bev_norm_log1[0]
        #     import cv2
        #     cv2.imshow('bev',bev_map)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     # plt.axis('off')
        #     # plt.savefig('/media/aiboy/DeepLearn/oacnn_nuscenes/bev_{}.png'.format(idx))
        #     # plt.show()
        #     print('end')

        # 简单通道cat
        output = torch.cat(enc_feats, dim=1)  # cat 点特征
        data_dict['logits'] = self.classifier(output)  # 分类器  每个点分配了标签
        data_dict['loss'] = 0.  # 交叉熵和lovasz损失
        data_dict = self.criterion(data_dict)

        return data_dict


if __name__ == '__main__':
    import yaml
    # config_file='/media/aiboy/DeepLearn/oacnn_nuscenes/config/nuscenes.yaml'
    # with open(config_file, 'r') as file:
    #     config = yaml.safe_load(file)
    config = {
        'model_params': {
            'input_dims': 3,
            'spatial_shape': [1000, 1000, 70],
            'scale_list': [2, 4, 8, 16],
            'hiden_size': 256,
            'num_classes': 17,
        },
        'dataset_params': {
            'ignore_label': 0,
            'max_volume_space': [50, 50, 3],
            'min_volume_space': [-50, -50, -4],
        },
        'train_params': {
            'lambda_seg2d': 1,
            'lambda_xm': 0.05,
            'lambda_lovasz': 1
        }
    }
    # Set device to CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create data dictionary
    data_dict = {
        'points': torch.rand((10000, 3)).to(device),  # 10000 points with 4 features each
        'labels': torch.randint(0, 17, (10000,)).to(device),  # 10000 labels
        'batch_idx': torch.zeros((10000,)).long().to(device),  # Batch index for each point
        'batch_size': 1  # Assuming a single batch
    }
    config={"max_num_points":10,
            "point_clou_range":[-40,-40,-1,40,40,5.4],
            "voxel_size":[0.1,0.1,0.1],
            "max_voxels":(90000,120000)}
    voxel=Voxelization(max_num_points=10,point_cloud_range=[-40,-40,-1,40,40,5.4],voxel_size=[0.1,0.1,0.1],max_voxels=(90000,120000))
    model = SPVCNN(input_dims=4,
                 spatial_shape=[800, 800, 64],
                 scale_list=[2, 4, 8, 16],
                 hide_size=256,
                 num_classes=17,
                 ignore_label=0,
                 max_volume_space=[40, 40, 5.4],
                 min_volume_space=[-40, -40, -1],
                 lambda_lovasz=1,
                 pretrained="/media/wj-hust/deeplearning/dzp/SADA_OCC_2/pretrain/spvcnn_occ.pth",).to(device)
    check1=model.state_dict()
    output_dict = model(data_dict)
    print(output_dict['loss'])
    print('end')
