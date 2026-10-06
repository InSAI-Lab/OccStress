# Modified from OccWorld
import os.path

import cv2
import copy
import time

import mmcv
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

from mmdet3d.models.losses.lovasz_softmax import lovasz_softmax
from mmdet3d.models.utils.vis_util import change_occupancy_to_bev
from mmcv.parallel import DataContainer
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@DETECTORS.register_module()
class IISceneTokenizer(CenterPoint):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 vq=None,
                 num_classes=18,
                 class_embeds_dim=8,
                 frame_number=2,
                 vq_channel=None,
                 # loss_cfg
                 grid_config=None,
                 empty_idx=None,
                 use_class_weights=False,
                 class_weights=None,
                 lovasz_loss=None,
                 focal_loss=None,
                 embed_loss_weight=None,
                 save_results=True,
                 results_type='occ3d',
                 save_root_override=None,
                 eval_aligned_history=False,
                 save_aligned_history_vis=False,
                 aligned_history_vis_dir='aligned_history_eval',
                 aligned_history_vis_interval=1,
                 aligned_history_vis_max_samples=20,
                 disable_history_at_train=False,
                 disable_history_at_test=False,
                 compare_history_at_test=False,
                 vote_mode_at_test='none',
                 vote_static_class_ids=None,
                 vote_dynamic_class_ids=None,
                 vote_history_min_agree=2,
                 vote_neighborhood_radius=1,
                 export_only=False,
                 return_aligned_history_metrics=False,
                 save_aligned_history_npz=False,
                 fusion_supervision_weight=0.0,
                 fusion_ranking_weight=0.0,
                 fusion_ranking_margin=0.5,
                 disagreement_gain_weight=0.0,
                 disagreement_stability_weight=0.0,
                 disagreement_gain_margin=0.05,
                 disagreement_stability_margin=0.0,
                 **kwargs):
        super(IISceneTokenizer, self).__init__(**kwargs)
        # ---------------------- init params ------------------------------
        self.num_classes = num_classes
        self.class_embeds_dim = class_embeds_dim

        # ---------------------- init Model ------------------------------
        start_time = time.time()
        self.encoder = builder.build_backbone(encoder)
        self.vq = builder.build_backbone(vq)
        end_time = time.time()
        self.decoder = builder.build_backbone(decoder)
        # Time module
        self.history_bev = None
        self.history_occ = None
        self.bev_aug = None
        self.frame_number = frame_number
        self.vq_channel = vq_channel
        x_config, y_config, z_config = grid_config['x'], grid_config['y'], grid_config['z']
        dx, bx, nx = gen_dx_bx(x_config, y_config, z_config)
        self.dx, self.bx, self.nx = dx, bx, nx
        # Embedding
        self.class_embeds = nn.Embedding(num_classes, class_embeds_dim)
        # Others
        self.save_results = save_results
        if self.save_results:
            if save_root_override is not None:
                self.save_root = save_root_override
                mmcv.mkdir_or_exist(self.save_root)
            elif results_type == 'occ3d':
                self.save_root = 'data/nuscenes/save_dir'
                mmcv.mkdir_or_exist('data/nuscenes/save_dir')
            elif results_type == 'waymo':
                self.save_root = 'data/waymo/save_dir'
                mmcv.mkdir_or_exist('data/waymo/save_dir')
            elif results_type == 'stcocc':
                self.save_root = 'data/nuscenes/save_dir_stc'
                mmcv.mkdir_or_exist('data/nuscenes/save_dir_stc')
        self.results_type = results_type
        self.eval_aligned_history = eval_aligned_history
        self.save_aligned_history_vis = save_aligned_history_vis
        self.aligned_history_vis_dir = aligned_history_vis_dir
        self.aligned_history_vis_interval = max(1, aligned_history_vis_interval)
        self.aligned_history_vis_max_samples = max(0, aligned_history_vis_max_samples)
        self.disable_history_at_train = disable_history_at_train
        self.disable_history_at_test = disable_history_at_test
        self.compare_history_at_test = compare_history_at_test
        self.vote_mode_at_test = vote_mode_at_test
        self.vote_static_class_ids = vote_static_class_ids or []
        self.vote_dynamic_class_ids = vote_dynamic_class_ids or []
        self.vote_history_min_agree = vote_history_min_agree
        self.vote_neighborhood_radius = vote_neighborhood_radius
        self.export_only = export_only
        self.return_aligned_history_metrics = return_aligned_history_metrics
        self.save_aligned_history_npz = save_aligned_history_npz
        self.aligned_history_vis_counter = 0
        self.fusion_supervision_weight = fusion_supervision_weight
        self.fusion_ranking_weight = fusion_ranking_weight
        self.fusion_ranking_margin = fusion_ranking_margin
        self.disagreement_gain_weight = disagreement_gain_weight
        self.disagreement_stability_weight = disagreement_stability_weight
        self.disagreement_gain_margin = disagreement_gain_margin
        self.disagreement_stability_margin = disagreement_stability_margin
        # Losses
        self.empty_idx = empty_idx
        self.use_class_weights = use_class_weights
        class_weight_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_weights = torch.tensor(
            np.array(class_weights),
            dtype=torch.float32,
            device=class_weight_device,
        )
        self.focal_loss = builder.build_loss(focal_loss)
        self.embed_loss_weight = embed_loss_weight
        self.history_reliability = None


    def _get_aligned_history_vis_root(self):
        vis_dir = self.aligned_history_vis_dir
        if self.compare_history_at_test:
            vis_dir = f'{vis_dir}_compare'
        if hasattr(self, 'save_root'):
            return os.path.join(self.save_root, vis_dir)
        return vis_dir

    def _make_occ_panel(self, occ_semantics, title):
        panel = change_occupancy_to_bev(
            occ_semantics, occ_size=occ_semantics.shape, free_cls=self.empty_idx)
        panel = cv2.resize(panel, (480, 480), interpolation=cv2.INTER_NEAREST)
        title_bar_h = 44
        panel_h, panel_w = panel.shape[:2]
        titled_panel = np.full((panel_h + title_bar_h, panel_w, 3), 255, dtype=np.uint8)
        titled_panel[title_bar_h:, :, :] = panel
        cv2.line(titled_panel, (0, title_bar_h - 1), (panel_w, title_bar_h - 1), (180, 180, 180), 1)
        cv2.putText(
            titled_panel,
            title,
            (8, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        return titled_panel

    def _make_text_panel(self, panel_shape, title, text_lines):
        panel_h, panel_w = panel_shape[:2]
        panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
        cv2.putText(panel, title, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.line(panel, (0, 56), (panel_w, 56), (180, 180, 180), 1)
        for line_idx, text in enumerate(text_lines):
            cv2.putText(
                panel,
                text,
                (16, 110 + line_idx * 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 20, 20),
                2,
            )
        return panel

    def _save_aligned_history_visualization(self,
                                            curr_gt_occ,
                                            curr_pred_occ,
                                            curr_pred_no_history_occ,
                                            history_gt_occ,
                                            history_raw_pred_occ,
                                            history_align_pred_occ,
                                            img_meta):
        vis_root = self._get_aligned_history_vis_root()
        scene_name = str(img_meta['scene_name']).zfill(3) if self.results_type == 'waymo' else str(img_meta['scene_name'])
        scene_root = os.path.join(vis_root, scene_name)
        mmcv.mkdir_or_exist(scene_root)
        sample_id = str(img_meta['sample_idx'])
        num_history = history_gt_occ.shape[0]
        reference_panel = self._make_occ_panel(curr_gt_occ, 't gt')
        page_slots = 2

        for page_idx, hist_start in enumerate(range(0, num_history, page_slots), start=1):
            page_hist_indices = list(range(hist_start, min(hist_start + page_slots, num_history)))
            panels = [
                reference_panel,
                self._make_occ_panel(curr_pred_occ, 't predict w/ history'),
                self._make_occ_panel(curr_pred_no_history_occ, 't predict no history'),
            ]

            for hist_idx in page_hist_indices:
                panels.extend([
                    self._make_occ_panel(history_gt_occ[hist_idx], f't-{hist_idx + 1} gt'),
                    self._make_occ_panel(history_raw_pred_occ[hist_idx], f't-{hist_idx + 1} raw predict'),
                    self._make_occ_panel(history_align_pred_occ[hist_idx], f't-{hist_idx + 1} align predict'),
                ])

            while len(panels) < 9:
                panels.append(self._make_text_panel(reference_panel.shape, 'Empty', ['']))

            row_images = [np.concatenate(panels[row_idx * 3:(row_idx + 1) * 3], axis=1) for row_idx in range(3)]
            canvas = np.concatenate(row_images, axis=0)

            if self.save_aligned_history_npz:
                np.savez(
                    os.path.join(scene_root, f'{sample_id}_page_{page_idx}.npz'),
                    curr_gt=curr_gt_occ.astype(np.uint8),
                    curr_pred_with_history=curr_pred_occ.astype(np.uint8),
                    curr_pred_no_history=curr_pred_no_history_occ.astype(np.uint8),
                    history_gt=history_gt_occ[page_hist_indices].astype(np.uint8),
                    history_raw_pred=history_raw_pred_occ[page_hist_indices].astype(np.uint8),
                    history_align_pred=history_align_pred_occ[page_hist_indices].astype(np.uint8),
                )
            cv2.imwrite(
                os.path.join(scene_root, f'{sample_id}_page_{page_idx}.png'),
                canvas,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )

    def _quantize_latent_for_decode(self, latent):
        temporal_context = latent.new_zeros((
            latent.shape[0],
            self.vq.recover_time,
            latent.shape[1],
            latent.shape[2],
            latent.shape[3],
        ))
        latent_q, _, _ = self.vq(latent, temporal_context, is_voxel=False)
        return latent_q

    def _compute_semantic_hist(self, pred, target):
        pred = pred.reshape(-1).astype(np.int64)
        target = target.reshape(-1).astype(np.int64)
        valid_mask = np.logical_and(target >= 0, target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[valid_mask] + pred[valid_mask],
            minlength=self.num_classes ** 2,
        )
        return hist.reshape(self.num_classes, self.num_classes).astype(np.int64)

    def _prepare_fusion_targets(self, pred_tensor, current_reliability, sampled_reliability):
        if current_reliability.dim() == 4:
            current_reliability = current_reliability[:, -1]
        current_reliability = current_reliability.to(pred_tensor.dtype)
        sampled_reliability = sampled_reliability.to(pred_tensor.dtype)
        if current_reliability.shape[-2:] != sampled_reliability.shape[-2:]:
            current_reliability = F.interpolate(
                current_reliability.unsqueeze(1),
                size=sampled_reliability.shape[-2:],
                mode='nearest',
            ).squeeze(1)

        target_clean = torch.cat([current_reliability.unsqueeze(1), sampled_reliability], dim=1)
        if target_clean.shape[-2:] != pred_tensor.shape[-2:]:
            target_clean = F.interpolate(target_clean, size=pred_tensor.shape[-2:], mode='nearest')
        return target_clean

    def fusion_supervision_loss(self, fusion_weights, current_reliability, sampled_reliability):
        pred_weights = fusion_weights.squeeze(2)
        target_clean = self._prepare_fusion_targets(pred_weights, current_reliability, sampled_reliability)

        supervised_mask = target_clean.min(dim=1).values < 0.999
        if not supervised_mask.any():
            return pred_weights.sum() * 0.0

        recency_logits = self.vq.recency_logits.view(1, -1, 1, 1).to(pred_weights.dtype).to(pred_weights.device)
        recency_prior = torch.exp(recency_logits).expand_as(pred_weights)
        target_scores = recency_prior * target_clean.clamp(min=0.0, max=1.0)
        target_den = target_scores.sum(dim=1, keepdim=True)
        fallback = recency_prior / recency_prior.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target_dist = torch.where(
            target_den > 1e-6,
            target_scores / target_den.clamp_min(1e-6),
            fallback,
        )

        per_cell_loss = -(target_dist * torch.log(pred_weights.clamp_min(1e-6))).sum(dim=1)
        supervised_mask = supervised_mask.to(per_cell_loss.dtype)
        return (per_cell_loss * supervised_mask).sum() / supervised_mask.sum().clamp_min(1.0)

    def fusion_ranking_loss(self, score_logits, current_reliability, sampled_reliability):
        pred_scores = score_logits.squeeze(2)
        target_clean = self._prepare_fusion_targets(pred_scores, current_reliability, sampled_reliability)

        recency_logits = self.vq.recency_logits.view(1, -1, 1, 1).to(pred_scores.dtype).to(pred_scores.device)
        total_scores = pred_scores + recency_logits

        clean_mask = target_clean > 0.5
        corrupt_mask = target_clean < 0.5
        supervised_mask = clean_mask.any(dim=1) & corrupt_mask.any(dim=1)
        if not supervised_mask.any():
            return total_scores.sum() * 0.0

        neg_inf = torch.finfo(total_scores.dtype).min
        best_clean = total_scores.masked_fill(~clean_mask, neg_inf).max(dim=1).values
        best_corrupt = total_scores.masked_fill(~corrupt_mask, neg_inf).max(dim=1).values
        margin_gap = best_clean - best_corrupt
        per_cell_loss = F.relu(self.fusion_ranking_margin - margin_gap)
        supervised_mask = supervised_mask.to(per_cell_loss.dtype)
        return (per_cell_loss * supervised_mask).sum() / supervised_mask.sum().clamp_min(1.0)

    def normalize_frame_reliability_map(self, frame_reliability_map, batch_size):
        while frame_reliability_map.dim() > 4 and frame_reliability_map.shape[0] == 1:
            frame_reliability_map = frame_reliability_map.squeeze(0)
        if frame_reliability_map.dim() == 3:
            frame_reliability_map = frame_reliability_map.unsqueeze(1)
        elif frame_reliability_map.dim() == 4 and frame_reliability_map.shape[0] != batch_size and frame_reliability_map.shape[1] == batch_size:
            frame_reliability_map = frame_reliability_map.transpose(0, 1)
        return frame_reliability_map

    def normalize_batched_tensor(self, tensor, batch_size):
        while tensor.dim() >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == batch_size:
            tensor = tensor.squeeze(0)
        return tensor

    def get_current_target(self, target_semantics):
        if target_semantics.dim() == 5:
            if target_semantics.shape[1] == 1:
                return target_semantics[:, 0]
            return target_semantics[:, -1]
        return target_semantics

    def resize_spatial_mask(self, mask, target_hw):
        while mask.dim() > 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.shape[-2:] == target_hw:
            return mask
        if mask.shape[-2:] == (target_hw[1], target_hw[0]):
            return mask.transpose(-2, -1)
        return F.interpolate(mask.unsqueeze(1), size=target_hw, mode='nearest').squeeze(1)

    def current_frame_ce_map(self, logits, target_current):
        logits_current = logits[:, 0].permute(0, 4, 1, 2, 3)
        return F.cross_entropy(
            logits_current,
            target_current.long(),
            ignore_index=255,
            reduction='none',
        )

    def class_mask(self, pred, class_ids):
        if not class_ids:
            return torch.zeros_like(pred, dtype=torch.bool)
        mask = pred == int(class_ids[0])
        for cls_id in class_ids[1:]:
            mask = mask | (pred == int(cls_id))
        return mask

    def static_majority_vote(self, current_pred, history_pred):
        vote_sources = torch.cat([current_pred.unsqueeze(1), history_pred], dim=1)
        best_count = torch.zeros_like(current_pred, dtype=torch.int16)
        best_class = current_pred.clone()
        for cls_id in self.vote_static_class_ids:
            cls_count = (vote_sources == int(cls_id)).sum(dim=1).to(best_count.dtype)
            update_mask = cls_count > best_count
            best_count = torch.where(update_mask, cls_count, best_count)
            best_class = torch.where(update_mask, torch.full_like(best_class, int(cls_id)), best_class)
        return best_class, best_count

    def dynamic_history_support(self, history_pred, class_id):
        radius = max(0, int(self.vote_neighborhood_radius))
        if radius == 0:
            return (history_pred == int(class_id)).sum(dim=1)
        hist_mask = (history_pred == int(class_id)).permute(0, 1, 4, 2, 3).reshape(
            -1, 1, history_pred.shape[2], history_pred.shape[3]
        )
        pooled = F.max_pool2d(
            hist_mask.float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ) > 0
        pooled = pooled.reshape(
            history_pred.shape[0], history_pred.shape[1], history_pred.shape[4], history_pred.shape[2], history_pred.shape[3]
        ).permute(0, 1, 3, 4, 2)
        return pooled.sum(dim=1)

    def vote_predictions_at_test(self, current_pred, history_pred):
        if self.vote_mode_at_test == 'none' or history_pred is None:
            return current_pred

        result = current_pred.clone()
        static_vote, static_count = self.static_majority_vote(current_pred, history_pred)
        current_dynamic_mask = self.class_mask(current_pred, self.vote_dynamic_class_ids)

        static_override_mask = (~current_dynamic_mask) & (static_count >= 2)
        result = torch.where(static_override_mask, static_vote, result)

        if self.vote_mode_at_test != 'dynamic_gated':
            return result

        best_dyn_support = torch.zeros_like(current_pred, dtype=torch.int16)
        best_dyn_class = torch.zeros_like(current_pred)
        for cls_id in self.vote_dynamic_class_ids:
            support = self.dynamic_history_support(history_pred, int(cls_id)).to(best_dyn_support.dtype)
            update_mask = support > best_dyn_support
            best_dyn_support = torch.where(update_mask, support, best_dyn_support)
            best_dyn_class = torch.where(update_mask, torch.full_like(best_dyn_class, int(cls_id)), best_dyn_class)

        allow_dynamic_override = (
            (~current_dynamic_mask)
            & (best_dyn_support >= int(self.vote_history_min_agree))
            & (static_count <= 1)
        )
        result = torch.where(allow_dynamic_override, best_dyn_class, result)
        return result

    def disagreement_gain_loss(self, logits_with_history, logits_no_history, target_semantics, current_reliability, sampled_reliability):
        target_current = self.get_current_target(target_semantics).to(logits_with_history.device)
        ce_with = self.current_frame_ce_map(logits_with_history, target_current)
        ce_no = self.current_frame_ce_map(logits_no_history, target_current)

        if current_reliability.dim() == 4:
            current_reliability = current_reliability[:, -1]
        current_reliability = self.resize_spatial_mask(
            current_reliability.to(ce_with.dtype),
            target_current.shape[1:3],
        )

        sampled_reliability = sampled_reliability.to(ce_with.dtype)
        if sampled_reliability.shape[-2:] != target_current.shape[1:3]:
            sampled_reliability = F.interpolate(
                sampled_reliability,
                size=target_current.shape[1:3],
                mode='nearest',
            )

        current_bad = current_reliability < 0.5
        current_good = current_reliability > 0.5
        history_bad = sampled_reliability.min(dim=1).values < 0.5
        history_good = sampled_reliability.max(dim=1).values > 0.5
        valid_mask = target_current != 255

        loss_dict = dict()
        if self.disagreement_gain_weight > 0:
            gain_mask = current_bad & history_good
            gain_mask_3d = gain_mask.unsqueeze(-1) & valid_mask
            if gain_mask_3d.any():
                gain_map = F.relu(ce_with - ce_no + self.disagreement_gain_margin)
                gain_mask_float = gain_mask_3d.to(gain_map.dtype)
                loss_dict['disagreement_gain_loss'] = self.disagreement_gain_weight * (
                    (gain_map * gain_mask_float).sum() / gain_mask_float.sum().clamp_min(1.0)
                )
            else:
                loss_dict['disagreement_gain_loss'] = ce_with.sum() * 0.0

        if self.disagreement_stability_weight > 0:
            stability_mask = current_good & history_bad
            stability_mask_3d = stability_mask.unsqueeze(-1) & valid_mask
            if stability_mask_3d.any():
                stability_map = F.relu(ce_with - ce_no + self.disagreement_stability_margin)
                stability_mask_float = stability_mask_3d.to(stability_map.dtype)
                loss_dict['disagreement_stability_loss'] = self.disagreement_stability_weight * (
                    (stability_map * stability_mask_float).sum() / stability_mask_float.sum().clamp_min(1.0)
                )
            else:
                loss_dict['disagreement_stability_loss'] = ce_with.sum() * 0.0
        return loss_dict

    def reconstruct_loss(self, pred, targ):
        # pred: [bs, T, W, H, Z, C]
        # targ: [bs, T, W, H, Z]
        # Change pred to [bs*T, c, w, h, z]
        bs, T, W, H, Z, C = pred.shape
        pred = pred.reshape(bs*T, W, H, Z, C).permute(0, 4, 1, 2, 3)
        targ = targ.reshape(bs*T, W, H, Z)

        loss_dict = dict()
        loss_reconstruct = self.focal_loss(pred, targ,
                                           None if not self.use_class_weights else self.class_weights,
                                           ignore_index=255)  # self-reconstruction loss
        loss_lovasz = lovasz_softmax(torch.softmax(pred, dim=1), targ, ignore=255)

        loss_dict['recon_loss'] = loss_reconstruct + loss_lovasz

        return loss_dict

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev

    def align_bev(self,
                  curr_bev,
                  img_metas,
                  curr_occ=None,
                  cache_bev=None,
                  cache_occ=None,
                  cache_reliability=None):
        # z_sampled: [bs, c, w, h]
        curr_bev = curr_bev.permute(0, 1, 3, 2).unsqueeze(2)   # change to [bs, c, h, w], z=1
        if cache_bev is None:
            cache_bev = curr_bev
        else:
            cache_bev = cache_bev.permute(0, 1, 3, 2).unsqueeze(2)
        bs, c, z, h, w = curr_bev.shape
        # prepare
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        start_mask = torch.as_tensor(start_of_sequence, device=curr_bev.device, dtype=torch.bool)
        curr_to_prev_ego_rt = torch.stack([torch.as_tensor(img_meta['curr_to_prev_ego_rt'], device=curr_bev.device) for img_meta in img_metas])
        bev_aug = torch.stack([img_meta['bda_mat'].to(curr_bev.device) for img_meta in img_metas])

        if self.history_bev is None:
            self.history_bev = cache_bev.repeat(1, self.frame_number, 1, 1, 1).clone()
            self.bev_aug = bev_aug.clone()

        if start_mask.any():
            self.history_bev[start_mask] = cache_bev[start_mask].repeat(1, self.frame_number, 1, 1, 1)
            self.bev_aug[start_mask] = bev_aug[start_mask]

        self.history_bev = self.history_bev.detach()
        tmp_bev = self.history_bev

        history_occ = None
        raw_history_bev = None
        sampled_reliability = None
        if self.eval_aligned_history and cache_occ is not None:
            if self.history_occ is None:
                self.history_occ = cache_occ.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1).clone()
            if start_mask.any():
                self.history_occ[start_mask] = cache_occ[start_mask].unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1)
            self.history_occ = self.history_occ.detach()
            history_occ = self.history_occ.clone()
            raw_history_bev = tmp_bev.reshape(bs, self.frame_number, c, z, h, w).permute(0, 1, 2, 3, 5, 4).squeeze(3).clone()

        tmp_reliability = None
        if cache_reliability is not None:
            batch_size = start_mask.shape[0]
            while cache_reliability.dim() > 3 and cache_reliability.shape[0] == 1:
                cache_reliability = cache_reliability.squeeze(0)
            if cache_reliability.dim() == 4 and cache_reliability.shape[0] != batch_size and cache_reliability.shape[1] == batch_size:
                cache_reliability = cache_reliability.transpose(0, 1)
            if cache_reliability.dim() == 4 and cache_reliability.shape[1] == 1:
                cache_reliability = cache_reliability.squeeze(1)
            if cache_reliability.dim() != 3 or cache_reliability.shape[0] != batch_size:
                cache_reliability = cache_reliability.reshape(batch_size, cache_reliability.shape[-2], cache_reliability.shape[-1])
            if cache_reliability.shape[-2:] != (h, w):
                cache_reliability = F.interpolate(
                    cache_reliability.unsqueeze(1).to(curr_bev.dtype),
                    size=(h, w),
                    mode='nearest',
                ).squeeze(1)
            if self.history_reliability is None:
                self.history_reliability = cache_reliability.unsqueeze(1).repeat(1, self.frame_number, 1, 1).clone()
            if start_mask.any():
                self.history_reliability[start_mask] = cache_reliability[start_mask].unsqueeze(1).repeat(1, self.frame_number, 1, 1)
            self.history_reliability = self.history_reliability.detach()
            tmp_reliability = self.history_reliability.unsqueeze(2)

        # align different time step
        grid = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid, self.dx, self.bx)

        rt_flow = (torch.inverse(feat2bev) @ self.bev_aug @ curr_to_prev_ego_rt @ torch.inverse(bev_aug) @ feat2bev)
        grid = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid

        normalize_factor = torch.tensor([w - 1.0, h - 1.0, 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # grid order is x, y, z
        grid[..., 2] = 0.0

        sampled_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode='bilinear')
        if tmp_reliability is not None:
            sampled_reliability = F.grid_sample(
                tmp_reliability.to(curr_bev.dtype),
                grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
                align_corners=True,
                mode='nearest',
            ).squeeze(2)

        bev_cat = torch.cat([cache_bev, sampled_bev], dim=1)
        self.history_bev = bev_cat[:, :-self.vq_channel, ...].detach().clone()
        if history_occ is not None:
            self.history_occ = torch.cat([cache_occ.unsqueeze(1), history_occ[:, :-1]], dim=1).detach().clone()
        if sampled_reliability is not None:
            self.history_reliability = torch.cat(
                [cache_reliability.unsqueeze(1), sampled_reliability[:, :-1]], dim=1).detach().clone()

        sampled_bev = sampled_bev.reshape(bs, self.frame_number, c, z, h, w).permute(0, 1, 2, 3, 5, 4).squeeze(3)  # change to w, h and squeeze z-axis

        return sampled_bev.clone(), history_occ, raw_history_bev, sampled_reliability.clone() if sampled_reliability is not None else None

    def forward_encoder(self, voxel_semantics):
        # voxel_semantics: [bs, T, W, H, Z]
        BS, T, H, W, Z = voxel_semantics.shape
        voxel_semantics = self.class_embeds(voxel_semantics)
        voxel_semantics = voxel_semantics.reshape(BS*T, H, W, Z * self.class_embeds_dim).permute(0, 3, 1, 2)
        z, shapes = self.encoder(voxel_semantics)
        return z, shapes

    def forward_decoder(self, z, shapes, input_shape):
        # z:[bs, C, H, W], input_shape: original shape of voxel_semantics
        logits = self.decoder(z, list(shapes))

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.class_embeds_dim)
        template = self.class_embeds.weight.T.unsqueeze(0)  # 1, expansion, cls
        similarity = torch.matmul(logits, template)  # -1, D, cls

        return similarity.reshape(bs, F, H, W, D, self.num_classes)

    def forward_test(self, voxel_semantics, img_metas, **kwargs):
                
        # --- MMCV compatibility: unwrap DataContainer ---
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data

        # batch_size = 1 时，mmcv 会再包一层 list
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        # ------------------------------------------------
        # 0. Prepare Input
        batch_size = len(img_metas)
        voxel_semantics = self.normalize_batched_tensor(voxel_semantics, batch_size)
        if voxel_semantics.dim() == 4:
            voxel_semantics = voxel_semantics.unsqueeze(1)
        model_device = self.class_embeds.weight.device
        voxel_semantics = voxel_semantics.to(model_device, non_blocking=True)
        bs, t, w, h, d = voxel_semantics.shape
        curr_occ = voxel_semantics[:, -1]
        voxel_semantics_clean = kwargs.get('voxel_semantics_clean', None)
        if voxel_semantics_clean is not None:
            voxel_semantics_clean = self.normalize_batched_tensor(voxel_semantics_clean, batch_size)
            if voxel_semantics_clean.dim() == 4:
                voxel_semantics_clean = voxel_semantics_clean.unsqueeze(1)
            voxel_semantics_clean = voxel_semantics_clean.to(model_device, non_blocking=True)
        curr_occ_clean = voxel_semantics_clean[:, -1] if voxel_semantics_clean is not None else curr_occ

        # 2. Process current voxel semantics
        start_time = time.time()
        curr_bev, shapes = self.forward_encoder(voxel_semantics)
        clean_curr_bev = None
        if voxel_semantics_clean is not None:
            clean_curr_bev, _ = self.forward_encoder(voxel_semantics_clean)

        # 3. Time fusion
        need_aligned_history = self.eval_aligned_history or self.vote_mode_at_test != 'none'
        need_compare = self.compare_history_at_test or self.vote_mode_at_test != 'none'

        sampled_bev, history_occ, raw_history_bev, _ = self.align_bev(
            curr_bev,
            img_metas,
            curr_occ=curr_occ if need_aligned_history else None,
            cache_bev=clean_curr_bev if clean_curr_bev is not None else curr_bev,
            cache_occ=curr_occ_clean if need_aligned_history else None,
        )

        # 4. vq
        zero_sampled_bev = torch.zeros_like(sampled_bev)
        if need_compare:
            z_sampled_with_history, loss, info = self.vq(curr_bev, sampled_bev, is_voxel=False)
            z_sampled_no_history, _, _ = self.vq(curr_bev, zero_sampled_bev, is_voxel=False)
            z_sampled = z_sampled_no_history if self.disable_history_at_test else z_sampled_with_history
        else:
            sampled_bev_for_model = zero_sampled_bev if self.disable_history_at_test else sampled_bev
            z_sampled, loss, info = self.vq(curr_bev, sampled_bev_for_model, is_voxel=False)
        end_time = time.time()

        # 5. Process Decoder
        logits = self.forward_decoder(z_sampled, shapes, (bs, 1, w, h, d))
        if need_compare:
            logits_with_history = self.forward_decoder(z_sampled_with_history, shapes, (bs, 1, w, h, d))
            logits_no_history = self.forward_decoder(z_sampled_no_history, shapes, (bs, 1, w, h, d))

        # 6. Preprocess logits
        output_dict = dict()
        pred = logits.softmax(-1).argmax(-1).cpu().numpy()
        pred_with_history = logits_with_history.softmax(-1).argmax(-1).cpu().numpy() if need_compare else pred
        pred_no_history = logits_no_history.softmax(-1).argmax(-1).cpu().numpy() if need_compare else None
        if need_compare:
            curr_occ_clean_np = curr_occ_clean.cpu().numpy().astype(np.uint8)
            current_with_history_hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            current_no_history_hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            for batch_idx in range(bs):
                current_with_history_hist += self._compute_semantic_hist(
                    pred_with_history[batch_idx, 0], curr_occ_clean_np[batch_idx])
                current_no_history_hist += self._compute_semantic_hist(
                    pred_no_history[batch_idx, 0], curr_occ_clean_np[batch_idx])
            output_dict['current_with_history_hist'] = current_with_history_hist
            output_dict['current_no_history_hist'] = current_no_history_hist
        voted_pred = None
        if need_aligned_history and history_occ is not None:
            history_occ_np = history_occ.cpu().numpy().astype(np.uint8)
            num_history = sampled_bev.shape[1]
            sampled_bev_flat = sampled_bev.reshape(bs * num_history, sampled_bev.shape[2], sampled_bev.shape[3], sampled_bev.shape[4])
            raw_history_bev_flat = raw_history_bev.reshape(bs * num_history, raw_history_bev.shape[2], raw_history_bev.shape[3], raw_history_bev.shape[4])

            raw_history_vq_latent = self._quantize_latent_for_decode(raw_history_bev_flat)
            raw_history_vq_logits = self.forward_decoder(raw_history_vq_latent, shapes, (bs, num_history, w, h, d))
            raw_history_vq_pred = raw_history_vq_logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)

            hist_vq_latent = self._quantize_latent_for_decode(sampled_bev_flat)
            hist_vq_logits = self.forward_decoder(hist_vq_latent, shapes, (bs, num_history, w, h, d))
            aligned_history_vq_pred = hist_vq_logits.softmax(-1).argmax(-1).cpu().numpy().astype(np.uint8)

            if self.vote_mode_at_test != 'none' and pred_no_history is not None:
                current_pred_tensor = torch.from_numpy(pred_no_history[:, 0].astype(np.int64))
                history_pred_tensor = torch.from_numpy(aligned_history_vq_pred.astype(np.int64))
                voted_pred = self.vote_predictions_at_test(current_pred_tensor, history_pred_tensor).unsqueeze(1).cpu().numpy().astype(np.uint8)

            if self.save_aligned_history_vis:
                for batch_idx, img_meta in enumerate(img_metas):
                    should_save = (
                        self.aligned_history_vis_counter < self.aligned_history_vis_max_samples
                        and self.aligned_history_vis_counter % self.aligned_history_vis_interval == 0
                    )
                    if should_save:
                        self._save_aligned_history_visualization(
                            curr_occ_clean[batch_idx].cpu().numpy().astype(np.uint8),
                            pred_with_history[batch_idx, 0].astype(np.uint8),
                            pred_no_history[batch_idx, 0].astype(np.uint8) if pred_no_history is not None else pred_with_history[batch_idx, 0].astype(np.uint8),
                            history_occ_np[batch_idx],
                            raw_history_vq_pred[batch_idx],
                            aligned_history_vq_pred[batch_idx],
                            img_meta,
                        )
                    self.aligned_history_vis_counter += 1
            if self.return_aligned_history_metrics:
                output_dict['aligned_history_pred_semantics'] = aligned_history_vq_pred
                output_dict['aligned_history_targ_semantics'] = history_occ_np
            if need_compare:
                align_to_current_hist = np.zeros((num_history, self.num_classes, self.num_classes), dtype=np.int64)
                for batch_idx in range(bs):
                    for hist_idx in range(num_history):
                        align_to_current_hist[hist_idx] += self._compute_semantic_hist(
                            aligned_history_vq_pred[batch_idx, hist_idx], curr_occ_clean_np[batch_idx])
                output_dict['align_to_current_hist'] = align_to_current_hist
        if self.save_results:
            # z_sampled: [bs, c, h, w]
            save_token = z_sampled[0].cpu().numpy()
            if self.results_type != 'waymo':
                mmcv.mkdir_or_exist(os.path.join(self.save_root, 'token_4f', str(img_metas[0]['scene_name'])))
                np.savez(os.path.join(self.save_root, 'token_4f', str(img_metas[0]['scene_name']), '{}.npz'.format(img_metas[0]['sample_idx'])), token=save_token)
            else:
                occ_path_idx = img_metas[0]['occ_path'].split('/')[-1].split('.')[0]
                mmcv.mkdir_or_exist(os.path.join(self.save_root, 'token_4f', str(img_metas[0]['scene_name']).zfill(3)))
                np.savez(os.path.join(self.save_root, 'token_4f', str(img_metas[0]['scene_name']).zfill(3), '{}.npz'.format(occ_path_idx)), token=save_token)

            # save pred
            # mmcv.mkdir_or_exist('save_dir/debug/{}'.format(img_metas[0]['scene_name']))
            # np.savez('save_dir/debug/{}/{}.npz'.format(img_metas[0]['scene_name'],img_metas[0]['sample_idx']), semantics=pred[0][0])

        if self.export_only:
            return [dict(
                index=[img_meta['index'] for img_meta in img_metas],
                time=end_time - start_time,
            )]

        if voted_pred is not None:
            output_dict['semantics'] = voted_pred
            output_dict['semantics_voted'] = voted_pred
        else:
            output_dict['semantics'] = pred.astype(np.uint8)
        if need_compare:
            output_dict['semantics_with_history'] = pred_with_history.astype(np.uint8)
            output_dict['semantics_no_history'] = pred_no_history.astype(np.uint8)
        output_dict['input_curr_semantics'] = curr_occ.cpu().numpy().astype(np.uint8)
        output_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        output_dict['time'] = end_time - start_time
        return [output_dict]

    def forward_train(self, voxel_semantics, img_metas, **kwargs):
        # 0. Prepare Input
        batch_size = len(img_metas)
        voxel_semantics = self.normalize_batched_tensor(voxel_semantics, batch_size)
        bs, t, w, h, d = voxel_semantics.shape
        voxel_semantics_clean = kwargs.get('voxel_semantics_clean', None)
        if voxel_semantics_clean is not None:
            voxel_semantics_clean = self.normalize_batched_tensor(voxel_semantics_clean, batch_size)
        target_semantics = voxel_semantics_clean if voxel_semantics_clean is not None else voxel_semantics
        frame_reliability_map = kwargs.get('frame_reliability_map', None)
        if frame_reliability_map is not None:
            frame_reliability_map = self.normalize_frame_reliability_map(frame_reliability_map, bs)
            frame_reliability_map = frame_reliability_map.to(voxel_semantics.device)

        # 2. Process current voxel semantics
        curr_bev, shapes = self.forward_encoder(voxel_semantics)

        # 3. Time fusion
        if self.disable_history_at_train:
            # Pure no-history training: do not build or update temporal cache.
            self.history_bev = None
            self.history_occ = None
            self.history_reliability = None
            self.bev_aug = None
            sampled_bev = curr_bev.new_zeros((
                bs,
                self.frame_number,
                curr_bev.shape[1],
                curr_bev.shape[2],
                curr_bev.shape[3],
            ))
            sampled_reliability = None
        else:
            sampled_bev, _, _, sampled_reliability = self.align_bev(
                curr_bev,
                img_metas,
                cache_reliability=frame_reliability_map[:, -1] if frame_reliability_map is not None else None,
            )

        # 4. vq
        z_sampled, loss, info = self.vq(curr_bev, sampled_bev, is_voxel=False)

        # 5. Process Decoder
        logits = self.forward_decoder(z_sampled, shapes, (bs, 1, w, h, d))
        logits_no_history = None
        if (
            (self.disagreement_gain_weight > 0 or self.disagreement_stability_weight > 0)
            and frame_reliability_map is not None
            and sampled_reliability is not None
            and not self.disable_history_at_train
        ):
            zero_sampled_bev = torch.zeros_like(sampled_bev)
            z_sampled_no_history, _, _ = self.vq(curr_bev, zero_sampled_bev, is_voxel=False)
            logits_no_history = self.forward_decoder(z_sampled_no_history, shapes, (bs, 1, w, h, d))

        # 6. Compute Loss
        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, target_semantics))
        loss_dict['embed_loss'] = self.embed_loss_weight * loss
        if (
            self.fusion_supervision_weight > 0
            and sampled_reliability is not None
            and isinstance(info, dict)
            and info.get('fusion_weights', None) is not None
        ):
            loss_dict['fusion_loss'] = self.fusion_supervision_weight * self.fusion_supervision_loss(
                info['fusion_weights'],
                frame_reliability_map,
                sampled_reliability,
            )
        if (
            self.fusion_ranking_weight > 0
            and sampled_reliability is not None
            and isinstance(info, dict)
            and info.get('score_logits', None) is not None
        ):
            loss_dict['fusion_rank_loss'] = self.fusion_ranking_weight * self.fusion_ranking_loss(
                info['score_logits'],
                frame_reliability_map,
                sampled_reliability,
            )
        if logits_no_history is not None:
            loss_dict.update(self.disagreement_gain_loss(
                logits,
                logits_no_history,
                target_semantics,
                frame_reliability_map,
                sampled_reliability,
            ))
        return loss_dict
