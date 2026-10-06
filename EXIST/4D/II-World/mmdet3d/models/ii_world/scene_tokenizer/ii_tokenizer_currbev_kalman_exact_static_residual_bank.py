import numpy as np
import torch

from mmcv.parallel import DataContainer
from mmdet.models import DETECTORS

from .ii_tokenizer_currbev_kalman_unified_motion_bank import IISceneTokenizerCurrBevKalmanUnifiedMotionBank


@DETECTORS.register_module()
class IISceneTokenizerCurrBevKalmanExactStaticResidualBank(IISceneTokenizerCurrBevKalmanUnifiedMotionBank):
    """Use baseline-exact ego warp for static alignment and GT residual flow only for dynamics.

    This route keeps the unified motion-bank contract but replaces the learned/static-flow
    branch with the exact stepwise ego-motion warp used by the baseline history cache.
    Dynamic motion is then injected only as a residual on top of that exact static source.
    """

    def _stack_bev_aug(self, img_metas, device, dtype):
        bev_aug = []
        for meta in img_metas:
            mat = meta.get('bda_mat', None)
            if mat is None:
                mat = np.eye(4, dtype=np.float32)
            elif torch.is_tensor(mat):
                mat = mat.detach().cpu().numpy()
            bev_aug.append(np.asarray(mat, dtype=np.float32))
        return torch.as_tensor(np.stack(bev_aug, axis=0), device=device, dtype=dtype)

    def _extract_stepwise_ego_rts(self, img_metas, seq_len, device, dtype):
        batch_stepwise = []
        for meta in img_metas:
            curr_rt = meta.get('curr_to_prev_ego_rt', None)
            if curr_rt is None:
                raise KeyError('img_meta is missing curr_to_prev_ego_rt required for exact static warp')
            if torch.is_tensor(curr_rt):
                curr_rt = curr_rt.detach().cpu().numpy()
            curr_rt = np.asarray(curr_rt, dtype=np.float32)

            previous = meta.get('previous_curr_to_prev_ego_rt', []) or []
            previous = [
                np.asarray(rt.detach().cpu().numpy() if torch.is_tensor(rt) else rt, dtype=np.float32)
                for rt in previous
            ]

            if len(previous) == seq_len:
                stepwise = previous
            elif len(previous) == seq_len - 1:
                stepwise = previous + [curr_rt]
            elif len(previous) == seq_len - 2:
                stepwise = [np.eye(4, dtype=np.float32)] + previous + [curr_rt]
            else:
                raise ValueError(
                    f'Unexpected previous_curr_to_prev_ego_rt length={len(previous)} for seq_len={seq_len}')

            batch_stepwise.append(np.stack(stepwise, axis=0))

        return torch.as_tensor(np.stack(batch_stepwise, axis=0), device=device, dtype=dtype)

    def _compute_exact_static_flow(self, step_rt, bev_aug, target_hw, dtype, device):
        batch_size = step_rt.shape[0]
        height, width = target_hw
        dummy = torch.zeros((batch_size, 1, 1, height, width), device=device, dtype=dtype)
        grid = self.generate_grid(dummy)
        feat2bev = self.generate_feat2bev(grid, self.dx, self.bx)

        rt_flow = (
            torch.inverse(feat2bev)
            @ bev_aug
            @ step_rt
            @ torch.inverse(bev_aug)
            @ feat2bev
        )
        grid = rt_flow.view(batch_size, 1, 1, 1, 4, 4) @ grid
        src = grid[..., :3, 0]
        src_x = src[..., 0].squeeze(-1)
        src_y = src[..., 1].squeeze(-1)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij',
        )
        x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, -1)
        y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1)

        flow = torch.stack([src_y - y_coords, src_x - x_coords], dim=1)
        valid = (
            (src_x >= 0.0) & (src_x <= (width - 1.0)) &
            (src_y >= 0.0) & (src_y <= (height - 1.0))
        ).to(dtype=dtype).unsqueeze(1)
        return flow, valid

    def _compute_exact_forward_static_flow(self, step_rt, bev_aug, target_hw, dtype, device):
        return self._compute_exact_static_flow(
            torch.inverse(step_rt.to(dtype)),
            bev_aug.to(dtype),
            target_hw,
            dtype=dtype,
            device=device,
        )

    def _build_exact_motion_step(self,
                                 step_rt,
                                 bev_aug,
                                 dynamic_residual_flow_step,
                                 dynamic_residual_valid_step,
                                 dynamic_residual_forward_flow_step,
                                 dynamic_residual_forward_valid_step,
                                 dynamic_mask_step,
                                 dynamic_source_mask_step,
                                 target_hw,
                                 dtype,
                                 device):
        exact_static_flow, static_valid_ratio = self._compute_exact_static_flow(
            step_rt.to(dtype),
            bev_aug.to(dtype),
            target_hw,
            dtype=dtype,
            device=device,
        )
        exact_forward_static_flow, _ = self._compute_exact_forward_static_flow(
            step_rt.to(dtype),
            bev_aug.to(dtype),
            target_hw,
            dtype=dtype,
            device=device,
        )
        dynamic_residual_latent, dynamic_residual_valid_ratio, dynamic_ratio = self._aggregate_dynamic_residual_to_latent(
            dynamic_residual_flow_step.to(dtype),
            dynamic_residual_valid_step,
            dynamic_mask_step,
            target_hw,
        )
        forward_dynamic_residual_latent, forward_dynamic_residual_valid_ratio, source_dynamic_ratio = self._aggregate_dynamic_residual_to_latent(
            dynamic_residual_forward_flow_step.to(dtype),
            dynamic_residual_forward_valid_step,
            dynamic_source_mask_step,
            target_hw,
        )
        dynamic_source_mask = (
            (source_dynamic_ratio > 0).to(dtype) *
            (forward_dynamic_residual_valid_ratio > 0).to(dtype)
        ).clamp(0.0, 1.0)
        dynamic_gate = dynamic_ratio * dynamic_residual_valid_ratio if self.unified_motion_use_dynamic_gate else dynamic_residual_valid_ratio
        return dict(
            static_flow=exact_static_flow,
            forward_static_flow=exact_forward_static_flow,
            static_valid_ratio=static_valid_ratio,
            dynamic_residual_flow=dynamic_residual_latent,
            dynamic_residual_valid_ratio=dynamic_residual_valid_ratio,
            forward_dynamic_residual_flow=forward_dynamic_residual_latent,
            forward_dynamic_residual_valid_ratio=forward_dynamic_residual_valid_ratio,
            dynamic_ratio=dynamic_ratio,
            source_dynamic_ratio=source_dynamic_ratio,
            dynamic_source_mask=dynamic_source_mask,
            forward_total_flow=exact_forward_static_flow + forward_dynamic_residual_latent,
            dynamic_gate=dynamic_gate.clamp(0.0, 1.0),
            step_rt=step_rt,
        )

    def _build_motion_step(self,
                           step_rt,
                           bev_aug,
                           dynamic_residual_flow_step,
                           dynamic_residual_valid_step,
                           dynamic_residual_forward_flow_step,
                           dynamic_residual_forward_valid_step,
                           dynamic_mask_step,
                           dynamic_source_mask_step,
                           target_hw,
                           dtype,
                           device):
        return self._build_exact_motion_step(
            step_rt,
            bev_aug,
            dynamic_residual_flow_step,
            dynamic_residual_valid_step,
            dynamic_residual_forward_flow_step,
            dynamic_residual_forward_valid_step,
            dynamic_mask_step,
            dynamic_source_mask_step,
            target_hw,
            dtype,
            device,
        )

    def _forward_unified_motion(self,
                                voxel_semantics,
                                voxel_semantics_clean,
                                img_metas,
                                oracle_static_flow_seq,
                                oracle_static_flow_valid_seq,
                                oracle_dynamic_residual_flow_seq,
                                oracle_dynamic_residual_flow_valid_seq,
                                oracle_dynamic_residual_flow_forward_seq,
                                oracle_dynamic_residual_flow_valid_forward_seq,
                                oracle_dynamic_mask_seq,
                                oracle_dynamic_source_mask_seq,
                                frame_reliability_map=None):
        batch_size, seq_len = voxel_semantics.shape[:2]
        if seq_len < 2:
            raise ValueError(f'Unified-motion route expects at least 2 frames, got {seq_len}')

        curr_target = voxel_semantics_clean[:, -1:]
        _, _, occ_h, occ_w, occ_z = curr_target.shape

        bev_seq = []
        shape_seq = []
        for frame_idx in range(seq_len):
            bev_k, shapes_k = self.forward_encoder(voxel_semantics[:, frame_idx:frame_idx + 1])
            bev_seq.append(bev_k)
            shape_seq.append(shapes_k)
        curr_bev = bev_seq[-1]
        curr_shapes = shape_seq[-1]
        clean_curr_bev, _ = self.forward_encoder(curr_target)

        stepwise_rts = self._extract_stepwise_ego_rts(img_metas, seq_len, curr_bev.device, curr_bev.dtype)
        bev_aug = self._stack_bev_aug(img_metas, curr_bev.device, curr_bev.dtype)

        motion_steps = []
        for step_idx in range(1, seq_len):
            motion_steps.append(self._build_motion_step(
                stepwise_rts[:, step_idx],
                bev_aug,
                oracle_dynamic_residual_flow_seq[:, step_idx - 1],
                oracle_dynamic_residual_flow_valid_seq[:, step_idx - 1],
                oracle_dynamic_residual_flow_forward_seq[:, step_idx - 1],
                oracle_dynamic_residual_flow_valid_forward_seq[:, step_idx - 1],
                oracle_dynamic_mask_seq[:, step_idx - 1],
                oracle_dynamic_source_mask_seq[:, step_idx - 1],
                curr_bev.shape[-2:],
                curr_bev.dtype,
                curr_bev.device,
            ))

        sampled_bev_motion, slot_debug = self._build_unified_history_bank(bev_seq, motion_steps)
        z_sampled, embed_loss, _ = self.vq(curr_bev, sampled_bev_motion, is_voxel=False)
        logits = self.forward_decoder(z_sampled, curr_shapes, (batch_size, 1, occ_h, occ_w, occ_z))

        reliability = frame_reliability_map
        posterior_mean = bev_seq[0]
        posterior_cov = self._initial_covariance(
            batch_size, posterior_mean.shape[-2], posterior_mean.shape[-1],
            posterior_mean.dtype, posterior_mean.device)
        step_debug = []
        last_prior_mean = posterior_mean
        last_prior_cov = posterior_cov
        last_gain_group = posterior_cov.new_zeros(posterior_cov.shape)
        last_motion_step = motion_steps[0]

        for step_idx in range(1, seq_len):
            motion_step = motion_steps[step_idx - 1]
            reliability_latent = None
            if reliability is not None:
                reliability_latent = self._pool_current_map(
                    reliability[:, step_idx:step_idx + 1],
                    posterior_mean.shape[-2:],
                )
            prior_mean, prior_cov, process_noise, static_prior, dynamic_prior, total_flow = self._predict_kalman_state_unified(
                posterior_mean, posterior_cov, motion_step)
            posterior_mean, posterior_cov, gain_group, observation_noise = self._update_kalman_state(
                prior_mean, prior_cov, bev_seq[step_idx], reliability_latent=reliability_latent)
            last_prior_mean = prior_mean
            last_prior_cov = prior_cov
            last_gain_group = gain_group
            last_motion_step = motion_step
            step_debug.append(dict(
                observation_bev=bev_seq[step_idx],
                reliability_latent=reliability_latent,
                process_noise=process_noise,
                observation_noise=observation_noise,
                prior_mean=prior_mean,
                posterior_mean=posterior_mean,
                gain_group=gain_group,
                static_prior=static_prior,
                dynamic_prior=dynamic_prior,
                total_flow=total_flow,
                static_flow=motion_step['static_flow'],
                dynamic_residual_flow=motion_step['dynamic_residual_flow'],
                static_valid_ratio=motion_step['static_valid_ratio'],
                dynamic_residual_valid_ratio=motion_step['dynamic_residual_valid_ratio'],
                dynamic_ratio=motion_step['dynamic_ratio'],
                dynamic_gate=motion_step['dynamic_gate'],
            ))

        debug = dict(
            bev_seq=bev_seq,
            shape_seq=shape_seq,
            curr_bev=curr_bev,
            curr_shapes=curr_shapes,
            clean_curr_bev=clean_curr_bev,
            sampled_bev_motion=sampled_bev_motion,
            slot_debug=slot_debug,
            motion_steps=motion_steps,
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            last_prior_mean=last_prior_mean,
            last_prior_cov=last_prior_cov,
            last_gain_group=last_gain_group,
            last_motion_step=last_motion_step,
            step_debug=step_debug,
        )
        return logits, embed_loss, curr_target, debug

    def forward_train(self,
                      voxel_semantics,
                      img_metas,
                      oracle_static_flow_seq,
                      oracle_static_flow_valid_seq,
                      oracle_dynamic_residual_flow_seq,
                      oracle_dynamic_residual_flow_valid_seq,
                      oracle_dynamic_mask_seq,
                      oracle_dynamic_source_mask_seq=None,
                      oracle_dynamic_residual_flow_forward_seq=None,
                      oracle_dynamic_residual_flow_valid_forward_seq=None,
                      frame_reliability_map=None,
                      voxel_semantics_clean=None,
                      **kwargs):
        if oracle_dynamic_residual_flow_forward_seq is None:
            oracle_dynamic_residual_flow_forward_seq = oracle_dynamic_residual_flow_seq
        if oracle_dynamic_residual_flow_valid_forward_seq is None:
            oracle_dynamic_residual_flow_valid_forward_seq = oracle_dynamic_residual_flow_valid_seq
        if oracle_dynamic_source_mask_seq is None:
            oracle_dynamic_source_mask_seq = oracle_dynamic_mask_seq
        return super().forward_train(
            voxel_semantics=voxel_semantics,
            img_metas=img_metas,
            oracle_static_flow_seq=oracle_static_flow_seq,
            oracle_static_flow_valid_seq=oracle_static_flow_valid_seq,
            oracle_dynamic_residual_flow_seq=oracle_dynamic_residual_flow_seq,
            oracle_dynamic_residual_flow_valid_seq=oracle_dynamic_residual_flow_valid_seq,
            oracle_dynamic_residual_flow_forward_seq=oracle_dynamic_residual_flow_forward_seq,
            oracle_dynamic_residual_flow_valid_forward_seq=oracle_dynamic_residual_flow_valid_forward_seq,
            oracle_dynamic_mask_seq=oracle_dynamic_mask_seq,
            oracle_dynamic_source_mask_seq=oracle_dynamic_source_mask_seq,
            frame_reliability_map=frame_reliability_map,
            voxel_semantics_clean=voxel_semantics_clean,
            **kwargs,
        )

    def forward_test(self,
                     voxel_semantics,
                     img_metas,
                     oracle_static_flow_seq,
                     oracle_static_flow_valid_seq,
                     oracle_dynamic_residual_flow_seq,
                     oracle_dynamic_residual_flow_valid_seq,
                     oracle_dynamic_mask_seq,
                     oracle_dynamic_source_mask_seq=None,
                     oracle_dynamic_residual_flow_forward_seq=None,
                     oracle_dynamic_residual_flow_valid_forward_seq=None,
                     frame_reliability_map=None,
                     voxel_semantics_clean=None,
                     **kwargs):
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        if oracle_dynamic_residual_flow_forward_seq is None:
            oracle_dynamic_residual_flow_forward_seq = oracle_dynamic_residual_flow_seq
        if oracle_dynamic_residual_flow_valid_forward_seq is None:
            oracle_dynamic_residual_flow_valid_forward_seq = oracle_dynamic_residual_flow_valid_seq
        if oracle_dynamic_source_mask_seq is None:
            oracle_dynamic_source_mask_seq = oracle_dynamic_mask_seq
        return super().forward_test(
            voxel_semantics=voxel_semantics,
            img_metas=img_metas,
            oracle_static_flow_seq=oracle_static_flow_seq,
            oracle_static_flow_valid_seq=oracle_static_flow_valid_seq,
            oracle_dynamic_residual_flow_seq=oracle_dynamic_residual_flow_seq,
            oracle_dynamic_residual_flow_valid_seq=oracle_dynamic_residual_flow_valid_seq,
            oracle_dynamic_residual_flow_forward_seq=oracle_dynamic_residual_flow_forward_seq,
            oracle_dynamic_residual_flow_valid_forward_seq=oracle_dynamic_residual_flow_valid_forward_seq,
            oracle_dynamic_mask_seq=oracle_dynamic_mask_seq,
            oracle_dynamic_source_mask_seq=oracle_dynamic_source_mask_seq,
            frame_reliability_map=frame_reliability_map,
            voxel_semantics_clean=voxel_semantics_clean,
            **kwargs,
        )
