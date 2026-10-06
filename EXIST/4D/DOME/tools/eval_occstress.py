#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch
from einops import rearrange
from mmengine import Config
from mmengine.registry import MODELS
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model  # noqa: E402,F401
from dataset.occstress_dataset import (  # noqa: E402
    FUTURE_OFFSETS,
    OBSERVED_OFFSETS,
    OccStressProtocolDataset,
)
from diffusion import create_diffusion  # noqa: E402


def build_prediction_exporter(dataset):
    if not os.environ.get('OCCSTRESS_PREDICTION_OUTPUT_ROOT'):
        return None
    raise NotImplementedError(
        'Qualitative voxel export is not part of this code snapshot. '
        'Unset OCCSTRESS_PREDICTION_OUTPUT_ROOT to evaluate metrics.')


def collate_occstress(samples):
    occupancy = torch.from_numpy(np.stack([item[0] for item in samples]))
    targets = torch.from_numpy(np.stack([item[1] for item in samples]))
    metadata = [item[2] for item in samples]
    return occupancy, targets, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate DOME on one OccStress protocol.')
    parser.add_argument(
        '--config', default=str(ROOT / 'config/train_dome.py'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base-info', required=True)
    parser.add_argument('--occstress-root', required=True)
    parser.add_argument('--nuscenes-root')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true',
                        help='Disable cuDNN autotuning and request deterministic Torch algorithms.')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--vae-checkpoint-sha256')
    parser.add_argument('--log-every', type=int, default=25)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def class_mapping_path(root, dataset):
    aliases = {'Occ3D-nuScenes': 'OccStress-nuScenes',
               'Occ3D-Waymo': 'OccStress-Waymo', 'UniOcc-CARLA': 'OccStress-CARLA'}
    name = aliases.get(dataset, dataset)
    if name not in ('OccStress-nuScenes', 'OccStress-Waymo', 'OccStress-CARLA'):
        raise ValueError('Unsupported dataset: ' + str(dataset))
    canonical = Path(root) / 'meta' / name / 'class_mapping.json'
    if canonical.is_file():
        return canonical
    legacy = Path(root) / 'meta/manual/class_mapping.json'
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(canonical)


def git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'not-a-git-checkout'


def protocol_name(path):
    path = Path(path).resolve()
    if 'protocols' in path.parts:
        return '/'.join(path.parts[path.parts.index('protocols') + 1:])
    return path.name


def token_seed(base_seed, token):
    digest = hashlib.sha256(f'{base_seed}:{token}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**31)


def load_state(module, checkpoint_path, preferred_key):
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    if preferred_key in checkpoint:
        state = checkpoint[preferred_key]
        state_key = preferred_key
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
        state_key = 'state_dict'
    else:
        state = checkpoint
        state_key = '<root>'

    module_keys = set(module.state_dict())
    if state and not module_keys.intersection(state):
        stripped = {
            key.removeprefix('module.'): value
            for key, value in state.items()
        }
        if module_keys.intersection(stripped):
            state = stripped
    incompatible = module.load_state_dict(state, strict=False)
    metadata = {
        'state_key': state_key,
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
    }
    if len(metadata['missing_keys']) >= len(module_keys):
        raise ValueError(
            f'{checkpoint_path} did not load any model parameters')
    del checkpoint, state
    return metadata


class HorizonMetrics:
    def __init__(self, horizons=6):
        self.semantic = np.zeros((horizons, 18, 18), dtype=np.int64)
        self.binary = np.zeros((horizons, 2, 2), dtype=np.int64)

    @staticmethod
    def _confusion(prediction, target, classes):
        encoded = target.reshape(-1) * classes + prediction.reshape(-1)
        # Torch 2.0 flags CUDA bincount even for exact integer counts.
        if torch.are_deterministic_algorithms_enabled():
            encoded = encoded.cpu()
        return torch.bincount(
            encoded, minlength=classes * classes).reshape(
                classes, classes).cpu().numpy()

    def update(self, prediction, target):
        if prediction.shape != target.shape or prediction.shape[1] != 6:
            raise ValueError(
                f'metric shape mismatch: {prediction.shape}, {target.shape}')
        for horizon in range(6):
            pred = prediction[:, horizon].long()
            gt = target[:, horizon].long()
            self.semantic[horizon] += self._confusion(pred, gt, 18)
            self.binary[horizon] += self._confusion(
                (pred != 17).long(), (gt != 17).long(), 2)

    @staticmethod
    def _semantic_metrics(confusion):
        per_class = []
        present_ious = []
        for class_index in range(17):
            seen = int(confusion[class_index, :].sum())
            positive = int(confusion[:, class_index].sum())
            correct = int(confusion[class_index, class_index])
            if seen == 0:
                per_class.append(None)
            else:
                iou = 100.0 * correct / (seen + positive - correct)
                per_class.append(iou)
                present_ious.append(iou)
        if not present_ious:
            raise ValueError('no present occupied classes in target')
        return float(np.mean(present_ious)), per_class

    @staticmethod
    def _binary_iou(confusion):
        seen = int(confusion[1, :].sum())
        positive = int(confusion[:, 1].sum())
        correct = int(confusion[1, 1])
        if seen == 0:
            return 100.0
        return 100.0 * correct / (seen + positive - correct)

    def result(self):
        results = {}
        for index, horizon in enumerate(FUTURE_OFFSETS):
            miou, per_class = self._semantic_metrics(self.semantic[index])
            results[str(horizon)] = {
                'miou': miou,
                'iou': self._binary_iou(self.binary[index]),
                'per_class_iou': per_class,
            }
        return results


def build_models(config, checkpoint, vae_checkpoint):
    world_model = MODELS.build(config.model.world_model).cuda().eval()
    vae = MODELS.build(config.model.vae).cuda().eval()
    vae.requires_grad_(False)
    world_load = load_state(
        world_model, checkpoint, 'ema' if config.get('ema', False)
        else 'state_dict')
    vae_load = load_state(vae, vae_checkpoint, 'state_dict')
    return world_model, vae, world_load, vae_load


def infer(world_model, vae, diffusion, config, sequence, metadata):
    batch_size = sequence.shape[0]
    encoded, shapes = vae.forward_encoder(sequence)
    encoded, _, _ = vae.sample_z(encoded)
    input_latents = encoded * config.model.vae.scaling_factor
    if input_latents.dim() == 5:
        input_latents = rearrange(
            input_latents, 'b c f h w -> b f c h w').contiguous()
    elif input_latents.dim() == 4:
        input_latents = rearrange(
            input_latents, '(b f) c h w -> b f c h w',
            b=batch_size).contiguous()
    else:
        raise ValueError(f'unexpected latent shape {input_latents.shape}')

    resolution = config.model.vae.encoder_cfg.resolution
    scale = 2 ** (len(config.model.vae.encoder_cfg.ch_mult) - 1)
    latent_size = resolution // scale
    noise_shape = (
        batch_size,
        config.end_frame,
        config.base_channel,
        latent_size,
        latent_size,
    )
    if config.sample.sample_method != 'ddpm':
        raise ValueError('OccStress evaluation currently requires official DDPM')
    latents = diffusion.p_sample_loop(
        world_model,
        noise_shape,
        None,
        clip_denoised=False,
        model_kwargs={'metas': metadata},
        progress=False,
        device='cuda',
        initial_cond_indices=list(range(config.sample.n_conds)),
        initial_cond_frames=input_latents,
    )
    latents = rearrange(
        latents / config.model.vae.scaling_factor,
        'b f c h w -> b c f h w')
    logits = vae.forward_decoder(
        latents,
        shapes=[[200, 200], [100, 100], [50, 50]],
        input_shape=[
            batch_size,
            config.end_frame,
            200,
            200,
            config._dim_,
        ],
    )
    return logits.argmax(dim=-1)


def main():
    args = parse_args()
    started = time.time()
    config = Config.fromfile(args.config)
    if (
            config.start_frame != 0 or
            config.mid_frame != 4 or
            config.end_frame != 10 or
            config.sample.n_conds != 4 or
            config.sample.num_sampling_steps != 20):
        raise ValueError(
            'expected official DOME H4/F6 configuration with 20 DDPM steps')

    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    dataset = OccStressProtocolDataset(
        protocol_path=args.protocol,
        base_info=args.base_info,
        occstress_root=args.occstress_root,
        nuscenes_root=args.nuscenes_root,
        max_samples=args.max_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    scene_shards = sorted({
        str(record['scene_name']).zfill(3) for record in dataset.records
    })
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_occstress,
    )
    prediction_exporter = build_prediction_exporter(dataset)
    world_model, vae, world_load, vae_load = build_models(
        config, args.checkpoint, args.vae_checkpoint)
    diffusion = create_diffusion(
        timestep_respacing=str(config.sample.num_sampling_steps),
        beta_start=config.schedule.beta_start,
        beta_end=config.schedule.beta_end,
        replace_cond_frames=config.replace_cond_frames,
        cond_frames_choices=config.cond_frames_choices,
        predict_xstart=config.schedule.get('predict_xstart', False),
    )

    metrics = HorizonMetrics()
    reconstruction = HorizonMetrics(horizons=6)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, (sequence, target, metadata) in enumerate(loader):
            token = metadata[0]['anchor_token']
            seed = token_seed(args.seed, token)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sequence = sequence.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            prediction = infer(
                world_model, vae, diffusion, config, sequence, metadata)
            if prediction_exporter is not None:
                prediction_exporter.export_record(
                    dataset.records[index],
                    prediction[0, config.mid_frame:config.end_frame])
            metrics.update(
                prediction[:, config.mid_frame:config.end_frame],
                target[:, config.mid_frame:config.end_frame])

            # Reuse the metric implementation for the current reconstructed
            # condition by repeating it over six slots.
            reconstructed_current = prediction[:, 3:4].expand(-1, 6, -1, -1, -1)
            current_target = target[:, 3:4].expand(-1, 6, -1, -1, -1)
            reconstruction.update(reconstructed_current, current_target)

            completed = index + 1
            if args.log_every and (
                    completed % args.log_every == 0 or
                    completed == len(dataset)):
                elapsed = time.time() - started
                print(
                    f'completed={completed}/{len(dataset)} '
                    f'rate={completed / elapsed:.3f} anchors/s',
                    flush=True)

    if prediction_exporter is not None:
        prediction_exporter.finish()

    horizon_metrics = metrics.result()
    reconstruction_metrics = reconstruction.result()['0.5']
    average_miou = float(np.mean([
        value['miou'] for value in horizon_metrics.values()
    ]))
    average_iou = float(np.mean([
        value['iou'] for value in horizon_metrics.values()
    ]))
    paper_indices = ('1.0', '2.0', '3.0')
    paper_average_miou = float(np.mean([
        horizon_metrics[key]['miou'] for key in paper_indices
    ]))
    paper_average_iou = float(np.mean([
        horizon_metrics[key]['iou'] for key in paper_indices
    ]))
    metric_values = [
        value
        for metrics_at_horizon in horizon_metrics.values()
        for value in (
            metrics_at_horizon['miou'],
            metrics_at_horizon['iou'],
        )
    ]
    if not all(math.isfinite(value) for value in metric_values):
        raise ValueError('evaluation produced a non-finite metric')

    elapsed = time.time() - started
    payload = {
        'status': 'success',
        'method': 'DOME',
        'dataset': dataset.dataset_name,
        'code_commit': git_commit(),
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': (
            args.checkpoint_sha256 or sha256(args.checkpoint)),
        'zero_shot_source_checkpoint': 'Occ3D-nuScenes',
        'vae_checkpoint': str(Path(args.vae_checkpoint).resolve()),
        'vae_checkpoint_sha256': (
            args.vae_checkpoint_sha256 or sha256(args.vae_checkpoint)),
        'world_model_load': world_load,
        'vae_load': vae_load,
        'protocol': str(Path(args.protocol).resolve()),
        'protocol_name': protocol_name(args.protocol),
        'protocol_sha256': sha256(args.protocol),
        'class_mapping_sha256': sha256(class_mapping_path(args.occstress_root, dataset.dataset_name)),
        'base_info': str(Path(args.base_info).resolve()),
        'occstress_root': str(Path(args.occstress_root).resolve()),
        'evaluated_records': len(dataset),
        'scene_shard_count': len(scene_shards),
        'scene_shards': scene_shards,
        'full_protocol_records': dataset.full_record_count,
        'num_shards': dataset.num_shards,
        'shard_index': dataset.shard_index,
        'observed_offsets_seconds': list(OBSERVED_OFFSETS),
        'ignored_offset_seconds': -2.0,
        'control_policy': (
            'carla_gt_future_ego_motion_poses'
            if dataset.dataset_name in ('UniOcc-CARLA', 'OccStress-CARLA')
            else 'official_gt_future_ego_motion_poses'),
        'coordinate_convention': (
            'Occ3D right-handed ego-local x-forward/y-left/z-up'),
        'metric_policy': 'present_gt_occupied_classes',
        'future_horizons_seconds': list(FUTURE_OFFSETS),
        'horizon_metrics': horizon_metrics,
        'average_miou': average_miou,
        'average_iou': average_iou,
        'paper_average_horizons_seconds': [1.0, 2.0, 3.0],
        'paper_average_miou': paper_average_miou,
        'paper_average_iou': paper_average_iou,
        'current_reconstruction_miou': reconstruction_metrics['miou'],
        'current_reconstruction_iou': reconstruction_metrics['iou'],
        'raw_confusion': {
            'semantic': metrics.semantic.tolist(),
            'binary': metrics.binary.tolist(),
            'reconstruction_semantic': reconstruction.semantic.tolist(),
            'reconstruction_binary': reconstruction.binary.tolist(),
        },
        'sampling': {
            'method': config.sample.sample_method,
            'steps': config.sample.num_sampling_steps,
            'base_seed': args.seed,
            'noise_policy': 'sha256(base_seed:anchor_token)',
            'deterministic_requested': args.deterministic,
        },
        'elapsed_seconds': elapsed,
        'anchors_per_second': len(dataset) / elapsed,
        'peak_gpu_memory_mib': (
            torch.cuda.max_memory_allocated() / (1024 * 1024)),
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
