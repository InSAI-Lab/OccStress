import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from mmengine import Config
from mmengine.registry import MODELS
from mmengine.runner import set_random_seed
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model  # noqa: F401
from dataset import get_dataloader
from diffusion import create_diffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal COME GPU smoke test.")
    parser.add_argument(
        "--py-config",
        default="configs/train_dome_v2.py",
        help="Config relative to the COME repo root.",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dir/dome_v2",
        help="Work dir relative to the COME repo root, or an absolute path.",
    )
    parser.add_argument(
        "--mode",
        choices=["world", "control"],
        default="world",
        help="Smoke test the world model or the controlnet pipeline.",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=2,
        help="Number of diffusion steps for the smoke test.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Validation sample index to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_existing_path(candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        path = resolve_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"No existing path found in candidates: {candidates}")


def pick_checkpoint(candidates, label):
    ckpt_path = resolve_existing_path(candidates)
    print(f"[smoke] using {label} checkpoint: {ckpt_path}")
    return ckpt_path


def load_model_checkpoint(module, ckpt_path, prefer_ema=False, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if prefer_ema and "ema" in ckpt:
        state_dict = ckpt["ema"]
        key = "ema"
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        key = "state_dict"
    elif "ema" in ckpt:
        state_dict = ckpt["ema"]
        key = "ema"
    else:
        state_dict = ckpt
        key = "raw"
    msg = module.load_state_dict(state_dict, strict=strict)
    print(f"[smoke] loaded {key} from {ckpt_path}: {msg}")
    return ckpt


def build_single_sample(cfg, sample_idx):
    cfg = deepcopy(cfg)
    cfg.train_loader.batch_size = 1
    cfg.val_loader.batch_size = 1
    cfg.train_loader.num_workers = 0
    cfg.val_loader.num_workers = 0
    cfg.train_loader.shuffle = False
    cfg.val_loader.shuffle = False
    cfg.val_dataset_config.test_mode = True

    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
    )
    for idx, batch in enumerate(val_loader):
        if idx == sample_idx:
            return batch
    raise IndexError(f"sample_idx={sample_idx} exceeds val loader length")


def encode_latents(vae, cfg, input_occs):
    bs = input_occs.shape[0]
    encoded_latent, _ = vae.forward_encoder(input_occs)
    encoded_latent, _, _ = vae.sample_z(encoded_latent)
    input_latents = encoded_latent * cfg.model.vae.scaling_factor
    if input_latents.dim() == 4:
        input_latents = rearrange(input_latents, "(b f) c h w -> b f c h w", b=bs).contiguous()
    elif input_latents.dim() == 5:
        input_latents = rearrange(input_latents, "b c f h w -> b f c h w", b=bs).contiguous()
    else:
        raise NotImplementedError(f"Unexpected latent dim: {input_latents.dim()}")
    return input_latents


def build_diffusion(cfg, num_sampling_steps):
    return create_diffusion(
        timestep_respacing=str(num_sampling_steps),
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        predict_xstart=cfg.schedule.get("predict_xstart", False),
    )


def maybe_add_bev_layout(model_kwargs, metas):
    if "data_bev" in metas[0]:
        data_bev = torch.stack([meta["data_bev"] for meta in metas]).cuda()
        model_kwargs["bev_layout"] = data_bev


def run_world_smoke(cfg, work_dir, args):
    world_model = MODELS.build(cfg.model.world_model).cuda().eval()
    vae = MODELS.build(cfg.model.vae).cuda().eval()
    vae.requires_grad_(False)

    model_ckpt = pick_checkpoint(
        [
            work_dir / "best_miou_world_model.pth",
            work_dir / "best_miou.pth",
            work_dir / "latest.pth",
        ],
        "world model",
    )
    load_model_checkpoint(world_model, model_ckpt, prefer_ema=cfg.get("ema", False), strict=False)

    vae_ckpt = pick_checkpoint([cfg.vae_load_from], "vae")
    load_model_checkpoint(vae, vae_ckpt, strict=True)

    input_occs, _, metas = build_single_sample(cfg, args.sample_idx)
    input_occs = input_occs.cuda()

    start = time.time()
    with torch.no_grad():
        input_latents = encode_latents(vae, cfg, input_occs)
        w = h = cfg.model.vae.encoder_cfg.resolution
        vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
        w //= vae_scale_factor
        h //= vae_scale_factor

        model_kwargs = {"metas": metas}
        maybe_add_bev_layout(model_kwargs, metas)

        diffusion = build_diffusion(cfg, args.num_sampling_steps)
        noise_shape = (
            input_occs.shape[0],
            input_occs.shape[1],
            cfg.base_channel,
            w,
            h,
        )
        n_conds = cfg.sample.get("n_conds", 0)
        initial_cond_indices = list(range(n_conds)) if n_conds else None
        latents = diffusion.p_sample_loop(
            world_model,
            noise_shape,
            None,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device="cuda",
            initial_cond_indices=initial_cond_indices,
            initial_cond_frames=input_latents,
        )
    elapsed = time.time() - start
    print(f"[smoke] world latents shape: {tuple(latents.shape)}")
    print(f"[smoke] world smoke finished in {elapsed:.2f}s")


def build_stage1_inputs(input_occs, metas, start_frame, mid_frame, end_frame):
    scene_range = (-40, -40, -1, 40, 40, 5.4)
    inputs_dict = {
        "source_occs": input_occs[:, start_frame:mid_frame].clone().permute(0, 1, 4, 3, 2),
        "target_occs": input_occs[:, mid_frame:end_frame].clone().permute(0, 1, 4, 3, 2),
        "source_metas": [],
        "target_metas": [],
        "metas": [],
    }
    for batch_idx in range(len(metas)):
        inputs_dict["source_metas"].append({"ego2global": []})
        inputs_dict["target_metas"].append({"ego2global": []})
        inputs_dict["metas"].append({"scene_range": scene_range})
        for frame_idx in range(end_frame - start_frame):
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(metas[batch_idx]["e2g_r"][frame_idx]).rotation_matrix
            ego2global[:3, 3] = metas[batch_idx]["e2g_t"][frame_idx]
            if frame_idx < mid_frame:
                inputs_dict["source_metas"][batch_idx]["ego2global"].append(ego2global)
            else:
                inputs_dict["target_metas"][batch_idx]["ego2global"].append(ego2global)
    return inputs_dict


def run_control_smoke(cfg, work_dir, args):
    controlnet = MODELS.build(cfg.model.world_model).cuda().eval()
    vae = MODELS.build(cfg.model.vae).cuda().eval()
    vae.requires_grad_(False)

    stage1_cfg = Config.fromfile(str(resolve_path(cfg.stage_one_config)))
    world_cfg = Config.fromfile(str(resolve_path(cfg.world_model_config)))
    stage1_model = MODELS.build(stage1_cfg.model).cuda().eval()
    stage1_model.requires_grad_(False)
    world_model = MODELS.build(world_cfg.model.world_model).cuda().eval()
    world_model.requires_grad_(False)

    control_ckpt = pick_checkpoint(
        [work_dir / "best_miou_controlnet.pth", work_dir / "latest.pth"],
        "controlnet",
    )
    world_ckpt = pick_checkpoint(
        [work_dir / "best_miou_world_model.pth", cfg.world_model_ckpt],
        "paired world model",
    )
    stage1_ckpt = pick_checkpoint([cfg.stage_one_ckpt], "stage1 forecaster")
    vae_ckpt = pick_checkpoint([cfg.vae_load_from], "vae")

    load_model_checkpoint(controlnet, control_ckpt, prefer_ema=cfg.get("ema", False), strict=False)
    load_model_checkpoint(world_model, world_ckpt, strict=False)
    load_model_checkpoint(stage1_model, stage1_ckpt, strict=True)
    load_model_checkpoint(vae, vae_ckpt, strict=True)

    input_occs, _, metas = build_single_sample(cfg, args.sample_idx)
    start_frame = cfg.get("start_frame", 0)
    mid_frame = cfg.get("mid_frame", 4)
    end_frame = cfg.get("end_frame", input_occs.shape[1])

    start = time.time()
    with torch.no_grad():
        stage1_inputs = build_stage1_inputs(input_occs, metas, start_frame, mid_frame, end_frame)
        stage1_outputs = stage1_model(stage1_inputs)
        future_occs_pred = stage1_outputs["sem_preds"].clone().permute(0, 1, 4, 3, 2)

        input_occs = input_occs.cuda()
        input_occs[:, mid_frame:end_frame] = future_occs_pred

        input_latents = encode_latents(vae, cfg, input_occs)
        stage1_invisible_mask = stage1_outputs["invisible_mask"].permute(0, 1, 4, 3, 2)

        w = h = cfg.model.vae.encoder_cfg.resolution
        vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
        w //= vae_scale_factor
        h //= vae_scale_factor

        model_kwargs = {
            "condition": input_latents,
            "metas": metas,
            "invisible_mask": stage1_invisible_mask,
        }
        maybe_add_bev_layout(model_kwargs, metas)

        diffusion = build_diffusion(cfg, args.num_sampling_steps)
        noise_shape = (
            input_occs.shape[0],
            end_frame,
            cfg.base_channel,
            w,
            h,
        )
        n_conds = cfg.sample.get("n_conds", 0)
        initial_cond_indices = list(range(n_conds)) if n_conds else None
        latents = diffusion.p_sample_loop(
            world_model,
            noise_shape,
            None,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device="cuda",
            initial_cond_indices=initial_cond_indices,
            initial_cond_frames=input_latents,
            controlnet=controlnet,
        )
    elapsed = time.time() - start
    print(f"[smoke] control latents shape: {tuple(latents.shape)}")
    print(f"[smoke] control smoke finished in {elapsed:.2f}s")


def main():
    args = parse_args()
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    os.chdir(ROOT)
    py_config = resolve_path(args.py_config)
    work_dir = resolve_path(args.work_dir)
    cfg = Config.fromfile(str(py_config))
    cfg.work_dir = str(work_dir)

    print(f"[smoke] repo root: {ROOT}")
    print(f"[smoke] config: {py_config}")
    print(f"[smoke] work_dir: {work_dir}")
    print(f"[smoke] mode: {args.mode}")
    print(f"[smoke] device: {torch.cuda.get_device_name(0)}")

    if args.mode == "world":
        run_world_smoke(cfg, work_dir, args)
    else:
        run_control_smoke(cfg, work_dir, args)


if __name__ == "__main__":
    main()
