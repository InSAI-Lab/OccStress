#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.history_fusion.visualize_occ_slot_builder_debug import build_occ_slots
from experiments.history_fusion.visualize_occ_slot_vote import _majority_vote_occ
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export no-history latent tokens for 5-frame occ-vote priors.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--save-root', required=True)
    parser.add_argument('--flow-root', default='data/nuscenes/occ_flow_gt')
    parser.add_argument('--history', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--require-flow', action='store_true', default=True)
    parser.add_argument('--max-fallback-ratio', type=float, default=0.2)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def distributed_info(args):
    if args.launcher == 'none':
        return 0, 1, 0
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    return rank, world_size, local_rank


def scene_name(info):
    return str(info.get('scene_name', os.path.basename(os.path.dirname(info['occ_path']))))


def sample_idx(info):
    return str(info.get('sample_idx', info['token']))


def token_path(save_root, info):
    return os.path.join(save_root, 'token_4f', scene_name(info), f'{sample_idx(info)}.npz')


def load_occ(info):
    occ_path = Path(info['occ_path'])
    if occ_path.suffix == '.npz':
        path = occ_path
    else:
        path = occ_path / 'labels.npz'
    return np.load(path)['semantics'].astype(np.uint8)


def trace_history(info_by_token, info, history):
    chain = [info]
    prev = info.get('prev')
    while prev and prev in info_by_token and len(chain) < history + 1:
        current = info_by_token[prev]
        chain.append(current)
        prev = current.get('prev')
    chain.reverse()
    return chain


def support_to_latent_confidence(support, out_hw):
    out_h, out_w = out_hw
    support_bev = support.max(axis=2).astype(np.float32) / 5.0
    in_h, in_w = support_bev.shape
    kernel_h = max(1, in_h // out_h)
    kernel_w = max(1, in_w // out_w)
    cropped = support_bev[:out_h * kernel_h, :out_w * kernel_w]
    pooled = cropped.reshape(out_h, kernel_h, out_w, kernel_w).mean(axis=(1, 3))
    return pooled[None].astype(np.float16)


def build_vote_occ(info_by_token, info, history, flow_root):
    curr_occ = load_occ(info)
    chain = trace_history(info_by_token, info, history)
    if len(chain) < history + 1:
        support = np.full(curr_occ.shape, 5, dtype=np.uint8)
        return curr_occ, support, True
    occ_seq = [load_occ(item) for item in chain]
    try:
        slots = build_occ_slots(chain, occ_seq, flow_root)
        slot_occ = np.stack(slots['scatter_slots'], axis=0).astype(np.uint8)
        voted_occ, support, _ = _majority_vote_occ(curr_occ, slot_occ)
        return voted_occ.astype(np.uint8), support.astype(np.uint8), False
    except Exception as exc:
        print(f'[warn] fallback current token={info.get("token")} reason={type(exc).__name__}: {exc}', flush=True)
        support = np.full(curr_occ.shape, 5, dtype=np.uint8)
        return curr_occ, support, True


def save_batch(model, batch_items, save_root, device):
    vote_occs = [item['vote_occ'] for item in batch_items]
    voxel = torch.from_numpy(np.stack(vote_occs)).to(device=device, dtype=torch.long).unsqueeze(1)
    with torch.no_grad():
        curr_bev, _ = model.forward_encoder(voxel)
        sampled_bev = curr_bev.new_zeros((curr_bev.shape[0], model.frame_number, *curr_bev.shape[1:]))
        z_sampled, _, _ = model.vq(curr_bev, sampled_bev, is_voxel=False)
    z_np = z_sampled.detach().cpu().numpy().astype(np.float32)

    for idx, item in enumerate(batch_items):
        out_path = token_path(save_root, item['info'])
        mmcv.mkdir_or_exist(os.path.dirname(out_path))
        confidence = support_to_latent_confidence(item['support'], z_np[idx].shape[-2:])
        np.savez(out_path, token=z_np[idx], vote_confidence=confidence)


def main():
    args = parse_args()
    rank, world_size, local_rank = distributed_info(args)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.save_results = False
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(device)
    model.eval()

    if args.require_flow:
        flow_root = Path(args.flow_root)
        has_flow = flow_root.exists() and any(flow_root.glob('*/*.npz'))
        if not has_flow:
            raise RuntimeError(f'Flow root is empty or missing: {args.flow_root}')

    data = mmcv.load(args.ann_file)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    info_by_token = {info['token']: info for info in infos}
    assigned = [(idx, info) for idx, info in enumerate(infos) if idx % world_size == rank]

    if rank == 0:
        print(
            f'[vote-export] ann={args.ann_file} total={len(infos)} world={world_size} '
            f'save_root={args.save_root} flow_root={args.flow_root}',
            flush=True,
        )

    pending = []
    done = 0
    skipped = 0
    fallback = 0
    processed = 0
    for local_idx, (global_idx, info) in enumerate(assigned, start=1):
        out_path = token_path(args.save_root, info)
        if os.path.exists(out_path):
            skipped += 1
            continue
        vote_occ, support, used_fallback = build_vote_occ(info_by_token, info, args.history, args.flow_root)
        fallback += int(used_fallback)
        processed += 1
        pending.append(dict(info=info, vote_occ=vote_occ, support=support))
        if len(pending) >= args.batch_size:
            save_batch(model, pending, args.save_root, device)
            done += len(pending)
            pending = []
        if local_idx % args.log_interval == 0:
            print(
                f'[vote-export][rank {rank}] local={local_idx}/{len(assigned)} '
                f'global_idx={global_idx} written={done} skipped={skipped} fallback={fallback}',
                flush=True,
            )

    if pending:
        save_batch(model, pending, args.save_root, device)
        done += len(pending)

    fallback_ratio = float(fallback) / float(max(processed, 1))
    print(
        f'[vote-export][rank {rank}] complete written={done} skipped={skipped} '
        f'fallback={fallback} fallback_ratio={fallback_ratio:.4f}',
        flush=True,
    )
    if processed > 0 and fallback_ratio > args.max_fallback_ratio:
        raise RuntimeError(
            f'Fallback ratio {fallback_ratio:.4f} exceeded max_fallback_ratio={args.max_fallback_ratio:.4f}. '
            'This usually means occ_flow_gt is incomplete.'
        )


if __name__ == '__main__':
    main()
