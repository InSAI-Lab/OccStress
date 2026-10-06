import argparse
import csv
import json
import math
import os
import random
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from mmdet.utils import setup_multi_processes
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze occ/latent geometry with official II-World weights')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--sample-limit', type=int, default=128)
    parser.add_argument('--strengths', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.04, 0.08])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_ann_file(cfg, override):
    if override is not None:
        return override
    if hasattr(cfg, 'data') and 'val' in cfg.data and 'ann_file' in cfg.data.val:
        return cfg.data.val.ann_file
    raise KeyError('Unable to resolve ann_file from config; pass --ann-file explicitly.')


def load_occ_semantics(occ_path: str) -> np.ndarray:
    occ_gt_label = os.path.join(occ_path, 'labels.npz')
    occ_labels = np.load(occ_gt_label)
    return occ_labels['semantics']


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(sorted_x):
        end = start + 1
        while end < len(sorted_x) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_corr(xs: List[float], ys: List[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 2:
        return float('nan')
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return float('nan')
    return float((rx * ry).sum() / denom)


def normalized_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.float() - b.float()
    return float(diff.pow(2).mean().sqrt().item())


def hamming_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    return float((a != b).float().mean().item())


def binary_iou_distance(a: np.ndarray, b: np.ndarray, empty_idx: int) -> float:
    valid = (a != 255) & (b != 255)
    a_occ = (a != empty_idx) & valid
    b_occ = (b != empty_idx) & valid
    union = np.logical_or(a_occ, b_occ).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a_occ, b_occ).sum()
    return float(1.0 - inter / union)


def semantic_miou_distance(a: np.ndarray, b: np.ndarray, num_classes: int) -> float:
    valid = (a != 255) & (b != 255)
    if valid.sum() == 0:
        return 0.0
    ious = []
    for cls_id in range(num_classes):
        a_mask = (a == cls_id) & valid
        b_mask = (b == cls_id) & valid
        union = np.logical_or(a_mask, b_mask).sum()
        if union == 0:
            continue
        inter = np.logical_and(a_mask, b_mask).sum()
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(1.0 - np.mean(ious))


def voxel_change_ratio(a: np.ndarray, b: np.ndarray) -> float:
    valid = (a != 255) & (b != 255)
    if valid.sum() == 0:
        return 0.0
    return float(((a != b) & valid).sum() / valid.sum())


def apply_block_dropout(curr_semantics: np.ndarray, strength: float, empty_idx: int, rng: np.random.RandomState) -> np.ndarray:
    perturbed = curr_semantics.copy()
    w, h, d = perturbed.shape
    total = max(1, int(round(strength * w * h * d)))
    side_xy = max(1, int(round(math.sqrt(max(1, total // max(d, 1))))))
    side_z = max(1, int(round(total / max(side_xy * side_xy, 1))))
    side_xy = min(side_xy, w, h)
    side_z = min(side_z, d)
    x0 = 0 if w == side_xy else rng.randint(0, w - side_xy + 1)
    y0 = 0 if h == side_xy else rng.randint(0, h - side_xy + 1)
    z0 = 0 if d == side_z else rng.randint(0, d - side_z + 1)
    perturbed[x0:x0 + side_xy, y0:y0 + side_xy, z0:z0 + side_z] = empty_idx
    return perturbed


def extract_current_only_features(model, voxel_semantics: np.ndarray) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    voxel_tensor = torch.from_numpy(voxel_semantics).long().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        curr_bev, _ = model.forward_encoder(voxel_tensor)
        vq = model.vq
        quant_in = vq.quant_conv(curr_bev)
        _, _, w, h = quant_in.shape

        shapes = []
        for i in range(vq.recover_stage):
            w_i = w // (2 ** (vq.recover_stage - i - 1))
            h_i = h // (2 ** (vq.recover_stage - i - 1))
            shapes.append((w_i, h_i))

        z_rest = quant_in.clone()
        z_hat = torch.zeros_like(z_rest)
        token_list = []

        for i in range(vq.recover_stage):
            z_rest_i = F.interpolate(z_rest, shapes[i], mode='bilinear', align_corners=False)
            z_q_i, _, (_, _, min_encoding_indices) = vq.forward_quantizer(
                z_rest_i, None, False, False, False)
            token_list.append(min_encoding_indices.reshape(-1).detach().cpu())
            z_q_i = F.interpolate(z_q_i, (w, h), mode='bilinear', align_corners=False)
            z_hat = z_hat + vq.recover_scale_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        for i in range(vq.recover_time):
            z_rest_i = z_rest
            z_q_i, _, (_, _, min_encoding_indices) = vq.forward_quantizer(
                z_rest_i, None, False, False, False)
            token_list.append(min_encoding_indices.reshape(-1).detach().cpu())
            z_hat = z_hat + vq.recover_time_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

        z_q = vq.post_quant_conv(z_hat)

    return dict(
        encoder=curr_bev.detach().cpu(),
        quantized=z_q.detach().cpu(),
        tokens=torch.cat(token_list, dim=0),
    )


def summarize_correlations(rows: List[Dict[str, float]], metric_key: str) -> Dict[str, float]:
    return dict(
        encoder_spearman=spearman_corr([r[metric_key] for r in rows], [r['encoder_l2'] for r in rows]),
        quantized_spearman=spearman_corr([r[metric_key] for r in rows], [r['quantized_l2'] for r in rows]),
        token_spearman=spearman_corr([r[metric_key] for r in rows], [r['token_hamming'] for r in rows]),
    )


def summarize_by_strength(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result = {}
    strengths = sorted({r['strength'] for r in rows})
    for strength in strengths:
        sub = [r for r in rows if r['strength'] == strength]
        result[f'{strength:.4f}'] = dict(
            count=len(sub),
            voxel_change_ratio=float(np.mean([r['voxel_change_ratio'] for r in sub])),
            binary_iou_distance=float(np.mean([r['binary_iou_distance'] for r in sub])),
            semantic_miou_distance=float(np.mean([r['semantic_miou_distance'] for r in sub])),
            encoder_l2=float(np.mean([r['encoder_l2'] for r in sub])),
            quantized_l2=float(np.mean([r['quantized_l2'] for r in sub])),
            token_hamming=float(np.mean([r['token_hamming'] for r in sub])),
        )
    return result


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)

    setup_multi_processes(cfg)
    set_seed(args.seed)

    ann_file = resolve_ann_file(cfg, args.ann_file)
    infos = mmcv.load(ann_file, file_format='pkl')['infos']
    infos = list(sorted(infos, key=lambda e: e['timestamp']))[:args.sample_limit]

    os.makedirs(args.out_dir, exist_ok=True)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.save_results = False
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this analysis script.')
    model = model.cuda()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    module = model.module

    rows = []
    rng = np.random.RandomState(args.seed)
    empty_idx = int(module.empty_idx)
    num_classes = int(module.num_classes)

    for sample_idx, info in enumerate(infos):
        base_occ = load_occ_semantics(info['occ_path']).astype(np.int64)
        base_feat = extract_current_only_features(module, base_occ)

        for strength in args.strengths:
            pert_occ = apply_block_dropout(base_occ, strength, empty_idx, rng)
            pert_feat = extract_current_only_features(module, pert_occ)
            rows.append(dict(
                sample_index=sample_idx,
                scene_name=str(info['occ_path']).split('/')[-2],
                sample_token=info['token'],
                strength=float(strength),
                voxel_change_ratio=voxel_change_ratio(base_occ, pert_occ),
                binary_iou_distance=binary_iou_distance(base_occ, pert_occ, empty_idx),
                semantic_miou_distance=semantic_miou_distance(base_occ, pert_occ, num_classes),
                encoder_l2=normalized_l2(base_feat['encoder'], pert_feat['encoder']),
                quantized_l2=normalized_l2(base_feat['quantized'], pert_feat['quantized']),
                token_hamming=hamming_ratio(base_feat['tokens'], pert_feat['tokens']),
            ))

    csv_path = os.path.join(args.out_dir, 'pair_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(
        source_commit='083911f97b66eceac6faf9c644a95660344687ac',
        checkpoint=os.path.abspath(args.checkpoint),
        ann_file=os.path.abspath(ann_file),
        sample_limit=args.sample_limit,
        strengths=[float(x) for x in args.strengths],
        total_pairs=len(rows),
        correlations=dict(
            voxel_change_ratio=summarize_correlations(rows, 'voxel_change_ratio'),
            binary_iou_distance=summarize_correlations(rows, 'binary_iou_distance'),
            semantic_miou_distance=summarize_correlations(rows, 'semantic_miou_distance'),
        ),
        by_strength=summarize_by_strength(rows),
    )

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
