#!/usr/bin/env python3
import argparse
import importlib
import os
from pathlib import Path

from mmcv import Config


def parse_args():
    repo_root = Path(__file__).resolve().parents[2] / "EXIST" / "3D" / "SDGOCC"
    parser = argparse.ArgumentParser(
        description="Build SDGOCC config/dataset/model and optionally fetch one sample")
    parser.add_argument(
        "--config",
        default=str(repo_root / "projects" / "configs" / "sdgocc" /
                    "sdgocc-r50-4d-stereo.py"),
        help="config file to validate")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="dataset split to build")
    parser.add_argument(
        "--load-interval",
        type=int,
        default=2048,
        help="dataset load_interval override for lightweight smoke checks")
    parser.add_argument(
        "--fetch-sample",
        action="store_true",
        help="materialize dataset[0] after building metadata")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if getattr(cfg, "plugin", False):
        importlib.import_module("projects.mmdet3d_plugin")

    split_cfg = cfg.data[args.split]
    split_cfg.load_interval = args.load_interval
    cfg.data.workers_per_gpu = 0
    cfg.data.samples_per_gpu = 1

    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    dataset = build_dataset(split_cfg)
    print(f"split={args.split}")
    print(f"dataset_type={type(dataset).__name__}")
    print(f"dataset_len={len(dataset)}")
    print(f"ann_file={split_cfg.ann_file}")
    print(f"data_root={split_cfg.data_root}")

    model = build_model(cfg.model)
    print(f"model_type={type(model).__name__}")

    if args.fetch_sample:
        sample = dataset[0]
        keys = sorted(sample.keys())
        print(f"sample_keys={keys}")


if __name__ == "__main__":
    main()
