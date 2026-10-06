import argparse
import os

import torch
from mmengine import Config
from mmengine.registry import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.config)
    if args.batch_size is not None:
        cfg.train_loader.batch_size = args.batch_size
        cfg.val_loader.batch_size = args.batch_size
    cfg.train_loader.num_workers = args.num_workers
    cfg.val_loader.num_workers = args.num_workers

    import model  # noqa: F401
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    train_loader, _ = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
    )

    input_occs, target_occs, metas = next(iter(train_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_occs = input_occs.to(device, non_blocking=True)
    target_occs = target_occs.to(device, non_blocking=True)

    model_instance = MODELS.build(cfg.model)
    model_instance.init_weights()
    model_instance.train().to(device)

    result_dict = model_instance(x=input_occs, metas=metas)
    loss_input = {
        "inputs": input_occs,
        "target_occs": target_occs,
    }
    for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
        loss_input[loss_input_key] = result_dict[loss_input_val]

    loss_func = OPENOCC_LOSS.build(cfg.loss).to(device)
    loss, loss_dict = loss_func(loss_input)
    loss.backward()

    allocated = 0.0
    reserved = 0.0
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    print(f"CONFIG={args.config}")
    print(f"BATCH_SIZE={cfg.train_loader.batch_size}")
    print(f"INPUT_SHAPE={tuple(input_occs.shape)}")
    print(f"TARGET_SHAPE={tuple(target_occs.shape)}")
    print(f"RESULT_KEYS={sorted(result_dict.keys())}")
    print(f"LOSS={loss.item():.6f}")
    print(f"LOSS_DICT={loss_dict}")
    print(f"CUDA_MAX_ALLOCATED_GB={allocated:.3f}")
    print(f"CUDA_MAX_RESERVED_GB={reserved:.3f}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
