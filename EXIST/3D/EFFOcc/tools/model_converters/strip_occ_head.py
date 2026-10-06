#!/usr/bin/env python3
"""Drop dataset-specific occupancy heads from an EFFOcc checkpoint."""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    kept = state_dict.__class__()
    removed = []
    for key, value in state_dict.items():
        normalized = key.removeprefix("module.")
        if normalized.startswith("occ_head."):
            removed.append(key)
        else:
            kept[key] = value
    if hasattr(state_dict, "_metadata"):
        kept._metadata = state_dict._metadata.copy()
    if not removed:
        raise RuntimeError("checkpoint contains no occ_head parameters")
    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = kept
        checkpoint.setdefault("meta", {})["stripped_occ_head"] = removed
        output = checkpoint
    else:
        output = kept
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + f".{os.getpid()}.tmp"
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(f"Removed {len(removed)} tensors; kept {len(kept)}")


if __name__ == "__main__":
    main()
