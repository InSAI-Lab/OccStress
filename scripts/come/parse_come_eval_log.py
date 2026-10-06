#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def extract_list(pattern: str, text: str) -> list[float] | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return list(ast.literal_eval(match.group(1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    text = log_path.read_text()

    eval_section = text.split("Evaluate results!")[-1]

    result = {
        "protocol": args.protocol,
        "log_path": str(log_path),
        "current_val_iou": extract_list(r"Current val iou is (\[.*?\])", text),
        "current_val_miou": extract_list(r"Current val miou is (\[.*?\])", text),
        "avg_val_iou": extract_float(r"avg val iou is ([0-9eE+\-.]+)", text),
        "avg_val_miou": extract_float(r"avg val miou is ([0-9eE+\-.]+)", text),
        "avg_tc": extract_float(r"Avg TC: ([0-9eE+\-.]+)", text),
        "fid": extract_float(r"FID: ([0-9eE+\-.]+)", text),
        "mmd": extract_float(r"MMD: ([0-9eE+\-.]+)", text),
        "occ_iou": extract_float(r"\|\s*Occ_IoU\s*\|\s*([0-9eE+\-.]+)", eval_section),
        "miou": extract_float(r"\|\s*MIoU\s*\|\s*([0-9eE+\-.]+)", eval_section),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
