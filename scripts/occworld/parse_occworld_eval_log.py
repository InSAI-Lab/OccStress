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

    result = {
        "protocol": args.protocol,
        "log_path": str(log_path),
        "current_val_iou": extract_list(r"Current val iou is (\[.*?\])", text),
        "current_val_miou": extract_list(r"Current val miou is (\[.*?\])", text),
        "avg_val_iou": extract_float(r"avg val iou is ([0-9eE+\-.]+)", text),
        "avg_val_miou": extract_float(r"avg val miou is ([0-9eE+\-.]+)", text),
        "avg_l2": extract_float(r"avg_l2 is ([0-9eE+\-.]+)", text),
        "avg_obj_col": extract_float(r"avg_obj_col is ([0-9eE+\-.]+)", text),
        "avg_obj_box_col": extract_float(r"avg_obj_box_col is ([0-9eE+\-.]+)", text),
        "avg_l2_single": extract_float(r"avg_l2_single is ([0-9eE+\-.]+)", text),
        "avg_obj_col_single": extract_float(r"avg_obj_col_single is ([0-9eE+\-.]+)", text),
        "avg_obj_box_col_single": extract_float(r"avg_obj_box_col_single is ([0-9eE+\-.]+)", text),
        "fps": extract_float(r"FPS is ([0-9eE+\-.]+)", text),
        "plan_reg_loss": extract_float(r"PlanRegLoss is ([0-9eE+\-.]+)", text),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
