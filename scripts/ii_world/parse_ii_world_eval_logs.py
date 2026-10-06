#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def extract_dict(text: str) -> dict | None:
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if not matches:
        return None
    try:
        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return None


def extract_table_metric(text: str, label: str, column_idx: int) -> float | None:
    pattern = rf"\|\s*{re.escape(label)}\s*\|\s*([0-9eE+\-.]+)\s*\|\s*([0-9eE+\-.]+)\s*\|"
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(column_idx))


def infer_job_id(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.log$", path.name)
    if not match:
        return None
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tokenizer-log", required=True)
    parser.add_argument("--world-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-id", type=int)
    args = parser.parse_args()

    tokenizer_log = Path(args.tokenizer_log)
    world_log = Path(args.world_log)
    tok_text = tokenizer_log.read_text()
    world_text = world_log.read_text()
    world_dict = extract_dict(world_text) or {}

    result = {
        "protocol": args.protocol,
        "job": args.job_id if args.job_id is not None else infer_job_id(world_log),
        "tokenizer_samples": extract_int(r"mIoU of (\d+) samples:", tok_text),
        "tokenizer_miou": extract_float(r"mIoU of \d+ samples: ([0-9eE+\-.]+)", tok_text),
        "avg_miou": extract_table_metric(world_text, "Average", 1),
        "avg_iou": extract_table_metric(world_text, "Average", 2),
        "miou_0s": world_dict.get("semantics_miou_time_0s", world_dict.get("semantics_miou")),
        "iou_0s": world_dict.get("binary_iou_time_0s", world_dict.get("binary_iou")),
        "miou_1.0s": world_dict.get("semantics_miou_time_1.0s"),
        "iou_1.0s": world_dict.get("binary_iou_time_1.0s"),
        "miou_2.0s": world_dict.get("semantics_miou_time_2.0s"),
        "iou_2.0s": world_dict.get("binary_iou_time_2.0s"),
        "miou_3.0s": world_dict.get("semantics_miou_time_3.0s"),
        "iou_3.0s": world_dict.get("binary_iou_time_3.0s"),
        "tokenizer_log": str(tokenizer_log),
        "world_log": str(world_log),
    }

    horizon_miou = {}
    horizon_iou = {}
    for key, value in sorted(world_dict.items()):
        if key.startswith("semantics_miou_time_"):
            label = key[len("semantics_miou_time_"):]
            result[f"miou_{label}"] = value
            horizon_miou[label] = value
        elif key.startswith("binary_iou_time_"):
            label = key[len("binary_iou_time_"):]
            result[f"iou_{label}"] = value
            horizon_iou[label] = value
    if horizon_miou:
        result["horizon_miou"] = horizon_miou
    if horizon_iou:
        result["horizon_iou"] = horizon_iou

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
