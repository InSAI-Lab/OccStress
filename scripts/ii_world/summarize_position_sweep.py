#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


POSITIONS = ["t-4", "t-3", "t-2", "t-1", "t"]
SLUGS = {"t-4": "tminus4", "t-3": "tminus3", "t-2": "tminus2", "t-1": "tminus1", "t": "t"}
HORIZONS = ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"]
TIME_KEYS = ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]


def load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def miou_vector(result: dict) -> list[float]:
    direct = result.get("current_val_miou")
    if isinstance(direct, list) and len(direct) >= len(HORIZONS):
        return [float(value) for value in direct[:len(HORIZONS)]]

    values = []
    for key in TIME_KEYS:
        value = result.get(f"miou_{key}")
        if value is None:
            value = (result.get("horizon_miou") or {}).get(key)
        if value is None:
            raise KeyError(f"Missing horizon mIoU {key} in {result.get('protocol')}")
        values.append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize position sweep results and draw a heatmap.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-heatmap", required=True)
    parser.add_argument("--corruption", default="semantic")
    parser.add_argument("--severity", default="hard")
    parser.add_argument("--model", default="I2-World")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    prefix = f"position_sweep_{args.corruption}_{args.severity}"
    clean = miou_vector(load_result(result_dir / f"{prefix}_clean_H4_F6_val_backbone.json"))
    corr = {}
    for position in POSITIONS:
        corr[position] = miou_vector(
            load_result(result_dir / f"{prefix}_{SLUGS[position]}_H4_F6_val_backbone.json")
        )

    absolute_drop = []
    relative_drop = []
    for position in POSITIONS:
        abs_row = [round(c - v, 6) for c, v in zip(clean, corr[position])]
        rel_row = [
            round(((c - v) / c * 100.0) if c else 0.0, 6)
            for c, v in zip(clean, corr[position])
        ]
        absolute_drop.append(abs_row)
        relative_drop.append(rel_row)

    output = {
        "model": args.model,
        "corruption": args.corruption,
        "severity": args.severity,
        "positions": POSITIONS,
        "horizons": HORIZONS,
        "time_keys": TIME_KEYS,
        "clean_miou": clean,
        "corr_miou": corr,
        "relative_drop": relative_drop,
        "absolute_drop": absolute_drop,
        "caption": (
            "We corrupt one input state at a time and measure the relative mIoU drop "
            "at each future horizon. Darker cells indicate stronger temporal error propagation."
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", *HORIZONS])
        for position, row in zip(POSITIONS, relative_drop):
            writer.writerow([position, *row])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(relative_drop, dtype=float).T
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(POSITIONS)), POSITIONS)
    ax.set_yticks(range(len(HORIZONS)), HORIZONS)
    ax.set_xlabel("Corrupted input position")
    ax.set_ylabel("Future prediction horizon")
    ax.set_title(f"Temporal Position Sensitivity of {args.model}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("relative mIoU drop (%)")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            text_color = "white" if matrix[y, x] >= matrix.max() * 0.55 else "#222222"
            ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center", color=text_color, fontsize=8)
    output_heatmap = Path(args.output_heatmap)
    output_heatmap.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_heatmap, dpi=220)
    plt.close(fig)
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")
    print(f"wrote {output_heatmap}")


if __name__ == "__main__":
    main()
