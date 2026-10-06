#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = [
    {
        "setting": "manual semantic hard",
        "model": "I2-World",
        "scope": "full",
        "json": "outputs/ii_world_position_sweep/semantic_hard/position_sweep_semantic_hard_heatmap.json",
        "png": "outputs/ii_world_position_sweep/semantic_hard/position_sweep_semantic_hard_heatmap.png",
        "csv": "outputs/ii_world_position_sweep/semantic_hard/position_sweep_semantic_hard_relative_drop.csv",
    },
    {
        "setting": "manual semantic hard",
        "model": "OccWorld",
        "scope": "full",
        "json": "outputs/occworld_position_sweep/semantic_hard/position_sweep_semantic_hard_heatmap.json",
        "png": "outputs/occworld_position_sweep/semantic_hard/position_sweep_semantic_hard_heatmap.png",
        "csv": "outputs/occworld_position_sweep/semantic_hard/position_sweep_semantic_hard_relative_drop.csv",
    },
    {
        "setting": "manual semantic hard",
        "model": "COME",
        "scope": "full",
        "json": "outputs/come_position_sweep/semantic_hard/position_sweep_semantic_hard_heatmap.json",
        "png": "outputs/come_position_sweep/semantic_hard/position_sweep_semantic_hard_heatmap.png",
        "csv": "outputs/come_position_sweep/semantic_hard/position_sweep_semantic_hard_relative_drop.csv",
    },
    {
        "setting": "upstream camera_only/stcocc Snow hard",
        "model": "I2-World",
        "scope": "full",
        "json": "outputs/ii_world_position_sweep/stcocc_snow_hard/position_sweep_stcocc_snow_hard_heatmap.json",
        "png": "outputs/ii_world_position_sweep/stcocc_snow_hard/position_sweep_stcocc_snow_hard_heatmap.png",
        "csv": "outputs/ii_world_position_sweep/stcocc_snow_hard/position_sweep_stcocc_snow_hard_relative_drop.csv",
    },
    {
        "setting": "upstream camera_only/stcocc Snow hard",
        "model": "OccWorld",
        "scope": "full",
        "json": "outputs/occworld_position_sweep/stcocc_snow_hard/position_sweep_stcocc_snow_hard_heatmap.json",
        "png": "outputs/occworld_position_sweep/stcocc_snow_hard/position_sweep_stcocc_snow_hard_heatmap.png",
        "csv": "outputs/occworld_position_sweep/stcocc_snow_hard/position_sweep_stcocc_snow_hard_relative_drop.csv",
    },
    {
        "setting": "upstream camera_only/stcocc Snow hard",
        "model": "COME",
        "scope": "sample512",
        "json": "outputs/come_position_sweep/stcocc_snow_hard_sample512/position_sweep_stcocc_snow_hard_heatmap.json",
        "png": "outputs/come_position_sweep/stcocc_snow_hard_sample512/position_sweep_stcocc_snow_hard_heatmap.png",
        "csv": "outputs/come_position_sweep/stcocc_snow_hard_sample512/position_sweep_stcocc_snow_hard_relative_drop.csv",
    },
    {
        "setting": "upstream pointcloud_fusion/sdgocc snow heavy",
        "model": "I2-World",
        "scope": "full",
        "json": "outputs/ii_world_position_sweep/sdgocc_snow_heavy/position_sweep_sdgocc_snow_heavy_heatmap.json",
        "png": "outputs/ii_world_position_sweep/sdgocc_snow_heavy/position_sweep_sdgocc_snow_heavy_heatmap.png",
        "csv": "outputs/ii_world_position_sweep/sdgocc_snow_heavy/position_sweep_sdgocc_snow_heavy_relative_drop.csv",
    },
    {
        "setting": "upstream pointcloud_fusion/sdgocc snow heavy",
        "model": "OccWorld",
        "scope": "full",
        "json": "outputs/occworld_position_sweep/sdgocc_snow_heavy/position_sweep_sdgocc_snow_heavy_heatmap.json",
        "png": "outputs/occworld_position_sweep/sdgocc_snow_heavy/position_sweep_sdgocc_snow_heavy_heatmap.png",
        "csv": "outputs/occworld_position_sweep/sdgocc_snow_heavy/position_sweep_sdgocc_snow_heavy_relative_drop.csv",
    },
    {
        "setting": "upstream pointcloud_fusion/sdgocc snow heavy",
        "model": "COME",
        "scope": "sample512",
        "json": "outputs/come_position_sweep/sdgocc_snow_heavy_sample512/position_sweep_sdgocc_snow_heavy_heatmap.json",
        "png": "outputs/come_position_sweep/sdgocc_snow_heavy_sample512/position_sweep_sdgocc_snow_heavy_heatmap.png",
        "csv": "outputs/come_position_sweep/sdgocc_snow_heavy_sample512/position_sweep_sdgocc_snow_heavy_relative_drop.csv",
    },
]

AVERAGE_OUTPUTS = {
    "I2-World": {
        "dir": "outputs/ii_world_position_sweep/method_average",
        "json": "position_sweep_method_average_heatmap.json",
        "csv": "position_sweep_method_average_relative_drop.csv",
        "png": "position_sweep_method_average_heatmap.png",
    },
    "OccWorld": {
        "dir": "outputs/occworld_position_sweep/method_average",
        "json": "position_sweep_method_average_heatmap.json",
        "csv": "position_sweep_method_average_relative_drop.csv",
        "png": "position_sweep_method_average_heatmap.png",
    },
    "COME": {
        "dir": "outputs/come_position_sweep/method_average",
        "json": "position_sweep_method_average_heatmap.json",
        "csv": "position_sweep_method_average_relative_drop.csv",
        "png": "position_sweep_method_average_heatmap.png",
    },
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_relative_csv(path: Path, positions: list[str], horizons: list[str], relative_drop: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", *horizons])
        for position, row in zip(positions, relative_drop):
            writer.writerow([position, *row])


def plot_heatmap(
    data: dict,
    output_path: Path,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    positions = data["positions"]
    horizons = data["horizons"]
    matrix = np.array(data["relative_drop"], dtype=float).T
    matrix = matrix[::-1, :]
    plot_horizons = list(reversed(horizons))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(positions)), positions)
    ax.set_yticks(range(len(plot_horizons)), plot_horizons)
    ax.set_xlabel("Corrupted input position")
    ax.set_ylabel("Future prediction horizon")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("relative mIoU drop (%)")
    tick_step = 15 if vmax <= 90 else 20
    cbar.set_ticks(np.arange(vmin, vmax + 0.001, tick_step))

    threshold = vmin + (vmax - vmin) * 0.55
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            text_color = "white" if value >= threshold else "#222222"
            ax.text(x, y, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def global_vmax(datasets: list[dict]) -> float:
    max_value = max(float(np.nanmax(np.array(data["relative_drop"], dtype=float))) for data in datasets)
    return float(math.ceil(max_value / 10.0) * 10.0)


def average_for_model(model: str, model_cases: list[dict], datasets_by_path: dict[str, dict]) -> dict:
    source_data = [datasets_by_path[case["json"]] for case in model_cases]
    positions = source_data[0]["positions"]
    horizons = source_data[0]["horizons"]
    relative = np.array([data["relative_drop"] for data in source_data], dtype=float)
    absolute = np.array([data["absolute_drop"] for data in source_data], dtype=float)
    clean = np.array([data["clean_miou"] for data in source_data], dtype=float)

    return {
        "model": model,
        "corruption": "method_average",
        "severity": "mixed",
        "positions": positions,
        "horizons": horizons,
        "time_keys": source_data[0].get("time_keys"),
        "source_heatmaps": [
            {
                "setting": case["setting"],
                "scope": case["scope"],
                "path": case["json"],
            }
            for case in model_cases
        ],
        "clean_miou": [round(float(value), 6) for value in clean.mean(axis=0)],
        "relative_drop": [[round(float(value), 6) for value in row] for row in relative.mean(axis=0)],
        "absolute_drop": [[round(float(value), 6) for value in row] for row in absolute.mean(axis=0)],
        "caption": (
            "Average relative mIoU drop across the three representative position-sweep settings. "
            "COME averages one full manual setting and two sample512 upstream diagnostics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw position sweep heatmaps with a shared color scale.")
    parser.add_argument("--root", default=".", help="OccStress-code repository root.")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="YlOrRd")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets_by_path = {case["json"]: read_json(root / case["json"]) for case in CASES}
    averages = {
        model: average_for_model(model, [case for case in CASES if case["model"] == model], datasets_by_path)
        for model in AVERAGE_OUTPUTS
    }
    all_datasets = list(datasets_by_path.values()) + list(averages.values())
    vmax = args.vmax if args.vmax is not None else global_vmax(all_datasets)

    plot_config = {
        "shared_colorbar": True,
        "vmin": args.vmin,
        "vmax": vmax,
        "cmap": args.cmap,
        "plot_horizon_order": "t+6 to t+1, with t+1 at the bottom",
        "negative_values": "Annotated values are preserved; color values below vmin are clipped to the shared scale.",
    }

    for case in CASES:
        data = datasets_by_path[case["json"]]
        data["plot_config"] = plot_config
        write_json(root / case["json"], data)
        title = f"{case['model']} Position Sensitivity ({case['setting']})"
        if case["scope"] != "full":
            title += f" [{case['scope']}]"
        plot_heatmap(data, root / case["png"], title=title, vmin=args.vmin, vmax=vmax, cmap=args.cmap)

    for model, data in averages.items():
        data["plot_config"] = plot_config
        cfg = AVERAGE_OUTPUTS[model]
        out_dir = root / cfg["dir"]
        write_json(out_dir / cfg["json"], data)
        write_relative_csv(out_dir / cfg["csv"], data["positions"], data["horizons"], data["relative_drop"])
        plot_heatmap(
            data,
            out_dir / cfg["png"],
            title=f"{model} Average Position Sensitivity",
            vmin=args.vmin,
            vmax=vmax,
            cmap=args.cmap,
        )

    print(f"shared_vmin={args.vmin}")
    print(f"shared_vmax={vmax}")
    for case in CASES:
        print(f"wrote {root / case['png']}")
    for model, cfg in AVERAGE_OUTPUTS.items():
        print(f"wrote {root / cfg['dir'] / cfg['png']}")


if __name__ == "__main__":
    main()
