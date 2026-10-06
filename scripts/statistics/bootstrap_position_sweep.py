#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Scene-clustered paired bootstrap for temporal position sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bootstrap_scene_metrics import (
    MODEL_NAMES,
    interval,
    protocol_score_features,
    score_weighted_features,
)


POSITIONS = ("t-4", "t-3", "t-2", "t-1", "t")
POSITION_SLUGS = {
    "t-4": "tminus4",
    "t-3": "tminus3",
    "t-2": "tminus2",
    "t-1": "tminus1",
    "t": "t",
}
SETTINGS = (
    "semantic_hard",
    "stcocc_snow_hard",
    "sdgocc_snow_heavy",
)


def find_position_file(root: Path, position: str) -> Path:
    slug = POSITION_SLUGS[position]
    matches = list(root.glob(f"position_sweep_*_{slug}_H4_F6_val_backbone.npz"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {position} file under {root}, found {matches}")
    return matches[0]


def find_clean_file(root: Path) -> Path:
    matches = list(root.glob("position_sweep_*_clean_H4_F6_val_backbone.npz"))
    if len(matches) != 1:
        raise ValueError(f"Expected one clean file under {root}, found {matches}")
    return matches[0]


def zero_policy(model: str, kind: str) -> str:
    if model == "ii_world" and kind == "semantic":
        return "exclude_zero"
    if model in ("occworld", "come"):
        return "absent_is_one"
    return "standard"


def load_features(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if data["semantic_hist"].shape != (150, 6, 18, 18):
            raise ValueError(f"Invalid semantic shape in {path}")
        if int(data["anchor_counts"].sum()) != 4519:
            raise ValueError(f"Invalid anchor count in {path}")
        semantic, binary = protocol_score_features(
            data["semantic_hist"],
            data["binary_hist"],
        )
        return data["scene_tokens"].astype(str).tolist(), semantic, binary


def format_ci(point: float, lower: float, upper: float) -> str:
    return f"{point:.2f} [{lower:.2f}, {upper:.2f}]"


def write_report(
    path: Path,
    rows: list[dict],
    averages: list[dict],
    contrasts: list[dict],
    metadata: dict,
    models: tuple[str, ...],
) -> None:
    lines = [
        "# Temporal Position-Sweep Uncertainty Analysis",
        "",
        "## Analysis Design",
        "",
        f"- Sampling unit: `{metadata['num_scenes']}` nuScenes validation scenes; `{metadata['num_anchors']}` paired anchors per protocol.",
        f"- Bootstrap: `{metadata['replicates']:,}` clustered paired resamples, seed `{metadata['seed']}`, percentile 95% confidence intervals.",
        "- Score: mIoU averaged at submission-native evaluator indices 1, 3, and 5 (physical 0.5/1.5/2.5 s for OccWorld/COME and 1/2/3 s for II-World).",
        "- Each cell below is `clean - single-position corrupted`; positive values indicate degradation.",
        "- Positions map to the five protocol inputs in chronological order: `t-4=-2.0 s`, `t-3=-1.5 s`, `t-2=-1.0 s`, `t-1=-0.5 s`, and `t=0 s`.",
        "- OccWorld and II-World observe all five positions. COME conditions on the four history states, so `t` is outside its model input and is an expected zero-effect control.",
        "",
    ]
    for setting in SETTINGS:
        lines.extend(
            [
                f"## {setting}",
                "",
                "| Model | t-4 | t-3 | t-2 | t-1 | t |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model in models:
            selected = {
                row["position"]: row
                for row in rows
                if row["score_type"] == "semantic"
                and row["setting"] == setting
                and row["model_key"] == model
            }
            cells = [
                format_ci(
                    selected[position]["drop"],
                    selected[position]["lower"],
                    selected[position]["upper"],
                )
                for position in POSITIONS
            ]
            lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## Average Across Three Fixed Settings",
            "",
            "| Model | t-4 | t-3 | t-2 | t-1 | t |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in models:
        selected = {
            row["position"]: row
            for row in averages
            if row["score_type"] == "semantic" and row["model_key"] == model
        }
        cells = [
            format_ci(
                selected[position]["drop"],
                selected[position]["lower"],
                selected[position]["upper"],
            )
            for position in POSITIONS
        ]
        lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Pre-Specified Position Contrasts",
            "",
            "`t - mean(t-4...t-1)` compares current-state sensitivity with mean historical sensitivity. Positive values mean current corruption is more damaging.",
            "",
            "| Model | Contrast | Difference [95% CI] | P(>0) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['model']} | {row['contrast']} | "
            f"{format_ci(row['point'], row['lower'], row['upper'])} | "
            f"{row['probability_positive']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The three corruption settings were fixed before this analysis; no post-hoc family selection was performed.",
            "- Intervals quantify scene-sampling uncertainty. They do not include manual corruption-instantiation variance because only one released semantic-hard realization is available.",
            f"- Full semantic and binary interval CSV: `{metadata['csv_path']}`",
            f"- Machine-readable summary: `{metadata['json_path']}`",
            f"- Bootstrap replicate archive: `{metadata['replicate_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_NAMES),
        default=list(MODEL_NAMES),
    )
    args = parser.parse_args()
    models = tuple(args.models)

    loaded = {}
    reference_scenes = None
    for model in models:
        loaded[model] = {}
        for setting in SETTINGS:
            setting_root = args.stats_root / model / setting
            loaded[model][setting] = {}
            for position, path in [
                ("clean", find_clean_file(setting_root)),
                *[
                    (position, find_position_file(setting_root, position))
                    for position in POSITIONS
                ],
            ]:
                scenes, semantic, binary = load_features(path)
                if reference_scenes is None:
                    reference_scenes = scenes
                elif scenes != reference_scenes:
                    raise ValueError(f"Scene order mismatch in {path}")
                loaded[model][setting][position] = {
                    "semantic": semantic,
                    "binary": binary,
                    "path": str(path.resolve()),
                }
    assert reference_scenes is not None

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(
        0,
        len(reference_scenes),
        size=(args.replicates, len(reference_scenes)),
        dtype=np.int16,
    )
    weights = np.zeros((args.replicates, len(reference_scenes)), dtype=np.int16)
    bootstrap_rows = np.repeat(np.arange(args.replicates), len(reference_scenes))
    np.add.at(weights, (bootstrap_rows, draws.reshape(-1)), 1)
    all_scenes = np.ones((1, len(reference_scenes)), dtype=np.int16)

    rows = []
    replicate_drops = {}
    for model in models:
        for setting in SETTINGS:
            for kind in ("semantic", "binary"):
                policy = zero_policy(model, kind)
                position_points = {}
                position_replicates = {}
                for position in ("clean", *POSITIONS):
                    features = loaded[model][setting][position][kind][None]
                    position_points[position] = score_weighted_features(
                        all_scenes,
                        features,
                        zero_policy=policy,
                    )[0, 0]
                    scores = np.empty(args.replicates, dtype=np.float64)
                    for start in range(0, args.replicates, args.batch_size):
                        stop = min(start + args.batch_size, args.replicates)
                        scores[start:stop] = score_weighted_features(
                            weights[start:stop],
                            features,
                            zero_policy=policy,
                        )[:, 0]
                    position_replicates[position] = scores

                for position in POSITIONS:
                    values = (
                        position_replicates["clean"]
                        - position_replicates[position]
                    )
                    point = float(
                        position_points["clean"] - position_points[position]
                    )
                    lower, upper = interval(values)
                    key = f"{model}_{setting}_{kind}_{POSITION_SLUGS[position]}"
                    replicate_drops[key] = values
                    rows.append(
                        {
                            "model_key": model,
                            "model": MODEL_NAMES[model],
                            "setting": setting,
                            "score_type": kind,
                            "position": position,
                            "clean_score": float(position_points["clean"]),
                            "corrupted_score": float(position_points[position]),
                            "drop": point,
                            "lower": lower,
                            "upper": upper,
                            "probability_positive": float(np.mean(values > 0)),
                        }
                    )

    averages = []
    for model in models:
        for kind in ("semantic", "binary"):
            for position in POSITIONS:
                selected = [
                    row
                    for row in rows
                    if row["model_key"] == model
                    and row["score_type"] == kind
                    and row["position"] == position
                ]
                keys = [
                    f"{model}_{row['setting']}_{kind}_{POSITION_SLUGS[position]}"
                    for row in selected
                ]
                values = np.stack(
                    [replicate_drops[key] for key in keys],
                    axis=1,
                ).mean(axis=1)
                lower, upper = interval(values)
                averages.append(
                    {
                        "model_key": model,
                        "model": MODEL_NAMES[model],
                        "score_type": kind,
                        "position": position,
                        "drop": float(np.mean([row["drop"] for row in selected])),
                        "lower": lower,
                        "upper": upper,
                        "probability_positive": float(np.mean(values > 0)),
                    }
                )

    contrasts = []
    for model in models:
        current = next(
            row
            for row in averages
            if row["model_key"] == model
            and row["score_type"] == "semantic"
            and row["position"] == "t"
        )
        history = [
            row
            for row in averages
            if row["model_key"] == model
            and row["score_type"] == "semantic"
            and row["position"] != "t"
        ]
        current_values = np.mean(
            [
                replicate_drops[
                    f"{model}_{setting}_semantic_t"
                ]
                for setting in SETTINGS
            ],
            axis=0,
        )
        history_values = np.mean(
            [
                replicate_drops[
                    f"{model}_{setting}_semantic_{POSITION_SLUGS[position]}"
                ]
                for setting in SETTINGS
                for position in POSITIONS[:-1]
            ],
            axis=0,
        )
        values = current_values - history_values
        lower, upper = interval(values)
        contrasts.append(
            {
                "model": MODEL_NAMES[model],
                "contrast": "t - mean(t-4...t-1)",
                "point": float(
                    current["drop"] - np.mean([row["drop"] for row in history])
                ),
                "lower": lower,
                "upper": upper,
                "probability_positive": float(np.mean(values > 0)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "position_sweep_clustered_intervals.csv"
    json_path = args.output_dir / "position_sweep_clustered_summary.json"
    replicate_path = args.output_dir / "position_sweep_clustered_replicates.npz"
    report_path = args.output_dir / "POSITION_SWEEP_UNCERTAINTY_ANALYSIS.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "method": "scene-level clustered paired percentile bootstrap",
        "num_scenes": len(reference_scenes),
        "num_anchors": 4519,
        "replicates": args.replicates,
        "seed": args.seed,
        "models": list(models),
        "stats_root": str(args.stats_root.resolve()),
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "replicate_path": str(replicate_path.resolve()),
    }
    json_path.write_text(
        json.dumps(
            {
                **metadata,
                "protocol_results": rows,
                "average_results": averages,
                "contrasts": contrasts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        replicate_path,
        scene_tokens=np.asarray(reference_scenes),
        scene_weights=weights,
        **replicate_drops,
    )
    write_report(report_path, rows, averages, contrasts, metadata, models)
    print(report_path)


if __name__ == "__main__":
    main()
