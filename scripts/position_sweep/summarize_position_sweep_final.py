#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Build the final five-method position-sweep summary from V2 and V3 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


METHODS = ("II-World", "OccWorld", "COME", "DOME", "GenieDrive")
HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
POSITIONS = ("tminus4", "tminus3", "tminus2", "tminus1", "t")
COMMON_POSITIONS = ("tminus3", "tminus2", "tminus1", "t")
POSITION_SECONDS = {
    "tminus4": -2.0,
    "tminus3": -1.5,
    "tminus2": -1.0,
    "tminus1": -0.5,
    "t": 0.0,
}
KNOWN_INPUT_OFFSETS = {
    "II-World": (-2.0, -1.5, -1.0, -0.5, 0.0),
    "OccWorld": (-2.0, -1.5, -1.0, -0.5, 0.0),
    "COME": (-1.5, -1.0, -0.5, 0.0),
    "DOME": (-1.5, -1.0, -0.5, 0.0),
    "GenieDrive": (-1.5, -1.0, -0.5, 0.0),
}

SETTINGS = {
    "manual_dropout_hard": {
        "label": "Dropout-hard",
        "domain": "manual",
        "source": "v3",
        "source_slug": "dropout_hard",
    },
    "manual_hole_hard": {
        "label": "Hole-hard",
        "domain": "manual",
        "source": "v3",
        "source_slug": "hole_hard",
    },
    "manual_misalignment_hard": {
        "label": "Misalignment-hard",
        "domain": "manual",
        "source": "v3",
        "source_slug": "misalignment_hard",
    },
    "stcocc_brightness_hard": {
        "label": "Brightness-hard",
        "domain": "camera",
        "source": "v3",
        "source_slug": "stcocc_brightness_hard",
    },
    "stcocc_camera_crash_hard": {
        "label": "CameraCrash-hard",
        "domain": "camera",
        "source": "v2",
        "source_slug": "stcocc_camera_crash_hard",
    },
    "stcocc_color_quant_hard": {
        "label": "ColorQuant-hard",
        "domain": "camera",
        "source": "v3",
        "source_slug": "stcocc_color_quant_hard",
    },
    "stcocc_fog_hard": {
        "label": "Fog-hard",
        "domain": "camera",
        "source": "v3",
        "source_slug": "stcocc_fog_hard",
    },
    "stcocc_frame_lost_hard": {
        "label": "FrameLost-hard",
        "domain": "camera",
        "source": "v3",
        "source_slug": "stcocc_frame_lost_hard",
    },
    "stcocc_low_light_hard": {
        "label": "LowLight-hard",
        "domain": "camera",
        "source": "v3",
        "source_slug": "stcocc_low_light_hard",
    },
    "stcocc_motion_blur_hard": {
        "label": "MotionBlur-hard",
        "domain": "camera",
        "source": "v2",
        "source_slug": "stcocc_motion_blur_hard",
    },
}

MANUAL_BASELINES = {
    "II-World": (
        "outputs/ii_world_position_sweep/semantic_hard/"
        "position_sweep_semantic_hard_clean_H4_F6_val_backbone.json"
    ),
    "OccWorld": (
        "outputs/occworld_position_sweep/semantic_hard/"
        "position_sweep_semantic_hard_clean_H4_F6_val_backbone.json"
    ),
    "COME": "outputs/future_aligned_v1/come/manual/clean/H4_F6_val_backbone.json",
    "DOME": (
        "outputs/position_sweep_v2/results/DOME/p0/semantic_hard/"
        "position_sweep_semantic_hard_clean_H4_F6_val_backbone.json"
    ),
    "GenieDrive": (
        "outputs/geniedrive_occstress/outputs/occstress/manual/clean/"
        "H4_F6_val_backbone.json"
    ),
}


@dataclass(frozen=True)
class Metrics:
    miou: tuple[float, ...]
    iou: tuple[float, ...]


@dataclass(frozen=True)
class Baseline:
    method: str
    domain: str
    path: Path
    metrics: Metrics
    payload: dict[str, Any]


@dataclass(frozen=True)
class Evaluation:
    method: str
    setting: str
    position: str
    path: Path
    metrics: Metrics
    input_offsets: tuple[float, ...]
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--v2-root", type=Path, default=Path("outputs/position_sweep_v2/results")
    )
    parser.add_argument(
        "--v3-root", type=Path, default=Path("outputs/position_sweep_v3/results")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/position_sweep_final_20260825"),
    )
    parser.add_argument(
        "--unobserved-tolerance",
        type=float,
        default=0.30,
        help="Maximum absolute average-mIoU delta at an unobserved position.",
    )
    return parser.parse_args()


def finite_tuple(values: Iterable[Any], label: str, path: Path) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != len(HORIZONS):
        raise ValueError(f"{path}: {label} has {len(result)} values, expected 6")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{path}: {label} contains a non-finite value")
    return result


def extract_metrics(method: str, payload: dict[str, Any], path: Path) -> Metrics:
    if "horizon_metrics" in payload:
        horizon_metrics = payload["horizon_metrics"]
        miou = [horizon_metrics[str(horizon)]["miou"] for horizon in HORIZONS]
        iou = [horizon_metrics[str(horizon)]["iou"] for horizon in HORIZONS]
    elif method == "II-World":
        miou = [payload[f"miou_{horizon:.1f}s"] for horizon in HORIZONS]
        iou = [payload[f"iou_{horizon:.1f}s"] for horizon in HORIZONS]
    elif "current_val_miou" in payload and "current_val_iou" in payload:
        miou = payload["current_val_miou"][:6]
        iou = payload["current_val_iou"][:6]
    else:
        raise ValueError(f"{path}: unsupported metric schema for {method}")
    return Metrics(
        finite_tuple(miou, "mIoU", path), finite_tuple(iou, "IoU", path)
    )


def mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def relative_drops(
    baseline: tuple[float, ...], corrupted: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(
        (clean - value) / clean * 100.0
        for clean, value in zip(baseline, corrupted)
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def visible_json_candidates(root: Path, filename: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(filename)
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def find_result(root: Path, source_slug: str, position: str) -> Path:
    filename = (
        f"position_sweep_{source_slug}_{position}_H4_F6_val_backbone.json"
    )
    candidates = visible_json_candidates(root, filename)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {filename} below {root}, found {len(candidates)}: "
            f"{candidates}"
        )
    return candidates[0]


def input_offsets(method: str, payload: dict[str, Any]) -> tuple[float, ...]:
    values = (
        payload.get("observed_offsets_seconds")
        or payload.get("input_time_offsets_sec")
        or payload.get("input_times_seconds")
        or KNOWN_INPUT_OFFSETS[method]
    )
    result = tuple(sorted(float(value) for value in values))
    if result != KNOWN_INPUT_OFFSETS[method]:
        raise ValueError(
            f"{method}: input offsets {result}, expected {KNOWN_INPUT_OFFSETS[method]}"
        )
    return result


def validate_formal_result(
    method: str, payload: dict[str, Any], path: Path
) -> None:
    if payload.get("status") != "success":
        raise ValueError(f"{path}: status={payload.get('status')!r}")
    if payload.get("method") != method:
        raise ValueError(
            f"{path}: method={payload.get('method')!r}, expected {method}"
        )
    if payload.get("evaluated_records") != 4519:
        raise ValueError(
            f"{path}: evaluated_records={payload.get('evaluated_records')!r}"
        )
    if method == "COME" and (
        payload.get("adapter_version") != "come_occstress_future_aligned_v2"
    ):
        raise ValueError(f"{path}: missing COME future-aligned v2 adapter marker")
    future = payload.get("future_time_offsets_sec")
    if future is not None and tuple(float(value) for value in future) != HORIZONS:
        raise ValueError(f"{path}: future offsets are {future}, expected {HORIZONS}")


def load_baselines(repo_root: Path, v2_root: Path) -> dict[tuple[str, str], Baseline]:
    baselines: dict[tuple[str, str], Baseline] = {}
    for method in METHODS:
        manual_path = repo_root / MANUAL_BASELINES[method]
        manual_payload = load_json(manual_path)
        manual_metrics = extract_metrics(method, manual_payload, manual_path)
        baselines[(method, "manual")] = Baseline(
            method, "manual", manual_path, manual_metrics, manual_payload
        )

        camera_path = find_result(
            v2_root / method, "stcocc_motion_blur_hard", "clean"
        )
        camera_payload = load_json(camera_path)
        validate_formal_result(method, camera_payload, camera_path)
        camera_metrics = extract_metrics(method, camera_payload, camera_path)
        baselines[(method, "camera")] = Baseline(
            method, "camera", camera_path, camera_metrics, camera_payload
        )
    return baselines


def load_evaluations(v2_root: Path, v3_root: Path) -> list[Evaluation]:
    evaluations = []
    for method in METHODS:
        for setting, spec in SETTINGS.items():
            result_root = (v2_root if spec["source"] == "v2" else v3_root) / method
            for position in POSITIONS:
                path = find_result(result_root, spec["source_slug"], position)
                payload = load_json(path)
                validate_formal_result(method, payload, path)
                evaluations.append(
                    Evaluation(
                        method=method,
                        setting=setting,
                        position=position,
                        path=path,
                        metrics=extract_metrics(method, payload, path),
                        input_offsets=input_offsets(method, payload),
                        payload=payload,
                    )
                )
    return evaluations


def checksum(payload: dict[str, Any]) -> str:
    return str(
        payload.get("checkpoint_sha256")
        or payload.get("provenance", {}).get("checkpoint_sha256")
        or ""
    )


def build_baseline_rows(
    baselines: dict[tuple[str, str], Baseline]
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for domain in ("manual", "camera"):
            baseline = baselines[(method, domain)]
            row: dict[str, Any] = {
                "method": method,
                "domain": domain,
                "average_miou": mean(baseline.metrics.miou),
                "average_iou": mean(baseline.metrics.iou),
                "checkpoint_sha256": checksum(baseline.payload),
                "result_path": str(baseline.path),
            }
            for horizon, miou, iou in zip(
                HORIZONS, baseline.metrics.miou, baseline.metrics.iou
            ):
                tag = str(horizon).replace(".", "p")
                row[f"miou_{tag}s"] = miou
                row[f"iou_{tag}s"] = iou
            rows.append(row)
    return rows


def build_drop_rows(
    evaluations: list[Evaluation],
    baselines: dict[tuple[str, str], Baseline],
) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        spec = SETTINGS[evaluation.setting]
        baseline = baselines[(evaluation.method, spec["domain"])]
        miou_drops = relative_drops(
            baseline.metrics.miou, evaluation.metrics.miou
        )
        iou_drops = relative_drops(baseline.metrics.iou, evaluation.metrics.iou)
        position_seconds = POSITION_SECONDS[evaluation.position]
        row: dict[str, Any] = {
            "method": evaluation.method,
            "setting": evaluation.setting,
            "setting_label": spec["label"],
            "domain": spec["domain"],
            "result_generation": spec["source"],
            "position": evaluation.position,
            "position_seconds": position_seconds,
            "in_input_window": position_seconds in evaluation.input_offsets,
            "evaluated_records": 4519,
            "input_offsets_seconds": json.dumps(evaluation.input_offsets),
            "clean_average_miou": mean(baseline.metrics.miou),
            "corrupted_average_miou": mean(evaluation.metrics.miou),
            "absolute_average_miou_drop": (
                mean(baseline.metrics.miou) - mean(evaluation.metrics.miou)
            ),
            "mean_relative_miou_drop_pct": mean(miou_drops),
            "clean_average_iou": mean(baseline.metrics.iou),
            "corrupted_average_iou": mean(evaluation.metrics.iou),
            "absolute_average_iou_drop": (
                mean(baseline.metrics.iou) - mean(evaluation.metrics.iou)
            ),
            "mean_relative_iou_drop_pct": mean(iou_drops),
            "baseline_result_path": str(baseline.path),
            "result_path": str(evaluation.path),
        }
        for horizon, miou, iou, miou_drop, iou_drop in zip(
            HORIZONS,
            evaluation.metrics.miou,
            evaluation.metrics.iou,
            miou_drops,
            iou_drops,
        ):
            tag = str(horizon).replace(".", "p")
            row[f"miou_{tag}s"] = miou
            row[f"iou_{tag}s"] = iou
            row[f"relative_miou_drop_{tag}s_pct"] = miou_drop
            row[f"relative_iou_drop_{tag}s_pct"] = iou_drop
        rows.append(row)
    return rows


def validate_coverage(
    evaluations: list[Evaluation], drop_rows: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    expected = len(METHODS) * len(SETTINGS) * len(POSITIONS)
    if len(evaluations) != expected:
        raise ValueError(f"loaded {len(evaluations)} evaluations, expected {expected}")
    keys = {(item.method, item.setting, item.position) for item in evaluations}
    if len(keys) != expected:
        raise ValueError("duplicate evaluation keys detected")

    unobserved = [row for row in drop_rows if not row["in_input_window"]]
    if len(unobserved) != 30:
        raise ValueError(f"found {len(unobserved)} unobserved rows, expected 30")
    max_delta_row = max(
        unobserved, key=lambda row: abs(float(row["absolute_average_miou_drop"]))
    )
    max_delta = abs(float(max_delta_row["absolute_average_miou_drop"]))
    if max_delta > tolerance:
        raise ValueError(
            "unobserved-position clean parity failed: "
            f"{max_delta:.4f} pp > {tolerance:.4f} pp at "
            f"{max_delta_row['method']} / {max_delta_row['setting']}"
        )
    return {
        "formal_evaluations": expected,
        "records_per_evaluation": 4519,
        "unobserved_position_rows": len(unobserved),
        "unobserved_max_abs_average_miou_delta_pp": max_delta,
        "unobserved_max_delta_method": max_delta_row["method"],
        "unobserved_max_delta_setting": max_delta_row["setting"],
        "unobserved_tolerance_pp": tolerance,
    }


def common_rows(drop_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in drop_rows
        if row["position"] in COMMON_POSITIONS and row["in_input_window"]
    ]
    expected = len(METHODS) * len(SETTINGS) * len(COMMON_POSITIONS)
    if len(rows) != expected:
        raise ValueError(f"common-visible rows={len(rows)}, expected {expected}")
    return rows


def setting_summary(common: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in common:
        grouped[(row["method"], row["setting"])].append(row)
    rows = []
    for method in METHODS:
        for setting, spec in SETTINGS.items():
            values = grouped[(method, setting)]
            worst = max(values, key=lambda row: row["mean_relative_miou_drop_pct"])
            rows.append(
                {
                    "method": method,
                    "setting": setting,
                    "setting_label": spec["label"],
                    "domain": spec["domain"],
                    "common_visible_position_count": len(values),
                    "mean_relative_miou_drop_pct": mean(
                        row["mean_relative_miou_drop_pct"] for row in values
                    ),
                    "mean_relative_iou_drop_pct": mean(
                        row["mean_relative_iou_drop_pct"] for row in values
                    ),
                    "worst_position": worst["position"],
                    "worst_position_seconds": worst["position_seconds"],
                    "worst_relative_miou_drop_pct": worst[
                        "mean_relative_miou_drop_pct"
                    ],
                }
            )
    return rows


def method_summary(
    settings: list[dict[str, Any]],
    common: list[dict[str, Any]],
    baselines: dict[tuple[str, str], Baseline],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        method_settings = [row for row in settings if row["method"] == method]
        manual = [row for row in method_settings if row["domain"] == "manual"]
        camera = [row for row in method_settings if row["domain"] == "camera"]
        method_cells = [row for row in common if row["method"] == method]
        worst = max(
            method_cells, key=lambda row: row["mean_relative_miou_drop_pct"]
        )
        rows.append(
            {
                "method": method,
                "manual_clean_average_miou": mean(
                    baselines[(method, "manual")].metrics.miou
                ),
                "camera_clean_average_miou": mean(
                    baselines[(method, "camera")].metrics.miou
                ),
                "manual_3_setting_mean_relative_miou_drop_pct": mean(
                    row["mean_relative_miou_drop_pct"] for row in manual
                ),
                "camera_7_setting_mean_relative_miou_drop_pct": mean(
                    row["mean_relative_miou_drop_pct"] for row in camera
                ),
                "all_10_setting_mean_relative_miou_drop_pct": mean(
                    row["mean_relative_miou_drop_pct"] for row in method_settings
                ),
                "all_10_setting_mean_relative_iou_drop_pct": mean(
                    row["mean_relative_iou_drop_pct"] for row in method_settings
                ),
                "worst_setting": worst["setting"],
                "worst_setting_label": worst["setting_label"],
                "worst_position": worst["position"],
                "worst_position_seconds": worst["position_seconds"],
                "worst_relative_miou_drop_pct": worst[
                    "mean_relative_miou_drop_pct"
                ],
            }
        )
    return rows


def position_summary(common: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for position in COMMON_POSITIONS:
            values = [
                row
                for row in common
                if row["method"] == method and row["position"] == position
            ]
            rows.append(
                {
                    "method": method,
                    "position": position,
                    "position_seconds": POSITION_SECONDS[position],
                    "setting_count": len(values),
                    "mean_relative_miou_drop_pct": mean(
                        row["mean_relative_miou_drop_pct"] for row in values
                    ),
                    "mean_relative_iou_drop_pct": mean(
                        row["mean_relative_iou_drop_pct"] for row in values
                    ),
                }
            )
    return rows


def cross_method_setting_summary(
    settings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for setting, spec in SETTINGS.items():
        values = [row for row in settings if row["setting"] == setting]
        rows.append(
            {
                "setting": setting,
                "setting_label": spec["label"],
                "domain": spec["domain"],
                "method_count": len(values),
                "cross_method_mean_relative_miou_drop_pct": mean(
                    row["mean_relative_miou_drop_pct"] for row in values
                ),
                "cross_method_mean_relative_iou_drop_pct": mean(
                    row["mean_relative_iou_drop_pct"] for row in values
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: row["cross_method_mean_relative_miou_drop_pct"],
        reverse=True,
    )


def load_legacy_partial_rows(repo_root: Path) -> list[dict[str, Any]]:
    source = (
        repo_root
        / "outputs/position_sweep_v2/summary_final_20260822/extended_setting_summary.csv"
    )
    accepted_settings = {"semantic_hard", "stcocc_snow_hard", "sdgocc_snow_heavy"}
    rows = []
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] not in METHODS or row["setting"] not in accepted_settings:
                continue
            rows.append(
                {
                    "method": row["method"],
                    "setting": row["setting"],
                    "setting_label": row["setting_label"],
                    "clean_average_miou": float(row["clean_average_miou"]),
                    "visible_position_count": int(row["visible_position_count"]),
                    "visible_positions_seconds": row["visible_positions_seconds"],
                    "visible_mean_relative_miou_drop_pct": float(
                        row["visible_mean_relative_miou_drop_pct"]
                    ),
                    "worst_visible_position": row["worst_visible_position"],
                    "worst_visible_position_seconds": float(
                        row["worst_visible_position_seconds"]
                    ),
                    "worst_visible_relative_miou_drop_pct": float(
                        row["worst_visible_relative_miou_drop_pct"]
                    ),
                    "source_summary": str(source),
                    "included_in_main_10_setting_mean": False,
                }
            )
    if len(rows) != 5:
        raise ValueError(f"legacy partial rows={len(rows)}, expected 5 from {source}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap_norm(matrix: np.ndarray) -> Normalize:
    low = float(np.nanmin(matrix))
    high = float(np.nanmax(matrix))
    if low < 0.0 < high:
        extent = max(abs(low), abs(high))
        return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    return Normalize(vmin=min(0.0, low), vmax=max(1.0, high))


def plot_heatmap(
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    output_base: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    width = max(7.5, 0.82 * len(xlabels) + 3.0)
    height = max(4.0, 0.52 * len(ylabels) + 2.2)
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    norm = heatmap_norm(matrix)
    image = axis.imshow(matrix, cmap="RdYlBu_r", norm=norm, aspect="auto")
    axis.set_xticks(range(len(xlabels)))
    axis.set_xticklabels(xlabels, rotation=25 if len(xlabels) > 5 else 0, ha="right")
    axis.set_yticks(range(len(ylabels)))
    axis.set_yticklabels(ylabels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=12)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.58 else "white",
                fontsize=8.5,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85, pad=0.025)
    colorbar.set_label("Mean relative mIoU drop (%)")
    for suffix in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_figures(
    output_root: Path,
    settings: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    common: list[dict[str, Any]],
) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    setting_lookup = {
        (row["method"], row["setting"]): row["mean_relative_miou_drop_pct"]
        for row in settings
    }
    setting_matrix = np.asarray(
        [
            [setting_lookup[(method, setting)] for setting in SETTINGS]
            for method in METHODS
        ],
        dtype=np.float64,
    )
    plot_heatmap(
        setting_matrix,
        [spec["label"].replace("-hard", "") for spec in SETTINGS.values()],
        list(METHODS),
        "Position-sweep robustness by corruption",
        figure_root / "method_by_setting_miou_drop",
        "Hard corruption",
        "Method",
    )

    position_lookup = {
        (row["method"], row["position"]): row["mean_relative_miou_drop_pct"]
        for row in positions
    }
    position_matrix = np.asarray(
        [
            [position_lookup[(method, position)] for position in COMMON_POSITIONS]
            for method in METHODS
        ],
        dtype=np.float64,
    )
    plot_heatmap(
        position_matrix,
        ["t-1.5s", "t-1.0s", "t-0.5s", "t"],
        list(METHODS),
        "Mean robustness over 10 hard corruptions",
        figure_root / "method_by_position_miou_drop",
        "Corrupted observation position",
        "Method",
    )

    common_lookup = {
        (row["method"], row["setting"], row["position"]): row[
            "mean_relative_miou_drop_pct"
        ]
        for row in common
    }
    for method in METHODS:
        matrix = np.asarray(
            [
                [
                    common_lookup[(method, setting, position)]
                    for position in COMMON_POSITIONS
                ]
                for setting in SETTINGS
            ],
            dtype=np.float64,
        )
        plot_heatmap(
            matrix,
            ["t-1.5s", "t-1.0s", "t-0.5s", "t"],
            [spec["label"] for spec in SETTINGS.values()],
            f"{method}: position sensitivity",
            figure_root / f"{method.lower().replace('-', '_')}_setting_by_position",
            "Corrupted observation position",
            "Hard corruption",
        )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(
    output_root: Path,
    validation: dict[str, Any],
    methods: list[dict[str, Any]],
    settings: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    cross_settings: list[dict[str, Any]],
    legacy_partial: list[dict[str, Any]],
) -> None:
    ranked = sorted(
        methods, key=lambda row: row["all_10_setting_mean_relative_miou_drop_pct"]
    )
    position_lookup = {
        (row["method"], row["position"]): row["mean_relative_miou_drop_pct"]
        for row in positions
    }
    main_rows = [
        [
            row["method"],
            f'{row["manual_clean_average_miou"]:.2f}',
            f'{row["camera_clean_average_miou"]:.2f}',
            f'{row["manual_3_setting_mean_relative_miou_drop_pct"]:.2f}',
            f'{row["camera_7_setting_mean_relative_miou_drop_pct"]:.2f}',
            f'{row["all_10_setting_mean_relative_miou_drop_pct"]:.2f}',
            f'{row["worst_setting_label"]} @ {float(row["worst_position_seconds"]):g}s',
            f'{row["worst_relative_miou_drop_pct"]:.2f}',
        ]
        for row in methods
    ]
    position_rows = [
        [row["method"]]
        + [
            f'{position_lookup[(row["method"], position)]:.2f}'
            for position in COMMON_POSITIONS
        ]
        for row in methods
    ]
    setting_rows = [
        [
            row["setting_label"],
            row["domain"],
            *[
                f'{next(item["mean_relative_miou_drop_pct"] for item in settings if item["method"] == method and item["setting"] == row["setting"]):.2f}'
                for method in METHODS
            ],
            f'{row["cross_method_mean_relative_miou_drop_pct"]:.2f}',
        ]
        for row in cross_settings
    ]
    legacy_rows = [
        [
            row["method"],
            row["setting_label"],
            str(row["visible_position_count"]),
            f'{row["visible_mean_relative_miou_drop_pct"]:.2f}',
            f'{row["worst_visible_position_seconds"]:g}s',
            f'{row["worst_visible_relative_miou_drop_pct"]:.2f}',
        ]
        for row in legacy_partial
    ]
    current_mean = mean(
        row["mean_relative_miou_drop_pct"]
        for row in positions
        if row["position"] == "t"
    )
    oldest_mean = mean(
        row["mean_relative_miou_drop_pct"]
        for row in positions
        if row["position"] == "tminus3"
    )
    lines = [
        "# Final position-sweep results",
        "",
        "Status: **FINAL (five methods, SparseWorld excluded)**.",
        "",
        "The primary comparison covers 10 hard corruption settings shared by II-World, OccWorld, COME, DOME, and GenieDrive. It uses the four observation positions visible to every method: t-1.5s, t-1.0s, t-0.5s, and t. Results at t-2.0s are retained in the detailed CSV but excluded from cross-method means because COME, DOME, and GenieDrive do not observe that frame.",
        "",
        "Manual corruptions use each method's GT-clean baseline. Camera corruptions use the corresponding STCOcc-clean baseline. Relative drop is computed independently at each of the six future horizons and then averaged.",
        "",
        "## Validation",
        "",
        f"- {validation['formal_evaluations']}/250 formal corrupted evaluations are valid; every evaluation contains 4,519 records and six future horizons.",
        f"- The 30 unobserved t-2.0s checks pass clean parity; maximum absolute average-mIoU delta is {validation['unobserved_max_abs_average_miou_delta_pp']:.4f} pp (tolerance {validation['unobserved_tolerance_pp']:.2f} pp).",
        "- The main comparison contains 200 common-visible cells: 5 methods x 10 settings x 4 positions.",
        "",
        "## Main rebuttal table",
        "",
        markdown_table(
            [
                "Method",
                "GT clean mIoU",
                "STCOcc clean mIoU",
                "Manual-3 drop (%)",
                "Camera-7 drop (%)",
                "All-10 drop (%)",
                "Worst setting/position",
                "Worst drop (%)",
            ],
            main_rows,
        ),
        "",
        "## Mean drop by corrupted position",
        "",
        markdown_table(
            ["Method", "t-1.5s", "t-1.0s", "t-0.5s", "t"], position_rows
        ),
        "",
        "## Mean drop by corruption",
        "",
        markdown_table(
            [
                "Setting",
                "Domain",
                *METHODS,
                "Method mean",
            ],
            setting_rows,
        ),
        "",
        "## Descriptive findings",
        "",
        f"- {ranked[0]['method']} has the lowest 10-setting mean relative mIoU drop ({ranked[0]['all_10_setting_mean_relative_miou_drop_pct']:.2f}%), while {ranked[-1]['method']} has the highest ({ranked[-1]['all_10_setting_mean_relative_miou_drop_pct']:.2f}%). This is a robustness ranking, not an absolute-accuracy ranking.",
        f"- Across methods and settings, corrupting the current observation gives a {current_mean:.2f}% mean drop, compared with {oldest_mean:.2f}% at t-1.5s.",
        f"- The largest cross-method setting mean is {cross_settings[0]['setting_label']} ({cross_settings[0]['cross_method_mean_relative_miou_drop_pct']:.2f}%).",
        "- Misalignment is strongly method-dependent: DOME drops by 36.43% on average, whereas the other four methods range from -0.01% to 3.90%. The overall DOME ranking is therefore substantially influenced by this one protocol, which remains included in the primary mean.",
        "- These are deterministic point estimates over the fixed 4,519-anchor validation set; statistical intervals should be reported separately where available.",
        "",
        "## Validated legacy partial results",
        "",
        "Semantic-hard, STCOcc Snow-hard, and SDGOcc Snow-heavy were previously evaluated for only a subset of methods. The validated full-anchor results are retained below but are not mixed into the five-method 10-setting mean.",
        "",
        markdown_table(
            [
                "Method",
                "Setting",
                "Visible positions",
                "Mean drop (%)",
                "Worst position",
                "Worst drop (%)",
            ],
            legacy_rows,
        ),
        "",
        "## Files",
        "",
        "- `baseline_metrics.csv`: GT-clean and STCOcc-clean metrics used for normalization.",
        "- `all_position_drops.csv`: all 250 corrupted evaluations, including t-2.0s.",
        "- `common_visible_by_position.csv`: the 200 cells used in cross-method summaries.",
        "- `setting_summary.csv`: per-method, per-setting averages over four common positions.",
        "- `method_summary.csv`: the main five-method summary.",
        "- `position_summary.csv`: per-method temporal sensitivity averaged over 10 settings.",
        "- `cross_method_setting_summary.csv`: corruption difficulty averaged over methods.",
        "- `legacy_partial_setting_summary.csv`: validated earlier settings lacking five-method coverage.",
        "- `figures/`: aggregate and per-method PNG/PDF heatmaps.",
        "",
    ]
    (output_root / "REBUTTAL_TABLES.md").write_text("\n".join(lines))


def write_latex(output_root: Path, methods: list[dict[str, Any]]) -> None:
    lines = [
        "% Five-method final position-sweep summary.",
        "% Relative mIoU drops are averaged over six future horizons and four common-visible positions.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Method & GT clean & STCOcc clean & Manual-3 $\\downarrow$ & Camera-7 $\\downarrow$ & All-10 $\\downarrow$ & Worst $\\downarrow$ \\\\",
        "    \\midrule",
    ]
    for row in methods:
        lines.append(
            "    {method} & {manual_clean:.2f} & {camera_clean:.2f} & {manual:.2f} & {camera:.2f} & {overall:.2f} & {worst:.2f} \\\\".format(
                method=row["method"],
                manual_clean=row["manual_clean_average_miou"],
                camera_clean=row["camera_clean_average_miou"],
                manual=row["manual_3_setting_mean_relative_miou_drop_pct"],
                camera=row["camera_7_setting_mean_relative_miou_drop_pct"],
                overall=row["all_10_setting_mean_relative_miou_drop_pct"],
                worst=row["worst_relative_miou_drop_pct"],
            )
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Position-sweep robustness over 10 hard corruptions. Drops (\\%) are relative to the matching GT-clean or STCOcc-clean baseline and are averaged over six future horizons and the four observation positions visible to all methods.}",
            "  \\label{tab:position-sweep-final}",
            "\\end{table*}",
            "",
        ]
    )
    (output_root / "REBUTTAL_TABLES.tex").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    v2_root = (repo_root / args.v2_root).resolve() if not args.v2_root.is_absolute() else args.v2_root
    v3_root = (repo_root / args.v3_root).resolve() if not args.v3_root.is_absolute() else args.v3_root
    output_root = (
        (repo_root / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    baselines = load_baselines(repo_root, v2_root)
    evaluations = load_evaluations(v2_root, v3_root)
    baseline_rows = build_baseline_rows(baselines)
    drops = build_drop_rows(evaluations, baselines)
    validation = validate_coverage(evaluations, drops, args.unobserved_tolerance)
    common = common_rows(drops)
    settings = setting_summary(common)
    methods = method_summary(settings, common, baselines)
    positions = position_summary(common)
    cross_settings = cross_method_setting_summary(settings)
    legacy_partial = load_legacy_partial_rows(repo_root)

    write_csv(output_root / "baseline_metrics.csv", baseline_rows)
    write_csv(output_root / "all_position_drops.csv", drops)
    write_csv(output_root / "common_visible_by_position.csv", common)
    write_csv(output_root / "setting_summary.csv", settings)
    write_csv(output_root / "method_summary.csv", methods)
    write_csv(output_root / "position_summary.csv", positions)
    write_csv(
        output_root / "cross_method_setting_summary.csv", cross_settings
    )
    write_csv(
        output_root / "legacy_partial_setting_summary.csv", legacy_partial
    )
    write_figures(output_root, settings, positions, common)
    write_markdown(
        output_root,
        validation,
        methods,
        settings,
        positions,
        cross_settings,
        legacy_partial,
    )
    write_latex(output_root, methods)

    summary = {
        "status": "final",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "settings": list(SETTINGS),
        "positions": list(POSITIONS),
        "common_positions": list(COMMON_POSITIONS),
        "future_horizons_seconds": list(HORIZONS),
        "validation": validation,
        "method_summary": methods,
        "cross_method_setting_summary": cross_settings,
        "legacy_partial_setting_summary": legacy_partial,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
