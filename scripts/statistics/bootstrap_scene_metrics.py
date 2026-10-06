#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Clustered paired bootstrap analysis for OccStress scene statistics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_NAMES = {
    "occworld": "OccWorld",
    "come": "COME",
    "ii_world": "II-World",
}
METRIC_ORDER = (
    "cs_i",
    "cs_p",
    "cs_o",
    "rs_i",
    "rs_p",
    "rs_o",
    "current",
    "burst",
    "history",
    "ucr",
    "mrs",
    "crf2",
)
METRIC_LABELS = {
    "cs_i": "CS-I",
    "cs_p": "CS-P",
    "cs_o": "CS-O",
    "rs_i": "RS-I",
    "rs_p": "RS-P",
    "rs_o": "RS-O",
    "current": "Current",
    "burst": "Burst",
    "history": "History",
    "ucr": "UCR",
    "mrs": "mRS",
    "crf2": "CRF-2",
}
MODE_LABELS = {
    "current_only": "current",
    "recent_burst": "burst",
    "history_only": "history",
}


@dataclass(frozen=True)
class Protocol:
    relative_path: str
    source: str
    family: str
    severity: str | None
    mode: str | None
    clean: bool
    traffic: bool


def classify(relative_path: str) -> Protocol:
    parts = Path(relative_path).parts
    filename = parts[-1]
    clean = "clean" in parts
    mode = next((value for key, value in MODE_LABELS.items() if filename.startswith(key)), None)
    if parts[0] == "manual":
        source = "o"
        traffic = len(parts) > 1 and parts[1] == "traffic"
        family = parts[1]
        severity = parts[2] if not clean and not traffic else None
    elif parts[:3] == ("upstream", "camera_only", "stcocc"):
        source = "i"
        traffic = False
        family = parts[3]
        severity = parts[4] if not clean else None
    elif parts[:3] == ("upstream", "pointcloud_fusion", "sdgocc"):
        source = "p"
        traffic = False
        family = parts[3]
        severity = parts[4] if not clean else None
    else:
        raise ValueError(f"Unknown protocol layout: {relative_path}")
    return Protocol(
        relative_path,
        source,
        family,
        severity,
        mode,
        clean,
        traffic,
    )


def family_groups(protocols: list[Protocol]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, protocol in enumerate(protocols):
        if protocol.clean:
            continue
        key = f"{protocol.source.upper()}:{protocol.family}"
        groups.setdefault(key, []).append(index)
    return groups


def protocol_score_features(
    semantic_hist: np.ndarray,
    binary_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    horizon_indices = np.asarray([1, 3, 5])
    semantic_hist = semantic_hist[:, horizon_indices]
    seen = semantic_hist[..., :17, :].sum(axis=-1)
    positive = semantic_hist[..., :, :17].sum(axis=-2)
    correct = np.diagonal(semantic_hist, axis1=-2, axis2=-1)[..., :17]
    semantic = np.stack((seen, positive, correct), axis=-1)

    binary_hist = binary_hist[:, horizon_indices]
    binary_seen = binary_hist[..., 1, :].sum(axis=-1)
    binary_positive = binary_hist[..., :, 1].sum(axis=-1)
    binary_correct = binary_hist[..., 1, 1]
    binary = np.stack((binary_seen, binary_positive, binary_correct), axis=-1)
    return semantic, binary


def score_weighted_features(
    weights: np.ndarray,
    features: np.ndarray,
    *,
    zero_policy: str,
) -> np.ndarray:
    protocols, scenes = features.shape[:2]
    trailing_shape = features.shape[2:]
    flat = features.reshape(protocols, scenes, -1).astype(np.float64)
    summed = np.einsum("bs,psf->bpf", weights, flat, optimize=True)
    summed = summed.reshape((len(weights), protocols) + trailing_shape)
    seen, positive, correct = (
        summed[..., 0],
        summed[..., 1],
        summed[..., 2],
    )
    union = seen + positive - correct
    iou = np.divide(
        correct,
        union,
        out=np.full_like(correct, np.nan),
        where=union != 0,
    )
    if zero_policy == "exclude_zero":
        iou[iou == 0] = np.nan
    elif zero_policy == "absent_is_one":
        iou[seen == 0] = 1.0
    elif zero_policy != "standard":
        raise ValueError(f"Unknown zero policy: {zero_policy}")
    if iou.ndim == 4:
        # Match the released evaluators: average semantic classes within each
        # horizon first, then average submission-native indices 1, 3, and 5.
        iou = np.nanmean(iou, axis=-1)
    return np.nanmean(iou, axis=-1) * 100.0


def aggregate_protocol_scores(
    scores: np.ndarray,
    protocols: list[Protocol],
) -> np.ndarray:
    def mean_where(predicate) -> np.ndarray:
        indices = [index for index, protocol in enumerate(protocols) if predicate(protocol)]
        if not indices:
            raise ValueError("Empty protocol aggregation")
        return scores[:, indices].mean(axis=1)

    clean = {
        source: mean_where(lambda p, source=source: p.source == source and p.clean)
        for source in "ipo"
    }
    robust = {
        source: mean_where(
            lambda p, source=source: (
                p.source == source and not p.clean and not p.traffic
            )
        )
        for source in "ipo"
    }
    temporal = {}
    for mode in ("current", "burst", "history"):
        source_values = [
            mean_where(
                lambda p, source=source, mode=mode: (
                    p.source == source
                    and p.mode == mode
                    and not p.clean
                    and not p.traffic
                )
            )
            for source in "ipo"
        ]
        temporal[mode] = np.mean(source_values, axis=0)

    mrs = np.mean([robust[source] for source in "ipo"], axis=0)
    ucr = (clean["i"] + clean["p"]) / (2.0 * clean["o"])
    crf2 = 5.0 * clean["o"] * mrs / (4.0 * clean["o"] + mrs)
    values = {
        "cs_i": clean["i"],
        "cs_p": clean["p"],
        "cs_o": clean["o"],
        "rs_i": robust["i"],
        "rs_p": robust["p"],
        "rs_o": robust["o"],
        **temporal,
        "ucr": ucr,
        "mrs": mrs,
        "crf2": crf2,
    }
    return np.stack([values[key] for key in METRIC_ORDER], axis=1)


def interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def format_ci(point: float, lower: float, upper: float, key: str) -> str:
    digits = 3 if key == "ucr" else 2
    return f"{point:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def load_model(
    stats_root: Path,
    model: str,
) -> tuple[
    list[Protocol],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    model_root = stats_root / model
    paths = sorted(model_root.rglob("*.npz"))
    if len(paths) != 184:
        raise ValueError(f"{model}: expected 184 scene-stat files, found {len(paths)}")

    protocols = []
    scene_tokens = None
    anchor_counts = None
    semantic_features = []
    binary_features = []
    provenance = []
    for path in paths:
        relative = path.relative_to(model_root).with_suffix("").as_posix() + ".pkl"
        protocols.append(classify(relative))
        with np.load(path) as data:
            tokens = data["scene_tokens"].astype(str).tolist()
            counts = data["anchor_counts"].astype(np.int64)
            if scene_tokens is None:
                scene_tokens = tokens
                anchor_counts = counts
            elif tokens != scene_tokens:
                raise ValueError(f"Scene order mismatch in {path}")
            elif not np.array_equal(counts, anchor_counts):
                raise ValueError(f"Per-scene anchor-count mismatch in {path}")
            if int(counts.sum()) != 4519:
                raise ValueError(f"Anchor count mismatch in {path}")
            semantic, binary = protocol_score_features(
                data["semantic_hist"],
                data["binary_hist"],
            )
            semantic_features.append(semantic)
            binary_features.append(binary)
            provenance.append(json.loads(data["metadata_json"].item()))
    assert scene_tokens is not None
    assert anchor_counts is not None
    return (
        protocols,
        scene_tokens,
        anchor_counts,
        np.stack(semantic_features),
        np.stack(binary_features),
        provenance,
    )


def write_report(
    output_path: Path,
    summary: dict,
    comparisons: list[dict],
    contrasts: list[dict],
    family_rows: list[dict],
    models: tuple[str, ...],
) -> None:
    lines = [
        "# OccStress Scene-Level Clustered Paired Bootstrap",
        "",
        "## Analysis Design",
        "",
        f"- Sampling unit: nuScenes validation scene (`{summary['num_scenes']}` clusters), not anchor.",
        f"- Anchors: `{summary['num_anchors']}` unique anchors per protocol; overlapping anchors remain inside their original scene cluster.",
        f"- Cluster sizes: {summary['cluster_anchor_count_min']}–{summary['cluster_anchor_count_max']} anchors per scene (median {summary['cluster_anchor_count_median']:.0f}).",
        f"- Bootstrap: `{summary['replicates']:,}` resamples, seed `{summary['seed']}`, percentile 95% confidence intervals.",
        "- Interval scope: aggregate and pre-specified paired contrasts support the main claims. Per-family and per-protocol intervals are marginal diagnostic intervals, not multiplicity-adjusted simultaneous intervals.",
        "- Pairing: every bootstrap replicate uses the same sampled scene multiplicities for all models, sources, clean/corrupted protocols, and temporal modes.",
        "- Anchor-population compatibility: clean and corrupted estimates use the same 4,519 complete-H4/F6 anchors. In particular, II-World's paired clean estimate is 39.78/50.14; the 39.35/49.71 clean value in the submitted table came from the native 6,019-anchor loader and is retained only as the published reference.",
        "- Metric reconstruction: per-scene semantic and binary confusion sufficient statistics are pooled before IoU is computed; the reported score averages submission-native evaluator indices 1, 3, and 5.",
        "- Native horizons: those indices are physical 0.5/1.5/2.5 s for OccWorld and COME because their submitted adapters include current reconstruction at index 0; they are physical 1/2/3 s for II-World.",
        "- Metric compatibility: each replicate preserves the released evaluator convention (OccWorld/COME assign IoU=1 to absent-GT classes; II-World excludes semantic classes whose pooled IoU is exactly zero).",
        "- Aggregation: Traffic is reported outside RS-O and is excluded from Current/Burst/History, matching the paper definition.",
        "",
        "## Main Semantic Results",
        "",
        "| Model | CS-I | CS-P | CS-O | RS-I | RS-P | RS-O | Current | Burst | History | UCR | mRS | CRF-2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        row = summary["models"][model]["semantic"]
        cells = [
            format_ci(row[key]["point"], row[key]["lower"], row[key]["upper"], key)
            for key in METRIC_ORDER
        ]
        lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Main Binary-IoU Results",
            "",
            "| Model | CS-I | CS-P | CS-O | RS-I | RS-P | RS-O | Current | Burst | History | mRS | CRF-2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    binary_order = tuple(key for key in METRIC_ORDER if key != "ucr")
    for model in models:
        row = summary["models"][model]["binary"]
        cells = [
            format_ci(row[key]["point"], row[key]["lower"], row[key]["upper"], key)
            for key in binary_order
        ]
        lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Corruption-Family Semantic Results",
            "",
            "Each cell is the family mean forecast mIoU with a scene-clustered 95% CI. `I`, `P`, and `O` denote STCOcc, SDGOcc, and manual occupancy corruption sources.",
            "",
            "| Source:Family | Protocols | "
            + " | ".join(MODEL_NAMES[model] for model in models)
            + " |",
            "| --- | ---: | " + " | ".join("---:" for _ in models) + " |",
        ]
    )
    for row in family_rows:
        cells = [
            format_ci(
                row[model]["point"],
                row[model]["lower"],
                row[model]["upper"],
                "family",
            )
            for model in models
        ]
        lines.append(
            f"| {row['family']} | {row['protocol_count']} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired Model Differences",
            "",
            "Positive values favor the first model under the submission-native score contract. Because OccWorld/COME and II-World use different physical horizon mappings, these are uncertainty estimates for the submitted comparisons, not a corrected common-time ranking. `P(>0)` is the paired bootstrap probability that the difference is positive.",
            "",
            "| Metric | Comparison | Difference [95% CI] | P(>0) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['metric']} | {row['first']} - {row['second']} | "
            f"{row['point']:.2f} [{row['lower']:.2f}, {row['upper']:.2f}] | "
            f"{row['probability_positive']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Within-Model Paired Contrasts",
            "",
            "| Model | Contrast | Difference [95% CI] | P(>0) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['model']} | {row['contrast']} | "
            f"{row['point']:.2f} [{row['lower']:.2f}, {row['upper']:.2f}] | "
            f"{row['probability_positive']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation And Scope",
            "",
            "- These intervals quantify validation-scene sampling uncertainty while respecting anchor overlap and clean/corrupted pairing.",
            "- II-World's submitted native clean result used 6,019 anchors (39.35/49.71), whereas OccStress protocols use 4,519 complete-H4/F6 anchors. The paired analysis uses the matched 4,519-anchor clean result (39.78/50.14), so the original table value is not silently mixed into paired differences.",
            "- The current release contains one deterministic realization per manual corruption setting. These intervals therefore do not estimate corruption-instantiation variance; that requires additional independently generated manual seeds.",
            "- Protocols are fixed a priori. No corruption family or severity was selected after inspecting these intervals.",
            "- The appendix's per-family and per-protocol 95% intervals are unadjusted marginal intervals. We do not use them as 184 simultaneous hypothesis tests.",
            "",
            "## Rebuttal-Ready Statement",
            "",
            "> Because temporally adjacent anchors overlap and are correlated, we replaced anchor-wise resampling with a scene-level clustered paired bootstrap. We resample the 150 validation scenes with replacement, preserve clean/corrupted anchor pairing, and use the same scene draws across models and protocols. We report 95% confidence intervals for clean, source-wise, protocol-wise, mRS, UCR, CRF-2, and paired model/protocol differences. The released manual benchmark contains one fixed corruption realization, so these intervals quantify scene-sampling uncertainty and do not claim corruption-instantiation variance.",
            "",
            "## Provenance",
            "",
            f"- Scene-stat root: `{summary['stats_root']}`",
            f"- Protocol inventory: `184 = 38 manual + 73 STCOcc + 73 SDGOcc`",
            f"- Bootstrap JSON: `{summary['json_path']}`",
            f"- Long-form CSV: `{summary['csv_path']}`",
            f"- Per-family CSV: `{summary['family_csv_path']}`",
            f"- Per-protocol CSV: `{summary['protocol_csv_path']}`",
            f"- Replicate archive: `{summary['replicate_path']}`",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_NAMES),
        default=list(MODEL_NAMES),
    )
    args = parser.parse_args()
    models = tuple(args.models)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_data = {}
    reference_protocols = None
    reference_scenes = None
    reference_anchor_counts = None
    provenance = {}
    for model in models:
        protocols, scenes, anchor_counts, semantic, binary, model_provenance = load_model(
            args.stats_root, model
        )
        protocol_paths = [protocol.relative_path for protocol in protocols]
        if reference_protocols is None:
            reference_protocols = protocol_paths
            reference_scenes = scenes
            reference_anchor_counts = anchor_counts
        elif (
            protocol_paths != reference_protocols
            or scenes != reference_scenes
            or not np.array_equal(anchor_counts, reference_anchor_counts)
        ):
            raise ValueError(f"Pairing mismatch for {model}")
        model_data[model] = (protocols, semantic, binary)
        provenance[model] = model_provenance

    assert reference_scenes is not None
    assert reference_anchor_counts is not None
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(
        0,
        len(reference_scenes),
        size=(args.replicates, len(reference_scenes)),
        dtype=np.int16,
    )
    weights = np.zeros((args.replicates, len(reference_scenes)), dtype=np.int16)
    rows = np.repeat(np.arange(args.replicates), len(reference_scenes))
    np.add.at(weights, (rows, draws.reshape(-1)), 1)

    replicate_results = {}
    point_results = {}
    protocol_replicates = {}
    protocol_points = {}
    for model, (protocols, semantic, binary) in model_data.items():
        replicate_results[model] = {
            "semantic": np.empty((args.replicates, len(METRIC_ORDER))),
            "binary": np.empty((args.replicates, len(METRIC_ORDER))),
        }
        protocol_replicates[model] = {
            "semantic": np.empty((args.replicates, len(protocols))),
            "binary": np.empty((args.replicates, len(protocols))),
        }
        point_results[model] = {}
        protocol_points[model] = {}
        all_scenes = np.ones((1, len(reference_scenes)), dtype=np.int16)
        for kind, features in (("semantic", semantic), ("binary", binary)):
            if kind == "semantic" and model == "ii_world":
                zero_policy = "exclude_zero"
            elif model in ("occworld", "come"):
                zero_policy = "absent_is_one"
            else:
                zero_policy = "standard"
            point_protocol = score_weighted_features(
                all_scenes,
                features,
                zero_policy=zero_policy,
            )
            protocol_points[model][kind] = point_protocol[0]
            point_results[model][kind] = aggregate_protocol_scores(
                point_protocol, protocols
            )[0]
            for start in range(0, args.replicates, args.batch_size):
                stop = min(start + args.batch_size, args.replicates)
                protocol_scores = score_weighted_features(
                    weights[start:stop],
                    features,
                    zero_policy=zero_policy,
                )
                protocol_replicates[model][kind][start:stop] = protocol_scores
                replicate_results[model][kind][start:stop] = aggregate_protocol_scores(
                    protocol_scores, protocols
                )

    json_path = args.output_dir / "clustered_bootstrap_summary.json"
    csv_path = args.output_dir / "clustered_bootstrap_intervals.csv"
    family_csv_path = args.output_dir / "clustered_bootstrap_family_intervals.csv"
    protocol_csv_path = args.output_dir / "clustered_bootstrap_protocol_intervals.csv"
    replicate_path = args.output_dir / "clustered_bootstrap_replicates.npz"
    report_path = args.output_dir / "STATISTICAL_UNCERTAINTY_ANALYSIS.md"
    summary = {
        "method": "scene-level clustered paired percentile bootstrap",
        "evaluation_contract": "submission_compatible_native_horizon",
        "score_horizon_indices": [1, 3, 5],
        "score_horizon_seconds": {
            "occworld": [0.5, 1.5, 2.5],
            "come": [0.5, 1.5, 2.5],
            "ii_world": [1.0, 2.0, 3.0],
        },
        "replicates": args.replicates,
        "seed": args.seed,
        "num_scenes": len(reference_scenes),
        "num_anchors": 4519,
        "cluster_anchor_count_min": int(reference_anchor_counts.min()),
        "cluster_anchor_count_median": float(np.median(reference_anchor_counts)),
        "cluster_anchor_count_max": int(reference_anchor_counts.max()),
        "cluster_anchor_count_mean": float(reference_anchor_counts.mean()),
        "cluster_anchor_count_std": float(reference_anchor_counts.std(ddof=1)),
        "stats_root": str(args.stats_root.resolve()),
        "json_path": str(json_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "family_csv_path": str(family_csv_path.resolve()),
        "protocol_csv_path": str(protocol_csv_path.resolve()),
        "replicate_path": str(replicate_path.resolve()),
        "models": {},
        "requested_models": list(models),
        "rank_probability_best": {},
        "provenance": provenance,
    }
    csv_rows = []
    for model in models:
        summary["models"][model] = {}
        for kind in ("semantic", "binary"):
            summary["models"][model][kind] = {}
            for index, key in enumerate(METRIC_ORDER):
                point = float(point_results[model][kind][index])
                lower, upper = interval(replicate_results[model][kind][:, index])
                cell = {"point": point, "lower": lower, "upper": upper}
                summary["models"][model][kind][key] = cell
                csv_rows.append(
                    {
                        "model": MODEL_NAMES[model],
                        "score_type": kind,
                        "metric": METRIC_LABELS[key],
                        **cell,
                    }
                )

    assert reference_protocols is not None
    protocol_rows = []
    family_rows_by_kind = []
    family_summary_rows = []
    groups = family_groups(model_data[models[0]][0])
    for model in models:
        for kind in ("semantic", "binary"):
            for protocol_index, relative_path in enumerate(reference_protocols):
                values = protocol_replicates[model][kind][:, protocol_index]
                lower, upper = interval(values)
                protocol = model_data[model][0][protocol_index]
                protocol_rows.append(
                    {
                        "model": MODEL_NAMES[model],
                        "score_type": kind,
                        "protocol": relative_path,
                        "source": protocol.source.upper(),
                        "family": protocol.family,
                        "severity": protocol.severity or "",
                        "mode": protocol.mode or "",
                        "clean": protocol.clean,
                        "traffic": protocol.traffic,
                        "point": float(protocol_points[model][kind][protocol_index]),
                        "lower": lower,
                        "upper": upper,
                    }
                )
            for family, indices in groups.items():
                values = protocol_replicates[model][kind][:, indices].mean(axis=1)
                lower, upper = interval(values)
                family_rows_by_kind.append(
                    {
                        "model": MODEL_NAMES[model],
                        "score_type": kind,
                        "family": family,
                        "protocol_count": len(indices),
                        "point": float(
                            protocol_points[model][kind][indices].mean()
                        ),
                        "lower": lower,
                        "upper": upper,
                    }
                )

    family_lookup = {
        (row["family"], row["model"]): row
        for row in family_rows_by_kind
        if row["score_type"] == "semantic"
    }
    for family, indices in groups.items():
        family_summary_rows.append(
            {
                "family": family,
                "protocol_count": len(indices),
                **{
                    model: family_lookup[(family, MODEL_NAMES[model])]
                    for model in models
                },
            }
        )
    summary["family_intervals"] = family_rows_by_kind

    comparisons = []
    model_pairs = (
        ("ii_world", "come"),
        ("ii_world", "occworld"),
        ("come", "occworld"),
    )
    comparison_keys = ("rs_i", "rs_p", "rs_o", "current", "burst", "history", "mrs", "crf2")
    for first, second in model_pairs:
        if first not in models or second not in models:
            continue
        for key in comparison_keys:
            index = METRIC_ORDER.index(key)
            values = (
                replicate_results[first]["semantic"][:, index]
                - replicate_results[second]["semantic"][:, index]
            )
            lower, upper = interval(values)
            comparisons.append(
                {
                    "metric": METRIC_LABELS[key],
                    "first": MODEL_NAMES[first],
                    "second": MODEL_NAMES[second],
                    "point": float(
                        point_results[first]["semantic"][index]
                        - point_results[second]["semantic"][index]
                    ),
                    "lower": lower,
                    "upper": upper,
                    "probability_positive": float(np.mean(values > 0)),
                }
            )

    contrasts = []
    contrast_keys = (
        ("CS-O - mRS", "cs_o", "mrs"),
        ("CS-I - RS-I", "cs_i", "rs_i"),
        ("CS-P - RS-P", "cs_p", "rs_p"),
        ("CS-O - RS-O", "cs_o", "rs_o"),
        ("History - Current", "history", "current"),
        ("History - Burst", "history", "burst"),
        ("RS-P - RS-I", "rs_p", "rs_i"),
    )
    for model in models:
        for label, first_key, second_key in contrast_keys:
            first_index = METRIC_ORDER.index(first_key)
            second_index = METRIC_ORDER.index(second_key)
            values = (
                replicate_results[model]["semantic"][:, first_index]
                - replicate_results[model]["semantic"][:, second_index]
            )
            lower, upper = interval(values)
            contrasts.append(
                {
                    "model": MODEL_NAMES[model],
                    "contrast": label,
                    "point": float(
                        point_results[model]["semantic"][first_index]
                        - point_results[model]["semantic"][second_index]
                    ),
                    "lower": lower,
                    "upper": upper,
                    "probability_positive": float(np.mean(values > 0)),
                }
            )

    for key in comparison_keys:
        index = METRIC_ORDER.index(key)
        stacked = np.stack(
            [replicate_results[model]["semantic"][:, index] for model in models],
            axis=1,
        )
        winners = np.argmax(stacked, axis=1)
        summary["rank_probability_best"][key] = {
            model: float(np.mean(winners == model_index))
            for model_index, model in enumerate(models)
        }
    summary["paired_model_differences"] = comparisons
    summary["within_model_contrasts"] = contrasts

    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "score_type", "metric", "point", "lower", "upper"),
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    with family_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "score_type",
                "family",
                "protocol_count",
                "point",
                "lower",
                "upper",
            ),
        )
        writer.writeheader()
        writer.writerows(family_rows_by_kind)
    with protocol_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "score_type",
                "protocol",
                "source",
                "family",
                "severity",
                "mode",
                "clean",
                "traffic",
                "point",
                "lower",
                "upper",
            ),
        )
        writer.writeheader()
        writer.writerows(protocol_rows)
    np.savez_compressed(
        replicate_path,
        metric_order=np.asarray(METRIC_ORDER),
        scene_tokens=np.asarray(reference_scenes),
        scene_weights=weights,
        **{
            f"{model}_{kind}": replicate_results[model][kind]
            for model in models
            for kind in ("semantic", "binary")
        },
        **{
            f"{model}_{kind}_protocol": protocol_replicates[model][kind]
            for model in models
            for kind in ("semantic", "binary")
        },
    )
    write_report(
        report_path,
        summary,
        comparisons,
        contrasts,
        family_summary_rows,
        models,
    )
    print(report_path)


if __name__ == "__main__":
    main()
