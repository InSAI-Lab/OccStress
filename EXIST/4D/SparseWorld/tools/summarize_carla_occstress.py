#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate and summarize SparseWorld-TC UniOcc-CARLA results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
INPUT_TIMES = (-2.0, -1.5, -1.0, -0.5, 0.0)
METRICS = tuple(
    f"{metric}_{horizon:.1f}s"
    for horizon in HORIZONS
    for metric in ("miou", "iou")
) + ("mean_six_miou", "mean_six_iou")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="occstress/protocols.json")
    parser.add_argument("--result-root", default="occstress/carla/results")
    parser.add_argument("--output-root", default="occstress/carla/summary")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def finite(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value!r}")
    return float(value)


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flatten_metrics(payload):
    horizons = payload.get("metrics", {}).get("horizons", [])
    if len(horizons) != len(HORIZONS):
        raise ValueError(f"horizon count={len(horizons)}")
    by_time = {float(item["seconds"]): item for item in horizons}
    if set(by_time) != set(HORIZONS):
        raise ValueError(f"horizon times={sorted(by_time)}")
    values = {}
    for horizon in HORIZONS:
        for metric in ("miou", "iou"):
            values[f"{metric}_{horizon:.1f}s"] = finite(
                by_time[horizon].get(metric), f"{metric}@{horizon}s"
            )
    mean_six = payload["metrics"]["mean_six_frames"]
    values["mean_six_miou"] = finite(mean_six.get("miou"), "mean mIoU")
    values["mean_six_iou"] = finite(mean_six.get("iou"), "mean IoU")
    return values


def mean_metrics(rows):
    return {metric: statistics.fmean(row[metric] for row in rows) for metric in METRICS}


def aggregate(rows, manifest):
    corrupt = [row for row in rows if row["corruption"] != "clean"]
    groups = [("scope", "clean", rows[:1]), ("scope", "robust_all", corrupt)]
    dimensions = (
        ("corruption", "corruption"),
        ("severity", "severity"),
        ("frame_protocol", "frame_protocol"),
    )
    for group_type, field in dimensions:
        names = list(dict.fromkeys(
            item[field] for item in manifest["protocols"] if item[field] is not None
        ))
        groups.extend(
            (group_type, name, [row for row in corrupt if row[field] == name])
            for name in names
        )
    return [
        {
            "group_type": group_type,
            "group": name,
            "protocol_count": len(group),
            **mean_metrics(group),
        }
        for group_type, name, group in groups
        if group
    ]


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    result_root = Path(args.result_root)
    output_root = Path(args.output_root)
    rows = []
    errors = []
    provenance = {
        field: set()
        for field in (
            "code_commit", "checkpoint_sha256", "base_info_sha256", "adapter_sha256"
        )
    }

    for expected in manifest["protocols"]:
        path = result_root / f"{expected['id']}.json"
        if not path.is_file():
            errors.append(f"missing: {expected['id']}")
            continue
        try:
            payload = json.loads(path.read_text())
            if payload.get("status") != "success":
                raise ValueError(f"status={payload.get('status')}")
            if payload.get("protocol") != expected:
                raise ValueError("protocol identity mismatch")
            if payload.get("method") != "SparseWorld-TC":
                raise ValueError(f"method={payload.get('method')}")
            if payload.get("dataset") != "UniOcc-CARLA":
                raise ValueError(f"dataset={payload.get('dataset')}")
            if payload.get("track") != "camera-only":
                raise ValueError(f"track={payload.get('track')}")
            if payload.get("sample_count") != 330:
                raise ValueError(f"sample_count={payload.get('sample_count')}")
            if payload.get("dataset_sample_count") != 330:
                raise ValueError(
                    f"dataset_sample_count={payload.get('dataset_sample_count')}"
                )
            if tuple(payload.get("input_times_seconds", [])) != INPUT_TIMES:
                raise ValueError(f"input times={payload.get('input_times_seconds')}")
            if tuple(payload.get("future_times_seconds", [])) != HORIZONS:
                raise ValueError(f"future times={payload.get('future_times_seconds')}")
            if len(payload.get("semantic_confusion", [])) != 6:
                raise ValueError("semantic confusion does not have six horizons")
            if len(payload.get("occupancy_confusion", [])) != 6:
                raise ValueError("occupancy confusion does not have six horizons")
            metrics = flatten_metrics(payload)
            runtime = payload["runtime"]
            source = payload["provenance"]
            for field in provenance:
                provenance[field].add(str(source.get(field)))
            rows.append({
                "protocol_id": expected["id"],
                "corruption": expected["corruption"],
                "severity": expected["severity"] or "",
                "frame_protocol": expected["frame_protocol"] or "",
                "sample_count": payload["sample_count"],
                **metrics,
                "elapsed_seconds": finite(runtime.get("elapsed_seconds"), "elapsed"),
                "samples_per_second": finite(
                    runtime.get("samples_per_second"), "throughput"
                ),
                "peak_gpu_memory_gib": finite(
                    runtime.get("peak_gpu_memory_gib"), "peak GPU memory"
                ),
                "gpu": runtime.get("gpu"),
                "hostname": runtime.get("hostname"),
                "slurm_job_id": runtime.get("slurm_job_id"),
            })
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid {expected['id']}: {error}")

    for field, values in provenance.items():
        if len(values) > 1:
            errors.append(f"inconsistent {field}: {sorted(values)}")
    complete = len(rows) == len(manifest["protocols"]) == 73 and not errors
    validation = {
        "complete": complete,
        "expected_protocols": 73,
        "valid_protocols": len(rows),
        "expected_anchors_per_protocol": 330,
        "valid_anchor_records": sum(row["sample_count"] for row in rows),
        "errors": errors,
        "provenance_values": {field: sorted(values) for field, values in provenance.items()},
    }
    write_json(output_root / "validation.json", validation)
    if not complete and not args.allow_incomplete:
        raise SystemExit(f"incomplete results: {len(rows)}/73; see {output_root / 'validation.json'}")

    if not rows:
        print(json.dumps(validation, indent=2))
        return
    aggregate_rows = aggregate(rows, manifest)
    clean = next(row for row in aggregate_rows if row["group"] == "clean")
    robust = next(row for row in aggregate_rows if row["group"] == "robust_all")
    summary = {
        **validation,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "SparseWorld-TC",
        "dataset": "UniOcc-CARLA",
        "track": "camera-only zero-shot",
        "evaluation_regime": "official Occ3D-nuScenes checkpoint without adaptation",
        "clean": clean,
        "robust_mean": robust,
        "degradation_from_clean": {
            metric: clean[metric] - robust[metric] for metric in METRICS
        },
        "by_corruption": {
            row["group"]: row for row in aggregate_rows
            if row["group_type"] == "corruption"
        },
        "by_severity": {
            row["group"]: row for row in aggregate_rows
            if row["group_type"] == "severity"
        },
        "by_frame_protocol": {
            row["group"]: row for row in aggregate_rows
            if row["group_type"] == "frame_protocol"
        },
    }
    write_json(output_root / "summary.json", summary)
    row_fields = (
        "protocol_id", "corruption", "severity", "frame_protocol", "sample_count",
        *METRICS, "elapsed_seconds", "samples_per_second", "peak_gpu_memory_gib",
        "gpu", "hostname", "slurm_job_id",
    )
    aggregate_fields = ("group_type", "group", "protocol_count", *METRICS)
    write_csv(output_root / "protocol_metrics.csv", rows, row_fields)
    write_csv(output_root / "aggregate_metrics.csv", aggregate_rows, aggregate_fields)
    print(json.dumps({
        "complete": complete,
        "protocols": len(rows),
        "anchor_records": validation["valid_anchor_records"],
        "clean_mean_six": {
            "miou": clean["mean_six_miou"], "iou": clean["mean_six_iou"]
        },
        "robust_mean_six": {
            "miou": robust["mean_six_miou"], "iou": robust["mean_six_iou"]
        },
        "output_root": str(output_root.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
