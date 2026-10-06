#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate and summarize SparseWorld-TC OccStress-Waymo camera results."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from summarize_carla_occstress import (  # noqa: E402
    HORIZONS,
    INPUT_TIMES,
    METRICS,
    aggregate,
    finite,
    flatten_metrics,
    write_csv,
    write_json,
)


EXPECTED_PROTOCOLS = 73
EXPECTED_SAMPLES = 5978


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="occstress/protocols.json")
    parser.add_argument("--result-root", default="occstress/waymo/results")
    parser.add_argument("--output-root", default="occstress/waymo/summary")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


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
            "code_commit", "checkpoint_sha256", "base_info_sha256",
            "pose_file_sha256", "adapter_sha256",
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
            if payload.get("dataset") != "OccStress-Waymo":
                raise ValueError(f"dataset={payload.get('dataset')}")
            if payload.get("track") != "camera-only":
                raise ValueError(f"track={payload.get('track')}")
            if payload.get("sample_count") != EXPECTED_SAMPLES:
                raise ValueError(f"sample_count={payload.get('sample_count')}")
            if payload.get("dataset_sample_count") != EXPECTED_SAMPLES:
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
    complete = (
        len(rows) == len(manifest["protocols"]) == EXPECTED_PROTOCOLS
        and not errors
    )
    validation = {
        "complete": complete,
        "expected_protocols": EXPECTED_PROTOCOLS,
        "valid_protocols": len(rows),
        "expected_anchors_per_protocol": EXPECTED_SAMPLES,
        "valid_anchor_records": sum(row["sample_count"] for row in rows),
        "errors": errors,
        "provenance_values": {
            field: sorted(values) for field, values in provenance.items()
        },
    }
    write_json(output_root / "validation.json", validation)
    if not complete and not args.allow_incomplete:
        raise SystemExit(
            f"incomplete results: {len(rows)}/{EXPECTED_PROTOCOLS}; "
            f"see {output_root / 'validation.json'}"
        )
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
        "dataset": "OccStress-Waymo",
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
