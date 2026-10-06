#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Validate position-sweep v3 metadata, assets, and temporal isolation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


POSITION_INDEX = {"t-4": 0, "t-3": 1, "t-2": 2, "t-1": 3, "t": 4}


def input_paths(record: dict) -> list[str]:
    return [item["occ_path"] for item in record["history"]] + [
        record["current_input"]["occ_path"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=4519)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    protocol_root = suite_root / "protocols"
    task_root = suite_root / "task_protocols"
    summary = {"status": "success", "settings": {}, "task_protocol_count": 0}

    for setting_dir in sorted(path for path in protocol_root.iterdir() if path.is_dir()):
        protocol_paths = sorted(setting_dir.glob("*.pkl"))
        if len(protocol_paths) != 6:
            raise ValueError(f"{setting_dir}: expected 6 protocols, found {len(protocol_paths)}")
        clean_paths = [path for path in protocol_paths if "_clean_H4_" in path.name]
        if len(clean_paths) != 1:
            raise ValueError(f"{setting_dir}: expected exactly one clean protocol")
        with clean_paths[0].open("rb") as handle:
            clean = pickle.load(handle)
        if len(clean) != args.expected_records:
            raise ValueError(
                f"{clean_paths[0]}: expected {args.expected_records} records"
            )
        clean_tokens = [record["anchor_token"] for record in clean]
        positions_seen = {"clean"}
        for path in protocol_paths:
            if path == clean_paths[0]:
                continue
            with path.open("rb") as handle:
                records = pickle.load(handle)
            if len(records) != args.expected_records:
                raise ValueError(f"{path}: expected {args.expected_records} records")
            position = records[0]["corruption"]["position"]
            if position in positions_seen:
                raise ValueError(f"{setting_dir}: duplicate position {position}")
            positions_seen.add(position)
            if [record["anchor_token"] for record in records] != clean_tokens:
                raise ValueError(f"{setting_dir}/{position}: anchor order differs")
            for index, (record, clean_record) in enumerate(zip(records, clean)):
                if len(record["history"]) != 4 or len(record["future_targets"]) != 6:
                    raise ValueError(f"{setting_dir}/{position}[{index}]: invalid H4/F6")
                if record["future_targets"] != clean_record["future_targets"]:
                    raise ValueError(f"{setting_dir}/{position}[{index}]: future target changed")
                pattern = ["T"] * 5
                if position != "clean":
                    pattern[POSITION_INDEX[position]] = "F"
                if record["corruption"]["pattern"] != pattern:
                    raise ValueError(f"{setting_dir}/{position}[{index}]: invalid pattern")

                changed_paths = [
                    value != clean_value
                    for value, clean_value in zip(
                        input_paths(record), input_paths(clean_record)
                    )
                ]
                is_misalignment = setting_dir.name == "manual_misalignment_hard"
                if is_misalignment:
                    if any(changed_paths):
                        raise ValueError(
                            f"{setting_dir}/{position}[{index}]: misalignment changed occupancy path"
                        )
                    metadata = record.get("misalignment")
                    if position == "clean":
                        if metadata:
                            raise ValueError(f"{setting_dir}/clean[{index}]: active metadata")
                    else:
                        expected_mask = [0, 0, 0, 0]
                        current_active = POSITION_INDEX[position] == 4
                        if not current_active:
                            expected_mask[POSITION_INDEX[position]] = 1
                        if metadata["affected_mask"] != expected_mask:
                            raise ValueError(f"{setting_dir}/{position}[{index}]: mask mismatch")
                        if bool(metadata["current_active"]) != current_active:
                            raise ValueError(
                                f"{setting_dir}/{position}[{index}]: current flag mismatch"
                            )
                else:
                    expected_changed = 0 if position == "clean" else 1
                    if sum(changed_paths) != expected_changed:
                        raise ValueError(
                            f"{setting_dir}/{position}[{index}]: changed {sum(changed_paths)} paths"
                        )

                if args.check_files:
                    for path_string in input_paths(record):
                        if not Path(path_string).is_file():
                            raise FileNotFoundError(path_string)
            del records

        expected_positions = {"clean", *POSITION_INDEX}
        if positions_seen != expected_positions:
            raise ValueError(f"{setting_dir}: invalid positions {positions_seen}")

        task_count = len(list((task_root / setting_dir.name).glob("*.pkl")))
        if task_count != 5:
            raise ValueError(f"{setting_dir}: expected 5 task links, found {task_count}")
        summary["task_protocol_count"] += task_count
        summary["settings"][setting_dir.name] = {
            "protocols": 6,
            "task_protocols": task_count,
            "records_per_protocol": args.expected_records,
            "anchor_order_endpoints": [clean_tokens[0], clean_tokens[-1]],
        }

    if len(summary["settings"]) != 8 or summary["task_protocol_count"] != 40:
        raise ValueError(f"invalid suite totals: {summary}")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
