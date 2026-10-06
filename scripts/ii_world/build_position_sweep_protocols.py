#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any


POSITIONS = [
    ("t-4", "tminus4", "history", 0),
    ("t-3", "tminus3", "history", 1),
    ("t-2", "tminus2", "history", 2),
    ("t-1", "tminus1", "history", 3),
    ("t", "t", "current", None),
]


def rewrite_paths(obj: Any, prefix_map: list[tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        for old, new in prefix_map:
            if obj.startswith(old):
                return new + obj[len(old):]
        return obj
    if isinstance(obj, list):
        return [rewrite_paths(item, prefix_map) for item in obj]
    if isinstance(obj, tuple):
        return tuple(rewrite_paths(item, prefix_map) for item in obj)
    if isinstance(obj, dict):
        return {key: rewrite_paths(value, prefix_map) for key, value in obj.items()}
    return obj


def corrupt_occ_path(root: Path, scene_name: str, token: str) -> str:
    return str(root / scene_name / token / "labels.npz")


def corrupt_event_path(root: Path | None, scene_name: str, token: str) -> str | None:
    if root is None:
        return None
    return str(root / scene_name / f"{token}.json")


def mark_history_corrupted(record: dict[str, Any], index: int, args: argparse.Namespace) -> None:
    token = record["history_tokens"][index]
    scene_name = record["scene_name"]
    hist = record["history"][index]
    hist["occ_source"] = args.corruption
    hist["source"] = args.corruption
    hist["occ_path"] = corrupt_occ_path(args.corrupt_occ_root, scene_name, token)
    event_path = corrupt_event_path(args.corrupt_event_root, scene_name, token)
    if event_path is not None:
        hist["event_path"] = event_path


def mark_current_corrupted(record: dict[str, Any], args: argparse.Namespace) -> None:
    token = record["anchor_token"]
    scene_name = record["scene_name"]
    current = copy.deepcopy(record.get("current_input", record["target"]))
    current["source"] = args.corruption
    current["occ_path"] = corrupt_occ_path(args.corrupt_occ_root, scene_name, token)
    event_path = corrupt_event_path(args.corrupt_event_root, scene_name, token)
    if event_path is not None:
        current["event_path"] = event_path
    record["current_input"] = current


def finalize_record(record: dict[str, Any], case_label: str, case_slug: str, args: argparse.Namespace) -> None:
    pattern = ["T", "T", "T", "T", "T"]
    if case_label != "clean":
        pattern_index = {"t-4": 0, "t-3": 1, "t-2": 2, "t-1": 3, "t": 4}[case_label]
        pattern[pattern_index] = "F"
    record["sample_id"] = (
        f"{record['anchor_token']}__H4__F6__positionsweep_"
        f"{args.corruption}_{args.severity}__{case_slug}"
    )
    record["corruption"] = {
        "type": args.corruption if case_label != "clean" else "clean",
        "severity": args.severity if case_label != "clean" else "clean",
        "frame_protocol": "position_sweep",
        "position": case_label,
        "pattern": pattern,
    }
    record["track"] = "analysis"
    record["subtrack"] = "position_sweep"
    record["source_model"] = args.source_model
    record["reference_protocol"] = (
        f"position_sweep/{args.corruption}/{args.severity}/{case_slug}"
    )


def build_case(records: list[dict[str, Any]], case: tuple[str, str, str, int | None], args: argparse.Namespace) -> list[dict[str, Any]]:
    label, slug, kind, index = case
    output = copy.deepcopy(records)
    for record in output:
        if kind == "history":
            assert index is not None
            mark_history_corrupted(record, index, args)
        elif kind == "current":
            mark_current_corrupted(record, args)
        finalize_record(record, label, slug, args)
    return output


def write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build single-position corruption protocols for temporal sensitivity analysis."
    )
    parser.add_argument("--root", required=True, help="OccStress-code repository root used for relative output paths.")
    parser.add_argument("--source-protocol", required=True, help="Clean H4/F6 protocol to use as base.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corruption", default="semantic")
    parser.add_argument("--severity", default="hard")
    parser.add_argument("--source-model", default="I2-World")
    parser.add_argument("--corrupt-occ-root", required=True)
    parser.add_argument("--corrupt-event-root")
    parser.add_argument(
        "--prefix-map",
        action="append",
        default=[],
        help="Path prefix rewrite in OLD=NEW form. Applied before constructing sweep cases.",
    )
    args = parser.parse_args()

    args.root = Path(args.root).resolve()
    args.source_protocol = Path(args.source_protocol).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.corrupt_occ_root = Path(args.corrupt_occ_root).resolve()
    args.corrupt_event_root = Path(args.corrupt_event_root).resolve() if args.corrupt_event_root else None

    prefix_map = []
    for item in args.prefix_map:
        old, sep, new = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --prefix-map entry: {item!r}")
        prefix_map.append((old, new))

    with args.source_protocol.open("rb") as handle:
        base_records = pickle.load(handle)
    base_records = rewrite_paths(base_records, prefix_map)

    cases = [("clean", "clean", "clean", None)] + POSITIONS
    manifest = {
        "model": args.source_model,
        "corruption": args.corruption,
        "severity": args.severity,
        "source_protocol": str(args.source_protocol),
        "positions": ["t-4", "t-3", "t-2", "t-1", "t"],
        "horizons": ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"],
        "protocols": {},
    }

    for label, slug, kind, index in cases:
        if kind == "clean":
            records = copy.deepcopy(base_records)
            for record in records:
                finalize_record(record, "clean", "clean", args)
        else:
            records = build_case(base_records, (label, slug, kind, index), args)
        out_path = args.output_dir / f"position_sweep_{args.corruption}_{args.severity}_{slug}_H4_F6_val_backbone.pkl"
        write_pickle(out_path, records)
        manifest["protocols"][label] = str(out_path)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    list_path = args.output_dir / "protocols.txt"
    list_path.write_text("\n".join(manifest["protocols"].values()) + "\n")
    print(f"wrote {len(cases)} protocols to {args.output_dir}")
    print(f"manifest={manifest_path}")
    print(f"protocol_list={list_path}")


if __name__ == "__main__":
    main()
