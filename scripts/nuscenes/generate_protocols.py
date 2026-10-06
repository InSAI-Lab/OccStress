#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from occstress.corruptions import misalignment as misalign
from occstress.corruptions import semantic
from occstress.datasets.construction import (
    clean_nuscenes_reference, manual_meta, manual_output, manual_relative,
    portable_reference, write_protocol,
)
from occstress.datasets.paths import data_root as shared_data_root
from occstress.protocols.layout import resolve_manual_protocol_path


CONTENT_CORRUPTIONS = {
    "semantic": ["easy", "mid", "hard"],
    "hole": ["easy", "mid", "hard"],
    "dropout": ["easy", "mid", "hard"],
}

FRAME_PROTOCOLS = ["history_k1", "current", "all_frame"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate protocol pkls for occupancy-state corruption benchmark.")
    parser.add_argument("--data-root", type=str, default="data/nuscenes")
    parser.add_argument("--subset-root", "--output-root", type=str, default=None,
                        help="Shared OccStress root; defaults to OCCSTRESS_DATA_ROOT.")
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=6)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--ann-file", type=str, default=None,
                        help="Optional pkl ann file used to restrict anchor tokens, e.g. val split only.")
    parser.add_argument("--protocol-suffix", type=str, default="val_backbone",
                        help="Optional suffix appended to protocol filenames and sample ids, e.g. val.")
    parser.add_argument("--backbone-path", type=str, default="",
                        help="Trusted clean backbone pkl defining the fixed anchor set.")
    parser.add_argument("--strict-backbone", action="store_true",
                        help="Error out when a backbone sample cannot be instantiated for a corruption protocol.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def history_tokens_for_anchor(sample, samples_by_token, history_length):
    cursor = sample
    history = []
    for _ in range(history_length):
        prev_token = cursor["prev"]
        if not prev_token:
            return None
        history.append(prev_token)
        cursor = samples_by_token[prev_token]
    history.reverse()
    return history


def future_tokens_for_anchor(sample, samples_by_token, future_length):
    cursor = sample
    future = []
    for _ in range(future_length):
        next_token = cursor["next"]
        if not next_token:
            return None
        future.append(next_token)
        cursor = samples_by_token[next_token]
    return future


from occstress.protocols.temporal import history_mask_for_protocol, target_active_for_protocol


def load_available_content(subset_root, corruption, severity):
    occ_root = manual_output(subset_root, "occ", corruption, severity)
    event_root = manual_output(subset_root, "events", corruption, severity)
    available = set()
    if not occ_root.exists() or not event_root.exists():
        return available
    for labels_path in occ_root.glob("*/*/labels.npz"):
        scene_name = labels_path.parent.parent.name
        token = labels_path.parent.name
        event_path = event_root / scene_name / f"{token}.json"
        if event_path.exists():
            available.add((scene_name, token))
    return available


def load_available_traffic(subset_root):
    occ_root = manual_output(subset_root, "occ", "traffic")
    event_root = manual_output(subset_root, "events", "traffic")
    available = set()
    if not occ_root.exists() or not event_root.exists():
        return available
    for labels_path in occ_root.glob("*/*/labels.npz"):
        scene_name = labels_path.parent.parent.name
        token = labels_path.parent.name
        event_path = event_root / scene_name / f"{token}.json"
        if event_path.exists():
            available.add((scene_name, token))
    return available


def clean_occ_relpath(data_root, scene_name, token):
    return clean_nuscenes_reference(scene_name, token)


def content_occ_relpath(subset_root, corruption, severity, scene_name, token):
    return manual_relative("occ", corruption, severity, scene_name, token, "labels.npz")


def content_event_relpath(subset_root, corruption, severity, scene_name, token):
    return manual_relative("events", corruption, severity, scene_name, f"{token}.json")


def traffic_occ_relpath(subset_root, scene_name, token):
    return manual_relative("occ", "traffic", scene_name, token, "labels.npz")


def traffic_event_relpath(subset_root, scene_name, token):
    return manual_relative("events", "traffic", scene_name, f"{token}.json")


def suffix_text(raw_suffix):
    if not raw_suffix:
        return ""
    return f"_{raw_suffix}"


def save_records(root, name, records, overwrite=False):
    # Keep temporal masks and record identifiers unchanged; only normalize filenames.
    name = name.replace("traffic_all_frame_", "traffic_")
    for old, new in (("history_k1", "recent_burst"), ("current", "current_only"),
                     ("all_frame", "history_only")):
        if not name.startswith("traffic_"):
            name = name.replace(f"_{old}_", f"_{new}_")
    path = resolve_manual_protocol_path(name, occstress_root=root)
    write_protocol(path, records, overwrite=overwrite)
    return portable_reference(path, root)


def retain_backbone_metadata(record, backbone):
    """Preserve auxiliary poses/controls; corruption fields override clean inputs."""
    if backbone is None:
        return record
    if isinstance(backbone, dict) and isinstance(record, dict):
        merged = copy.deepcopy(backbone)
        for key, value in record.items():
            merged[key] = retain_backbone_metadata(value, backbone.get(key))
        return merged
    if isinstance(backbone, list) and isinstance(record, list) and len(backbone) == len(record):
        return [retain_backbone_metadata(value, base) for value, base in zip(record, backbone)]
    return copy.deepcopy(record)


def build_content_record(sample, scene_name, history_tokens, future_tokens, history_length, future_length, subset_root, data_root, corruption, severity, frame_protocol, k, sample_suffix=""):
    protocol_mask = history_mask_for_protocol(history_length, frame_protocol, k)
    target_active = target_active_for_protocol(frame_protocol)
    history = []
    for idx, token in enumerate(history_tokens):
        use_corrupt = bool(protocol_mask[idx])
        history.append(
            {
                "index": idx,
                "token": token,
                "occ_source": corruption if use_corrupt else "clean",
                "occ_path": content_occ_relpath(subset_root, corruption, severity, scene_name, token) if use_corrupt else clean_occ_relpath(data_root, scene_name, token),
                "event_path": content_event_relpath(subset_root, corruption, severity, scene_name, token) if use_corrupt else None,
                "rt_source": "clean",
            }
        )
    return {
        "sample_id": f"{sample['token']}__H{history_length}__F{future_length}__{corruption}_{severity}__{frame_protocol}{sample_suffix}",
        "scene_name": scene_name,
        "scene_token": sample["scene_token"],
        "anchor_token": sample["token"],
        "anchor_timestamp": sample["timestamp"],
        "history_length": history_length,
        "history_tokens": history_tokens,
        "future_length": future_length,
        "future_tokens": future_tokens,
        "frame_protocol": frame_protocol,
        "k": k,
        "current_input": {
            "token": sample["token"],
            "source": corruption if target_active else "clean",
            "occ_path": content_occ_relpath(subset_root, corruption, severity, scene_name, sample["token"]) if target_active else clean_occ_relpath(data_root, scene_name, sample["token"]),
            "event_path": content_event_relpath(subset_root, corruption, severity, scene_name, sample["token"]) if target_active else None,
            "rt_source": "clean",
        },
        "target": {
            "token": sample["token"],
            "source": "clean",
            "occ_path": clean_occ_relpath(data_root, scene_name, sample["token"]),
            "event_path": None,
        },
        "future_targets": [
            {
                "index": idx,
                "token": token,
                "source": "clean",
                "occ_path": clean_occ_relpath(data_root, scene_name, token),
            }
            for idx, token in enumerate(future_tokens)
        ],
        "history": history,
        "corruption": {
            "type": corruption,
            "severity": severity,
            "frame_protocol": frame_protocol,
            "k": k,
        },
    }


def build_traffic_record(sample, scene_name, history_tokens, future_tokens, history_length, future_length, subset_root, data_root, sample_suffix=""):
    history = []
    for idx, token in enumerate(history_tokens):
        history.append(
            {
                "index": idx,
                "token": token,
                "occ_source": "traffic",
                "occ_path": traffic_occ_relpath(subset_root, scene_name, token),
                "event_path": traffic_event_relpath(subset_root, scene_name, token),
                "rt_source": "clean",
            }
        )
    return {
        "sample_id": f"{sample['token']}__H{history_length}__F{future_length}__traffic__all_frame{sample_suffix}",
        "scene_name": scene_name,
        "scene_token": sample["scene_token"],
        "anchor_token": sample["token"],
        "anchor_timestamp": sample["timestamp"],
        "history_length": history_length,
        "history_tokens": history_tokens,
        "future_length": future_length,
        "future_tokens": future_tokens,
        "frame_protocol": "all_frame",
        "k": None,
        "current_input": {
            "token": sample["token"],
            "source": "traffic",
            "occ_path": traffic_occ_relpath(subset_root, scene_name, sample["token"]),
            "event_path": traffic_event_relpath(subset_root, scene_name, sample["token"]),
            "rt_source": "clean",
        },
        "target": {
            "token": sample["token"],
            "source": "traffic",
            "occ_path": traffic_occ_relpath(subset_root, scene_name, sample["token"]),
            "event_path": traffic_event_relpath(subset_root, scene_name, sample["token"]),
        },
        "future_targets": [
            {
                "index": idx,
                "token": token,
                "source": "traffic",
                "occ_path": traffic_occ_relpath(subset_root, scene_name, token),
                "event_path": traffic_event_relpath(subset_root, scene_name, token),
            }
            for idx, token in enumerate(future_tokens)
        ],
        "history": history,
        "corruption": {
            "type": "traffic",
            "frame_protocol": "all_frame",
            "scope": "full_mirrored_scene",
            "mirror_axis": "y",
            "mirror_occupancy": True,
            "mirror_motion": True,
            "note": "Compatibility alias: traffic_all_frame means occupancy, targets, and ego-motion metadata are mirrored.",
        },
    }


def build_misalignment_record(sample, scene_name, history_tokens, future_tokens, history_length, future_length, data_root, severity, frame_protocol, k, metadata, sample_suffix=""):
    affected_mask = np.array(history_mask_for_protocol(history_length, frame_protocol, k), dtype=np.uint8)
    delta_rt, dx_m, dy_m, yaw_deg = misalign.build_delta_series(sample["token"], severity, history_length, affected_mask)
    current_active = target_active_for_protocol(frame_protocol)
    if current_active:
        current_mask = np.array([1], dtype=np.uint8)
        curr_delta_rt, curr_dx_m, curr_dy_m, curr_yaw_deg = misalign.build_delta_series(
            sample["token"] + "__current", severity, 1, current_mask)
        curr_delta_rt = curr_delta_rt[0]
        curr_dx_m = float(curr_dx_m[0])
        curr_dy_m = float(curr_dy_m[0])
        curr_yaw_deg = float(curr_yaw_deg[0])
    else:
        curr_delta_rt = np.eye(4, dtype=np.float32)
        curr_dx_m = 0.0
        curr_dy_m = 0.0
        curr_yaw_deg = 0.0

    history = []
    for idx, token in enumerate(history_tokens):
        history.append(
            {
                "index": idx,
                "token": token,
                "occ_source": "clean",
                "occ_path": clean_occ_relpath(data_root, scene_name, token),
                "rt_source": "misalignment" if int(affected_mask[idx]) == 1 else "clean",
            }
        )
    return {
        "sample_id": f"{sample['token']}__H{history_length}__F{future_length}__misalignment_{severity}__{frame_protocol}{sample_suffix}",
        "scene_name": scene_name,
        "scene_token": sample["scene_token"],
        "anchor_token": sample["token"],
        "anchor_timestamp": sample["timestamp"],
        "history_length": history_length,
        "history_tokens": history_tokens,
        "future_length": future_length,
        "future_tokens": future_tokens,
        "frame_protocol": frame_protocol,
        "k": k,
        "current_input": {
            "token": sample["token"],
            "source": "clean",
            "occ_path": clean_occ_relpath(data_root, scene_name, sample["token"]),
            "event_path": None,
            "rt_source": "misalignment" if current_active else "clean",
        },
        "target": {
            "token": sample["token"],
            "source": "clean",
            "occ_path": clean_occ_relpath(data_root, scene_name, sample["token"]),
        },
        "future_targets": [
            {
                "index": idx,
                "token": token,
                "source": "clean",
                "occ_path": clean_occ_relpath(data_root, scene_name, token),
            }
            for idx, token in enumerate(future_tokens)
        ],
        "history": history,
        "corruption": {
            "type": "misalignment",
            "severity": severity,
            "frame_protocol": frame_protocol,
            "k": k,
            "mode": misalign.MISALIGNMENT_CONFIGS[severity]["mode"],
        },
        "misalignment": {
            "affected_mask": affected_mask.tolist(),
            "dx_m": [float(x) for x in dx_m],
            "dy_m": [float(y) for y in dy_m],
            "yaw_deg": [float(v) for v in yaw_deg],
            "current_active": current_active,
            "current_dx_m": curr_dx_m,
            "current_dy_m": curr_dy_m,
            "current_yaw_deg": curr_yaw_deg,
            # Store transformation matrices as plain nested lists so protocol
            # files remain readable in older numpy environments such as the
            # legacy MMCV 1.7.2 stack used by II-World eval.
            "delta_rt": delta_rt.tolist(),
            "current_delta_rt": curr_delta_rt.tolist(),
        },
    }


def main():
    args = parse_args()
    if args.protocol_suffix == "val_backbone" and not args.backbone_path:
        raise ValueError("The formal val_backbone suite requires --backbone-path")
    nusc_root = os.path.join(args.data_root, "v1.0-trainval")
    metadata = semantic.collect_indices(nusc_root)
    samples_by_token = {row["token"]: row for row in metadata["samples"]}
    subset_root = shared_data_root(args.subset_root)
    data_root = args.data_root

    availability = {}
    for corruption, severities in CONTENT_CORRUPTIONS.items():
        for severity in severities:
            availability[(corruption, severity)] = load_available_content(subset_root, corruption, severity)
    traffic_available = load_available_traffic(subset_root)

    allowed_anchor_tokens = None
    if args.ann_file:
        with open(args.ann_file, "rb") as handle:
            ann = pickle.load(handle)
        allowed_anchor_tokens = {info['token'] for info in ann['infos']}
    backbone_records = None
    if args.backbone_path:
        with open(args.backbone_path, "rb") as handle:
            backbone_records = pickle.load(handle)
        args.strict_backbone = True
        if not backbone_records:
            raise ValueError("Clean backbone is empty")
        for record in backbone_records:
            if (len(record["history_tokens"]) != args.history_length
                    or len(record["future_tokens"]) != args.future_length):
                raise ValueError("Backbone input/future window disagrees with construction arguments")

    file_suffix = suffix_text(args.protocol_suffix)
    sample_suffix = f"__{args.protocol_suffix}" if args.protocol_suffix else ""

    summary = {
        "history_length": args.history_length,
        "future_length": args.future_length,
        "k": args.k,
        "ann_file": Path(args.ann_file).name if args.ann_file else None,
        "protocol_suffix": args.protocol_suffix,
        "files": {},
    }
    backbone_by_anchor = {record["anchor_token"]: record for record in (backbone_records or [])}

    def preserve(records):
        return [retain_backbone_metadata(record, backbone_by_anchor.get(record["anchor_token"]))
                for record in records]

    for corruption, severities in CONTENT_CORRUPTIONS.items():
        for severity in severities:
            available = availability[(corruption, severity)]
            for frame_protocol in FRAME_PROTOCOLS:
                records = []
                missing_samples = []
                protocol_mask = history_mask_for_protocol(args.history_length, frame_protocol, args.k)
                required_positions = [i for i, flag in enumerate(protocol_mask) if flag == 1]
                require_target = target_active_for_protocol(frame_protocol)
                source_iter = backbone_records if backbone_records is not None else metadata["samples"]
                for source_sample in source_iter:
                    if backbone_records is not None:
                        sample = samples_by_token[source_sample["anchor_token"]]
                        scene_name = source_sample["scene_name"]
                        history_tokens = list(source_sample["history_tokens"])
                        future_tokens = list(source_sample["future_tokens"])
                        sample_id = source_sample["sample_id"]
                    else:
                        sample = source_sample
                        if allowed_anchor_tokens is not None and sample["token"] not in allowed_anchor_tokens:
                            continue
                        scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
                        history_tokens = history_tokens_for_anchor(sample, samples_by_token, args.history_length)
                        future_tokens = future_tokens_for_anchor(sample, samples_by_token, args.future_length)
                        sample_id = sample["token"]
                    if history_tokens is None or future_tokens is None:
                        if args.strict_backbone and backbone_records is not None:
                            missing_samples.append((sample_id, "missing_history_or_future"))
                        continue
                    ok = True
                    for idx in required_positions:
                        if (scene_name, history_tokens[idx]) not in available:
                            ok = False
                            break
                    if ok and require_target and (scene_name, sample["token"]) not in available:
                        ok = False
                    if not ok:
                        if args.strict_backbone and backbone_records is not None:
                            missing_samples.append((sample_id, "missing_corruption_frame"))
                        continue
                    records.append(
                        build_content_record(
                            sample, scene_name, history_tokens, future_tokens, args.history_length, args.future_length, subset_root, data_root,
                            corruption, severity, frame_protocol, args.k, sample_suffix=sample_suffix
                        )
                    )
                if missing_samples:
                    preview = ", ".join(f"{sid}:{reason}" for sid, reason in missing_samples[:5])
                    raise RuntimeError(f"{corruption}/{severity}/{frame_protocol} missing {len(missing_samples)} backbone samples. Examples: {preview}")
                out_name = f"{corruption}_{severity}_{frame_protocol}_H{args.history_length}_F{args.future_length}{file_suffix}.pkl"
                reference = save_records(subset_root, out_name, preserve(records), args.overwrite)
                summary["files"][reference] = len(records)

    for severity in ["easy", "mid", "hard"]:
        for frame_protocol in FRAME_PROTOCOLS:
            records = []
            source_iter = backbone_records if backbone_records is not None else metadata["samples"]
            for source_sample in source_iter:
                if backbone_records is not None:
                    sample = samples_by_token[source_sample["anchor_token"]]
                    scene_name = source_sample["scene_name"]
                    history_tokens = list(source_sample["history_tokens"])
                    future_tokens = list(source_sample["future_tokens"])
                else:
                    sample = source_sample
                    if allowed_anchor_tokens is not None and sample["token"] not in allowed_anchor_tokens:
                        continue
                    scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
                    history_tokens = history_tokens_for_anchor(sample, samples_by_token, args.history_length)
                    future_tokens = future_tokens_for_anchor(sample, samples_by_token, args.future_length)
                if history_tokens is None or future_tokens is None:
                    continue
                records.append(
                    build_misalignment_record(
                        sample, scene_name, history_tokens, future_tokens, args.history_length, args.future_length, data_root,
                        severity, frame_protocol, args.k, metadata, sample_suffix=sample_suffix
                    )
                )
            out_name = f"misalignment_{severity}_{frame_protocol}_H{args.history_length}_F{args.future_length}{file_suffix}.pkl"
            reference = save_records(subset_root, out_name, preserve(records), args.overwrite)
            summary["files"][reference] = len(records)

    traffic_records = []
    missing_traffic = []
    source_iter = backbone_records if backbone_records is not None else metadata["samples"]
    for source_sample in source_iter:
        if backbone_records is not None:
            sample = samples_by_token[source_sample["anchor_token"]]
            scene_name = source_sample["scene_name"]
            history_tokens = list(source_sample["history_tokens"])
            future_tokens = list(source_sample["future_tokens"])
            sample_id = source_sample["sample_id"]
        else:
            sample = source_sample
            if allowed_anchor_tokens is not None and sample["token"] not in allowed_anchor_tokens:
                continue
            scene_name = metadata["scene_by_token"][sample["scene_token"]]["name"]
            history_tokens = history_tokens_for_anchor(sample, samples_by_token, args.history_length)
            future_tokens = future_tokens_for_anchor(sample, samples_by_token, args.future_length)
            sample_id = sample["token"]
        if history_tokens is None or future_tokens is None:
            if args.strict_backbone and backbone_records is not None:
                missing_traffic.append((sample_id, "missing_history_or_future"))
            continue
        required_traffic_tokens = list(history_tokens) + [sample["token"]] + list(future_tokens)
        ok = all((scene_name, token) in traffic_available for token in required_traffic_tokens)
        if not ok:
            if args.strict_backbone and backbone_records is not None:
                missing_traffic.append((sample_id, "missing_traffic_frame"))
            continue
        traffic_records.append(build_traffic_record(sample, scene_name, history_tokens, future_tokens, args.history_length, args.future_length, subset_root, data_root, sample_suffix=sample_suffix))
    if missing_traffic:
        preview = ", ".join(f"{sid}:{reason}" for sid, reason in missing_traffic[:5])
        raise RuntimeError(f"traffic/all_frame missing {len(missing_traffic)} backbone samples. Examples: {preview}")
    out_name = f"traffic_all_frame_H{args.history_length}_F{args.future_length}{file_suffix}.pkl"
    reference = save_records(subset_root, out_name, preserve(traffic_records), args.overwrite)
    summary["files"][reference] = len(traffic_records)

    meta_dir = manual_meta(subset_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary_path = meta_dir / f"protocol_summary_H{args.history_length}_F{args.future_length}_k{args.k}{file_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
