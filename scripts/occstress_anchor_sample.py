#!/usr/bin/env python3
"""Create anchor-level OccStress sample candidates and review pages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from occstress_sample_review import CLASS_COLORS, choose_numeric_array, semantic_bev


PROTOCOL_SUFFIX = "_H4_F6_val_backbone.pkl"
PROTOCOL_ORDER = ("current_only", "recent_burst", "history_only")
FRAME_LABELS = ("t-4", "t-3", "t-2", "t-1", "t", "t+1", "t+2", "t+3", "t+4", "t+5", "t+6")


def load_pickle(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} is {type(data).__name__}, expected list")
    return data


def dump_pickle(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def protocol_mode(filename: str) -> str | None:
    if filename == "H4_F6_val_backbone.pkl":
        return "clean"
    if not filename.endswith(PROTOCOL_SUFFIX):
        return None
    return filename[: -len(PROTOCOL_SUFFIX)]


def category_from_protocol(rel_path: Path) -> tuple[str, str, str, str] | None:
    parts = rel_path.parts
    if not parts:
        return None
    mode = protocol_mode(parts[-1])
    if mode is None or mode == "clean":
        return None
    if parts[0] == "manual":
        if len(parts) == 3 and parts[1] == "traffic":
            category_id = "manual__traffic"
            return category_id, "manual/traffic", "traffic", "all"
        if len(parts) == 4 and parts[1] != "clean":
            corruption, severity = parts[1], parts[2]
            category_id = f"manual__{corruption}__{severity}"
            return category_id, f"manual/{corruption}/{severity}", corruption, severity
    if parts[0] == "upstream" and len(parts) == 6:
        _, subtrack, model, corruption, severity, _ = parts
        if corruption == "clean":
            return None
        category_id = f"upstream__{subtrack}__{model}__{corruption}__{severity}"
        return category_id, f"upstream/{subtrack}/{model}/{corruption}/{severity}", corruption, severity
    return None


def discover_categories(protocol_root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in sorted(protocol_root.rglob("*.pkl")):
        rel_path = path.relative_to(protocol_root)
        parsed = category_from_protocol(rel_path)
        if parsed is None:
            continue
        category_id, group, corruption, severity = parsed
        mode = protocol_mode(path.name)
        if category_id not in grouped:
            grouped[category_id] = {
                "category_id": category_id,
                "group": group,
                "corruption": corruption,
                "severity": severity,
                "protocols": {},
            }
        grouped[category_id]["protocols"][mode] = rel_path.as_posix()

    categories = []
    for category in grouped.values():
        modes = set(category["protocols"])
        if category["category_id"] == "manual__traffic":
            if "history_only" not in modes:
                raise FileNotFoundError("manual traffic history_only protocol missing")
        elif modes != set(PROTOCOL_ORDER):
            raise FileNotFoundError(f"{category['category_id']} protocol modes incomplete: {sorted(modes)}")
        categories.append(category)

    categories.sort(key=lambda x: x["category_id"])
    for idx, category in enumerate(categories, start=1):
        category["category_index"] = idx
    return categories


def stable_rng(seed: int, category_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{category_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def select_anchor_records(records: list[dict[str, Any]], seed: int, category_id: str, count: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"{category_id} has only {len(records)} records")
    indices = list(range(len(records)))
    stable_rng(seed, category_id).shuffle(indices)
    selected = [records[i] for i in indices[:count]]
    return selected


def strip_known_prefix(path_value: str, project_root: Path) -> str:
    path_value = path_value.strip()
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            marker = "/OccStress-code/"
            if marker in path_value:
                return path_value.split(marker, 1)[1]
    return path_value


def source_and_sample_rel(path_value: str, project_root: Path) -> tuple[Path, str]:
    rel = strip_known_prefix(path_value, project_root)
    if rel.startswith("data/OccStress/"):
        return project_root / rel, rel[len("data/OccStress/") :]
    if rel.startswith("data/OccStress/"):
        return project_root / rel, rel[len("data/OccStress/") :]
    if rel.startswith("data/OccStress_HF/"):
        tail = rel[len("data/OccStress_HF/") :]
        return project_root / rel, tail
    if rel.startswith("data/nuscenes/gts/"):
        tail = rel[len("data/nuscenes/gts/") :]
        return project_root / rel, f"occ/manual/clean/{tail}"
    if rel.startswith(("occ/", "events/", "cache/", "meta/", "protocols/")):
        return project_root / "data" / "OccStress" / rel, rel
    if rel.startswith("sample/"):
        return project_root / "data" / "OccStress_HF" / rel, rel[len("sample/") :]
    raise ValueError(f"Unsupported path reference: {path_value}")


def rewrite_paths_and_collect(
    obj: Any,
    project_root: Path,
    file_map: dict[str, Path],
) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key.endswith("_path") and isinstance(value, str) and value:
                src, sample_rel = source_and_sample_rel(value, project_root)
                file_map[sample_rel] = src
                out[key] = sample_rel
            else:
                out[key] = rewrite_paths_and_collect(value, project_root, file_map)
        return out
    if isinstance(obj, list):
        return [rewrite_paths_and_collect(value, project_root, file_map) for value in obj]
    return obj


def copy_file_map(file_map: dict[str, Path], out_root: Path) -> tuple[int, list[dict[str, str]]]:
    copied = 0
    missing: list[dict[str, str]] = []
    for sample_rel, src in sorted(file_map.items()):
        dst = out_root / sample_rel
        if not src.exists():
            missing.append({"sample_rel": sample_rel, "source": str(src)})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
        copied += 1
    return copied, missing


def copy_misalignment_cache(
    project_root: Path,
    out_root: Path,
    category: dict[str, Any],
    selected_records: list[dict[str, Any]],
) -> list[str]:
    if category["corruption"] != "misalignment":
        return []
    cache_paths: list[str] = []
    severity = category["severity"]
    for record in selected_records:
        scene = record["scene_name"]
        token = record["anchor_token"]
        src_dir = project_root / "data" / "OccStress" / "cache" / "manual" / "misalignment" / severity / scene
        for src in sorted(src_dir.glob(f"{token}__H4__misalignment_{severity}__*.npz")):
            sample_rel = f"cache/manual/misalignment/{severity}/{scene}/{src.name}"
            dst = out_root / sample_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or src.stat().st_size != dst.stat().st_size:
                shutil.copy2(src, dst)
            cache_paths.append(sample_rel)
    return cache_paths


def sequence_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for idx, hist in enumerate(record.get("history", [])):
        frames.append(
            {
                "label": FRAME_LABELS[idx],
                "role": "input",
                "token": hist.get("token"),
                "source": hist.get("occ_source") or hist.get("source"),
                "occ_path": hist.get("occ_path"),
                "event_path": hist.get("event_path"),
            }
        )
    current = record.get("current_input", {})
    frames.append(
        {
            "label": "t",
            "role": "input",
            "token": current.get("token"),
            "source": current.get("source"),
            "occ_path": current.get("occ_path"),
            "event_path": current.get("event_path"),
        }
    )
    for idx, future in enumerate(record.get("future_targets", []), start=1):
        frames.append(
            {
                "label": f"t+{idx}",
                "role": "future_gt",
                "token": future.get("token"),
                "source": future.get("source"),
                "occ_path": future.get("occ_path"),
                "event_path": future.get("event_path"),
            }
        )
    return frames


def rel_protocol_output_path(protocol_rel: str) -> str:
    return f"protocols/{protocol_rel}"


def build_candidates(
    project_root: Path,
    protocol_root: Path,
    out_root: Path,
    seed: int,
    count: int,
    force: bool,
) -> None:
    if out_root.exists() and force:
        backup = out_root.with_name(f"{out_root.name}.frame_level_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        out_root.rename(backup)
        print(f"moved existing candidates to {backup}")
    out_root.mkdir(parents=True, exist_ok=True)

    categories = discover_categories(protocol_root)
    if len(categories) != 61:
        raise RuntimeError(f"Expected 61 categories, found {len(categories)}")

    manifest_categories = []
    manifest_entries = []
    total_protocol_records = 0
    total_file_map: dict[str, Path] = {}
    clean_anchor_by_group: dict[str, set[str]] = defaultdict(set)

    for category in categories:
        reference_mode = "history_only" if category["category_id"] == "manual__traffic" else "current_only"
        reference_rel = category["protocols"][reference_mode]
        reference_records = load_pickle(protocol_root / reference_rel)
        selected_reference = select_anchor_records(reference_records, seed, category["category_id"], count)
        selected_tokens = [record["anchor_token"] for record in selected_reference]
        selected_token_set = set(selected_tokens)

        mode_entries: dict[str, dict[str, Any]] = {}
        protocol_manifest: dict[str, str] = {}
        first_records_by_token: dict[str, dict[str, Any]] = {}

        for mode in [m for m in PROTOCOL_ORDER if m in category["protocols"]]:
            protocol_rel = category["protocols"][mode]
            records = load_pickle(protocol_root / protocol_rel)
            by_token = {record["anchor_token"]: record for record in records}
            missing_tokens = [token for token in selected_tokens if token not in by_token]
            if missing_tokens:
                raise RuntimeError(f"{category['category_id']} {mode} missing selected anchors: {missing_tokens[:5]}")
            subset_raw = [by_token[token] for token in selected_tokens]
            subset = []
            for record in subset_raw:
                file_map: dict[str, Path] = {}
                rewritten = rewrite_paths_and_collect(copy.deepcopy(record), project_root, file_map)
                total_file_map.update(file_map)
                subset.append(rewritten)
                first_records_by_token.setdefault(rewritten["anchor_token"], rewritten)
            out_protocol_rel = rel_protocol_output_path(protocol_rel)
            dump_pickle(out_root / out_protocol_rel, subset)
            protocol_manifest[mode] = out_protocol_rel
            total_protocol_records += len(subset)
            mode_entries[mode] = {record["anchor_token"]: record for record in subset}

        cache_paths = copy_misalignment_cache(project_root, out_root, category, selected_reference)
        if category["group"].startswith("manual/"):
            clean_anchor_by_group["manual"].update(selected_tokens)
        elif category["group"].startswith("upstream/camera_only/stcocc/"):
            clean_anchor_by_group["upstream/camera_only/stcocc"].update(selected_tokens)
        elif category["group"].startswith("upstream/pointcloud_fusion/sdgocc/"):
            clean_anchor_by_group["upstream/pointcloud_fusion/sdgocc"].update(selected_tokens)

        category_entry = {
            **{k: category[k] for k in ("category_id", "category_index", "group", "corruption", "severity")},
            "n_candidates": count,
            "protocols": protocol_manifest,
        }
        manifest_categories.append(category_entry)

        for rank, token in enumerate(selected_tokens, start=1):
            rec0 = first_records_by_token[token]
            entry_protocols = {}
            for mode, token_map in mode_entries.items():
                record = token_map[token]
                entry_protocols[mode] = {
                    "sample_id": record.get("sample_id"),
                    "protocol_path": protocol_manifest[mode],
                    "frame_protocol": record.get("frame_protocol"),
                    "sequence": sequence_from_record(record),
                }
            manifest_entries.append(
                {
                    "category_id": category["category_id"],
                    "category_index": category["category_index"],
                    "candidate_rank": rank,
                    "group": category["group"],
                    "corruption": category["corruption"],
                    "severity": category["severity"],
                    "scene": rec0.get("scene_name"),
                    "anchor_token": token,
                    "sample_id": rec0.get("sample_id"),
                    "protocols": entry_protocols,
                    "cache_paths": [p for p in cache_paths if f"/{rec0.get('scene_name')}/" in p and token in p],
                    "selected": False,
                }
            )

    write_clean_baseline_protocols(protocol_root, out_root, project_root, clean_anchor_by_group, total_file_map)
    copied, missing = copy_file_map(total_file_map, out_root)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced files; first: {missing[:5]}")

    copy_meta(project_root, out_root)

    manifest = {
        "created_at_remote": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "n_per_category": count,
        "category_count": len(manifest_categories),
        "candidate_count": len(manifest_entries),
        "protocol_record_count": total_protocol_records,
        "copied_file_count": copied,
        "categories": manifest_categories,
        "entries": manifest_entries,
    }
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "candidate_anchor_index.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)
    with (manifest_dir / "candidate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_count": len(manifest_entries),
                "category_count": len(manifest_categories),
                "n_per_category": count,
                "protocol_record_count": total_protocol_records,
                "copied_file_count": copied,
                "missing_file_count": len(missing),
                "anchor_level": True,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    write_readme(out_root)
    print(json.dumps(json.load(open(manifest_dir / "candidate_summary.json")), indent=2))


def write_clean_baseline_protocols(
    protocol_root: Path,
    out_root: Path,
    project_root: Path,
    anchors_by_group: dict[str, set[str]],
    total_file_map: dict[str, Path],
) -> None:
    baseline_paths = {
        "manual": "manual/clean/H4_F6_val_backbone.pkl",
        "upstream/camera_only/stcocc": "upstream/camera_only/stcocc/clean/H4_F6_val_backbone.pkl",
        "upstream/pointcloud_fusion/sdgocc": "upstream/pointcloud_fusion/sdgocc/clean/H4_F6_val_backbone.pkl",
    }
    for group, rel in baseline_paths.items():
        anchors = anchors_by_group.get(group)
        src = protocol_root / rel
        if not anchors or not src.exists():
            continue
        records = load_pickle(src)
        subset = []
        for record in records:
            if record.get("anchor_token") in anchors:
                file_map: dict[str, Path] = {}
                subset.append(rewrite_paths_and_collect(copy.deepcopy(record), project_root, file_map))
                total_file_map.update(file_map)
        dump_pickle(out_root / "protocols" / rel, subset)


def copy_meta(project_root: Path, out_root: Path) -> None:
    src = project_root / "data" / "OccStress" / "meta"
    if src.exists():
        shutil.copytree(src, out_root / "meta", dirs_exist_ok=True)


def write_readme(out_root: Path) -> None:
    text = """# OccStress Candidate Sample

This directory is an anchor-level sample candidate set.

Each candidate corresponds to one H4_F6 protocol anchor with 4 history inputs,
1 current input, and 6 future target frames. Non-traffic categories include all
three temporal protocols: current_only, recent_burst, and history_only. Traffic
uses its full-sequence protocol.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def frame_image(sample_root: Path, rel_path: str | None, cell_size: int) -> Image.Image:
    if not rel_path:
        return missing_image(cell_size, "missing")
    path = sample_root / rel_path
    if not path.exists():
        return missing_image(cell_size, "missing")
    try:
        key, arr = choose_numeric_array(path)
        labels = semantic_bev(arr)
        rgb = CLASS_COLORS[labels]
        rgb = np.rot90(rgb, 1)
        img = Image.fromarray(rgb, mode="RGB")
        return img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
    except Exception:  # noqa: BLE001
        return missing_image(cell_size, "error")


def missing_image(cell_size: int, text: str) -> Image.Image:
    img = Image.new("RGB", (cell_size, cell_size), (255, 246, 244))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, cell_size - 1, cell_size - 1], outline=(180, 35, 24))
    draw.text((6, cell_size // 2 - 6), text, fill=(180, 35, 24))
    return img


def draw_cell_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    draw.rectangle([x, y, x + 71, y + 16], fill=(255, 255, 255))
    draw.text((x + 3, y + 2), text, fill=(24, 33, 47), font=font)


def source_badge(source: str | None) -> tuple[str, tuple[int, int, int]]:
    if source in (None, "", "clean"):
        return "clean", (89, 99, 118)
    return str(source), (24, 120, 111)


def reference_mode(entry: dict[str, Any]) -> str:
    if "current_only" in entry["protocols"]:
        return "current_only"
    if "history_only" in entry["protocols"]:
        return "history_only"
    return next(iter(entry["protocols"]))


def render_sheet(
    sample_root: Path,
    entry: dict[str, Any],
    out_path: Path,
    cell_size: int,
    display_mode: str,
) -> None:
    if display_mode == "reference":
        modes = [reference_mode(entry)]
    elif display_mode == "all":
        modes = [mode for mode in PROTOCOL_ORDER if mode in entry["protocols"]]
    else:
        raise ValueError(display_mode)
    if not modes:
        modes = list(entry["protocols"].keys())
    image_cache: dict[str, Image.Image] = {}

    def cached_frame(rel_path: str | None) -> Image.Image:
        cache_key = rel_path or "__missing__"
        if cache_key not in image_cache:
            image_cache[cache_key] = frame_image(sample_root, rel_path, cell_size)
        return image_cache[cache_key]

    left = 128
    gap = 6
    row_h = cell_size + 38
    header_h = 30
    width = left + len(FRAME_LABELS) * (cell_size + gap) + gap
    height = header_h + len(modes) * row_h + 8
    img = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for col, label in enumerate(FRAME_LABELS):
        x = left + col * (cell_size + gap)
        draw.text((x + 4, 8), label, fill=(24, 33, 47), font=font)

    for row, mode in enumerate(modes):
        y = header_h + row * row_h
        draw.text((10, y + 6), mode, fill=(24, 33, 47), font=font)
        sequence = entry["protocols"][mode]["sequence"]
        for col, frame in enumerate(sequence[: len(FRAME_LABELS)]):
            x = left + col * (cell_size + gap)
            frame_img = cached_frame(frame.get("occ_path"))
            img.paste(frame_img, (x, y))
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline=(217, 222, 231))
            label = frame.get("label") or FRAME_LABELS[col]
            draw_cell_label(draw, (x, y), label, font)
            badge, color = source_badge(frame.get("source"))
            draw.text((x + 3, y + cell_size + 4), badge[:18], fill=color, font=font)
            role = "in" if frame.get("role") == "input" else "gt"
            draw.text((x + 3, y + cell_size + 18), role, fill=(104, 115, 134), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)


def render_sheet_task(args: tuple[str, dict[str, Any], str, int, str]) -> str:
    sample_root, entry, out_path, cell_size, display_mode = args
    render_sheet(Path(sample_root), entry, Path(out_path), cell_size, display_mode)
    return out_path


def annotated_frame(
    sample_root: Path,
    frame: dict[str, Any],
    frame_size: int,
    index: int,
    total: int,
) -> Image.Image:
    img = frame_image(sample_root, frame.get("occ_path"), frame_size).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    label = frame.get("label") or f"{index + 1}/{total}"
    source, color = source_badge(frame.get("source"))
    role = "input" if frame.get("role") == "input" else "future GT"
    title = f"{label}  {role}  {source}"
    draw.rectangle([0, 0, frame_size - 1, 25], fill=(255, 255, 255))
    draw.text((7, 7), title[:44], fill=color, font=font)
    draw.rectangle([0, 0, frame_size - 1, frame_size - 1], outline=(24, 33, 47))
    return img


def render_animation(
    sample_root: Path,
    entry: dict[str, Any],
    out_path: Path,
    frame_size: int,
    display_mode: str,
    duration_ms: int,
    animation_format: str,
    quality: int,
) -> None:
    mode = reference_mode(entry) if display_mode == "reference" else next(iter(entry["protocols"]))
    sequence = entry["protocols"][mode]["sequence"][: len(FRAME_LABELS)]
    frames = [
        annotated_frame(sample_root, frame, frame_size, idx, len(sequence))
        for idx, frame in enumerate(sequence)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if animation_format == "webp":
        frames[0].save(
            out_path,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=4,
        )
    elif animation_format == "gif":
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    else:
        raise ValueError(animation_format)


def render_animation_task(args: tuple[str, dict[str, Any], str, int, str, int, str, int]) -> str:
    sample_root, entry, out_path, frame_size, display_mode, duration_ms, animation_format, quality = args
    render_animation(Path(sample_root), entry, Path(out_path), frame_size, display_mode, duration_ms, animation_format, quality)
    return out_path


def build_review(
    sample_root: Path,
    out_dir: Path,
    cell_size: int,
    gif_size: int,
    duration_ms: int,
    force: bool,
    display_mode: str,
    media: str,
    animation_format: str,
    quality: int,
    workers: int,
) -> None:
    manifest_path = sample_root / "manifests" / "candidate_anchor_index.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = out_dir / "sheets"
    gifs_dir = out_dir / "animations"

    categories_by_id = {c["category_id"]: {**c, "items": []} for c in data["categories"]}
    render_tasks: list[tuple[str, dict[str, Any], str, int, str]] = []
    gif_tasks: list[tuple[str, dict[str, Any], str, int, str, int, str, int]] = []
    for idx, entry in enumerate(data["entries"], start=1):
        category_id = entry["category_id"]
        sheet_path = sheets_dir / category_id / f"{int(entry['candidate_rank']):02d}.png"
        animation_ext = "webp" if animation_format == "webp" else "gif"
        gif_path = gifs_dir / category_id / f"{int(entry['candidate_rank']):02d}.{animation_ext}"
        if media in {"sheet", "both"} and (force or not sheet_path.exists()):
            render_tasks.append((str(sample_root), entry, str(sheet_path), cell_size, display_mode))
        if media in {"gif", "both"} and (force or not gif_path.exists()):
            gif_tasks.append(
                (str(sample_root), entry, str(gif_path), gif_size, display_mode, duration_ms, animation_format, quality)
            )
        item = {
            "category_id": category_id,
            "category_index": entry["category_index"],
            "candidate_rank": entry["candidate_rank"],
            "group": entry["group"],
            "corruption": entry["corruption"],
            "severity": entry["severity"],
            "scene": entry["scene"],
            "anchor_token": entry["anchor_token"],
            "sample_id": entry["sample_id"],
            "protocols": entry["protocols"],
            "display_mode": display_mode,
            "display_protocol": reference_mode(entry) if display_mode == "reference" else "all",
            "sheet": sheet_path.relative_to(out_dir).as_posix() if media in {"sheet", "both"} else None,
            "animation": gif_path.relative_to(out_dir).as_posix() if media in {"gif", "both"} else None,
        }
        categories_by_id[category_id]["items"].append(item)

    if workers > 1:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(render_sheet_task, task) for task in render_tasks]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 100 == 0:
                    print(f"rendered {done}/{len(render_tasks)}")
    else:
        for done, task in enumerate(render_tasks, start=1):
            render_sheet_task(task)
            if done % 100 == 0:
                print(f"rendered {done}/{len(render_tasks)}")

    if workers > 1:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(render_animation_task, task) for task in gif_tasks]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 100 == 0:
                    print(f"rendered gifs {done}/{len(gif_tasks)}")
    else:
        for done, task in enumerate(gif_tasks, start=1):
            render_animation_task(task)
            if done % 100 == 0:
                print(f"rendered gifs {done}/{len(gif_tasks)}")

    review_categories = []
    for category in sorted(categories_by_id.values(), key=lambda c: c["category_index"]):
        category["items"].sort(key=lambda x: int(x["candidate_rank"]))
        review_categories.append(category)

    review = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_candidate_index": "manifests/candidate_anchor_index.json",
        "anchor_level": True,
        "seed": data.get("seed"),
        "n_per_category": data.get("n_per_category"),
        "selection_target_per_category": 10,
        "category_count": len(review_categories),
        "candidate_count": sum(len(c["items"]) for c in review_categories),
        "frame_labels": FRAME_LABELS,
        "display_mode": display_mode,
        "media": media,
        "categories": review_categories,
    }
    with (out_dir / "review_index.json").open("w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=True, indent=2)
    write_review_html(out_dir / "review.html", review)
    print(f"wrote {out_dir / 'review.html'}")


def write_review_html(out_path: Path, review: dict[str, Any]) -> None:
    payload = json.dumps(review, ensure_ascii=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OccStress Anchor Sample Review</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #18212f;
      --muted: #687386;
      --line: #d9dee7;
      --accent: #18786f;
      --accent-soft: #dcefeb;
      --warn: #a15c00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    header {{
      min-height: 56px;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 10;
      flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    button, select {{
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: white; }}
    .spacer {{ flex: 1; }}
    .status {{ color: var(--muted); white-space: nowrap; }}
    .app {{ display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: calc(100vh - 56px); }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 56px);
      position: sticky;
      top: 56px;
    }}
    main {{ padding: 16px; min-width: 0; }}
    .category-button {{
      width: 100%;
      height: auto;
      min-height: 48px;
      text-align: left;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
      padding: 8px 10px;
    }}
    .category-button.active {{ background: var(--accent-soft); border-color: #9ccdc6; }}
    .category-name {{ overflow-wrap: anywhere; font-weight: 650; line-height: 1.2; }}
    .category-meta {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .pill {{
      border-radius: 999px;
      padding: 2px 8px;
      background: #eef1f5;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .pill.done {{ background: var(--accent-soft); color: var(--accent); font-weight: 650; }}
    .pill.warn {{ background: #fff2d8; color: var(--warn); font-weight: 650; }}
    h2 {{ margin: 0; font-size: 20px; letter-spacing: 0; overflow-wrap: anywhere; }}
    .subhead {{ color: var(--muted); margin-top: 4px; }}
    .heading {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: start; margin-bottom: 12px; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-width: 0; }}
    .card.selected {{ border-color: var(--accent); outline: 2px solid var(--accent-soft); }}
    .card-top {{ display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 8px; align-items: start; margin-bottom: 8px; }}
    input[type="checkbox"] {{ width: 18px; height: 18px; accent-color: var(--accent); }}
    .rank {{ font-weight: 700; margin-bottom: 2px; }}
    .path {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; line-height: 1.25; }}
    .sheet {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; display: block; }}
    .animation {{
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      image-rendering: pixelated;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      display: block;
    }}
    @media (max-width: 980px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ position: static; max-height: 260px; border-right: 0; border-bottom: 1px solid var(--line); }}
      main {{ padding: 12px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>OccStress Anchor Sample Review</h1>
    <span class="status" id="globalStatus"></span>
    <div class="spacer"></div>
    <select id="categorySelect"></select>
    <button id="incompleteButton">Next incomplete</button>
    <button class="primary" id="exportButton">Export JSON</button>
  </header>
  <div class="app">
    <aside id="sidebar"></aside>
    <main>
      <div class="heading">
        <div>
          <h2 id="categoryTitle"></h2>
          <div class="subhead" id="categorySubhead"></div>
        </div>
        <span class="pill" id="categoryCount"></span>
      </div>
      <div class="toolbar">
        <button id="clearCategory">Clear category</button>
        <button id="selectFirstTen">Select first 10</button>
      </div>
      <div class="grid" id="cards"></div>
    </main>
  </div>
  <script id="review-data" type="application/json">{payload}</script>
  <script>
    const review = JSON.parse(document.getElementById('review-data').textContent);
    const target = review.selection_target_per_category || 10;
    const storageKey = 'occstress-anchor-sample-review-' + (review.seed || 'default');
    const selected = new Map();
    let current = 0;
    function selectedSet(categoryId) {{
      if (!selected.has(categoryId)) selected.set(categoryId, new Set());
      return selected.get(categoryId);
    }}
    function loadState() {{
      try {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;
        const obj = JSON.parse(raw);
        for (const [category, ranks] of Object.entries(obj)) selected.set(category, new Set(ranks.map(Number)));
      }} catch (err) {{ console.warn(err); }}
    }}
    function saveState() {{
      const obj = {{}};
      for (const [category, ranks] of selected.entries()) obj[category] = Array.from(ranks).sort((a, b) => a - b);
      localStorage.setItem(storageKey, JSON.stringify(obj));
    }}
    function totalSelected() {{
      let total = 0;
      for (const ranks of selected.values()) total += ranks.size;
      return total;
    }}
    function categoryStatus(category) {{
      const count = selectedSet(category.category_id).size;
      return [count === target ? 'done' : 'warn', count + '/' + target];
    }}
    function setCurrent(index) {{
      current = Math.max(0, Math.min(review.categories.length - 1, index));
      render();
    }}
    function renderSidebar() {{
      const sidebar = document.getElementById('sidebar');
      const select = document.getElementById('categorySelect');
      sidebar.textContent = '';
      select.textContent = '';
      review.categories.forEach((category, index) => {{
        const [klass, text] = categoryStatus(category);
        const button = document.createElement('button');
        button.className = 'category-button' + (index === current ? ' active' : '');
        button.onclick = () => setCurrent(index);
        button.innerHTML = `<span><span class="category-name">${{category.category_index}}. ${{category.category_id}}</span><span class="category-meta">${{category.group}}</span></span><span class="pill ${{klass}}">${{text}}</span>`;
        sidebar.appendChild(button);
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${{category.category_index}}. ${{category.category_id}} (${{text}})`;
        select.appendChild(option);
      }});
      select.value = String(current);
    }}
    function renderGlobalStatus() {{
      const complete = review.categories.filter(c => selectedSet(c.category_id).size === target).length;
      document.getElementById('globalStatus').textContent = `${{complete}}/${{review.categories.length}} categories, ${{totalSelected()}} selected`;
    }}
    function renderCards() {{
      const category = review.categories[current];
      const set = selectedSet(category.category_id);
      document.getElementById('categoryTitle').textContent = category.category_id;
      document.getElementById('categorySubhead').textContent = `${{category.group}} / ${{category.corruption}} / ${{category.severity}}`;
      const [klass, text] = categoryStatus(category);
      const pill = document.getElementById('categoryCount');
      pill.className = 'pill ' + klass;
      pill.textContent = text;
      const cards = document.getElementById('cards');
      cards.textContent = '';
      category.items.forEach(item => {{
        const checked = set.has(Number(item.candidate_rank));
        const card = document.createElement('section');
        card.className = 'card' + (checked ? ' selected' : '');
        const mediaSrc = item.animation || item.sheet;
        const mediaClass = item.animation ? 'animation' : 'sheet';
        const mediaAlt = item.animation ? 'H4 F6 sequence animation' : 'H4 F6 sequence sheet';
        card.innerHTML = `<div class="card-top"><input type="checkbox" ${{checked ? 'checked' : ''}}><div><div class="rank">#${{String(item.candidate_rank).padStart(2, '0')}} ${{item.scene}} / ${{item.anchor_token}}</div><div class="path">${{item.sample_id}}</div></div></div><img class="${{mediaClass}}" src="${{mediaSrc}}" loading="lazy" alt="${{mediaAlt}}">`;
        const checkbox = card.querySelector('input');
        checkbox.addEventListener('change', () => {{
          const rank = Number(item.candidate_rank);
          if (checkbox.checked && set.size >= target && !set.has(rank)) {{
            checkbox.checked = false;
            alert('This category already has ' + target + ' samples.');
            return;
          }}
          if (checkbox.checked) set.add(rank);
          else set.delete(rank);
          saveState();
          render();
        }});
        cards.appendChild(card);
      }});
    }}
    function render() {{
      renderSidebar();
      renderGlobalStatus();
      renderCards();
    }}
    function exportSelection() {{
      const payload = {{
        created_at: new Date().toISOString(),
        anchor_level: true,
        source_seed: review.seed,
        target_per_category: target,
        category_count: review.categories.length,
        selected_count: totalSelected(),
        complete: review.categories.every(c => selectedSet(c.category_id).size === target),
        categories: review.categories.map(category => {{
          const ranks = selectedSet(category.category_id);
          return {{
            category_id: category.category_id,
            category_index: category.category_index,
            group: category.group,
            corruption: category.corruption,
            severity: category.severity,
            selected_ranks: Array.from(ranks).sort((a, b) => a - b),
            selected_items: category.items.filter(item => ranks.has(Number(item.candidate_rank)))
          }};
        }})
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'occstress_anchor_sample_selection.json';
      a.click();
      URL.revokeObjectURL(url);
    }}
    document.getElementById('categorySelect').addEventListener('change', event => setCurrent(Number(event.target.value)));
    document.getElementById('incompleteButton').addEventListener('click', () => {{
      const next = review.categories.findIndex((category, index) => index > current && selectedSet(category.category_id).size !== target);
      if (next >= 0) setCurrent(next);
      else {{
        const first = review.categories.findIndex(category => selectedSet(category.category_id).size !== target);
        if (first >= 0) setCurrent(first);
      }}
    }});
    document.getElementById('clearCategory').addEventListener('click', () => {{
      selectedSet(review.categories[current].category_id).clear();
      saveState();
      render();
    }});
    document.getElementById('selectFirstTen').addEventListener('click', () => {{
      const category = review.categories[current];
      selected.set(category.category_id, new Set(category.items.slice(0, target).map(item => Number(item.candidate_rank))));
      saveState();
      render();
    }});
    document.getElementById('exportButton').addEventListener('click', exportSelection);
    loadState();
    render();
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def collect_path_values(obj: Any, paths: set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.endswith("_path") and isinstance(value, str) and value:
                paths.add(value)
            else:
                collect_path_values(value, paths)
    elif isinstance(obj, list):
        for value in obj:
            collect_path_values(value, paths)


def assert_relative_path(path_value: str) -> None:
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Non-relative or unsafe path in mini sample: {path_value}")


def copy_candidate_relative(candidate_root: Path, out_root: Path, rel_path: str) -> bool:
    assert_relative_path(rel_path)
    src = candidate_root / rel_path
    dst = out_root / rel_path
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def subset_protocol_by_sample_ids(protocol_path: Path, sample_ids: set[str]) -> list[dict[str, Any]]:
    records = load_pickle(protocol_path)
    subset = [record for record in records if record.get("sample_id") in sample_ids]
    if len(subset) != len(sample_ids):
        found = {record.get("sample_id") for record in subset}
        missing = sorted(sample_ids - found)
        raise RuntimeError(f"{protocol_path} missing {len(missing)} records: {missing[:5]}")
    return subset


def subset_clean_protocol_by_tokens(protocol_path: Path, anchor_tokens: set[str]) -> list[dict[str, Any]]:
    records = load_pickle(protocol_path)
    subset = [record for record in records if record.get("anchor_token") in anchor_tokens]
    if not subset:
        return []
    found = {record.get("anchor_token") for record in subset}
    missing = sorted(anchor_tokens - found)
    if missing:
        raise RuntimeError(f"{protocol_path} missing {len(missing)} clean anchors: {missing[:5]}")
    return subset


def build_mini(
    candidate_root: Path,
    out_root: Path,
    seed: int,
    count: int,
    force: bool,
) -> None:
    index_path = candidate_root / "manifests" / "candidate_anchor_index.json"
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("entries"):
        raise ValueError(f"No entries in {index_path}")

    if out_root.exists():
        if not force:
            raise FileExistsError(out_root)
        backup = out_root.with_name(f"{out_root.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        out_root.rename(backup)
        print(f"moved existing mini sample to {backup}")
    out_root.mkdir(parents=True, exist_ok=True)

    entries_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in data["entries"]:
        entries_by_category[entry["category_id"]].append(entry)

    selected_entries: list[dict[str, Any]] = []
    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in sorted(data["categories"], key=lambda c: int(c["category_index"])):
        category_id = category["category_id"]
        entries = list(entries_by_category[category_id])
        if len(entries) < count:
            raise ValueError(f"{category_id} has only {len(entries)} candidate entries")
        rng = stable_rng(seed, category_id)
        rng.shuffle(entries)
        chosen = sorted(entries[:count], key=lambda e: int(e["candidate_rank"]))
        selected_by_category[category_id] = chosen
        selected_entries.extend(chosen)

    rel_paths_to_copy: set[str] = set()
    protocol_records_written = 0
    protocol_record_counts: dict[str, int] = {}
    protocols_to_sample_ids: dict[str, set[str]] = defaultdict(set)
    clean_anchor_tokens: dict[str, set[str]] = defaultdict(set)

    for entry in selected_entries:
        group = entry["group"]
        if group.startswith("manual/"):
            clean_anchor_tokens["manual"].add(entry["anchor_token"])
        elif group.startswith("upstream/camera_only/stcocc/"):
            clean_anchor_tokens["upstream/camera_only/stcocc"].add(entry["anchor_token"])
        elif group.startswith("upstream/pointcloud_fusion/sdgocc/"):
            clean_anchor_tokens["upstream/pointcloud_fusion/sdgocc"].add(entry["anchor_token"])

        for protocol in entry["protocols"].values():
            protocols_to_sample_ids[protocol["protocol_path"]].add(protocol["sample_id"])
        for cache_path in entry.get("cache_paths") or []:
            rel_paths_to_copy.add(cache_path)

    for protocol_rel, sample_ids in sorted(protocols_to_sample_ids.items()):
        subset = subset_protocol_by_sample_ids(candidate_root / protocol_rel, sample_ids)
        dump_pickle(out_root / protocol_rel, subset)
        protocol_records_written += len(subset)
        protocol_record_counts[protocol_rel] = len(subset)
        for record in subset:
            collect_path_values(record, rel_paths_to_copy)

    clean_protocols = {
        "manual": "protocols/manual/clean/H4_F6_val_backbone.pkl",
        "upstream/camera_only/stcocc": "protocols/upstream/camera_only/stcocc/clean/H4_F6_val_backbone.pkl",
        "upstream/pointcloud_fusion/sdgocc": "protocols/upstream/pointcloud_fusion/sdgocc/clean/H4_F6_val_backbone.pkl",
    }
    clean_protocol_counts: dict[str, int] = {}
    for group, protocol_rel in clean_protocols.items():
        tokens = clean_anchor_tokens.get(group, set())
        src = candidate_root / protocol_rel
        if not tokens or not src.exists():
            continue
        subset = subset_clean_protocol_by_tokens(src, tokens)
        dump_pickle(out_root / protocol_rel, subset)
        clean_protocol_counts[protocol_rel] = len(subset)
        for record in subset:
            collect_path_values(record, rel_paths_to_copy)

    missing_paths = []
    copied_files = 0
    for rel_path in sorted(rel_paths_to_copy):
        if not rel_path:
            continue
        if copy_candidate_relative(candidate_root, out_root, rel_path):
            copied_files += 1
        else:
            missing_paths.append(rel_path)
    if missing_paths:
        raise FileNotFoundError(f"Missing {len(missing_paths)} selected mini files: {missing_paths[:10]}")

    for rel_dir in ("meta",):
        src = candidate_root / rel_dir
        if src.exists():
            shutil.copytree(src, out_root / rel_dir, dirs_exist_ok=True)

    mini_categories = []
    for category in sorted(data["categories"], key=lambda c: int(c["category_index"])):
        category_id = category["category_id"]
        chosen = selected_by_category[category_id]
        mini_categories.append(
            {
                **{k: category.get(k) for k in ("category_id", "category_index", "group", "corruption", "severity")},
                "n_selected": len(chosen),
                "selected_ranks": [entry["candidate_rank"] for entry in chosen],
                "selected_anchor_tokens": [entry["anchor_token"] for entry in chosen],
            }
        )

    mini_index = {
        "created_at_remote": datetime.now().isoformat(timespec="seconds"),
        "source_candidate_index": "sample/candidates/manifests/candidate_anchor_index.json",
        "source_seed": data.get("seed"),
        "selection_seed": seed,
        "n_per_category": count,
        "category_count": len(mini_categories),
        "candidate_count": len(selected_entries),
        "protocol_record_count": protocol_records_written,
        "clean_protocol_record_count": sum(clean_protocol_counts.values()),
        "categories": mini_categories,
        "entries": selected_entries,
        "protocol_record_counts": protocol_record_counts,
        "clean_protocol_record_counts": clean_protocol_counts,
    }
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "mini_index.json").open("w", encoding="utf-8") as f:
        json.dump(mini_index, f, ensure_ascii=True, indent=2)
    summary = {
        "mini": True,
        "anchor_level": True,
        "category_count": len(mini_categories),
        "candidate_count": len(selected_entries),
        "n_per_category": count,
        "selection_seed": seed,
        "protocol_record_count": protocol_records_written,
        "clean_protocol_record_count": sum(clean_protocol_counts.values()),
        "copied_file_count": copied_files,
        "missing_file_count": 0,
    }
    with (manifest_dir / "mini_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    (out_root / "README.md").write_text(
        "# OccStress Mini Sample\n\n"
        "Anchor-level mini set sampled from OccStress candidate samples.\n"
        f"It contains {count} random anchors per minimal category and preserves "
        "the same protocol/data layout with relative paths only.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def validate_mini(mini_root: Path) -> dict[str, Any]:
    index_path = mini_root / "manifests" / "mini_index.json"
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    missing_paths = []
    absolute_paths = []
    bad_sequences = []
    protocol_counts = {}
    for protocol_path in sorted((mini_root / "protocols").rglob("*.pkl")):
        records = load_pickle(protocol_path)
        rel = protocol_path.relative_to(mini_root).as_posix()
        protocol_counts[rel] = len(records)
        for record in records:
            paths: set[str] = set()
            collect_path_values(record, paths)
            if record.get("history") is not None and record.get("future_targets") is not None:
                n_frames = len(record.get("history", [])) + 1 + len(record.get("future_targets", []))
                if n_frames != 11:
                    bad_sequences.append((rel, record.get("sample_id"), n_frames))
            for path_value in paths:
                if Path(path_value).is_absolute() or ".." in Path(path_value).parts:
                    absolute_paths.append(path_value)
                elif not (mini_root / path_value).exists():
                    missing_paths.append(path_value)
    category_counts = {category["category_id"]: category["n_selected"] for category in data["categories"]}
    return {
        "category_count": data["category_count"],
        "candidate_count": data["candidate_count"],
        "category_count_min": min(category_counts.values()),
        "category_count_max": max(category_counts.values()),
        "protocol_file_count": len(protocol_counts),
        "missing_path_count": len(set(missing_paths)),
        "absolute_path_count": len(set(absolute_paths)),
        "bad_sequence_count": len(bad_sequences),
        "protocol_record_count": sum(protocol_counts.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-candidates")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--protocol-root", type=Path, required=True)
    build.add_argument("--out-root", type=Path, required=True)
    build.add_argument("--seed", type=int, default=20260506)
    build.add_argument("--count", type=int, default=30)
    build.add_argument("--force", action="store_true")

    review = subparsers.add_parser("build-review")
    review.add_argument("--sample-root", type=Path, required=True)
    review.add_argument("--out-dir", type=Path, required=True)
    review.add_argument("--cell-size", type=int, default=72)
    review.add_argument("--gif-size", type=int, default=320)
    review.add_argument("--duration-ms", type=int, default=550)
    review.add_argument("--display-mode", choices=["reference", "all"], default="reference")
    review.add_argument("--media", choices=["sheet", "gif", "both"], default="sheet")
    review.add_argument("--animation-format", choices=["gif", "webp"], default="gif")
    review.add_argument("--quality", type=int, default=78)
    review.add_argument("--workers", type=int, default=1)
    review.add_argument("--force", action="store_true")

    mini = subparsers.add_parser("build-mini")
    mini.add_argument("--candidate-root", type=Path, required=True)
    mini.add_argument("--out-root", type=Path, required=True)
    mini.add_argument("--seed", type=int, default=20260506)
    mini.add_argument("--count", type=int, default=10)
    mini.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate-mini")
    validate.add_argument("--mini-root", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-candidates":
        build_candidates(
            args.project_root.resolve(),
            args.protocol_root.resolve(),
            args.out_root.resolve(),
            args.seed,
            args.count,
            args.force,
        )
    elif args.command == "build-review":
        build_review(
            args.sample_root.resolve(),
            args.out_dir.resolve(),
            args.cell_size,
            args.gif_size,
            args.duration_ms,
            args.force,
            args.display_mode,
            args.media,
            args.animation_format,
            args.quality,
            max(1, args.workers),
        )
    elif args.command == "build-mini":
        build_mini(args.candidate_root.resolve(), args.out_root.resolve(), args.seed, args.count, args.force)
    elif args.command == "validate-mini":
        print(json.dumps(validate_mini(args.mini_root.resolve()), indent=2))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
