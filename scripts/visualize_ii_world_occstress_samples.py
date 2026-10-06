#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_misalignment_review_sample import voxelize_transformed
from generate_semantic_noise_subset import build_preview
from occstress_layout import normalize_protocol_name, resolve_manual_protocol_path


ROWS = [
    ("clean", "clean_H4_F6_val_backbone", "Clean"),
    ("semantic", "semantic_hard_current_H4_F6_val_backbone", "Semantic\nhard current"),
    ("hole", "hole_hard_current_H4_F6_val_backbone", "Hole\nhard current"),
    ("dropout", "dropout_hard_current_H4_F6_val_backbone", "Dropout\nhard current"),
    ("misalignment", "misalignment_hard_current_H4_F6_val_backbone", "Misalignment\nhard current"),
    ("traffic", "traffic_all_frame_H4_F6_val_backbone", "Traffic\nmirrored sequence"),
]

COLUMN_LABELS = ["H-4", "H-3", "H-2", "H-1", "Current", "F+1", "F+2", "F+3", "F+4", "F+5", "F+6"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize OccStress occupancy sequences without model predictions.")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default=str(REPO_ROOT / "data/OccStress/protocols"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "outputs/ii_world_occstress_visuals"),
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--fig-width", type=float, default=28.0)
    parser.add_argument("--fig-height", type=float, default=15.0)
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_occ_npz(path):
    p = Path(path)
    return p if p.suffix == ".npz" else p / "labels.npz"


def load_semantics(path):
    return np.load(resolve_occ_npz(path), allow_pickle=True)["semantics"]


def resolve_protocol_path(protocol_name, protocol_dir=None):
    protocol_name = normalize_protocol_name(protocol_name)
    nested_path = resolve_manual_protocol_path(protocol_name, root=REPO_ROOT, must_exist=False)
    if nested_path.exists():
        return nested_path

    if protocol_dir is not None:
        flat_path = Path(protocol_dir) / f"{protocol_name}.pkl"
        if flat_path.exists():
            return flat_path

    return resolve_manual_protocol_path(protocol_name, root=REPO_ROOT, must_exist=True)


def select_indices(records, count):
    if count >= len(records):
        return list(range(len(records)))
    return np.linspace(0, len(records) - 1, count, dtype=int).tolist()


def build_row_frames(record, row_type):
    history = []
    if row_type == "misalignment":
        mis = record["misalignment"]
        for hist, delta_rt in zip(record["history"], mis["delta_rt"]):
            sem = load_semantics(hist["occ_path"])
            if hist.get("rt_source") == "misalignment":
                sem = voxelize_transformed(sem, np.asarray(delta_rt, dtype=np.float32))
            history.append(sem)
        current = load_semantics(record["current_input"]["occ_path"])
        if mis.get("current_active"):
            current = voxelize_transformed(current, np.asarray(mis["current_delta_rt"], dtype=np.float32))
    else:
        history = [load_semantics(h["occ_path"]) for h in record["history"]]
        current = load_semantics(record["current_input"]["occ_path"])
    future = [load_semantics(f["occ_path"]) for f in record["future_targets"]]
    return history + [current] + future


def plot_image(ax, semantics, title=None, ylabel=None):
    ax.imshow(build_preview(semantics))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, rotation=0, ha="right", va="center", labelpad=28)


def save_sample_figure(out_path, scene_name, anchor_token, rows_payload, fig_width, fig_height):
    nrows = len(rows_payload)
    ncols = len(COLUMN_LABELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.04, wspace=0.035, hspace=0.22)
    fig.suptitle(f"OccStress Occupancy Sequences\nscene={scene_name}  anchor={anchor_token}", fontsize=15, y=0.975)

    for row_idx, payload in enumerate(rows_payload):
        for col_idx, sem in enumerate(payload["frames"]):
            plot_image(
                axes[row_idx, col_idx],
                sem,
                title=COLUMN_LABELS[col_idx] if row_idx == 0 else None,
                ylabel=payload["label"] if col_idx == 0 else None,
            )

    fig.text(0.012, 0.5, "Rows: clean + 5 state corruptions", rotation=90, va="center", fontsize=11)
    fig.text(0.5, 0.015, "Columns: 4 history frames, 1 current frame, 6 future GT frames", ha="center", fontsize=11)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bundles = {}
    for row_type, protocol_name, label in ROWS:
        protocol_path = resolve_protocol_path(protocol_name, args.protocol_dir)
        records = load_pickle(protocol_path)
        bundles[row_type] = {
            "label": label,
            "protocol_name": protocol_name,
            "records": records,
            "record_by_anchor": {r["anchor_token"]: r for r in records},
        }

    clean_records = bundles["clean"]["records"]
    selected = [clean_records[i] for i in select_indices(clean_records, args.num_samples)]

    index_rows = []
    for clean_record in selected:
        anchor = clean_record["anchor_token"]
        scene_name = clean_record["scene_name"]
        rows_payload = []
        for row_type, _, _ in ROWS:
            bundle = bundles[row_type]
            record = bundle["record_by_anchor"][anchor]
            rows_payload.append(
                {
                    "row_type": row_type,
                    "label": bundle["label"],
                    "protocol_name": bundle["protocol_name"],
                    "frames": build_row_frames(record, row_type),
                }
            )

        file_name = f"{scene_name}__{anchor}.png"
        out_path = output_root / file_name
        save_sample_figure(out_path, scene_name, anchor, rows_payload, args.fig_width, args.fig_height)
        index_rows.append(
            {
                "scene_name": scene_name,
                "anchor_token": anchor,
                "sample_id": clean_record["sample_id"],
                "image": str(out_path),
                "protocols": {p["row_type"]: p["protocol_name"] for p in rows_payload},
            }
        )

    with (output_root / "index.json").open("w") as f:
        json.dump(
            {
                "num_samples": len(index_rows),
                "row_types": [row[0] for row in ROWS],
                "columns": COLUMN_LABELS,
                "samples": index_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
