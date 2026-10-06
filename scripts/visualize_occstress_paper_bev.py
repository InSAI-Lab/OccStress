#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IIWORLD_OCC3D_COLORS_RGB = np.array(
    [
        [0, 0, 0],          # others
        [255, 120, 50],     # barrier
        [255, 192, 203],    # bicycle
        [255, 255, 0],      # bus
        [0, 150, 245],      # car
        [0, 255, 255],      # construction_vehicle
        [255, 127, 0],      # motorcycle
        [255, 0, 0],        # pedestrian
        [255, 240, 150],    # traffic_cone
        [135, 60, 0],       # trailer
        [160, 32, 240],     # truck
        [255, 0, 255],      # driveable_surface
        [139, 137, 137],    # other_flat
        [75, 0, 75],        # sidewalk
        [150, 240, 80],     # terrain
        [230, 230, 250],    # manmade
        [0, 175, 0],        # vegetation
        [255, 255, 255],    # free
    ],
    dtype=np.uint8,
)

FREE_LABEL = 17
IGNORE_LABEL = 255

DEFAULT_ANCHOR = "7a903ae4cbb0466093fc117258a31efc"
DEFAULT_ROWS = [
    {
        "key": "gt",
        "label": "GT",
        "root": "data/nuscenes/gts",
    },
    {
        "key": "camera_only_fog_hard",
        "label": "Camera-Only Fog hard",
        "root": "data/OccStress/occ/upstream/camera_only/stcocc/Fog/hard",
    },
    {
        "key": "pointcloud_fusion_fog_heavy",
        "label": "Point-Fusion fog heavy",
        "root": "data/OccStress/occ/upstream/pointcloud_fusion/sdgocc/fog/heavy",
    },
    {
        "key": "semantic_hard",
        "label": "Semantic hard",
        "root": "data/OccStress/occ/manual/semantic/hard",
    },
    {
        "key": "dropout_hard",
        "label": "Dropout hard",
        "root": "data/OccStress/occ/manual/dropout/hard",
    },
]

COLUMN_SPECS = [
    ("t-4", "history", 0),
    ("t-3", "history", 1),
    ("t-2", "history", 2),
    ("t-1", "history", 3),
    ("t", "anchor", None),
    ("t+2", "future", 1),
    ("t+4", "future", 3),
    ("t+6", "future", 5),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create paper BEV tiles for one OccStress H4_F6 sample.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("data/OccStress_HF/protocols/manual/clean/H4_F6_val_backbone.pkl"),
        help="Clean H4_F6 protocol pkl used to resolve the sample sequence.",
    )
    parser.add_argument("--anchor-token", default=DEFAULT_ANCHOR)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_occ_bev"))
    parser.add_argument("--tile-size", type=int, default=1000)
    parser.add_argument("--sheet-tile-size", type=int, default=320)
    parser.add_argument("--sheet-pad", type=int, default=12)
    parser.add_argument(
        "--tile-border-px",
        type=int,
        default=3,
        help="Draw a fixed outer border on each tile so external layout tools keep the same visual extent.",
    )
    parser.add_argument(
        "--crop-source",
        choices=["gt", "all", "full"],
        default="gt",
        help="Use one fixed BEV crop for all tiles. gt uses the GT row extent, all uses every row, full keeps the full 80m x 80m grid.",
    )
    parser.add_argument("--crop-padding-cells", type=int, default=10)
    parser.add_argument("--no-sheet", action="store_true")
    return parser.parse_args()


def resolve_under_repo(repo_root, path):
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def manifest_path(repo_root, path):
    path = Path(path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def load_records(protocol_path):
    with protocol_path.open("rb") as f:
        return pickle.load(f)


def find_record(records, anchor_token):
    for record in records:
        if record["anchor_token"] == anchor_token:
            return record
    raise KeyError(f"Anchor token not found in protocol: {anchor_token}")


def frame_tokens(record):
    tokens = []
    for label, source, index in COLUMN_SPECS:
        if source == "history":
            token = record["history_tokens"][index]
        elif source == "anchor":
            token = record["anchor_token"]
        elif source == "future":
            token = record["future_tokens"][index]
        else:
            raise ValueError(source)
        tokens.append({"label": label, "token": token})
    return tokens


def labels_path(row_root, scene_name, token):
    return row_root / scene_name / token / "labels.npz"


def load_semantics(path):
    with np.load(path, allow_pickle=True) as data:
        return data["semantics"]


def bev_labels_from_semantics(semantics):
    valid = (semantics != FREE_LABEL) & (semantics != IGNORE_LABEL)
    z = np.arange(semantics.shape[2], dtype=np.float32).reshape(1, 1, -1)
    score = valid.astype(np.float32) * (z + 1.0)
    selected = np.argmax(score, axis=2)
    bev = np.take_along_axis(semantics, selected[:, :, None], axis=2).squeeze(-1)
    bev[~valid.any(axis=2)] = FREE_LABEL
    return bev[::-1, ::-1]


def crop_box_from_bevs(bevs, padding_cells):
    bevs = list(bevs)
    if not bevs:
        raise ValueError("Cannot determine a crop from an empty BEV list.")
    first = bevs[0]
    height, width = first.shape

    occupied = np.zeros((height, width), dtype=bool)
    for bev in bevs:
        occupied |= (bev != FREE_LABEL) & (bev != IGNORE_LABEL)

    coords = np.argwhere(occupied)
    if coords.size == 0:
        return [0, 0, width, height]

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    box_w = int(x1 - x0)
    box_h = int(y1 - y0)
    side = min(max(box_w, box_h) + 2 * int(padding_cells), min(width, height))

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return [left, top, left + side, top + side]


def image_from_bev_labels(bev, crop_box):
    left, top, right, bottom = crop_box
    cropped = bev[top:bottom, left:right]
    return IIWORLD_OCC3D_COLORS_RGB[cropped]


def resize_tile(image, size):
    return Image.fromarray(image).resize((size, size), Image.Resampling.NEAREST)


def add_tile_border(image, border_px):
    if border_px <= 0:
        return image
    draw = ImageDraw.Draw(image)
    for offset in range(border_px):
        draw.rectangle(
            [offset, offset, image.width - 1 - offset, image.height - 1 - offset],
            outline=(35, 35, 35),
        )
    return image


def font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_centered(draw, box, text, fill, font_obj):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.text((x, y), text, fill=fill, font=font_obj)


def save_sheet(out_path, rows, columns, tile_paths, tile_size, pad):
    row_label_width = 310
    column_label_height = 58
    width = row_label_width + len(columns) * tile_size + (len(columns) + 1) * pad
    height = column_label_height + len(rows) * tile_size + (len(rows) + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    col_font = font(28)
    row_font = font(24)

    for col_idx, column in enumerate(columns):
        x0 = row_label_width + pad + col_idx * (tile_size + pad)
        draw_centered(draw, (x0, 0, x0 + tile_size, column_label_height), column["label"], (0, 0, 0), col_font)

    for row_idx, row in enumerate(rows):
        y0 = column_label_height + pad + row_idx * (tile_size + pad)
        draw_centered(draw, (0, y0, row_label_width - 6, y0 + tile_size), row["label"], (0, 0, 0), row_font)
        for col_idx, column in enumerate(columns):
            x0 = row_label_width + pad + col_idx * (tile_size + pad)
            tile = Image.open(tile_paths[(row["key"], column["label"])]).resize(
                (tile_size, tile_size), Image.Resampling.NEAREST
            )
            sheet.paste(tile, (x0, y0))

    sheet.save(out_path)


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    protocol_path = resolve_under_repo(repo_root, args.protocol)
    output_root = resolve_under_repo(repo_root, args.output_root)
    records = load_records(protocol_path)
    record = find_record(records, args.anchor_token)
    scene_name = record["scene_name"]
    columns = frame_tokens(record)

    sample_root = output_root / f"{scene_name}__{record['anchor_token']}"
    tile_root = sample_root / "tiles"
    tile_root.mkdir(parents=True, exist_ok=True)

    rows = []
    resolved_rows = []
    bev_cache = {}
    source_paths = {}
    for row in DEFAULT_ROWS:
        row_root = resolve_under_repo(repo_root, row["root"])
        resolved_row = dict(row)
        resolved_row["root_path"] = row_root
        resolved_rows.append(resolved_row)
        rows.append(row)
        for column in columns:
            npz_path = labels_path(row_root, scene_name, column["token"])
            if not npz_path.exists():
                raise FileNotFoundError(npz_path)
            bev_cache[(row["key"], column["label"])] = bev_labels_from_semantics(load_semantics(npz_path))
            source_paths[(row["key"], column["label"])] = npz_path

    if args.crop_source == "full":
        first_bev = next(iter(bev_cache.values()))
        crop_box = [0, 0, first_bev.shape[1], first_bev.shape[0]]
    elif args.crop_source == "gt":
        crop_box = crop_box_from_bevs(
            [bev_cache[("gt", column["label"])] for column in columns],
            args.crop_padding_cells,
        )
    else:
        crop_box = crop_box_from_bevs(list(bev_cache.values()), args.crop_padding_cells)

    crop_side_cells = crop_box[2] - crop_box[0]
    tile_paths = {}
    manifest = {
        "scene_name": scene_name,
        "anchor_token": record["anchor_token"],
        "sample_id": record["sample_id"],
        "protocol": manifest_path(repo_root, protocol_path),
        "columns": columns,
        "rows": [],
        "color_map": "II-World Occ3D RGB class colors from tools/vis_occ_3d.py",
        "projection": "highest occupied voxel along z, free/255 ignored, image flipped x/y to match existing BEV previews",
        "tile_border_px": args.tile_border_px,
        "crop": {
            "source": args.crop_source,
            "padding_cells": args.crop_padding_cells,
            "bev_pixel_box_xyxy": crop_box,
            "side_cells": crop_side_cells,
            "voxel_size_m": [0.4, 0.4, 0.4],
            "world_extent_before_crop_m": [-40.0, -40.0, 40.0, 40.0],
            "meters_per_tile_pixel": 0.4 * crop_side_cells / args.tile_size,
            "note": "The same crop box and scale are applied to every single tile and to the sheet.",
        },
    }

    for row in resolved_rows:
        row_root = row["root_path"]
        row_dir = tile_root / row["key"]
        row_dir.mkdir(parents=True, exist_ok=True)
        row_entry = {"key": row["key"], "label": row["label"], "root": manifest_path(repo_root, row_root), "tiles": []}

        for column in columns:
            cache_key = (row["key"], column["label"])
            npz_path = source_paths[cache_key]
            image = resize_tile(image_from_bev_labels(bev_cache[cache_key], crop_box), args.tile_size)
            image = add_tile_border(image, args.tile_border_px)
            out_name = f"{column['label'].replace('+', 'p').replace('-', 'm')}.png"
            out_path = row_dir / out_name
            image.save(out_path)
            tile_paths[(row["key"], column["label"])] = out_path
            row_entry["tiles"].append(
                {
                    "frame": column["label"],
                    "token": column["token"],
                    "source_npz": manifest_path(repo_root, npz_path),
                    "image": manifest_path(repo_root, out_path),
                }
            )
        manifest["rows"].append(row_entry)

    if not args.no_sheet:
        sheet_path = sample_root / "grid_5x8.png"
        save_sheet(sheet_path, rows, columns, tile_paths, args.sheet_tile_size, args.sheet_pad)
        manifest["sheet"] = manifest_path(repo_root, sheet_path)

    manifest_file = sample_root / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"output": str(sample_root), "tiles": len(DEFAULT_ROWS) * len(columns)}, indent=2))


if __name__ == "__main__":
    main()
