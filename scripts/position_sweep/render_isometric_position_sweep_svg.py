#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Render editable isometric SVG slabs for the final position sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

METHODS = ("OccWorld", "COME", "II-World", "DOME", "GenieDrive")
METHOD_SLUGS = {
    "OccWorld": "occworld",
    "COME": "come",
    "II-World": "ii_world",
    "DOME": "dome",
    "GenieDrive": "geniedrive",
}
METHOD_DISPLAY_LABELS = {
    "OccWorld": "OccWorld",
    "COME": "COME",
    "II-World": "I\u00b2-World",
    "DOME": "DOME",
    "GenieDrive": "GenieDrive",
}
POSITIONS = ("tminus3", "tminus2", "tminus1", "t")
FULL_POSITIONS = ("tminus4", *POSITIONS)
POSITION_SECONDS = dict(zip(FULL_POSITIONS, (-2.0, -1.5, -1.0, -0.5, 0.0)))
FIVE_STATE_METHODS = {"OccWorld", "II-World"}
Matrix = list[list[float | None]]
POSITION_LABELS = {
    "tminus4": "t-2.0s",
    "tminus3": "t-1.5s",
    "tminus2": "t-1.0s",
    "tminus1": "t-0.5s",
    "t": "t",
}
HORIZONS = ("0p5", "1p0", "1p5", "2p0", "2p5", "3p0")
# The near-to-right arrow in the paper layout runs opposite HORIZON_VECTOR.
# Reversing display rows makes that arrow increase from +0.5s to +3.0s.
DISPLAY_HORIZONS = tuple(reversed(HORIZONS))
HORIZON_INDEX = {horizon: index for index, horizon in enumerate(HORIZONS)}
HORIZON_LABELS = {
    "0p5": "+0.5s",
    "1p0": "+1.0s",
    "1p5": "+1.5s",
    "2p0": "+2.0s",
    "2p5": "+2.5s",
    "3p0": "+3.0s",
}
EXPECTED_SETTINGS = {
    "manual_dropout_hard",
    "manual_hole_hard",
    "manual_misalignment_hard",
    "stcocc_brightness_hard",
    "stcocc_camera_crash_hard",
    "stcocc_color_quant_hard",
    "stcocc_fog_hard",
    "stcocc_frame_lost_hard",
    "stcocc_low_light_hard",
    "stcocc_motion_blur_hard",
}

# ColorBrewer YlOrRd-9, matching the palette used by the original heatmap.
COLOR_STOPS = (
    (0.000, "#ffffcc"),
    (0.125, "#ffeda0"),
    (0.250, "#fed976"),
    (0.375, "#feb24c"),
    (0.500, "#fd8d3c"),
    (0.625, "#fc4e2a"),
    (0.750, "#e31a1c"),
    (0.875, "#bd0026"),
    (1.000, "#800026"),
)

VIEWBOX_WIDTH = 1180
VIEWBOX_HEIGHT = 390
ORIGIN = (685.0, 18.0)
POSITION_VECTOR = (105.0, 30.0)
HORIZON_VECTOR = (-105.0, 30.0)
THICKNESS = 40.0


def svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Defaults to the matching common-visible or all-position CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to separate directories for four- and five-position SVGs.",
    )
    parser.add_argument(
        "--include-tminus4",
        action="store_true",
        help="Show all five positions, with not-observed placeholders where needed.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Shared color maximum in relative mIoU-drop percent (default: next 5).",
    )
    return parser.parse_args()


def parse_hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def as_hex(rgb: tuple[float, float, float]) -> str:
    channels = [max(0, min(255, round(channel))) for channel in rgb]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def interpolate_color(value: float, vmax: float) -> str:
    unit = max(0.0, min(1.0, value / vmax))
    for (left_x, left_color), (right_x, right_color) in zip(
        COLOR_STOPS, COLOR_STOPS[1:]
    ):
        if unit <= right_x:
            fraction = (unit - left_x) / (right_x - left_x)
            left = parse_hex(left_color)
            right = parse_hex(right_color)
            return as_hex(
                tuple(
                    left[channel]
                    + fraction * (right[channel] - left[channel])
                    for channel in range(3)
                )
            )
    return COLOR_STOPS[-1][1]


def shade(color: str, factor: float) -> str:
    return as_hex(tuple(channel * factor for channel in parse_hex(color)))


def point(column: int, row: int) -> tuple[float, float]:
    return (
        ORIGIN[0]
        + column * POSITION_VECTOR[0]
        + row * HORIZON_VECTOR[0],
        ORIGIN[1]
        + column * POSITION_VECTOR[1]
        + row * HORIZON_VECTOR[1],
    )


def points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def add_text(
    parent: ET.Element,
    x: float,
    y: float,
    content: str,
    **attrs: str,
) -> ET.Element:
    element = ET.SubElement(
        parent,
        svg_tag("text"),
        {"x": f"{x:.1f}", "y": f"{y:.1f}", **attrs},
    )
    element.text = content
    return element


def is_observed(method: str, position: str) -> bool:
    return position != "tminus4" or method in FIVE_STATE_METHODS


def load_matrices(
    path: Path, positions: tuple[str, ...] = POSITIONS
) -> tuple[dict[str, Matrix], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_rows = len(METHODS) * len(EXPECTED_SETTINGS) * len(positions)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} aligned rows, found {len(rows)}")

    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["method"], row["setting"], row["position"])
        if key in keyed:
            raise ValueError(f"Duplicate aligned result: {key}")
        keyed[key] = row
        if row["method"] not in METHODS:
            raise ValueError(f"Unexpected method: {row['method']}")
        if row["setting"] not in EXPECTED_SETTINGS:
            raise ValueError(f"Unexpected setting: {row['setting']}")
        if row["position"] not in positions:
            raise ValueError(f"Unexpected position: {row['position']}")
        observed = is_observed(row["method"], row["position"])
        seconds = POSITION_SECONDS[row["position"]]
        offsets = json.loads(row["input_offsets_seconds"])
        if (
            row["in_input_window"] != str(observed)
            or (seconds in offsets) != observed
            or float(row["position_seconds"]) != seconds
        ):
            raise ValueError(f"Input-window metadata mismatch for {key}")
        if int(row["evaluated_records"]) != 4519:
            raise ValueError(f"Unexpected record count for {key}")

    expected_keys = {
        (method, setting, position)
        for method in METHODS
        for setting in EXPECTED_SETTINGS
        for position in positions
    }
    if set(keyed) != expected_keys:
        missing = sorted(expected_keys - set(keyed))
        extra = sorted(set(keyed) - expected_keys)
        raise ValueError(f"Protocol alignment failed; missing={missing}, extra={extra}")

    accumulators: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    generations: set[str] = set()
    for row in rows:
        generations.add(row["result_generation"])
        if not is_observed(row["method"], row["position"]):
            continue
        for horizon in HORIZONS:
            value = float(row[f"relative_miou_drop_{horizon}s_pct"])
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite value for {row['method']}, {row['setting']}, "
                    f"{row['position']}, {horizon}"
                )
            accumulators[(row["method"], row["position"], horizon)].append(value)

    matrices: dict[str, Matrix] = {}
    for method in METHODS:
        matrices[method] = []
        for position in positions:
            if not is_observed(method, position):
                matrices[method].append([None] * len(HORIZONS))
                continue
            position_values = []
            for horizon in HORIZONS:
                values = accumulators[(method, position, horizon)]
                if len(values) != len(EXPECTED_SETTINGS):
                    raise ValueError(
                        f"Expected {len(EXPECTED_SETTINGS)} values for "
                        f"{method}/{position}/{horizon}, found {len(values)}"
                    )
                position_values.append(sum(values) / len(values))
            matrices[method].append(position_values)
    return matrices, sorted(generations)


def add_style(root: ET.Element) -> None:
    style = ET.SubElement(root, svg_tag("style"))
    style.text = """
      .cell-top, .cell-side { stroke: #858994; stroke-width: 2.4;
        stroke-linejoin: round; vector-effect: non-scaling-stroke; }
      .cell-side { stroke-width: 2.6; }
      .method-label { fill: #111827; font-family: Georgia, 'Times New Roman', serif;
        font-size: 42px; font-style: italic; font-weight: 700; letter-spacing: 0; }
      .axis-label { fill: #248866; font-family: Arial, Helvetica, sans-serif;
        font-size: 25px; font-weight: 600; letter-spacing: 0; }
      .tick-label { fill: #30343b; font-family: Arial, Helvetica, sans-serif;
        font-size: 22px; letter-spacing: 0; }
      .legend-title { fill: #20242b; font-family: Arial, Helvetica, sans-serif;
        font-size: 24px; font-weight: 600; letter-spacing: 0; }
    """


def add_metadata(
    root: ET.Element,
    method: str | None,
    vmax: float,
    generations: list[str],
    positions: tuple[str, ...] = POSITIONS,
) -> None:
    title = ET.SubElement(root, svg_tag("title"))
    title.text = (
        f"{method} position-wise temporal sensitivity"
        if method
        else "Five-method position-wise temporal sensitivity"
    )
    description = ET.SubElement(root, svg_tag("desc"))
    description.text = (
        "Relative semantic mIoU drop, averaged independently at each future "
        "horizon over 10 aligned hard-corruption settings. Methods share four "
        "visible corrupted-input positions, six future horizons, and 4,519 "
        f"anchors. Shared color range: 0-{vmax:g}%. Result generations: "
        f"{', '.join(generations)}."
    )
    if positions == FULL_POSITIONS:
        description.text += (
            " Five input positions are displayed. Gray hatching and N/O mean "
            "not observed by the model, not zero drop or a missing experiment. "
            "Each frame step is 0.5s; t-4 corresponds to t-2.0s."
        )


def value_color(value: float | None, vmax: float) -> str:
    return "#e6e8eb" if value is None else interpolate_color(value, vmax)


def add_unobserved_mark(
    parent: ET.Element, corners: list[tuple[float, float]]
) -> None:
    p00, p10, _, p01 = corners

    # Explicit line segments stay editable and avoid SVG pattern/clip dependencies.
    def at(u: float, v: float) -> tuple[float, float]:
        return tuple(
            p00[axis] + u * (p10[axis] - p00[axis])
            + v * (p01[axis] - p00[axis])
            for axis in (0, 1)
        )

    for index in range(8):
        intercept = -0.28 + index * 0.18
        lower = max(0.0, -intercept / 0.35)
        upper = min(1.0, (1.0 - intercept) / 0.35)
        if lower >= upper:
            continue
        start = at(lower, 0.35 * lower + intercept)
        end = at(upper, 0.35 * upper + intercept)
        ET.SubElement(
            parent, svg_tag("line"),
            {
                "x1": f"{start[0]:.2f}", "y1": f"{start[1]:.2f}",
                "x2": f"{end[0]:.2f}", "y2": f"{end[1]:.2f}",
                "stroke": "#afb4bc", "stroke-width": "1.5",
            },
        )
    center = at(0.5, 0.5)
    ET.SubElement(
        parent, svg_tag("rect"),
        {
            "x": f"{center[0] - 20:.2f}", "y": f"{center[1] - 10:.2f}",
            "width": "40", "height": "20", "fill": "#e6e8eb", "stroke": "none",
        },
    )
    add_text(
        parent, center[0], center[1] + 5, "N/O",
        **{
            "text-anchor": "middle", "font-family": "Arial, sans-serif",
            "font-size": "16", "font-weight": "700", "fill": "#505763",
            "stroke": "none",
        },
    )


def add_slab(
    parent: ET.Element,
    method: str,
    matrix: Matrix,
    vmax: float,
    positions: tuple[str, ...] = POSITIONS,
) -> ET.Element:
    slab = ET.SubElement(
        parent,
        svg_tag("g"),
        {"id": f"slab-{METHOD_SLUGS[method]}", "data-method": method},
    )

    slug = METHOD_SLUGS[method]
    sides = ET.SubElement(slab, svg_tag("g"), {"id": f"{slug}-thickness-faces"})
    last_position = len(positions) - 1
    last_horizon = len(DISPLAY_HORIZONS) - 1

    # Two exposed boundaries form the editable slab thickness.
    for display_index, horizon in enumerate(DISPLAY_HORIZONS):
        top_left = point(last_position + 1, display_index)
        top_right = point(last_position + 1, display_index + 1)
        value = matrix[last_position][HORIZON_INDEX[horizon]]
        fill = shade(value_color(value, vmax), 0.79)
        ET.SubElement(
            sides,
            svg_tag("polygon"),
            {
                "id": f"{slug}-side-position-t-{horizon}",
                "class": "cell-side",
                "points": points_attr(
                    [
                        top_left,
                        top_right,
                        (top_right[0], top_right[1] + THICKNESS),
                        (top_left[0], top_left[1] + THICKNESS),
                    ]
                ),
                "fill": fill,
                "stroke": "#858994",
                "stroke-width": "2.6",
                "stroke-linejoin": "round",
            },
        )

    for position_index, position in enumerate(positions):
        top_left = point(position_index, last_horizon + 1)
        top_right = point(position_index + 1, last_horizon + 1)
        boundary_horizon = DISPLAY_HORIZONS[last_horizon]
        value = matrix[position_index][HORIZON_INDEX[boundary_horizon]]
        fill = shade(value_color(value, vmax), 0.88)
        ET.SubElement(
            sides,
            svg_tag("polygon"),
            {
                "id": f"{slug}-side-horizon-{boundary_horizon}-{position}",
                "class": "cell-side",
                "points": points_attr(
                    [
                        top_left,
                        top_right,
                        (top_right[0], top_right[1] + THICKNESS),
                        (top_left[0], top_left[1] + THICKNESS),
                    ]
                ),
                "fill": fill,
                "stroke": "#858994",
                "stroke-width": "2.6",
                "stroke-linejoin": "round",
            },
        )

    tops = ET.SubElement(slab, svg_tag("g"), {"id": f"{slug}-heatmap-cells"})
    for position_index, position in enumerate(positions):
        position_group = ET.SubElement(
            tops,
            svg_tag("g"),
            {
                "id": f"{slug}-position-{position}",
                "data-position": POSITION_LABELS[position],
            },
        )
        for display_index, horizon in enumerate(DISPLAY_HORIZONS):
            value = matrix[position_index][HORIZON_INDEX[horizon]]
            p00 = point(position_index, display_index)
            p10 = point(position_index + 1, display_index)
            p11 = point(position_index + 1, display_index + 1)
            p01 = point(position_index, display_index + 1)
            cell = ET.SubElement(
                position_group,
                svg_tag("g"),
                {
                    "id": f"{slug}-cell-{position}-{horizon}",
                    "data-horizon": HORIZON_LABELS[horizon],
                    "data-drop": "" if value is None else f"{value:.6f}",
                    "data-observed": str(value is not None).lower(),
                },
            )
            tooltip = ET.SubElement(cell, svg_tag("title"))
            tooltip.text = (
                f"{method}: input {POSITION_LABELS[position]}, forecast "
                f"{HORIZON_LABELS[horizon]}, "
                + ("N/O: input state not observed" if value is None
                   else f"relative mIoU drop {value:.4f}%")
            )
            ET.SubElement(
                cell,
                svg_tag("polygon"),
                {
                    "class": "cell-top",
                    "points": points_attr([p00, p10, p11, p01]),
                    "fill": value_color(value, vmax),
                    "stroke": "#858994",
                    "stroke-width": "2.4",
                    "stroke-linejoin": "round",
                },
            )
            if value is None:
                add_unobserved_mark(cell, [p00, p10, p11, p01])
    return slab


def base_svg(width: int, height: int) -> ET.Element:
    return ET.Element(
        svg_tag("svg"),
        {
            "version": "1.1",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )


def write_svg(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def render_individual(
    method: str,
    matrix: Matrix,
    vmax: float,
    generations: list[str],
    output_path: Path,
    positions: tuple[str, ...] = POSITIONS,
) -> None:
    extra = len(positions) - len(POSITIONS)
    root = base_svg(VIEWBOX_WIDTH + 105 * extra, VIEWBOX_HEIGHT + 30 * extra)
    add_metadata(root, method, vmax, generations, positions)
    add_style(root)
    add_slab(root, method, matrix, vmax, positions)
    write_svg(root, output_path)


def add_legend(root: ET.Element, vmax: float, x: float, y: float) -> None:
    add_text(
        root,
        x - 20,
        y - 28,
        "Relative mIoU",
        **{"class": "legend-title", "text-anchor": "middle"},
    )
    add_text(
        root,
        x - 20,
        y + 2,
        "drop (%)",
        **{"class": "legend-title", "text-anchor": "middle"},
    )
    bar_y = y + 28
    bar_height = 300.0
    strip_count = 100
    strip_height = bar_height / strip_count
    legend_strips = ET.SubElement(
        root, svg_tag("g"), {"id": "legend-color-strips"}
    )
    for index in range(strip_count):
        value = vmax * (index + 0.5) / strip_count
        ET.SubElement(
            legend_strips,
            svg_tag("rect"),
            {
                "x": f"{x:.1f}",
                "y": f"{bar_y + bar_height - (index + 1) * strip_height:.1f}",
                "width": "34",
                "height": f"{strip_height + 0.1:.1f}",
                "fill": interpolate_color(value, vmax),
                "stroke": "none",
            },
        )
    ET.SubElement(
        root,
        svg_tag("rect"),
        {
            "x": f"{x:.1f}",
            "y": f"{bar_y:.1f}",
            "width": "34",
            "height": f"{bar_height:.1f}",
            "fill": "none",
            "stroke": "#858994",
            "stroke-width": "1.8",
        },
    )
    for index in range(6):
        value = vmax * index / 5
        tick_y = y + 328 - 300 * index / 5
        ET.SubElement(
            root,
            svg_tag("line"),
            {
                "x1": f"{x + 34:.1f}",
                "x2": f"{x + 44:.1f}",
                "y1": f"{tick_y:.1f}",
                "y2": f"{tick_y:.1f}",
                "stroke": "#444950",
                "stroke-width": "1.8",
            },
        )
        add_text(
            root,
            x + 52,
            tick_y + 7,
            f"{value:g}",
            **{"class": "tick-label"},
        )


def render_combined(
    matrices: dict[str, Matrix],
    vmax: float,
    generations: list[str],
    output_path: Path,
    positions: tuple[str, ...] = POSITIONS,
) -> None:
    extra = len(positions) - len(POSITIONS)
    width = 1540 + 105 * extra
    panel_spacing = 310 + 90 * extra
    top = 25
    height = top + panel_spacing * len(METHODS) + 155 + 45 * extra
    root = base_svg(width, height)
    add_metadata(root, None, vmax, generations, positions)
    add_style(root)

    for index, method in enumerate(METHODS):
        y = top + index * panel_spacing
        panel = ET.SubElement(
            root,
            svg_tag("g"),
            {
                "id": f"panel-{METHOD_SLUGS[method]}",
                "transform": f"translate(210 {y})",
            },
        )
        add_slab(panel, method, matrices[method], vmax, positions)
        add_text(
            root,
            45,
            y + 168,
            METHOD_DISPLAY_LABELS[method],
            **{"class": "method-label"},
        )

    add_legend(root, vmax, 1375 + 105 * extra, 72)
    footer_y = height - 82 - 45 * extra
    add_text(
        root,
        width / 2,
        footer_y,
        ("Corrupted input positions: t-4, t-3, t-2, t-1, t (0.5s steps)"
         if extra else "Corrupted input positions: t-1.5s, t-1.0s, t-0.5s, t"),
        **{"class": "axis-label", "text-anchor": "middle"},
    )
    add_text(
        root,
        width / 2,
        footer_y + 39,
        "Future prediction horizons: +0.5s to +3.0s (0.5s steps)",
        **{"class": "axis-label", "text-anchor": "middle"},
    )
    if extra:
        add_text(
            root, width / 2, footer_y + 78,
            "Gray hatching / N/O: not observed by the model",
            **{"class": "tick-label", "text-anchor": "middle"},
        )
    write_svg(root, output_path)


def write_matrix_csv(
    path: Path, matrices: dict[str, Matrix],
    positions: tuple[str, ...] = POSITIONS,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "position", "position_seconds", "in_input_window"]
            + [f"relative_miou_drop_{horizon}s_pct" for horizon in HORIZONS]
        )
        for method in METHODS:
            for position, values in zip(positions, matrices[method]):
                writer.writerow(
                    [method, position, POSITION_SECONDS[position],
                     is_observed(method, position)]
                    + ["" if value is None else f"{value:.6f}" for value in values]
                )


def write_manifest(
    path: Path,
    input_path: Path,
    matrices: dict[str, Matrix],
    vmax: float,
    generations: list[str],
    positions: tuple[str, ...] = POSITIONS,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(input_path.resolve()),
        "source_csv_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "metric": "relative semantic mIoU drop (%)",
        "aggregation": (
            "arithmetic mean over 10 protocol-aligned hard-corruption settings, "
            "computed independently for every method, input position, and horizon"
        ),
        "methods": list(METHODS),
        "settings": sorted(EXPECTED_SETTINGS),
        "common_positions": [POSITION_LABELS[position] for position in POSITIONS],
        "displayed_positions": [POSITION_LABELS[position] for position in positions],
        "future_horizons": [HORIZON_LABELS[horizon] for horizon in HORIZONS],
        "display_axes": {
            "short_left_to_near_edge": (
                f"corrupted input position: {POSITION_LABELS[positions[0]]} to t"
            ),
            "long_near_to_right_edge": "future prediction horizon: +0.5s to +3.0s",
        },
        "records_per_evaluation": 4519,
        "formal_evaluations_used": (
            sum(is_observed(m, p) for m in METHODS for p in positions)
            * len(EXPECTED_SETTINGS)
        ),
        "source_evaluations_validated": len(METHODS) * len(EXPECTED_SETTINGS) * len(positions),
        "common_visible_evaluations": len(METHODS) * len(EXPECTED_SETTINGS) * len(POSITIONS),
        "placeholder_cells": sum(value is None for matrix in matrices.values()
                                 for row in matrix for value in row),
        "result_generations": generations,
        "color_scale": {"palette": "ColorBrewer YlOrRd-9", "min": 0, "max": vmax},
        "unobserved_policy": (
            "COME, DOME, and GenieDrive do not observe t-2.0s. When displayed, "
            "those cells are null/N/O with gray hatching, not zero-valued. "
            "Cross-method comparisons use the four common visible positions."
        ),
        "under_range_cells": [
            {"method": method, "position": POSITION_LABELS[position],
             "horizon": HORIZON_LABELS[horizon], "value": value}
            for method, matrix in matrices.items()
            for position, values in zip(positions, matrix)
            for horizon, value in zip(HORIZONS, values)
            if value is not None and value < 0
        ],
        "under_range_policy": "Negative drops retain their values and use the 0% endpoint color.",
        "matrices": {
            method: {
                POSITION_LABELS[position]: {
                    HORIZON_LABELS[horizon]: value
                    for horizon, value in zip(HORIZONS, values)
                }
                for position, values in zip(positions, matrix)
            }
            for method, matrix in matrices.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    positions = FULL_POSITIONS if args.include_tminus4 else POSITIONS
    result_root = Path("outputs/position_sweep_final_20260825")
    if args.input is None:
        args.input = result_root / (
            "all_position_drops.csv" if args.include_tminus4
            else "common_visible_by_position.csv"
        )
    if args.output_dir is None:
        args.output_dir = result_root / (
            "isometric_svg_five_positions" if args.include_tminus4 else "isometric_svg"
        )
    matrices, generations = load_matrices(args.input, positions)
    observed_max = max(
        value
        for matrix in matrices.values()
        for position_values in matrix
        for value in position_values
        if value is not None
    )
    vmax = args.vmax if args.vmax is not None else math.ceil(observed_max / 5) * 5
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    if observed_max > vmax:
        raise ValueError(f"Observed maximum {observed_max:.3f} exceeds vmax={vmax:g}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        render_individual(
            method,
            matrices[method],
            vmax,
            generations,
            args.output_dir / f"{METHOD_SLUGS[method]}_position_sweep.svg",
            positions,
        )

    render_combined(
        matrices,
        vmax,
        generations,
        args.output_dir / "position_sweep_five_method_preview.svg",
        positions,
    )
    write_matrix_csv(args.output_dir / "position_sweep_matrices.csv", matrices, positions)
    write_manifest(
        args.output_dir / "manifest.json",
        args.input,
        matrices,
        vmax,
        generations,
        positions,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "methods": list(METHODS),
                "observed_max": observed_max,
                "vmax": vmax,
                "aligned_rows": (
                    len(METHODS) * len(EXPECTED_SETTINGS) * len(positions)
                ),
                "displayed_positions": list(positions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
