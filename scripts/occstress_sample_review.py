#!/usr/bin/env python3
"""Build and apply a lightweight review set for OccStress samples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PREFERRED_KEYS = (
    "semantics",
    "labels",
    "voxel_semantics",
    "occ",
    "occupancy",
    "target",
)

CLASS_COLORS = np.array(
    [
        [245, 247, 250],
        [242, 142, 43],
        [89, 161, 79],
        [237, 201, 72],
        [176, 122, 161],
        [255, 157, 167],
        [156, 117, 95],
        [186, 176, 172],
        [118, 183, 178],
        [76, 120, 168],
        [225, 87, 89],
        [157, 205, 156],
        [183, 143, 207],
        [214, 97, 107],
        [128, 177, 211],
        [251, 128, 114],
        [141, 211, 199],
        [232, 234, 237],
        [204, 204, 204],
    ],
    dtype=np.uint8,
)


def load_index(sample_root: Path) -> dict[str, Any]:
    index_path = sample_root / "manifests" / "candidate_index.json"
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "entries" not in data:
        raise ValueError(f"Missing entries in {index_path}")
    return data


def choose_numeric_array(npz_path: Path) -> tuple[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        for key in PREFERRED_KEYS:
            if key in data.files:
                arr = np.asarray(data[key])
                if np.issubdtype(arr.dtype, np.number):
                    return key, arr
        for key in data.files:
            arr = np.asarray(data[key])
            if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
                return key, arr
    raise ValueError(f"No numeric array found in {npz_path}")


def squeeze_to_volume(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        if arr.shape[0] <= 12:
            arr = arr[-1]
        elif arr.shape[-1] <= 12:
            arr = arr[..., -1]
        else:
            arr = arr.reshape((-1,) + arr.shape[-3:])[-1]
    if arr.ndim > 4:
        arr = arr.reshape((-1,) + arr.shape[-3:])[-1]
    return arr


def semantic_bev(arr: np.ndarray) -> np.ndarray:
    arr = squeeze_to_volume(arr)
    if arr.ndim == 2:
        labels = arr.astype(np.int64, copy=False)
    elif arr.ndim == 3:
        vertical_axis = int(np.argmin(arr.shape))
        if vertical_axis != 2:
            arr = np.moveaxis(arr, vertical_axis, 2)
        vol = arr.astype(np.int64, copy=False)
        occupied = (vol != 17) & (vol != 255) & (vol != 0)
        if not np.any(occupied):
            occupied = vol != 0
        rev = occupied[..., ::-1]
        top_from_end = np.argmax(rev, axis=2)
        top = occupied.shape[2] - 1 - top_from_end
        x_idx = np.arange(vol.shape[0])[:, None]
        y_idx = np.arange(vol.shape[1])[None, :]
        labels = vol[x_idx, y_idx, top]
        labels = np.where(occupied.any(axis=2), labels, 17)
    else:
        flat = np.reshape(arr, arr.shape[-2:])
        labels = flat.astype(np.int64, copy=False)
    labels = np.nan_to_num(labels, nan=18, posinf=18, neginf=18).astype(np.int64)
    labels = np.mod(labels, len(CLASS_COLORS))
    return labels


def render_array(arr: np.ndarray, out_path: Path, size: int) -> None:
    labels = semantic_bev(arr)
    rgb = CLASS_COLORS[labels]
    rgb = np.rot90(rgb, 1)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((size, size), Image.Resampling.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)


def render_npz(npz_path: Path, out_path: Path, size: int) -> dict[str, Any]:
    key, arr = choose_numeric_array(npz_path)
    render_array(arr, out_path, size)
    return {"key": key, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def normalize_entry_for_output(entry: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "category_id",
        "category_index",
        "candidate_rank",
        "group",
        "corruption",
        "severity",
        "scene",
        "token",
        "primary_kind",
        "primary_path",
        "clean_path",
        "event_path",
        "protocol_paths",
    }
    return {k: entry.get(k) for k in keep}


def build_review(sample_root: Path, out_dir: Path, thumb_size: int, force: bool) -> None:
    data = load_index(sample_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbs"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []

    entries = data["entries"]
    for idx, entry in enumerate(entries, start=1):
        category = entry["category_id"]
        rank = int(entry["candidate_rank"])
        category_dir = thumbs_dir / category
        primary_thumb = category_dir / f"{rank:02d}_primary.png"
        clean_thumb = category_dir / f"{rank:02d}_clean.png"
        item = normalize_entry_for_output(entry)
        item["thumb_primary"] = str(primary_thumb.relative_to(out_dir))
        item["thumb_clean"] = str(clean_thumb.relative_to(out_dir)) if entry.get("clean_path") else None
        item["primary_array"] = None
        item["clean_array"] = None

        primary_path = sample_root / entry["primary_path"]
        if force or not primary_thumb.exists():
            try:
                item["primary_array"] = render_npz(primary_path, primary_thumb, thumb_size)
            except Exception as exc:  # noqa: BLE001
                item["thumb_primary"] = None
                errors.append(
                    {
                        "category_id": category,
                        "candidate_rank": str(rank),
                        "path": entry["primary_path"],
                        "error": repr(exc),
                    }
                )
        else:
            item["primary_array"] = {"cached": True}

        clean_path_value = entry.get("clean_path")
        if clean_path_value:
            clean_path = sample_root / clean_path_value
            if force or not clean_thumb.exists():
                try:
                    item["clean_array"] = render_npz(clean_path, clean_thumb, thumb_size)
                except Exception as exc:  # noqa: BLE001
                    item["thumb_clean"] = None
                    errors.append(
                        {
                            "category_id": category,
                            "candidate_rank": str(rank),
                            "path": clean_path_value,
                            "error": repr(exc),
                        }
                    )
            else:
                item["clean_array"] = {"cached": True}

        groups[category].append(item)
        if idx % 100 == 0:
            print(f"rendered {idx}/{len(entries)}")

    categories = []
    for category in sorted(groups, key=lambda c: groups[c][0].get("category_index", 0)):
        first = groups[category][0]
        categories.append(
            {
                "category_id": category,
                "category_index": first.get("category_index"),
                "group": first.get("group"),
                "corruption": first.get("corruption"),
                "severity": first.get("severity"),
                "count": len(groups[category]),
                "items": sorted(groups[category], key=lambda x: int(x["candidate_rank"])),
            }
        )

    review = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_candidate_index": "manifests/candidate_index.json",
        "seed": data.get("seed"),
        "n_per_category": data.get("n_per_category"),
        "selection_target_per_category": 10,
        "category_count": len(categories),
        "candidate_count": sum(len(c["items"]) for c in categories),
        "categories": categories,
        "errors": errors,
    }

    with (out_dir / "review_index.json").open("w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=True, indent=2)
    write_html(out_dir / "review.html", review)
    print(f"wrote {out_dir / 'review.html'}")
    print(f"errors: {len(errors)}")


def write_html(out_path: Path, review: dict[str, Any]) -> None:
    payload = json.dumps(review, ensure_ascii=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OccStress Sample Review</title>
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
      --bad: #b42318;
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
      height: 56px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 20;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 650;
      letter-spacing: 0;
    }}
    button, select {{
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .spacer {{ flex: 1; }}
    .status {{
      color: var(--muted);
      white-space: nowrap;
    }}
    .app {{
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-height: calc(100vh - 56px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 56px);
      position: sticky;
      top: 56px;
    }}
    main {{
      padding: 18px;
      min-width: 0;
    }}
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
    .category-button.active {{
      background: var(--accent-soft);
      border-color: #9ccdc6;
    }}
    .category-name {{
      overflow-wrap: anywhere;
      font-weight: 600;
      line-height: 1.2;
    }}
    .category-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }}
    .pill {{
      border-radius: 999px;
      padding: 2px 8px;
      background: #eef1f5;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .pill.done {{
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 650;
    }}
    .pill.warn {{
      background: #fff2d8;
      color: var(--warn);
      font-weight: 650;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .heading {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }}
    h2 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.25;
      overflow-wrap: anywhere;
      letter-spacing: 0;
    }}
    .subhead {{
      color: var(--muted);
      margin-top: 4px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .card.selected {{
      border-color: var(--accent);
      outline: 2px solid var(--accent-soft);
    }}
    .card-top {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      margin-bottom: 8px;
    }}
    input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
    .rank {{
      font-weight: 700;
      margin-bottom: 2px;
    }}
    .path {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      line-height: 1.25;
    }}
    .thumbs {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .thumb-block {{
      min-width: 0;
    }}
    .thumb-label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .thumb {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: contain;
      image-rendering: pixelated;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
    }}
    .missing {{
      width: 100%;
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--bad);
      background: #fff6f4;
    }}
    @media (max-width: 980px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{
        position: static;
        max-height: 260px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      header {{ flex-wrap: wrap; height: auto; min-height: 56px; padding: 10px 12px; }}
      main {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>OccStress Sample Review</h1>
    <span class="status" id="globalStatus"></span>
    <div class="spacer"></div>
    <select id="categorySelect" aria-label="Category"></select>
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
    const storageKey = 'occstress-sample-review-' + (review.seed || 'default');
    const selected = new Map();
    let current = 0;

    function keyOf(item) {{
      return item.category_id + '::' + item.candidate_rank;
    }}

    function loadState() {{
      try {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;
        const obj = JSON.parse(raw);
        for (const [category, ranks] of Object.entries(obj)) {{
          selected.set(category, new Set(ranks.map(Number)));
        }}
      }} catch (err) {{
        console.warn(err);
      }}
    }}

    function saveState() {{
      const obj = {{}};
      for (const [category, ranks] of selected.entries()) {{
        obj[category] = Array.from(ranks).sort((a, b) => a - b);
      }}
      localStorage.setItem(storageKey, JSON.stringify(obj));
    }}

    function selectedSet(categoryId) {{
      if (!selected.has(categoryId)) selected.set(categoryId, new Set());
      return selected.get(categoryId);
    }}

    function totalSelected() {{
      let total = 0;
      for (const ranks of selected.values()) total += ranks.size;
      return total;
    }}

    function categoryStatus(category) {{
      const count = selectedSet(category.category_id).size;
      if (count === target) return ['done', count + '/' + target];
      return ['warn', count + '/' + target];
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
        button.innerHTML = `
          <span>
            <span class="category-name">${{category.category_index}}. ${{category.category_id}}</span>
            <span class="category-meta">${{category.group}} / ${{category.corruption}} / ${{category.severity}}</span>
          </span>
          <span class="pill ${{klass}}">${{text}}</span>`;
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
      document.getElementById('globalStatus').textContent =
        `${{complete}}/${{review.categories.length}} categories, ${{totalSelected()}} selected`;
    }}

    function imageOrMissing(src, alt) {{
      if (!src) return '<div class="missing">missing</div>';
      return `<img class="thumb" src="${{src}}" alt="${{alt}}" loading="lazy">`;
    }}

    function renderCards() {{
      const category = review.categories[current];
      const set = selectedSet(category.category_id);
      document.getElementById('categoryTitle').textContent = category.category_id;
      document.getElementById('categorySubhead').textContent =
        `${{category.group}} / ${{category.corruption}} / ${{category.severity}}`;
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
        card.innerHTML = `
          <div class="card-top">
            <input type="checkbox" ${{checked ? 'checked' : ''}} aria-label="Select sample">
            <div>
              <div class="rank">#${{String(item.candidate_rank).padStart(2, '0')}} ${{item.scene}} / ${{item.token}}</div>
              <div class="path">${{item.primary_path}}</div>
            </div>
          </div>
          <div class="thumbs">
            <div class="thumb-block">
              <div class="thumb-label">candidate</div>
              ${{imageOrMissing(item.thumb_primary, 'candidate')}}
            </div>
            <div class="thumb-block">
              <div class="thumb-label">clean</div>
              ${{imageOrMissing(item.thumb_clean, 'clean')}}
            </div>
          </div>`;
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

    function setCurrent(index) {{
      current = Math.max(0, Math.min(review.categories.length - 1, index));
      render();
    }}

    function render() {{
      renderSidebar();
      renderGlobalStatus();
      renderCards();
    }}

    function exportSelection() {{
      const payload = {{
        created_at: new Date().toISOString(),
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
      a.download = 'occstress_sample_selection.json';
      a.click();
      URL.revokeObjectURL(url);
    }}

    document.getElementById('categorySelect').addEventListener('change', event => {{
      setCurrent(Number(event.target.value));
    }});
    document.getElementById('incompleteButton').addEventListener('click', () => {{
      const next = review.categories.findIndex((category, index) =>
        index > current && selectedSet(category.category_id).size !== target);
      if (next >= 0) setCurrent(next);
      else {{
        const first = review.categories.findIndex(category =>
          selectedSet(category.category_id).size !== target);
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


def iter_selected_items(selection: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for category in selection.get("categories", []):
        selected_items = category.get("selected_items", [])
        items.extend(selected_items)
    return items


def copy_relative(src_root: Path, dst_root: Path, rel_path: str) -> bool:
    if not rel_path:
        return False
    src = src_root / rel_path
    dst = dst_root / rel_path
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def apply_selection(candidate_root: Path, selection_json: Path, out_root: Path, force: bool) -> None:
    if out_root.exists() and force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with selection_json.open("r", encoding="utf-8") as f:
        selection = json.load(f)
    items = iter_selected_items(selection)
    copied = 0
    missing: list[str] = []
    for item in items:
        paths: list[str] = []
        for key in ("primary_path", "clean_path", "event_path"):
            value = item.get(key)
            if value:
                paths.append(value)
        for value in item.get("protocol_paths") or []:
            if value:
                paths.append(value)
        for rel_path in paths:
            if copy_relative(candidate_root, out_root, rel_path):
                copied += 1
            else:
                missing.append(rel_path)

    for rel_path in (
        "meta",
        "manifests/candidate_summary.json",
        "manifests/candidate_index.json",
        "README.md",
    ):
        src = candidate_root / rel_path
        dst = out_root / rel_path
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection_json, manifest_dir / "sample_selection.json")
    with (manifest_dir / "sample_selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "selected_items": len(items),
                "copied_files": copied,
                "missing_files": missing,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    print(f"selected_items: {len(items)}")
    print(f"copied_files: {copied}")
    print(f"missing_files: {len(missing)}")
    if missing:
        print("first_missing:", missing[:10])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-review")
    build.add_argument("--sample-root", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--thumb-size", type=int, default=192)
    build.add_argument("--force", action="store_true")

    apply = subparsers.add_parser("apply-selection")
    apply.add_argument("--candidate-root", type=Path, required=True)
    apply.add_argument("--selection-json", type=Path, required=True)
    apply.add_argument("--out-root", type=Path, required=True)
    apply.add_argument("--force", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-review":
        build_review(args.sample_root, args.out_dir, args.thumb_size, args.force)
    elif args.command == "apply-selection":
        apply_selection(args.candidate_root, args.selection_json, args.out_root, args.force)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
