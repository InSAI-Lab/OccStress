#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Build the compact CVT-Occ annotation used by OccStress-Waymo runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SCENES = 202
EXPECTED_FRAMES = 7_998


def install_numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to read annotation pickles written by NumPy 2.x."""
    import numpy as np

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=Path,
        help=(
            "Optional official Waymo annotation. When omitted, create the "
            "minimal image index required by the occupancy test loader."
        ),
    )
    parser.add_argument("--frame-index-root", type=Path, required=True)
    parser.add_argument("--output-annotation", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selected_sample_ids(frame_index_root: Path) -> list[int]:
    paths = sorted(frame_index_root.glob("[0-9][0-9][0-9].json"))
    if len(paths) != EXPECTED_SCENES:
        raise ValueError(
            f"expected {EXPECTED_SCENES} frame indexes, got {len(paths)}"
        )
    sample_ids = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for frame in payload["frames"]:
            sample_ids.append(
                1_000_000
                + int(frame["scene_index"]) * 1_000
                + int(frame["frame_index"])
            )
    if len(sample_ids) != EXPECTED_FRAMES or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise ValueError(
            f"expected {EXPECTED_FRAMES} unique frames, got {len(sample_ids)}"
        )
    return sample_ids


def main() -> int:
    args = parse_args()
    sample_ids = selected_sample_ids(args.frame_index_root)
    if args.annotation is None:
        annotation = None
        compact = [
            {
                "image": {
                    "image_idx": sample_id,
                    "image_path": "",
                }
            }
            for sample_id in sample_ids
        ]
    else:
        selected = set(sample_ids)
        install_numpy_pickle_compatibility()
        with args.annotation.open("rb") as handle:
            annotation = pickle.load(handle)
        if not isinstance(annotation, list):
            raise TypeError(
                f"expected a list annotation, got {type(annotation).__name__}"
            )

        by_sample_id = {
            int(info["image"]["image_idx"]): info
            for info in annotation
            if int(info["image"]["image_idx"]) in selected
        }
        missing = sorted(selected - set(by_sample_id))
        if missing:
            raise ValueError(
                "official annotation is missing "
                f"{len(missing)} frames: {missing[:8]}"
            )
        compact = [by_sample_id[sample_id] for sample_id in sample_ids]
    atomic_pickle(args.output_annotation, compact)
    payload = {
        "annotation_mode": (
            "minimal-frame-index"
            if args.annotation is None
            else "official-subset"
        ),
        "frame_count": len(compact),
        "frame_index_root": str(args.frame_index_root.resolve()),
        "output_annotation": str(args.output_annotation.resolve()),
        "output_sha256": sha256_file(args.output_annotation),
        "scene_count": EXPECTED_SCENES,
        "source_annotation": (
            None if args.annotation is None else str(args.annotation.resolve())
        ),
        "source_count": None if annotation is None else len(annotation),
        "source_sha256": (
            None if args.annotation is None else sha256_file(args.annotation)
        ),
        "status": "success",
    }
    atomic_json(args.output_manifest, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
