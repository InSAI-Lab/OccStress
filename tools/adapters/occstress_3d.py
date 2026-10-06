from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .occstress_paths import data_root, resolve_occstress_path


def load_protocol(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_occstress_path(path)
    with resolved.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected a list of protocol records, got {type(records)!r}")
    return records


def labels_npz_path(*parts: str, occstress_root: str | Path | None = None) -> Path:
    return data_root(occstress_root) / "occ" / Path(*parts) / "labels.npz"


def save_labels_npz(path: str | Path, semantics: np.ndarray, **extra_arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"semantics": semantics}
    arrays.update(extra_arrays)
    np.savez_compressed(path, **arrays)
    return path


def iter_label_files(root: str | Path) -> Iterator[Path]:
    root = resolve_occstress_path(root)
    yield from sorted(root.rglob("labels.npz"))
