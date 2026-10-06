"""Output paths for portable dataset construction, independent of model stacks."""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

from .paths import data_root, dataset_name


def manual_relative(kind, *parts, dataset="nuscenes"):
    if kind not in {"occ", "events", "cache"}:
        raise ValueError(f"Unsupported manual asset kind: {kind}")
    relative = Path(kind, "manual", dataset_name(dataset), *parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid manual asset path: {relative}")
    return relative.as_posix()


def manual_output(root, kind, *parts, dataset="nuscenes"):
    return data_root(root) / manual_relative(kind, *parts, dataset=dataset)


def manual_meta(root, dataset="nuscenes"):
    return data_root(root) / "meta" / dataset_name(dataset) / "manual"


def clean_nuscenes_reference(scene, token):
    return f"external/OccStress-nuScenes/gts/{scene}/{token}/labels.npz"


def portable_reference(path, root):
    # Do not dereference asset symlinks: their logical mount stays inside the dataset.
    path = Path(path).expanduser()
    base = data_root(root)
    if ".." in path.parts:
        raise ValueError(f"Parent traversal is not a release reference: {path}")
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    try:
        return path.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"Output must be inside the shared OccStress root: {path}") from exc


def write_protocol(path, records, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
