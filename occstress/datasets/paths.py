"""Portable roots shared by the release tools; no data is downloaded here."""
from __future__ import annotations

import os
from pathlib import Path


DATASETS = {
    'nuscenes': ('OccStress-nuScenes', 'OCCSTRESS_DATA_ROOT'),
    'waymo': ('OccStress-Waymo', 'OCCSTRESS_DATA_ROOT'),
    'carla': ('OccStress-CARLA', 'OCCSTRESS_DATA_ROOT'),
}
PREFIX_DATASETS = {value[0]: key for key, value in DATASETS.items()}
TRACKS = {'manual', 'upstream', 'position_sweep'}


def dataset_key(dataset: str) -> str:
    key = dataset.lower()
    if key.startswith('occstress-'):
        key = key[len('occstress-'):]
    if key not in DATASETS:
        raise ValueError(f'Unsupported OccStress dataset: {dataset}')
    return key


def dataset_name(dataset: str) -> str:
    return DATASETS[dataset_key(dataset)][0]


def repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    if start is not None:
        return Path(start).expanduser().resolve()
    candidate = Path(os.environ.get('OCCSTRESS_CODE_ROOT',
                                    Path(__file__).resolve().parents[2])).expanduser().resolve()
    if not (candidate / 'configs/evaluation_contract.json').is_file():
        raise RuntimeError('Checkout required: set OCCSTRESS_CODE_ROOT or use an editable install')
    return candidate


def data_root(root: str | os.PathLike[str] | None = None, *,
              dataset: str = 'nuscenes', code_root=None) -> Path:
    key = dataset_key(dataset)
    _, variable = DATASETS[key]
    if root is not None:
        return Path(root).expanduser().resolve()
    if variable in os.environ:
        return Path(os.environ[variable]).expanduser().resolve()
    return (repo_root(code_root) / 'data' / 'OccStress').resolve()


def resolve_occstress_path(path: str | os.PathLike[str], *, code_root=None,
                           occstress_root=None, dataset: str | None = None,
                           external_root=None) -> Path:
    path = Path(path).expanduser()
    if '..' in path.parts:
        raise ValueError(f'Parent traversal is not a dataset path: {path}')
    if dataset is not None:
        names = {PREFIX_DATASETS[part] for part in path.parts if part in PREFIX_DATASETS}
        if names and names != {dataset_key(dataset)}:
            raise ValueError(f'Path {path} conflicts with dataset={dataset}')
    if path.is_absolute():
        return path
    parts = path.parts
    if parts[:2] == ('data', 'OccStress'):
        parts = parts[2:]
        path = Path(*parts)
    named_dataset = None
    if parts and parts[0] in {'occ', 'protocols', 'events', 'cache'}:
        if len(parts) >= 2 and parts[1] in TRACKS:
            named_dataset = PREFIX_DATASETS.get(parts[2]) if len(parts) >= 3 else None
            if named_dataset is None:
                raise ValueError(f'Missing OccStress dataset namespace: {path}')
    elif parts and parts[0] in {'meta', 'external'}:
        named_dataset = PREFIX_DATASETS.get(parts[1]) if len(parts) >= 2 else None
        if named_dataset is None:
            raise ValueError(f'Missing OccStress dataset namespace: {path}')
    if named_dataset and dataset is not None and dataset_key(dataset) != named_dataset:
        raise ValueError(f'Path {path} conflicts with dataset={dataset}')
    key = dataset_key(dataset or named_dataset or 'nuscenes')
    base = data_root(occstress_root, dataset=key, code_root=code_root)
    external_root = external_root or os.environ.get('OCCSTRESS_EXTERNAL_ROOT')
    if parts and parts[0] == 'external' and external_root:
        return Path(external_root).expanduser().joinpath(*parts[1:]).resolve()
    if parts and parts[0] in {'occ', 'protocols', 'meta', 'events', 'cache', 'external', 'manifests'}:
        return (base / path).resolve()
    # Non-benchmark files, e.g. official nuScenes info PKLs, are checkout-relative.
    return (repo_root(code_root) / path).resolve()


def manual_protocol_path(*parts: str, occstress_root=None, dataset='nuscenes') -> Path:
    return data_root(occstress_root, dataset=dataset) / 'protocols' / 'manual' / dataset_name(dataset) / Path(*parts)


def release_occ_path(path, *, occstress_root=None, dataset=None):
    """Resolve namespaced release assets before any legacy model cache fallback."""
    parts = Path(path).parts
    if parts[:2] == ('data', 'OccStress'):
        parts = parts[2:]
    if parts and parts[0] in {'occ', 'external'}:
        result = resolve_occstress_path(Path(*parts), occstress_root=occstress_root,
                                       dataset=dataset)
        return result if result.suffix == '.npz' else result / 'labels.npz'
    return None


def upstream_protocol_path(subtrack: str, source_model: str, *parts: str,
                           occstress_root=None, dataset='nuscenes') -> Path:
    if subtrack not in {'camera_only', 'camera_fusion', 'pointcloud_fusion'}:
        raise ValueError(f'Unsupported upstream subtrack: {subtrack}')
    return data_root(occstress_root, dataset=dataset) / 'protocols' / 'upstream' / dataset_name(dataset) / subtrack / source_model / Path(*parts)


def occ_path(*parts: str, occstress_root=None, dataset='nuscenes') -> Path:
    if not parts or parts[0] not in {'manual', 'upstream'}:
        raise ValueError('Occupancy assets require a manual or upstream track')
    return data_root(occstress_root, dataset=dataset) / 'occ' / parts[0] / dataset_name(dataset) / Path(*parts[1:])
