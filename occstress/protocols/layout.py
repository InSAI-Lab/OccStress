#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import os
import sys
from pathlib import Path

from occstress.datasets.paths import DATASETS, data_root, dataset_name, repo_root

DEFAULT_ROOT = repo_root()
DATASET_DIRS = {key: value[0] for key, value in DATASETS.items()}

MANUAL_TRACK = "manual"
UPSTREAM_TRACK = "upstream"
MANUAL_FAMILIES_WITH_SEVERITY = {"semantic", "hole", "dropout", "misalignment"}
MANUAL_FAMILIES_NO_SEVERITY = {"traffic"}
UPSTREAM_SUBTRACKS = {"camera_only", "camera_fusion", "pointcloud_fusion"}


def _as_path(root):
    return Path(root).resolve()


def dataset_occstress_root(dataset="nuscenes", root=DEFAULT_ROOT, occstress_root=None):
    return data_root(occstress_root, dataset=dataset, code_root=root)


def manual_protocol_root(root=DEFAULT_ROOT, dataset="nuscenes", occstress_root=None):
    return dataset_occstress_root(dataset, root, occstress_root) / "protocols" / MANUAL_TRACK / dataset_name(dataset)


def upstream_protocol_root(subtrack, source_model, root=DEFAULT_ROOT, dataset="nuscenes", occstress_root=None):
    if subtrack not in UPSTREAM_SUBTRACKS:
        raise ValueError(f"Unsupported upstream subtrack: {subtrack}")
    return dataset_occstress_root(dataset, root, occstress_root) / "protocols" / UPSTREAM_TRACK / dataset_name(dataset) / subtrack / source_model


def normalize_protocol_name(protocol_name):
    return protocol_name[:-4] if protocol_name.endswith(".pkl") else protocol_name


def _split_protocol_name(protocol_name):
    protocol_name = normalize_protocol_name(protocol_name)
    if protocol_name.startswith("clean_"):
        return ("clean", None, protocol_name[len("clean_"):])

    for family in sorted(MANUAL_FAMILIES_WITH_SEVERITY, key=len, reverse=True):
        prefix = f"{family}_"
        if protocol_name.startswith(prefix):
            rest = protocol_name[len(prefix):]
            severity, _, tail = rest.partition("_")
            if severity in {"easy", "mid", "hard"} and tail:
                return (family, severity, tail)

    for family in MANUAL_FAMILIES_NO_SEVERITY:
        prefix = f"{family}_"
        if protocol_name.startswith(prefix):
            tail = protocol_name[len(prefix):]
            if tail:
                return (family, None, tail)

    raise ValueError(f"Unsupported manual protocol name: {protocol_name}")


def manual_protocol_relpath(protocol_name, dataset="nuscenes"):
    family, severity, tail = _split_protocol_name(protocol_name)
    base = Path(MANUAL_TRACK) / dataset_name(dataset)
    if family == "clean":
        return base / "clean" / f"{tail}.pkl"
    if severity is None:
        return base / family / f"{tail}.pkl"
    return base / family / severity / f"{tail}.pkl"


def resolve_manual_protocol_path(
    protocol_name, root=DEFAULT_ROOT, must_exist=False, dataset="nuscenes", occstress_root=None
):
    root = _as_path(root)
    rel = manual_protocol_relpath(protocol_name, dataset)
    protocol_root = dataset_occstress_root(dataset, root, occstress_root) / "protocols"
    path = protocol_root / rel
    if must_exist and not path.exists():
        name = normalize_protocol_name(protocol_name)
        if not name.endswith("_backbone"):
            fallback = protocol_root / manual_protocol_relpath(f"{name}_backbone", dataset)
            if fallback.exists():
                return fallback
        raise FileNotFoundError(path)
    return path


def resolve_upstream_protocol_path(
    subtrack, source_model, corruption, severity, protocol_stem, root=DEFAULT_ROOT, must_exist=False,
    dataset="nuscenes", occstress_root=None
):
    root = _as_path(root)
    protocol_stem = normalize_protocol_name(protocol_stem)
    if corruption == "clean":
        path = upstream_protocol_root(subtrack, source_model, root=root, dataset=dataset, occstress_root=occstress_root) / "clean" / f"{protocol_stem}.pkl"
    else:
        if not severity:
            raise ValueError("severity is required for non-clean upstream protocols")
        path = (
            upstream_protocol_root(subtrack, source_model, root=root, dataset=dataset, occstress_root=occstress_root)
            / corruption
            / severity
            / f"{protocol_stem}.pkl"
        )
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def manual_protocol_name_from_path(path, root=DEFAULT_ROOT, dataset="nuscenes", occstress_root=None):
    root = _as_path(root)
    path = Path(path)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    rel = path.relative_to(dataset_occstress_root(dataset, root, occstress_root) / "protocols")
    parts = rel.parts
    if len(parts) < 4 or parts[:2] != (MANUAL_TRACK, dataset_name(dataset)):
        raise ValueError(f"Path is not under manual protocol root: {path}")
    parts = (parts[0], *parts[2:])

    stem = path.stem
    if parts[1] == "clean":
        return f"clean_{stem}"
    if parts[1] in MANUAL_FAMILIES_NO_SEVERITY:
        return f"{parts[1]}_{stem}"
    if parts[1] in MANUAL_FAMILIES_WITH_SEVERITY and len(parts) >= 4:
        return f"{parts[1]}_{parts[2]}_{stem}"
    raise ValueError(f"Unsupported manual protocol path: {path}")


def protocol_key_from_path(path, root=DEFAULT_ROOT, dataset="nuscenes", occstress_root=None):
    root = _as_path(root)
    path = Path(path)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()

    rel = path.relative_to(dataset_occstress_root(dataset, root, occstress_root) / "protocols")
    parts = rel.parts
    if len(parts) < 3 or parts[1] != dataset_name(dataset):
        raise ValueError(f"Invalid protocol path: {path}")

    if parts[0] == MANUAL_TRACK:
        return manual_protocol_name_from_path(
            path, root=root, dataset=dataset, occstress_root=occstress_root
        )

    if parts[0] == UPSTREAM_TRACK:
        parts = (parts[0], *parts[2:])
        if len(parts) < 5:
            raise ValueError(f"Unsupported upstream protocol path: {path}")
        subtrack = parts[1]
        source_model = parts[2]
        if parts[3] == "clean":
            return f"upstream__{subtrack}__{source_model}__clean__{path.stem}"
        if len(parts) < 6:
            raise ValueError(f"Unsupported upstream protocol path: {path}")
        corruption = parts[3]
        severity = parts[4]
        return f"upstream__{subtrack}__{source_model}__{corruption}__{severity}__{path.stem}"

    raise ValueError(f"Unsupported protocol track in path: {path}")


def main():
    parser = argparse.ArgumentParser(description="Helpers for the unified OccStress dataset layout.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-manual-protocol")
    resolve_parser.add_argument("protocol_name")
    resolve_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    resolve_parser.add_argument("--dataset", default="nuscenes", choices=sorted(DATASET_DIRS))
    resolve_parser.add_argument("--occstress-root")
    resolve_parser.add_argument("--must-exist", action="store_true")

    upstream_resolve_parser = subparsers.add_parser("resolve-upstream-protocol")
    upstream_resolve_parser.add_argument("subtrack")
    upstream_resolve_parser.add_argument("source_model")
    upstream_resolve_parser.add_argument("corruption")
    upstream_resolve_parser.add_argument("severity")
    upstream_resolve_parser.add_argument("protocol_stem")
    upstream_resolve_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    upstream_resolve_parser.add_argument("--dataset", default="nuscenes", choices=sorted(DATASET_DIRS))
    upstream_resolve_parser.add_argument("--occstress-root")
    upstream_resolve_parser.add_argument("--must-exist", action="store_true")

    name_parser = subparsers.add_parser("name-from-manual-path")
    name_parser.add_argument("path")
    name_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    name_parser.add_argument("--dataset", default="nuscenes", choices=sorted(DATASET_DIRS))
    name_parser.add_argument("--occstress-root")

    protocol_key_parser = subparsers.add_parser("protocol-key-from-path")
    protocol_key_parser.add_argument("path")
    protocol_key_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    protocol_key_parser.add_argument("--dataset", default="nuscenes", choices=sorted(DATASET_DIRS))
    protocol_key_parser.add_argument("--occstress-root")

    root_parser = subparsers.add_parser("manual-protocol-root")
    root_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    root_parser.add_argument("--dataset", default="nuscenes", choices=sorted(DATASET_DIRS))
    root_parser.add_argument("--occstress-root")

    args = parser.parse_args()

    if args.command == "resolve-manual-protocol":
        print(resolve_manual_protocol_path(
            args.protocol_name, root=args.root, must_exist=args.must_exist,
            dataset=args.dataset, occstress_root=args.occstress_root))
    elif args.command == "name-from-manual-path":
        print(manual_protocol_name_from_path(
            args.path, root=args.root, dataset=args.dataset, occstress_root=args.occstress_root))
    elif args.command == "protocol-key-from-path":
        print(protocol_key_from_path(
            args.path, root=args.root, dataset=args.dataset, occstress_root=args.occstress_root))
    elif args.command == "resolve-upstream-protocol":
        print(
            resolve_upstream_protocol_path(
                args.subtrack,
                args.source_model,
                args.corruption,
                args.severity,
                args.protocol_stem,
                root=args.root,
                must_exist=args.must_exist,
                dataset=args.dataset,
                occstress_root=args.occstress_root,
            )
        )
    elif args.command == "manual-protocol-root":
        print(manual_protocol_root(
            args.root, dataset=args.dataset, occstress_root=args.occstress_root))


if __name__ == "__main__":
    main()
