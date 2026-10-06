#!/usr/bin/env python3
"""Select published shards without downloading any dataset payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.datasets.paths import dataset_name


def select_packages(index, dataset, track="all", source=None, revision="main"):
    name = dataset_name(dataset)
    if track not in {"all", "manual", "upstream"}:
        raise ValueError("Choose all, manual or upstream")
    if source and (track != "upstream" or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", source)):
        raise ValueError("--source requires --track upstream and SUBTRACK/MODEL")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", revision):
        raise ValueError("Invalid dataset revision")
    selected, seen = [], set()
    metadata = {f"metadata/{name}.tar.zst", "metadata/common.tar.zst"}
    for item in index["packages"]:
        relative = item["path"]
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or relative in seen
                or "\\" in relative or str(path) != relative
                or any(part.startswith(".") for part in path.parts)):
            raise ValueError(f"Invalid or duplicate package path: {relative}")
        seen.add(relative)
        asset = any(relative.startswith(f"archives/{t}/{name}/")
                    for t in (("manual", "upstream") if track == "all" else (track,)))
        if source:
            asset = relative.startswith(f"archives/upstream/{name}/{source}/")
        if relative not in metadata and not asset:
            continue
        if item["status"] not in {"pending", "available"}:
            raise ValueError(f"Unknown package status: {relative}")
        if item["status"] == "available":
            if (not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256") or "")
                    or type(item.get("bytes")) is not int or item["bytes"] <= 0):
                raise ValueError(f"Available package is missing size/checksum: {relative}")
        selected.append({**item, "url": f"https://huggingface.co/datasets/insailab/OccStress/resolve/{revision}/{relative}"})
    if not metadata.issubset(seen) or not any(row["path"].startswith("archives/") for row in selected):
        raise ValueError("No complete package selection for the requested dataset/track/source")
    return selected


def resolve_revision(revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    with urlopen(f"https://huggingface.co/api/datasets/insailab/OccStress/revision/{revision}",
                 timeout=60) as response:
        commit = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hub did not return an immutable dataset revision")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["nuscenes", "waymo", "carla"])
    parser.add_argument("--track", default="all", choices=["all", "manual", "upstream"])
    parser.add_argument("--source", help="Upstream SUBTRACK/MODEL, e.g. camera_fusion/effocc")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=Path, help="Previously downloaded packages.json")
    group.add_argument("--fetch", action="store_true", help="Fetch the small public index only")
    parser.add_argument("--revision", default="main", help="Use a dataset commit for repeatable downloads")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.revision):
        parser.error("Invalid revision")
    if args.fetch:
        args.revision = resolve_revision(args.revision)
        url = f"https://huggingface.co/datasets/insailab/OccStress/resolve/{args.revision}/packages.json"
        with urlopen(url, timeout=60) as response:
            index = json.load(response)
    else:
        index = json.loads(args.index.read_text())
    packages = select_packages(index, args.dataset, args.track, args.source, args.revision)
    missing = [item["path"] for item in packages if item["status"] != "available"]
    print(json.dumps({"dataset": dataset_name(args.dataset), "track": args.track,
                      "source": args.source, "revision": args.revision,
                      "ready": not missing, "pending": missing, "packages": packages}, indent=2))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
