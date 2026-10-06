#!/usr/bin/env python3
"""Download one immutable package selection and verify archive checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.list_data_packages import select_packages


def validated_selection(selection):
    revision = selection.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Selection must pin a commit; regenerate it with --fetch")
    rows = select_packages({"packages": selection["packages"]}, selection["dataset"],
                           selection.get("track", "all"), selection.get("source"), revision)
    if len(rows) != len(selection["packages"]):
        raise ValueError("Selection includes packages outside the requested dataset/track")
    if any(row["status"] != "available" for row in rows):
        raise ValueError("Selected packages are still pending; no payload downloaded")
    if any(not row["path"].endswith(".tar.zst") for row in rows):
        raise ValueError("Only dataset archives may be downloaded")
    return rows


def matches(path, row):
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == row["sha256"]


def destination(root, relative):
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink in download path: {relative}")
    return path


def download(selection, output_dir, verify_only=False, fetch=None):
    rows = validated_selection(selection)
    root = Path(output_dir).resolve()
    receipt = root / "download-receipt.json"
    if receipt.is_symlink():
        raise ValueError("Download receipt must not be a symlink")
    if receipt.exists() and json.loads(receipt.read_text())["revision"] != selection["revision"]:
        raise ValueError("Use a different download directory for a different dataset revision")
    if fetch is None and not verify_only:
        from huggingface_hub import hf_hub_download
        fetch = hf_hub_download
    for row in rows:
        path = destination(root, row["path"])
        if not matches(path, row):
            if verify_only:
                raise ValueError(f"Missing or invalid archive: {row['path']}")
            print(f"Downloading {row['path']}", flush=True)
            fetch(repo_id="insailab/OccStress", repo_type="dataset",
                  revision=selection["revision"], filename=row["path"],
                  local_dir=str(root), force_download=path.exists())
            path = destination(root, row["path"])
            if not matches(path, row):
                raise ValueError(f"Archive checksum/size mismatch: {row['path']}")
        print(f"Verified {row['path']}", flush=True)
    report = {"dataset": selection["dataset"], "revision": selection["revision"],
              "track": selection.get("track", "all"), "source": selection.get("source"),
              "status": "verified", "packages": rows,
              "archive_bytes": sum(row["bytes"] for row in rows)}
    if not verify_only:
        root.mkdir(parents=True, exist_ok=True)
        temporary = destination(root, "download-receipt.json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(receipt)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True,
                        help="JSON from tools/list_data_packages.py --fetch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check locally, without network access")
    args = parser.parse_args()
    report = download(json.loads(args.selection.read_text()), args.output_dir, args.verify_only)
    print(json.dumps({key: value for key, value in report.items() if key != "packages"}, indent=2))


if __name__ == "__main__":
    main()
