#!/usr/bin/env python3
"""Check one user-supplied weight against the recorded identity; never loads pickle."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(item, path):
    path = Path(path)
    expected = item.get("expected_sha256") or item.get("sha256")
    size = item.get("expected_bytes")
    if size is not None and path.stat().st_size != size:
        return {"id": item["id"], "status": "mismatch", "reason": "file size"}
    if not expected:
        return {"id": item["id"], "status": "unverified",
                "reason": "No recorded checksum; matching a name or size is not identity verification"}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    return {"id": item["id"], "status": "match" if actual == expected else "mismatch",
            "sha256": actual, "expected_sha256": expected,
            "identity_basis": item.get("checksum_basis", item.get("identity_status"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    rows = json.loads((ROOT / "configs/resources.json").read_text())["checkpoints"]
    item = next((row for row in rows if row["id"] == args.id), None)
    if item is None:
        parser.error("Unknown checkpoint id")
    report = verify(item, args.file)
    print(json.dumps(report, indent=2))
    return int(report["status"] != "match")


if __name__ == "__main__":
    raise SystemExit(main())
