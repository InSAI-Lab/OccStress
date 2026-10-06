import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.download_data import download, validated_selection
from tools.list_data_packages import resolve_revision, select_packages


class DownloadTests(unittest.TestCase):
    def selection(self):
        paths = ["metadata/common.tar.zst", "metadata/OccStress-Waymo.tar.zst",
                 "archives/manual/OccStress-Waymo/part-00000.tar.zst"]
        return {"dataset": "waymo", "track": "manual", "source": None,
                "revision": "a" * 40,
                "packages": [{"path": p, "status": "available", "bytes": 4,
                              "sha256": hashlib.sha256(b"data").hexdigest()} for p in paths]}

    def test_download_resume_and_offline_verification(self):
        selection, calls = self.selection(), []
        def fetch(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
        with tempfile.TemporaryDirectory() as directory:
            report = download(selection, directory, fetch=fetch)
            self.assertEqual(report["archive_bytes"], 12)
            self.assertEqual(len(calls), 3)
            download(selection, directory, fetch=fetch)
            download(selection, directory, verify_only=True)
            self.assertEqual(len(calls), 3)
            path = Path(directory) / selection["packages"][0]["path"]
            path.write_bytes(b"bad!")
            with self.assertRaisesRegex(ValueError, "invalid archive"):
                download(selection, directory, verify_only=True)
            download(selection, directory, fetch=fetch)
            self.assertTrue(calls[-1]["force_download"])

    def test_pending_mutable_revision_and_foreign_paths_fail_before_fetch(self):
        for transform in [lambda s: s.update(revision="main"),
                          lambda s: s["packages"][0].update(status="pending"),
                          lambda s: s["packages"][0].update(path="metadata/../secret"),
                          lambda s: s["packages"][0].update(bytes=True)]:
            selection = self.selection()
            transform(selection)
            with self.assertRaises(ValueError):
                validated_selection(selection)
        selection = self.selection()
        selection["packages"].append({**selection["packages"][0],
                                     "path": "archives/manual/OccStress-CARLA/a.tar.zst"})
        with self.assertRaisesRegex(ValueError, "outside"):
            validated_selection(selection)

    def test_wrong_content_never_gets_a_verified_receipt(self):
        def fetch(**kwargs):
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"bad!")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                download(self.selection(), directory, fetch=fetch)
            self.assertFalse((Path(directory) / "download-receipt.json").exists())

    def test_no_symlink_writes_or_version_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                download(self.selection(), root, verify_only=True)
            (root / "download-receipt.json").write_text(json.dumps({"revision": "b" * 40}))
            with self.assertRaisesRegex(ValueError, "different"):
                download(self.selection(), root, verify_only=True)

    def test_reject_ambiguous_archive_paths(self):
        for value in ("metadata//common.tar.zst", "metadata/./common.tar.zst",
                      "metadata/.hidden/a.tar.zst", "metadata\\common.tar.zst"):
            selection = self.selection()
            selection["packages"][0]["path"] = value
            with self.assertRaises(ValueError):
                select_packages(selection, "waymo", "manual")

    def test_fixed_revision_needs_no_resolution(self):
        with patch("tools.list_data_packages.urlopen") as request:
            self.assertEqual(resolve_revision("a" * 40), "a" * 40)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
