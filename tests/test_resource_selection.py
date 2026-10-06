import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.check_checkpoint import verify
from tools.list_data_packages import select_packages


class ResourceSelectionTests(unittest.TestCase):
    def index(self):
        paths = ["metadata/common.tar.zst", "metadata/OccStress-Waymo.tar.zst",
                 "archives/manual/OccStress-Waymo/part-00000.tar.zst",
                 "archives/upstream/OccStress-Waymo/camera_fusion/effocc/part-00000.tar.zst",
                 "archives/manual/OccStress-CARLA/part-00000.tar.zst"]
        return {"packages": [dict(path=path, status="available", bytes=12, sha256="a"*64) for path in paths]}

    def test_manual_selection_has_both_metadata_archives_and_no_other_track(self):
        rows = select_packages(self.index(), "waymo", "manual", revision="b"*40)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all("/resolve/" + "b"*40 + "/" in r["url"] for r in rows))
        self.assertFalse(any("/upstream/" in r["path"] for r in rows))

    def test_source_filter_keeps_metadata(self):
        rows = select_packages(self.index(), "waymo", "upstream", "camera_fusion/effocc")
        self.assertEqual(len(rows), 3)
        with self.assertRaises(ValueError):
            select_packages(self.index(), "waymo", "upstream", "camera_only/missing")

    def test_partial_or_unverified_payload_is_not_silently_available(self):
        index = self.index()
        index["packages"][2].update(status="pending", sha256=None, bytes=None)
        self.assertEqual(select_packages(index, "waymo", "manual")[-1]["status"], "pending")
        index["packages"][2]["status"] = "available"
        with self.assertRaises(ValueError):
            select_packages(index, "waymo", "manual")

    def test_invalid_paths_duplicates_and_revisions_are_rejected(self):
        for value in ("../escape.tar.zst", "/absolute.tar.zst"):
            index = self.index()
            index["packages"][2]["path"] = value
            with self.assertRaises(ValueError):
                select_packages(index, "waymo", "manual")
        index = self.index()
        index["packages"].append(copy.deepcopy(index["packages"][0]))
        with self.assertRaises(ValueError):
            select_packages(index, "waymo", "manual")
        with self.assertRaises(ValueError):
            select_packages(self.index(), "waymo", revision="../main")

    def test_checkpoint_match_mismatch_and_unknown_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            path.write_bytes(b"not-a-pickle")
            item = {"id": "fixture"}
            self.assertEqual(verify(item, path)["status"], "unverified")
            item["expected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(verify(item, path)["status"], "match")
            path.write_bytes(b"changed")
            self.assertEqual(verify(item, path)["status"], "mismatch")
