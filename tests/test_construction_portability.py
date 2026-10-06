import collections
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from occstress.corruptions import dropout, hole, semantic, traffic
from occstress.datasets.construction import manual_output, portable_reference, write_protocol
from occstress.datasets.paths import resolve_occstress_path
from scripts.nuscenes import build_clean_backbone, generate_protocols
from scripts.waymo import build_effocc_waymo_upstream_protocols as point
from scripts.waymo import build_effocc_waymo_camera_upstream_protocols as camera


class ConstructionPortabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "OccStress"
        self.env = patch.dict(os.environ, {"OCCSTRESS_DATA_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def call(self, module, argv):
        with patch.object(sys, "argv", [module.__name__, *argv]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_manual_defaults_use_shared_environment_root(self):
        for module in (semantic, dropout, hole, traffic):
            flags = [] if module is traffic else ["--severity", "hard"]
            with patch.object(sys, "argv", ["generate", *flags]):
                args = module.parse_args()
            self.assertIsNone(args.output_root)
            self.assertEqual(manual_output(args.output_root, "occ", "traffic"),
                             self.root / "occ/manual/OccStress-nuScenes/traffic")

    def test_traffic_generator_writes_portable_events_without_changing_mirror(self):
        raw = self.root.parent / "raw"
        path = raw / "gts/scene/anchor/labels.npz"
        path.parent.mkdir(parents=True)
        original = np.arange(24, dtype=np.uint8).reshape(3, 4, 2) % 18
        np.savez_compressed(path, semantics=original, mask_camera=original, mask_lidar=original)
        with patch.object(sys, "argv", ["traffic", "--data-root", str(raw)]):
            args = traffic.parse_args()
        sample = {"token": "anchor", "scene_token": "scene-token"}
        metadata = {"scene_by_token": {"scene-token": {"name": "scene"}}}
        summary = {"processed": 0, "changed_voxels_total": 0,
                   "class_histogram_total": collections.Counter()}
        self.assertTrue(traffic.process_sample(sample, metadata, args, summary))
        event = json.loads((self.root / "events/manual/OccStress-nuScenes/traffic/scene/anchor.json").read_text())
        with np.load(self.root / event["occ_path_corrupt"]) as payload:
            for key in ("semantics", "mask_camera", "mask_lidar"):
                np.testing.assert_array_equal(payload[key], original[:, ::-1, :])
        self.assertEqual(event["occ_path_clean"], "external/OccStress-nuScenes/gts/scene/anchor/labels.npz")

    def test_nuscenes_backbone_and_37_protocols_relocate(self):
        samples = [dict(token=str(i), prev=str(i-1) if i else "",
                        next=str(i+1) if i < 10 else "", scene_token="s", timestamp=i)
                   for i in range(11)]
        metadata = {"samples": samples, "scene_by_token": {"s": {"name": "scene"}}}
        ann = self.root.parent / "val.pkl"
        ann.write_bytes(pickle.dumps({"infos": [{"token": "4"}]}))
        with patch.object(semantic, "collect_indices", return_value=metadata):
            self.call(build_clean_backbone, ["--ann-file", str(ann)])
        backbone = self.root / "protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl"
        self.assertEqual(len(pickle.loads(backbone.read_bytes())), 1)
        base_rows = pickle.loads(backbone.read_bytes())
        base_rows[0]["future_trajectory"] = [[i, 0] for i in range(6)]
        base_rows[0]["command"] = [0, 0, 1]
        write_protocol(backbone, base_rows, overwrite=True)
        for family in ("semantic", "hole", "dropout", "traffic"):
            for severity in ([""] if family == "traffic" else ["easy", "mid", "hard"]):
                parts = [family, *([severity] if severity else [])]
                for sample in samples:
                    output = manual_output(self.root, "occ", *parts, "scene", sample["token"], "labels.npz")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output, semantics=np.zeros((2, 2, 1), dtype=np.uint8))
                    event = manual_output(self.root, "events", *parts, "scene", sample["token"] + ".json")
                    event.parent.mkdir(parents=True, exist_ok=True)
                    event.write_text("{}")
        with patch.object(semantic, "collect_indices", return_value=metadata):
            self.call(generate_protocols, ["--backbone-path", str(backbone)])
        protocols = list((self.root / "protocols/manual/OccStress-nuScenes").rglob("*.pkl"))
        self.assertEqual(len(protocols), 38)
        records = []
        for path in protocols:
            rows = pickle.loads(path.read_bytes())
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["future_targets"]), 6)
            self.assertEqual(rows[0]["future_trajectory"], base_rows[0]["future_trajectory"])
            self.assertEqual(rows[0]["command"], base_rows[0]["command"])
            records.extend(rows)
        moved = self.root.with_name("relocated")
        shutil.move(self.root, moved)
        for row in records:
            for frame in [*row["history"], row["current_input"], row["target"], *row["future_targets"]]:
                ref = frame["occ_path"]
                self.assertFalse(Path(ref).is_absolute())
                if ref.startswith("occ/"):
                    self.assertTrue(resolve_occstress_path(ref, occstress_root=moved).is_file())
                else:
                    self.assertTrue(ref.startswith("external/OccStress-nuScenes/"))
        traffic_path = moved / "protocols/manual/OccStress-nuScenes/traffic/H4_F6_val_backbone.pkl"
        self.assertTrue(traffic_path.is_file())
        row = pickle.loads((moved / "protocols/manual/OccStress-nuScenes/semantic/hard/recent_burst_H4_F6_val_backbone.pkl").read_bytes())[0]
        self.assertEqual([f["occ_source"] != "clean" for f in row["history"]], [False, False, False, True])
        self.assertEqual(row["current_input"]["source"], "semantic")

    def test_waymo_both_full_builders_preserve_targets_controls_and_portability(self):
        def frame(index):
            return {"token": str(index), "occ_path": f"external/OccStress-Waymo/gts/000/{index}/labels.npz"}
        row = dict(sample_id="4__clean", anchor_token="4", scene_name="000",
                   history_length=4, future_length=6, history=[frame(i) for i in range(4)],
                   current_input=frame(4), target=frame(4), future_targets=[frame(i) for i in range(5, 11)],
                   pose=[[1, 0], [0, 1]], future_trajectory=[[i, 0] for i in range(6)], command=[0, 0, 1])
        for f in [row["target"], *row["future_targets"]]:
            f["occ_source"] = "clean"
        backbone = self.root / "protocols/manual/OccStress-Waymo/clean/H4_F6_val_backbone.pkl"
        write_protocol(backbone, [row])
        for module in (point, camera):
            base = self.root / "occ/upstream/OccStress-Waymo" / module.SUBTRACK / "effocc"
            meta = self.root / "meta/OccStress-Waymo/upstream" / module.SUBTRACK / "effocc"
            indexes = meta / "frame_index"
            indexes.mkdir(parents=True)
            (indexes / "000.json").write_text(json.dumps({"frames": [{"token": str(i)} for i in range(11)]}))
            for corruption, severity in module.settings():
                setting = module.setting_root(base, corruption, severity) / "000"
                for i in range(11):
                    path = setting / str(i) / "labels.npz"
                    path.parent.mkdir(parents=True)
                    np.savez_compressed(path, semantics=np.zeros(module.VOXEL_SHAPE, dtype=np.uint8))
                (setting / ".done.json").write_text(json.dumps(dict(status="success", adapter_version=module.ADAPTER_VERSION,
                    frames=11, voxel_shape=list(module.VOXEL_SHAPE), checkpoint_sha256="a"*64)))
            args = ["--expected-scenes", "1", "--expected-frames", "11", "--expected-anchors", "1"]
            self.call(module, args)
            manifest = json.loads((meta / "protocols.json").read_text())
            self.assertEqual(len(manifest["protocols"]), 73)
            for item in manifest["protocols"]:
                records = pickle.loads((self.root / item["path"]).read_bytes())
                for key in ("target", "future_targets", "pose", "command", "future_trajectory"):
                    self.assertEqual(records[0][key], row[key])
                for f in [*records[0]["history"], records[0]["current_input"]]:
                    self.assertTrue(f["occ_path"].startswith("occ/upstream/OccStress-Waymo/"))
                    self.assertTrue((self.root / f["occ_path"]).is_file())
            with self.assertRaises(FileExistsError):
                self.call(module, args)
            self.call(module, ["--spec-only"])
            self.assertEqual(json.loads((meta / "protocols.json").read_text())["status"], "success")

    def test_portable_paths_and_protocol_writes_are_guarded(self):
        path = self.root / "protocols/fixture.pkl"
        write_protocol(path, [1])
        with self.assertRaises(FileExistsError):
            write_protocol(path, [2])
        self.assertEqual(pickle.loads(path.read_bytes()), [1])
        with self.assertRaises(ValueError):
            portable_reference(self.root.parent / "escape", self.root)
        with self.assertRaises(ValueError):
            portable_reference(self.root / "../escape", self.root)


if __name__ == "__main__":
    unittest.main()
