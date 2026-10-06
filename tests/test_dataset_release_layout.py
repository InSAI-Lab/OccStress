import ast
import hashlib
import io
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from occstress.adapters.forecasting import command_plan
from occstress.datasets.metadata import load_metadata
from occstress.datasets.paths import release_occ_path
from tools.validate_occstress_dataset import validate
from occstress.distribution import withheld

def extracted(path, name, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    scope = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name)
    env = {'os': os, 'Path': Path}
    future = [item for item in tree.body
              if isinstance(item, ast.ImportFrom) and item.module == '__future__']
    exec(compile(ast.Module(body=future + [node], type_ignores=[]), path, 'exec'), env)
    function = env[name]
    return function.__func__ if isinstance(function, staticmethod) else function


class ReleaseAdapterTests(unittest.TestCase):
    def test_dome_class_map_uses_dataset_namespace(self):
        resolve = extracted('EXIST/4D/DOME/tools/eval_occstress.py', 'class_mapping_path')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'meta/OccStress-Waymo/class_mapping.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            for name in ('Occ3D-Waymo', 'OccStress-Waymo'):
                self.assertEqual(resolve(root, name), path)
            with self.assertRaises(FileNotFoundError):
                resolve(root, 'OccStress-CARLA')

    @unittest.skipIf(withheld(ROOT, 'EXIST/4D/SparseWorld'), 'SparseWorld source withheld pending permission')
    def test_sparseworld_camera_paths_use_external_mount(self):
        source = 'EXIST/4D/SparseWorld/occstress/carla_dataset.py'
        resolve = extracted(source, '_camera_path', 'SparseWorldCarlaDataset')
        frame = {'cams': {'CAM_FRONT': {'data_path':
                 'external/OccStress-CARLA/native_dataset/scene/camera.png'}}}
        with patch.dict(os.environ, {'OCCSTRESS_EXTERNAL_ROOT': '/official'}, clear=True):
            self.assertEqual(resolve(frame, 'CAM_FRONT'),
                             '/official/OccStress-CARLA/native_dataset/scene/camera.png')

    @unittest.skipIf(withheld(ROOT, 'EXIST/4D/SparseWorld'), 'SparseWorld source withheld pending permission')
    def test_sparseworld_private_namespace_does_not_shadow_sdk(self):
        native = ROOT / 'EXIST/4D/SparseWorld'
        with patch.object(sys, 'path', [str(native), *sys.path]):
            import occstress
            package = importlib.import_module('occstress_adapters')
            self.assertEqual(Path(occstress.__file__).resolve().parent, ROOT / 'occstress')
            self.assertIn(str(native / 'occstress'), package.__path__)
            spec = importlib.machinery.PathFinder.find_spec('carla_dataset', package.__path__)
            self.assertEqual(Path(spec.origin), native / 'occstress/carla_dataset.py')

    def test_new_paths_bypass_stale_native_caches(self):
        frame = {'token': 't', 'occ_source': 'clean',
                 'occ_path': 'occ/upstream/OccStress-Waymo/camera_only/cvtocc/clean/s/t/labels.npz'}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            root = Path(temporary)
            cached = root / 'old/s/t/labels.npz'
            cached.parent.mkdir(parents=True)
            cached.touch()
            os.environ.update(OCCSTRESS_DATA_ROOT=str(root), OCCSTRESS_WAYMO_CLEAN_OCC_CACHE=str(root / 'old'))
            expected = root / frame['occ_path']
            for path in ('EXIST/4D/II-World/mmdet3d/datasets/waymo_occstress_world_dataset.py',
                         'EXIST/4D/GenieDrive/occ_gen/mmdet3d/datasets/waymo_occstress_world_dataset.py'):
                self.assertEqual(Path(extracted(path, '_frame_occ_path')(frame, 's')), expected)
            self.assertEqual(extracted('EXIST/4D/COME/dataset/dataset.py', '_waymo_frame_occ_path')(frame, 's'), expected)
            adapter = types.SimpleNamespace(occstress_root=root)
            self.assertEqual(extracted('EXIST/4D/DOME/dataset/occstress_dataset.py', '_resolve_occ_path',
                                       'OccStressProtocolDataset')(adapter, frame, 's'), expected)
            self.assertEqual(extracted('EXIST/4D/OccWorld/dataset/dataset.py', '_resolve_occ_path',
                                       'OccStressCARLASceneDataset')(adapter, frame), expected)

    def test_controls_json_does_not_change_numeric_values(self):
        controls = {'metadata': {'dataset': 'OccStress-Waymo'}, 'infos': {
            's': [{'token': 't', 'pose_mat': [[1.0, -0.25], [0.0, 1.0]], 'pose_mode': [0, 1, 0]}]}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'meta/OccStress-Waymo/controls.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(controls))
            self.assertEqual(load_metadata('meta/OccStress-Waymo/controls.json', occstress_root=root), controls)
            path = root / 'old.pkl'
            path.write_bytes(pickle.dumps(controls))
            with self.assertRaises(ValueError):
                load_metadata(path)
            self.assertEqual(load_metadata(path, trusted_pickle=True), controls)

    def test_native_arguments_resolve_before_changing_cwd(self):
        _, command, _ = command_plan('geniedrive', 'waymo', 'python', [
            '--protocol=protocols/manual/OccStress-Waymo/clean/p.pkl',
            '--base-info', 'meta/OccStress-Waymo/controls.json'],
            environ={'OCCSTRESS_DATA_ROOT': '/datasets/OccStress'})
        self.assertIn('--protocol=/datasets/OccStress/protocols/manual/OccStress-Waymo/clean/p.pkl', command)
        self.assertEqual(command[-1], '/datasets/OccStress/meta/OccStress-Waymo/controls.json')

    def test_mmcv_named_streams_preserve_ownership_and_pickle_trust(self):
        controls = {'infos': {'s': [{'token': 't', 'pose_mode': [0, 1, 0]}]}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'controls.json').write_text(json.dumps(controls))
            (root / 'official.pkl').write_bytes(pickle.dumps(controls))
            for mode in ('r', 'rb'):
                with (root / 'controls.json').open(mode) as handle:
                    self.assertEqual(load_metadata(handle), controls)
                    self.assertFalse(handle.closed)
            with (root / 'official.pkl').open('rb') as handle:
                with self.assertRaises(ValueError):
                    load_metadata(handle)
                self.assertEqual(handle.tell(), 0)
                self.assertEqual(load_metadata(handle, trusted_pickle=True), controls)
                self.assertFalse(handle.closed)
        with self.assertRaises(ValueError):
            load_metadata(io.BytesIO(b'{}'))

    def test_all_waymo_tokenizer_branches_use_shared_metadata_loader(self):
        path = ROOT / 'EXIST/4D/II-World/mmdet3d/datasets/waymo_occstress_world_dataset.py'
        tree = ast.parse(path.read_text())
        for name in ('OccStressWaymoWorldDataset', 'OccStressWaymoTokenizerDataset',
                     'OccStressWaymoFutureTokenizerDataset'):
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'load_annotations')
            calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'load_metadata']
            self.assertEqual(len(calls), 1, name)

    def test_external_namespace_not_clean_prediction_cache(self):
        with patch.dict(os.environ, {'OCCSTRESS_DATA_ROOT': '/datasets/OccStress'}, clear=True):
            self.assertEqual(release_occ_path('external/OccStress-CARLA/canonical_gt/s/t'),
                             Path('/datasets/OccStress/external/OccStress-CARLA/canonical_gt/s/t/labels.npz'))

    def test_partial_payload_is_not_reported_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = 'OccStress-Waymo'
            (root / 'manifests').mkdir()
            meta = root / 'meta' / name
            meta.mkdir(parents=True)
            mapping = b'{}'
            (meta / 'class_mapping.json').write_bytes(mapping)
            (meta / 'dataset.json').write_text(json.dumps({'dataset': name, 'anchor_count': 5978,
                'class_mapping_sha256': hashlib.sha256(mapping).hexdigest()}))
            protocol = 'protocols/manual/' + name + '/clean/p.pkl'
            (root / protocol).parent.mkdir(parents=True)
            (root / protocol).touch()
            (root / 'manifests/protocols.json').write_text(json.dumps([
                {'dataset': name, 'path': protocol, 'anchors': 5978}]))
            self.assertEqual(validate(root)[0], [])
            self.assertIn(name + ': payload assembly incomplete', validate(root, require_payloads=True)[0])


if __name__ == '__main__':
    unittest.main()
