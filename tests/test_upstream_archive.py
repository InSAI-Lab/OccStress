import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
from occstress.distribution import withheld


class UpstreamArchiveTest(unittest.TestCase):
    def test_stcocc_absolute_and_relative_corruption_roots(self):
        path = ROOT / 'EXIST/3D/STCOcc/mmdet3d/datasets/nuscenes_dataset.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_update_cam_data_path')
        namespace = {'osp': os.path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        for root, expected in (
            ('/corrupt/Brightness', '/corrupt/Brightness/hard/CAM_FRONT/frame.jpg'),
            ('Brightness', '/raw/nuscenes/Brightness/hard/CAM_FRONT/frame.jpg'),
            (None, '/raw/nuscenes/samples/CAM_FRONT/frame.jpg'),
        ):
            with self.subTest(root=root):
                info = {'cams': {'CAM_FRONT': {'data_path': '/raw/nuscenes/samples/CAM_FRONT/frame.jpg'}}}
                obj = SimpleNamespace(use_corner_case_data=root, corner_case_degree='hard')
                result = namespace['_update_cam_data_path'](obj, info)
                self.assertIs(result, info)
                self.assertEqual(result['cams']['CAM_FRONT']['data_path'], expected)

    def test_alocc_absolute_corruption_root_does_not_duplicate_clean_root(self):
        path = ROOT / 'EXIST/3D/ALOcc/mmdet3d/datasets/nuscenes_dataset.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == 'update_data_path')
        namespace = {'osp': os.path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        camera = {'data_path': '/raw/nuscenes/samples/CAM_FRONT/frame.jpg'}
        obj = SimpleNamespace(data_infos=[{'cams': {'CAM_FRONT': camera}}],
                              use_corner_case_data='/corrupt/Brightness', corner_case_degree='hard')
        namespace['update_data_path'](obj)
        self.assertEqual(camera['data_path'], '/corrupt/Brightness/hard/CAM_FRONT/frame.jpg')

    def test_effocc_git_metadata_is_optional_in_source_archive(self):
        path = ROOT / 'scripts/waymo/export_effocc_waymo_upstream.py'
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('git_commit', 'git_is_dirty')]
        namespace = dict(os=os, Path=Path, subprocess=subprocess)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(namespace['git_commit'](Path(tmp)))
            self.assertIsNone(namespace['git_is_dirty'](Path(tmp)))
        with patch.dict(os.environ, {'EFFOCC_CODE_COMMIT': 'abc', 'EFFOCC_CODE_DIRTY': '0'}):
            self.assertEqual(namespace['git_commit'](ROOT), 'abc')
            self.assertIs(namespace['git_is_dirty'](ROOT), False)

    def test_effocc_smoke_named_sdk_modules_are_not_test_files(self):
        root = ROOT / 'EXIST/3D/EFFOcc/mmdet3d'
        for relative in ('core/bbox/coders/smoke_bbox_coder.py',
                         'models/dense_heads/smoke_mono3d_head.py',
                         'models/detectors/smoke_mono3d.py'):
            with self.subTest(module=relative):
                self.assertTrue((root / relative).is_file())

    @unittest.skipIf(withheld(ROOT, 'EXIST/4D/SparseWorld'), 'SparseWorld source withheld pending permission')
    def test_sparseworld_direct_protocol_does_not_hash_unshipped_files(self):
        path = ROOT / 'EXIST/4D/SparseWorld/tools/occstress_eval.py'
        tree = ast.parse(path.read_text())
        body = next(node.body for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
        start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'adapter_paths' for t in node.targets))
        stop = next(i for i in range(start + 1, len(body)) if isinstance(body[i], ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'result_payload' for t in body[i].targets))
        fragment = ast.Module(body=body[start:stop], type_ignores=[])
        root = ROOT / 'EXIST/4D/SparseWorld'
        for index in (None, 0):
            namespace = dict(Path=Path, REPO_ROOT=root,
                             args=SimpleNamespace(manifest='optional.json', protocol_index=index))
            exec(compile(fragment, str(path), 'exec'), namespace)
            paths = namespace['adapter_paths']
            self.assertEqual(len(paths), 3 if index is None else 4)
            self.assertTrue(all(p.is_file() for p in paths[:3]))
            self.assertFalse(any(p.suffix == '.slurm' for p in paths))
            if index is not None:
                self.assertEqual(paths[-1], root / 'optional.json')
