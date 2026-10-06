import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.adapters.occstress_paths import data_root, repo_root, resolve_occstress_path
from tools.release_status import inspect, resource_issues
from tools.check_release import scan
from scripts.occstress_layout import (protocol_key_from_path, resolve_manual_protocol_path,
                               resolve_upstream_protocol_path)


class PathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_root_does_not_depend_on_cwd(self):
        self.assertEqual(repo_root(), ROOT)

    def test_explicit_root_beats_environment(self):
        with patch.dict(os.environ, {'OCCSTRESS_CODE_ROOT': '/wrong'}):
            self.assertEqual(repo_root(self.root), self.root)

    def test_checkout_prefix_and_namespaced_assets(self):
        for prefix in ('', 'data/OccStress/'):
            path = 'occ/manual/OccStress-nuScenes/semantic/hard/s/t/labels.npz'
            result = resolve_occstress_path(prefix + path, code_root=self.root)
            self.assertEqual(result, self.root / 'data/OccStress' / path)

    def test_external_mount_wins_over_stale_checkout(self):
        stale = self.root / 'data/OccStress/occ/a.npz'
        stale.parent.mkdir(parents=True)
        stale.touch()
        with patch.dict(os.environ, {'OCCSTRESS_DATA_ROOT': str(self.root / 'external')}):
            self.assertEqual(resolve_occstress_path('data/OccStress/occ/a.npz', code_root=self.root),
                             self.root / 'external/occ/a.npz')

    def test_shared_root_keeps_dataset_namespaces_separate(self):
        with patch.dict(os.environ, {'OCCSTRESS_DATA_ROOT': str(self.root / 'shared')}):
            self.assertEqual(data_root(dataset='Waymo', code_root=self.root), self.root / 'shared')
        with self.assertRaises(ValueError):
            resolve_occstress_path('occ/manual/OccStress-Waymo/a.npz', dataset='nuscenes')
        with self.assertRaises(ValueError):
            resolve_occstress_path('/datasets/OccStress/occ/manual/OccStress-Waymo/a.npz', dataset='nuscenes')

    def test_relative_assets_use_explicit_mount(self):
        for prefix in ('occ/manual/OccStress-Waymo', 'protocols/manual/OccStress-CARLA', 'meta/OccStress-nuScenes'):
            self.assertEqual(resolve_occstress_path(f'{prefix}/a', occstress_root=self.root), self.root / prefix / 'a')

    def test_external_gt_can_be_mounted_separately(self):
        with patch.dict(os.environ, {'OCCSTRESS_EXTERNAL_ROOT': str(self.root / 'private')}):
            result = resolve_occstress_path('external/OccStress-Waymo/native_gt/validation-data/001/a.npz')
            self.assertEqual(result, self.root / 'private/OccStress-Waymo/native_gt/validation-data/001/a.npz')

    def test_missing_namespace_and_traversal_fail(self):
        for path in ('occ/manual/semantic/a.npz', 'external/unknown/a.npz', '../a.npz'):
            with self.assertRaises(ValueError):
                resolve_occstress_path(path, occstress_root=self.root)

    def test_protocol_roundtrip_all_datasets(self):
        for dataset in ('nuScenes', 'Waymo', 'CARLA'):
            manual = resolve_manual_protocol_path('semantic_hard_current_H4_F6_val_backbone',
                                                  root=self.root, dataset=dataset)
            self.assertEqual(protocol_key_from_path(manual, root=self.root, dataset=dataset),
                             'semantic_hard_current_H4_F6_val_backbone')
            upstream = resolve_upstream_protocol_path('camera_only', 'source', 'Snow', 'hard',
                                                       'current_H4_F6_val_backbone', root=self.root, dataset=dataset)
            self.assertIn('upstream__camera_only__source__Snow__hard',
                          protocol_key_from_path(upstream, root=self.root, dataset=dataset))

    def test_missing_protocol_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_manual_protocol_path('clean_H4_F6_val', root=self.root, must_exist=True)

    def test_backbone_fallback(self):
        path = resolve_manual_protocol_path('clean_H4_F6_val_backbone', root=self.root)
        path.parent.mkdir(parents=True)
        path.touch()
        self.assertEqual(resolve_manual_protocol_path('clean_H4_F6_val', root=self.root, must_exist=True), path)


class RegistryTests(unittest.TestCase):
    def test_release_check_detects_syntax_error(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'bad.py').write_text('for key in []\n    pass\n')
            self.assertTrue(any('Python syntax' in item for item in scan(Path(directory), syntax=True)))

    def test_release_check_does_not_follow_external_link(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'external').symlink_to('/tmp', target_is_directory=True)
            self.assertTrue(any('symlink' in item for item in scan(Path(directory))))

    def test_declared_entrypoints_exist(self):
        report = inspect(ROOT)
        self.assertEqual(report['issues'], [])
        self.assertFalse(report['gpu_reproduction_verified'])
        self.assertTrue(report['pending_resources'])

    def test_pending_resources_are_explicit(self):
        manifest = json.loads((ROOT / 'configs/resources.json').read_text())
        self.assertEqual(resource_issues(manifest), [])
        item = next(row for row in manifest['checkpoints'] if row['status'] == 'pending')
        item['status'] = 'available'
        self.assertTrue(resource_issues(manifest))
        item.update(url='https://drive.google.com/file/d/example/view', sha256='a' * 64,
                    redistribution_review='cleared')
        self.assertEqual(resource_issues(manifest), [])

    def test_official_reference_is_not_a_verified_download(self):
        manifest = json.loads((ROOT / 'configs/resources.json').read_text())
        item = next(row for row in manifest['checkpoints'] if row['status'] == 'external_reference')
        self.assertEqual(resource_issues(manifest), [])
        item.update(status='available', sha256='a' * 64)
        self.assertTrue(resource_issues(manifest))
        item['identity_status'] = 'exact_benchmark_file_verified'
        self.assertEqual(resource_issues(manifest), [])
        item['redistribution_review'] = 'cleared'
        self.assertTrue(resource_issues(manifest))

    def test_repository_address_does_not_publish_pending_datasets(self):
        manifest = json.loads((ROOT / 'configs/resources.json').read_text())
        self.assertEqual(manifest['dataset_repository_url'], 'https://huggingface.co/datasets/insailab/OccStress')
        self.assertEqual(resource_issues(manifest), [])
        for row in manifest['datasets']:
            self.assertTrue(row['url'].endswith('/packages.json'))
            if row['status'] == 'publishing':
                self.assertIsNone(row['sha256'])
            else:
                self.assertEqual(row['status'], 'available')
                self.assertRegex(row['sha256'], '^[0-9a-f]{64}$')
                self.assertRegex(row['revision'], '^[0-9a-f]{40}$')
                self.assertIn('/resolve/' + row['revision'] + '/', row['url'])
        pending = manifest['datasets'][0]
        pending.update(status='publishing', url=manifest['dataset_package_index_url'], sha256=None)
        self.assertEqual(resource_issues(manifest), [])
        pending['status'] = 'available'
        self.assertTrue(resource_issues(manifest))
        pending['status'] = 'publishing'
        manifest['dataset_repository_url'] = 'https://example.com/datasets/insailab/OccStress'
        self.assertTrue(resource_issues(manifest))

    def test_available_dataset_rejects_mutable_index_even_with_checksum(self):
        manifest = json.loads((ROOT / 'configs/resources.json').read_text())
        row = manifest['datasets'][0]
        row.update(status='available', sha256='a' * 64, revision='b' * 40,
                   url='https://huggingface.co/datasets/insailab/OccStress/resolve/main/packages.json')
        self.assertTrue(resource_issues(manifest))
        row['url'] = row['url'].replace('/main/', '/' + row['revision'] + '/')
        self.assertEqual(resource_issues(manifest), [])

    def test_resource_cannot_escape_checkout(self):
        manifest = json.loads((ROOT / 'configs/resources.json').read_text())
        manifest['datasets'][0]['local_path'] = '../outside'
        self.assertTrue(resource_issues(manifest))

    def test_sparseworld_is_camera_only(self):
        methods = json.loads((ROOT / 'configs/methods.json').read_text())['methods']
        item = next(m for m in methods if m['id'] == 'sparseworld-tc')
        self.assertEqual(item['tracks'], ['camera_direct'])


class TemporalTests(unittest.TestCase):
    def test_iiworld_config_respects_protocol_environment(self):
        method_root = ROOT / 'EXIST/4D/II-World'
        with patch.dict(os.environ, {'OCCSTRESS_CODE_ROOT': str(ROOT),
                                     'PROTOCOL_PATH': '/synthetic/protocol.pkl',
                                     'OCCSTRESS_IIWORLD_TOKEN_ROOT': '/synthetic/tokens'}, clear=True):
            sys.path.insert(0, str(method_root))
            try:
                result = runpy.run_path(str(method_root / 'configs/world_model/ii_generate_world_occstress.py'))
                self.assertEqual(result['protocol_path'], '/synthetic/protocol.pkl')
                self.assertEqual(result['occstress_token_root'], '/synthetic/tokens')
                for module in result['custom_imports']['imports']:
                    self.assertTrue((method_root / (module.replace('.', '/') + '.py')).is_file())
            finally:
                sys.path.remove(str(method_root))

    def test_configs_default_to_actual_future(self):
        for name, config, current in [('OccWorld', 'config/occworld_occstress.py', 5),
                                      ('COME', 'configs/local_eval_controlnet_occstress.py', 4)]:
            with patch.dict(os.environ, {'OCCSTRESS_CODE_ROOT': str(ROOT)}, clear=True):
                values = runpy.run_path(str(ROOT / 'EXIST/4D' / name / config))
            self.assertTrue(values['future_aligned'])
            self.assertEqual(values['mid_frame'], current)
            self.assertEqual(values['end_frame'] - values['mid_frame'], 6)
            self.assertEqual(values['val_dataset_config']['return_len'], current + 6)

    def test_actual_adapter_sequence_no_duplicate_current(self):
        # Exercise the real sequence-builder AST without importing CUDA frameworks.
        record = dict(sample_id='synthetic', scene_name='scene',
                      history=[dict(token=f'h{i}', occ_path=f'h{i}') for i in range(4)],
                      current_input=dict(token='current', occ_path='current_corrupted'),
                      target=dict(token='current', occ_path='current_clean'),
                      future_targets=[dict(token=f'f{i}', occ_path=f'f{i}') for i in range(1, 7)])
        for name, observed in [('OccWorld', 5), ('COME', 4)]:
            path = ROOT / 'EXIST/4D' / name / 'dataset/dataset.py'
            tree = ast.parse(path.read_text())
            methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                       and node.name == '_build_sequence']
            self.assertEqual(len(methods), 1)
            namespace = {}
            exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
            adapter = types.SimpleNamespace(future_aligned=True, return_len=observed + 6,
                                             offset=0, _clean_occ_path=lambda scene, token: token)
            result = namespace['_build_sequence'](adapter, record)
            self.assertEqual([row['token'] for row in result[observed:]], [f'f{i}' for i in range(1, 7)])
            self.assertEqual(sum(row['token'] == 'current' for row in result), 1)

    def test_independent_seed_preserves_legacy_default(self):
        for name in ('semantic_noise', 'dropout', 'hole'):
            path = ROOT / 'occstress/corruptions' / ({'semantic_noise': 'semantic'}.get(name, name) + '.py')
            tree = ast.parse(path.read_text())
            method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'stable_seed')
            namespace = {'hashlib': hashlib, 'REALIZATION_SEED': None}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
            original = namespace['stable_seed']('test', 'hard', 'token')
            self.assertEqual(original, namespace['stable_seed']('test', 'hard', 'token'))
            namespace['REALIZATION_SEED'] = 3
            first = namespace['stable_seed']('test', 'hard', 'token')
            self.assertNotEqual(original, first)
            self.assertEqual(first, namespace['stable_seed']('test', 'hard', 'token'))


if __name__ == '__main__':
    unittest.main()
