import argparse
import ast
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_formal_configs import config_closure, issues, make_lock
from tools.install_environment import install_plan
from tools.run_formal import command_plan
from tools.adapters.occstress_paths import upstream_protocol_path
from occstress.distribution import withheld


def functions_from(path, names, namespace=None):
    """Exercise real pure functions without loading model/CUDA imports."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('Missing test function')
    scope = dict(namespace or {})
    future = [node for node in tree.body
              if isinstance(node, ast.ImportFrom) and node.module == '__future__']
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class FormalConfigTests(unittest.TestCase):
    def test_dome_determinism_is_explicit_opt_in(self):
        parse = functions_from(ROOT / 'EXIST/4D/DOME/tools/eval_occstress.py',
                               ['parse_args'], {'argparse': argparse, 'ROOT': ROOT})['parse_args']
        arguments = ['eval', '--checkpoint', 'world.pth', '--vae-checkpoint', 'vae.pth',
                     '--protocol', 'p.pkl', '--base-info', 'b.pkl', '--occstress-root', '/data',
                     '--output-json', 'out.json']
        with patch.object(sys, 'argv', arguments):
            self.assertFalse(parse().deterministic)
        with patch.object(sys, 'argv', arguments + ['--deterministic']):
            self.assertTrue(parse().deterministic)

    def test_lock_is_current(self):
        self.assertEqual(issues(ROOT), [])
        self.assertGreater(len(make_lock(ROOT)['files']), 60)

    def test_cycle_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / 'first.py', root / 'second.py'
            first.write_text("_base_ = ['second.py']\n")
            second.write_text("_base_ = 'first.py'\n")
            with self.assertRaisesRegex(ValueError, 'Cyclic'):
                config_closure(first, root)
            first.write_text("_base_ = '../outside.py'\n")
            with self.assertRaises(ValueError):
                config_closure(first, root)

    def test_contract_offsets_and_tracks(self):
        contract = json.loads((ROOT / 'configs/evaluation_contract.json').read_text())
        self.assertEqual(contract['timeline']['future_horizons_seconds'], [.5, 1, 1.5, 2, 2.5, 3])
        self.assertEqual(contract['timeline']['paper_average_horizons_seconds'], [1, 2, 3])
        self.assertEqual(contract['models']['geniedrive']['input_offsets_seconds'], [-1.5, -1, -.5, 0])
        self.assertEqual(contract['models']['sparseworld-tc']['tracks'], ['camera_direct'])
        self.assertEqual(contract['datasets']['waymo']['anchors'], 5978)

    def test_launch_plans_use_declared_configs(self):
        contract = json.loads((ROOT / 'configs/evaluation_contract.json').read_text())
        for name, spec in contract['models'].items():
            if withheld(ROOT, spec['directory']):
                with self.assertRaisesRegex(ValueError, 'Pending upstream permission'):
                    command_plan(name, 'nuscenes', 'python', [], environ={})
                continue
            for dataset, config in spec['configs'].items():
                with self.subTest(model=name, dataset=dataset):
                    directory, command, env = command_plan(name, dataset, '/env/bin/python', [], environ={})
                    self.assertIn(str(directory / config), command)
                    self.assertEqual(command[0], '/env/bin/python')
                    self.assertEqual(env['OCCSTRESS_DATASET'], dataset)
                    self.assertEqual(env['PYTHONNOUSERSITE'], '1')

    def test_native_parser_accepts_injected_flags(self):
        contract = json.loads((ROOT / 'configs/evaluation_contract.json').read_text())
        for model, spec in contract['models'].items():
            if withheld(ROOT, spec['directory']):
                continue
            for dataset, entry in spec['entrypoints'].items():
                if model == 'iiworld' and dataset == 'nuscenes':
                    continue  # Native OpenMMLab test.py uses a positional config.
                tree = ast.parse((ROOT / spec['directory'] / entry).read_text())
                flags = {arg.value for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'add_argument' for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
                command = command_plan(model, dataset, 'python', [], environ={})[1]
                for argument in command[2:]:
                    if argument.startswith('--'):
                        self.assertIn(argument, flags, (model, dataset))

    def test_legacy_and_override_blocked(self):
        for value in ('0', 'false', 'no'):
            with self.assertRaises(ValueError):
                command_plan('come', 'waymo', 'python', [], environ={'OCCSTRESS_COME_FUTURE_ALIGNED': value})
        for flag in ('--config=x.py', '--cfg-options', '--dataset=carla'):
            with self.assertRaises(ValueError):
                command_plan('dome', 'waymo', 'python', [flag], environ={})

    def test_environment_and_tokenizer_isolation(self):
        _, command, env = command_plan('iiworld', 'carla', 'python', [], environ={
            'PYTHONPATH': '/other-model', 'OCCSTRESS_WAYMO_PROTOCOL': '/wrong.pkl',
            'OCCSTRESS_DATA_ROOT': '/mounted/OccStress'})
        self.assertNotIn('/other-model', env['PYTHONPATH'])
        self.assertNotIn('OCCSTRESS_WAYMO_PROTOCOL', env)
        self.assertEqual(env['OCCSTRESS_DATA_ROOT'], '/mounted/OccStress')
        self.assertTrue(command[command.index('--tokenizer-config') + 1].endswith('ii_scene_tokenizer_carla_occstress.py'))
        with self.assertRaises(ValueError):
            command_plan('iiworld', 'waymo', 'python', [], stage='tokenizer', environ={})


class UpstreamContractTests(unittest.TestCase):
    def test_exporter_allows_release_archive_without_git(self):
        funcs = functions_from(ROOT / 'scripts/waymo/export_effocc_waymo_upstream.py',
                               ['git_commit', 'git_is_dirty'], {'subprocess': subprocess, 'Path': Path, 'os': os})
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(funcs['git_commit'](Path(temp)))
            self.assertIsNone(funcs['git_is_dirty'](Path(temp)))

    def setUp(self):
        masks = functions_from(ROOT / 'occstress/protocols/temporal.py',
                               ['history_mask_for_protocol', 'target_active_for_protocol'])
        self.builder = runpy.run_path(str(ROOT / 'scripts/build_upstream_occstress_protocol.py'))
        self.masks = masks

    def test_pattern_names_match_real_generator(self):
        contract = json.loads((ROOT / 'configs/evaluation_contract.json').read_text())
        for spec in contract['temporal_protocols'].values():
            alias = spec['legacy_filename']
            active = self.masks['history_mask_for_protocol'](4, alias, 1)
            active.append(self.masks['target_active_for_protocol'](alias))
            self.assertEqual([idx - 4 for idx, flag in enumerate(active) if flag], spec['slots'])

    def test_dataset_specific_paths_and_summary_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(output_occstress_root_resolved=root / 'OccStress', dataset='carla', subtrack='camera_only',
                                      source_model='flashocc_carla', corruption='Snow', severity='hard',
                                      frame_protocol='current')
            output = self.builder['protocol_output_path'](args, Path('H4_F6_val_backbone.pkl'), root)
            self.assertEqual(output, root / 'OccStress/protocols/upstream/OccStress-CARLA/camera_only/flashocc_carla/Snow/hard/current_H4_F6_val_backbone.pkl')
            summary = self.builder['summary_output_path'](args, output, root)
            args.corruption = 'Fog'
            self.assertNotEqual(summary, self.builder['summary_output_path'](args, output, root))
            args.corruption = 'clean'
            self.assertEqual(self.builder['protocol_output_path'](args, Path('H4_F6_val_backbone.pkl'), root).parent.name, 'clean')

    def test_record_preserves_targets_and_temporal_placement(self):
        record = dict(sample_id='anchor', anchor_token='a', scene_name='s', dataset='UniOcc-CARLA',
                      history_length=4, future_length=6,
                      history=[dict(token=f'h{i}') for i in range(4)],
                      current_input=dict(token='a'), target=dict(token='a', occ_path='gt/a'),
                      future_targets=[dict(token=f'f{i}', occ_path=f'gt/f{i}') for i in range(6)])
        original = copy.deepcopy(record)
        with tempfile.TemporaryDirectory() as tmp:
            clean, corrupt = Path(tmp) / 'clean', Path(tmp) / 'corrupt'
            for root in (clean, corrupt):
                for token in ('h0', 'h1', 'h2', 'h3', 'a'):
                    file = root / 's' / token / 'labels.npz'
                    file.parent.mkdir(parents=True)
                    file.touch()
            for mode, expected in [('current', [0, 0, 0, 0, 1]),
                                   ('history_k1', [0, 0, 0, 1, 1]),
                                   ('all_frame', [1, 1, 1, 1, 0])]:
                args = argparse.Namespace(subtrack='camera_only', source_model='test',
                                          corruption='Snow', severity='hard', frame_protocol=mode,
                                          history_k=1, target_root_resolved=None)
                missing = []
                out = self.builder['build_record'](record, args, clean, corrupt, missing)
                self.assertEqual(missing, [])
                refs = out['history'] + [out['current_input']]
                self.assertEqual([int(ref['prediction_variant'] != 'clean') for ref in refs], expected)
                self.assertEqual(out['future_targets'], original['future_targets'])
                self.assertEqual(out['dataset'], 'UniOcc-CARLA')
            self.assertEqual(record, original)

    def test_fusion_camera_is_not_camera_only(self):
        specs = json.loads((ROOT / 'configs/upstream_methods.json').read_text())['sources']
        spec = next(item for item in specs if item['id'] == 'effocc-waymo-camera-stress')
        self.assertIn('lidar', spec['sensor_inputs'])
        self.assertEqual(spec['corrupted_sensor'], 'camera')
        path = upstream_protocol_path('camera_fusion', 'effocc', 'clean', dataset='waymo')
        self.assertIn('camera_fusion', path.parts)

    def test_iiworld_asset_remount_is_dataset_specific(self):
        source = ROOT / 'EXIST/4D/II-World/mmdet3d/datasets/waymo_occstress_world_dataset.py'
        functions = functions_from(source, ['_occ_npz', '_frame_occ_path'], {'os': os, 'Path': Path})
        frame = dict(token='frame', occ_source='corrupted', occ_path='occ/manual/OccStress-CARLA/semantic/hard/frame/labels.npz')
        with patch.dict(os.environ, {'OCCSTRESS_DATASET': 'carla',
                                     'OCCSTRESS_DATA_ROOT': '/mounted/OccStress'}, clear=True):
            self.assertEqual(functions['_frame_occ_path'](frame, '000'),
                             '/mounted/OccStress/occ/manual/OccStress-CARLA/semantic/hard/frame/labels.npz')

    def test_iiworld_raw_label_mapping_matches_contract(self):
        source = ROOT / 'EXIST/4D/II-World/mmdet3d/datasets/waymo_occstress_world_dataset.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_load_semantics')
        assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'mapping' for t in node.targets))
        actual = {str(key): value for key, value in ast.literal_eval(assignment.value).items()}
        contract = json.loads((ROOT / 'configs/evaluation_contract.json').read_text())
        self.assertEqual(actual, contract['datasets']['waymo']['raw_to_occ3d'])


class EnvironmentPlanTests(unittest.TestCase):
    def test_explicit_host_compiler_and_clean_flags(self):
        plan = install_plan('dome', Path('/new/env'), Path('/cuda'), '8.0',
                            cc='/usr/bin/gcc', cxx='/usr/bin/g++', clean_build_flags=True)
        self.assertEqual(plan['env']['CC'], '/usr/bin/gcc')
        self.assertEqual(plan['env']['CXX'], '/usr/bin/g++')
        self.assertEqual(plan['env']['CFLAGS'], '')
        self.assertEqual(plan['env']['LDFLAGS'], '')

    def test_all_automatic_plans_have_existing_build_sources(self):
        profiles = json.loads((ROOT / 'environments/profiles.json').read_text())
        for name, spec in profiles['models'].items():
            if withheld(ROOT, spec['directory']):
                with self.assertRaisesRegex(ValueError, 'Pending upstream permission'):
                    install_plan(name, Path('/new/env'), Path('/cuda'), '8.0')
                continue
            if spec.get('manual_install_only'):
                continue
            plan = install_plan(name, Path('/new/env'), Path('/cuda'), '8.0')
            self.assertFalse(plan['gpu_reproduction_verified'])
            self.assertTrue(plan['commands'][0]['command'][1] == 'create')
            self.assertIn('--override-channels', plan['commands'][0]['command'])
            self.assertIn('conda-forge', plan['commands'][0]['command'])
            for build in spec['builds']:
                self.assertTrue((ROOT / spec['directory'] / build / 'setup.py').is_file(), name)
            for step in plan['commands']:
                self.assertNotIn('-e', step['command'])

    @unittest.skipIf(withheld(ROOT, 'EXIST/3D/CVT-Occ'), 'CVT-Occ source withheld pending permission')
    def test_cvt_dependency_is_bundled_runtime_base(self):
        profiles = json.loads((ROOT / 'environments/profiles.json').read_text())
        spec = profiles['models']['cvtocc']
        dependency = ROOT / spec['dependency_directory']
        self.assertTrue((dependency / 'mmdet3d/version.py').is_file())
        self.assertTrue((dependency / 'LICENSE').is_file())
        self.assertIn('CVT-Occ/dependencies', str(dependency))
        self.assertTrue(spec['dependency_path_only'])

    def test_sdg_legacy_exception_is_explicit(self):
        reason = ('Pending upstream permission' if withheld(ROOT, 'EXIST/3D/SDGOCC')
                  else 'native installation')
        with self.assertRaisesRegex(ValueError, reason):
            install_plan('sdgocc', Path('/new/env'), Path('/cuda'), '8.0')


if __name__ == '__main__':
    unittest.main()
