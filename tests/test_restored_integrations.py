import copy
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from occstress.distribution import withheld

ROOT = Path(__file__).resolve().parents[1]


def backbone():
    def frame(token):
        return dict(token=token, occ_path=f'external/OccStress-Waymo/gts/000/{token}/labels.npz',
                    occ_source='clean')
    return [dict(scene_name='000', anchor_token='a', sample_id='a__clean',
                 history_length=4, future_length=6, history=[frame(f'h{i}') for i in range(4)],
                 current_input=frame('a'), target=frame('a'),
                 future_targets=[frame(f'f{i}') for i in range(6)], command=[0, 0, 1])]


@unittest.skipIf(withheld(ROOT, 'EXIST/3D/CVT-Occ'), 'CVT-Occ omitted in filtered build')
class RestoredCVTTests(unittest.TestCase):
    def test_all_temporal_patterns_have_portable_inputs_and_unchanged_gt(self):
        from scripts.waymo import build_cvtocc_waymo_upstream_protocols as cvt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exports = root / 'occ/upstream/OccStress-Waymo/camera_only/cvtocc'
            reference = backbone()
            saved = copy.deepcopy(reference)
            for pattern, mask in [('clean', [0, 0, 0, 0, 0]),
                                  ('current_only', [0, 0, 0, 0, 1]),
                                  ('recent_burst', [0, 0, 0, 1, 1]),
                                  ('history_only', [1, 1, 1, 1, 0])]:
                corruption = 'clean' if pattern == 'clean' else 'Snow'
                row = cvt.build_protocol(reference, exports, corruption, 'hard', pattern, root)[0]
                frames = [*row['history'], row['current_input']]
                self.assertEqual([int(f['prediction_variant'] != 'clean') for f in frames], mask)
                self.assertTrue(all(f['occ_path'].startswith('occ/upstream/OccStress-Waymo/') for f in frames))
                for key in ('target', 'future_targets', 'command'):
                    self.assertEqual(row[key], reference[0][key])
            self.assertEqual(reference, saved)
            reference[0]['future_targets'][0]['occ_path'] = '/unportable/gt.npz'
            with self.assertRaisesRegex(ValueError, 'portable clean backbone'):
                cvt.build_protocol(reference, exports, 'clean', 'clean', 'clean', root)

    def test_explicit_root_and_native_config_overrides(self):
        from scripts.waymo import build_cvtocc_waymo_upstream_protocols as cvt
        with patch.dict(os.environ, {'OCCSTRESS_DATA_ROOT': '/unused'}):
            with patch.object(sys, 'argv', ['build', '--occstress-root', '/datasets/OccStress']):
                args = cvt.parse_args()
        self.assertEqual(args.occ_root, Path('/datasets/OccStress/occ/upstream/OccStress-Waymo/camera_only/cvtocc'))
        self.assertFalse(args.overwrite)
        environment = {'OCCSTRESS_CODE_ROOT': '/code', 'OCCSTRESS_DATA_ROOT': '/data',
                       'OCCSTRESS_EXTERNAL_ROOT': '/external', 'CVTOCC_FRAME_INDEX_ROOT': '/index'}
        with patch.dict(os.environ, environment, clear=True):
            cfg = runpy.run_path(str(ROOT / 'EXIST/3D/CVT-Occ/projects/configs/occstress/cvtocc_waymo_local_2hz.py'))
        self.assertEqual(cfg['occ_gt_root'], '/external/OccStress-Waymo/gts/')
        self.assertEqual(cfg['frame_index_root'], '/index')
        self.assertTrue((ROOT / 'EXIST/3D/CVT-Occ/projects/configs/occstress' / cfg['_base_'][0]).is_file())


@unittest.skipIf(withheld(ROOT, 'EXIST/3D/FusionOcc'), 'FusionOcc omitted in filtered build')
class RestoredFusionTests(unittest.TestCase):
    def test_shared_builder_receives_namespaced_root_and_backbone(self):
        from scripts.fusionocc import build_occstress_upstream_protocols as fusion
        args = SimpleNamespace(root=ROOT, occstress_root=Path('/datasets/OccStress'),
                               occ_root=None, backbone_protocol=None, subtrack='pointcloud_fusion',
                               source_model='fusionocc', target_root=None, overwrite=False)
        with patch.object(fusion.subprocess, 'run') as run:
            fusion.run_builder(args, 'Snow', 'hard', 'history_k1')
        command = run.call_args.args[0]
        def value(flag):
            return command[command.index(flag) + 1]
        self.assertEqual(value('--dataset'), 'nuscenes')
        self.assertEqual(value('--occstress-root'), '/datasets/OccStress')
        self.assertEqual(value('--backbone-protocol'), '/datasets/OccStress/protocols/manual/OccStress-nuScenes/clean/H4_F6_val_backbone.pkl')
        self.assertEqual(value('--clean-input-root'), '/datasets/OccStress/occ/upstream/OccStress-nuScenes/pointcloud_fusion/fusionocc/clean')
        self.assertNotIn('--target-root', command)
        args.backbone_protocol = Path('/datasets/OccStress/protocols/custom.pkl')
        with patch.object(fusion.subprocess, 'run') as run:
            fusion.run_builder(args, 'clean', 'clean')
        command = run.call_args.args[0]
        self.assertEqual(value('--backbone-protocol'), str(args.backbone_protocol))
