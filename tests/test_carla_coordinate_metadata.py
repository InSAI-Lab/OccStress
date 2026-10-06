import copy
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CarlaCoordinateMetadataTests(unittest.TestCase):
    def test_path_rewrite_preserves_namespaced_relative_layout(self):
        rewrite = runpy.run_path(str(ROOT / 'scripts/carla/prepare_carla_model_view.py'))['rewrite_occ_path']
        with TemporaryDirectory() as temp:
            source, target = Path(temp) / 'source/OccStress', Path(temp) / 'target/OccStress'
            relative = Path('occ/manual/OccStress-CARLA/traffic/scene/token/labels.npz')
            expected = str(target / relative)
            for path in (str(relative), str(source / relative), str(Path(temp) / 'previous/OccStress' / relative)):
                with self.subTest(path=path):
                    self.assertEqual(rewrite(path, source, target), expected)
            with self.assertRaises(ValueError):
                rewrite(str(Path(temp) / 'other/data/labels.npz'), source, target)
            with self.assertRaises(ValueError):
                rewrite('../other/labels.npz', source, target)

    def test_traffic_diagnostic_controls_follow_coordinate_conversion(self):
        module = runpy.run_path(str(ROOT / 'scripts/carla/prepare_carla_model_view.py'))
        with TemporaryDirectory() as temp:
            source, target = Path(temp) / 'source', Path(temp) / 'target'
            frame = {'token': 't', 'occ_path': str(source / 'occ/traffic/s/t/labels.npz')}
            record = {'history': [copy.deepcopy(frame)], 'future_targets': [copy.deepcopy(frame)],
                      'current_input': copy.deepcopy(frame), 'target': copy.deepcopy(frame),
                      'traffic_mirror': True, 'traffic_transform': {
                          'current_gt_ego_fut_trajs': [[1., -2.]] * 6,
                          'current_gt_ego_fut_cmd': [1., 0., 0.],
                          'current_gt_ego_fut_masks': [1.] * 6,
                          'command_permutation': [1, 0, 2]}}
            original = copy.deepcopy(record)
            result = module['convert_record'](record, source, target)
            np.testing.assert_array_equal(result['traffic_transform']['current_gt_ego_fut_trajs'], [[1., 2.]] * 6)
            self.assertEqual(result['traffic_transform']['current_gt_ego_fut_cmd'], [0., 1., 0.])
            self.assertEqual(result['traffic_transform']['command_permutation'], [1, 0, 2])
            self.assertEqual(result['traffic_transform']['current_gt_ego_fut_masks'], [1.] * 6)
            self.assertEqual(record, original)
