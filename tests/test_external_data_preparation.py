import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import prepare_external_data as prep


class ExternalPreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.shared = self.base / 'OccStress'
        self.external = self.base / 'external'
        self.raw = self.base / 'raw'
        self.raw.mkdir()

    def fixture(self):
        info = self.raw / 'scene_infos.pkl'
        info.write_bytes(b'not unpickled by this tool')
        meta = self.shared / 'meta/OccStress-CARLA'
        meta.mkdir(parents=True)
        (meta / 'dataset.json').write_text(json.dumps({'source_scene_info_sha256': hashlib.sha256(info.read_bytes()).hexdigest()}))
        frames = []
        self.native = np.full((200, 200, 16), 10, dtype=np.uint8)
        self.native[3, 1, 2] = 1
        for index in range(2):
            token = f'carla-val-Town05-{index:03d}'
            scene = 'scene_Town05'
            path = self.raw / scene / f'{index}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, occ_label=self.native, occ_mask_camera=self.native == 1,
                                unused_object_metadata=np.array([{'do_not_decode': True}], dtype=object))
            frames.append(dict(scene_name=scene, token=token, frame_idx=index,
                               occ_path=f'external/OccStress-CARLA/canonical_gt/{scene}/{token}/labels.npz'))
        controls = {'infos': {'scene_Town05': frames},
                    'metadata': {'model_view': True, 'coordinate_convention': 'Occ3D right-handed ego-local x-forward/y-left/z-up'}}
        control_path = meta / 'controls.json'
        control_path.write_text(json.dumps(controls))
        return control_path

    def convert(self, **kwargs):
        with patch.object(prep, 'EXPECTED_SCENES', {'scene_Town05': 2}):
            return prep.carla_gt(self.shared, self.external, self.raw, **kwargs)

    def test_conversion_and_resume_preserve_controls_and_raw(self):
        controls = self.fixture()
        before = controls.read_bytes()
        raw = (self.raw / 'scene_Town05/0.npz').read_bytes()
        report = self.convert()
        self.assertEqual((report['status'], report['created']), ('complete', 2))
        target = self.external / 'OccStress-CARLA/canonical_gt/scene_Town05/carla-val-Town05-000/labels.npz'
        with np.load(target) as result:
            self.assertEqual(result['semantics'][3, 198, 2], 4)
            self.assertEqual(result['semantics'][3, 1, 2], 17)
            self.assertTrue(result['infov'][3, 198, 2])
        self.assertEqual(self.convert()['reused'], 2)
        self.assertEqual(controls.read_bytes(), before)
        self.assertEqual((self.raw / 'scene_Town05/0.npz').read_bytes(), raw)

    def test_partial_conversion_does_not_write_complete_receipt(self):
        self.fixture()
        self.assertEqual(self.convert(max_frames=1)['status'], 'partial')
        self.assertFalse((self.external / 'OccStress-CARLA/canonical_gt/.prepared.json').exists())

    def test_mismatching_existing_output_is_not_overwritten(self):
        self.fixture()
        self.convert()
        target = next(self.external.rglob('labels.npz'))
        np.savez_compressed(target, semantics=np.zeros((1,)), infov=np.zeros((1,)))
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Existing GT differs'):
            self.convert()
        self.assertEqual(target.read_bytes(), before)

    def test_wrong_source_revision_fails_before_writes(self):
        self.fixture()
        (self.raw / 'scene_infos.pkl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source revision'):
            self.convert()
        self.assertFalse(self.external.exists())

    def test_wrong_grid_and_unknown_label_rejected(self):
        self.fixture()
        path = self.raw / 'scene_Town05/0.npz'
        np.savez_compressed(path, occ_label=np.full((200, 200, 16), 255), occ_mask_camera=np.zeros((200, 200, 16)))
        with self.assertRaisesRegex(ValueError, 'unmapped'):
            self.convert()
        np.savez_compressed(path, occ_label=np.zeros((2, 2, 1), dtype=np.uint8), occ_mask_camera=np.zeros((2, 2, 1)))
        with self.assertRaisesRegex(ValueError, 'grid'):
            self.convert()

    def test_mount_is_idempotent_and_never_replaces_other_source(self):
        report = prep.mount(self.external, 'nuscenes', 'gts', self.raw)
        self.assertEqual(report['status'], 'mounted')
        self.assertEqual(prep.mount(self.external, 'nuscenes', 'gts', self.raw)['status'], 'already_mounted')
        other = self.base / 'other'
        other.mkdir()
        with self.assertRaises(FileExistsError):
            prep.mount(self.external, 'nuscenes', 'gts', other)
        with self.assertRaises(ValueError):
            prep.mount(self.external, 'nuscenes', '../bad', self.raw)

    def test_waymo_mount_requires_voxel04_parent(self):
        with self.assertRaisesRegex(ValueError, 'validation-data'):
            prep.mount(self.external, 'waymo', 'native_gt', self.raw)
        (self.raw / 'validation-data').mkdir()
        self.assertEqual(prep.mount(self.external, 'waymo', 'native_gt', self.raw)['status'], 'mounted')

    def test_conversion_refuses_external_symlink_destination(self):
        self.fixture()
        other = self.base / 'outside'
        other.mkdir()
        parent = self.external / 'OccStress-CARLA'
        parent.mkdir(parents=True)
        (parent / 'canonical_gt').symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.convert()
        self.assertEqual(list(other.iterdir()), [])

    def test_check_requires_trust_and_all_gt_references(self):
        self.fixture()
        self.convert()
        value = 'external/OccStress-CARLA/canonical_gt/scene_Town05/carla-val-Town05-000/labels.npz'
        def frame(token):
            return {'token': token, 'occ_path': value}
        row = dict(scene_name='scene_Town05', anchor_token='a', history=[frame(f'h{i}') for i in range(4)],
                   current_input=frame('a'), target=frame('a'), future_targets=[frame(f'f{i}') for i in range(6)])
        backbone = self.shared / 'protocols/manual/OccStress-CARLA/clean/H4_F6_val_backbone.pkl'
        backbone.parent.mkdir(parents=True)
        backbone.write_bytes(pickle.dumps([row]))
        with patch.dict(prep.ANCHORS, {'carla': 1}):
            with self.assertRaises(ValueError):
                prep.check(self.shared, self.external, 'carla')
            self.assertEqual(prep.check(self.shared, self.external, 'carla', True)['status'], 'passed')
            next(self.external.rglob('labels.npz')).unlink()
            with self.assertRaises(FileNotFoundError):
                prep.check(self.shared, self.external, 'carla', True)
