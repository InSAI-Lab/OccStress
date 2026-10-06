import contextlib
import copy
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from occstress.datasets.paths import dataset_name, resolve_occstress_path
from occstress.protocols.validation import load_records
from scripts import build_upstream_occstress_protocol as shared
from scripts.alocc import build_nuscenes_occstress_protocols as alocc
from scripts.carla import build_carla_upstream_protocols as carla

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = 'H4_F6_val_backbone'


def fixture(root, dataset, sources, external=None):
    name = dataset_name(dataset)
    records = []
    for index in range(2):
        scene, anchor = 'scene' + str(index), 'a' + str(index)
        gt_kind = 'canonical_gt' if dataset == 'carla' else 'gts'

        def frame(token):
            relative = Path('external') / name / gt_kind / scene / token / 'labels.npz'
            path = (external / Path(*relative.parts[1:])) if external else root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, semantics=np.zeros((2, 2, 1), dtype=np.uint8))
            return dict(token=token, occ_path=relative.as_posix(), occ_source='clean')

        records.append(dict(dataset=name, scene_name=scene, anchor_token=anchor,
                            sample_id=anchor + '__clean', history_length=4, future_length=6,
                            history=[frame('h' + str(i)) for i in range(4)],
                            current_input=frame(anchor), target=frame(anchor),
                            future_targets=[frame('f' + str(i)) for i in range(6)],
                            corruption={'type': 'clean'}, pose=[[1, 0], [0, 1]],
                            command=[0, 0, 1], future_trajectory=[[i, i / 2] for i in range(6)]))
    backbone = root / 'protocols/manual' / name / 'clean' / (BACKBONE + '.pkl')
    backbone.parent.mkdir(parents=True)
    backbone.write_bytes(pickle.dumps(records))
    for subtrack, source, corruptions, severities in sources:
        base = root / 'occ/upstream' / name / subtrack / source
        for corruption, severity in [('clean', None), *[(c, s) for c in corruptions for s in severities]]:
            setting = base / corruption / severity if severity else base / corruption
            for row in records:
                for entry in [*row['history'], row['current_input']]:
                    path = setting / row['scene_name'] / entry['token'] / 'labels.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, semantics=np.full((2, 2, 1), int(corruption != 'clean'), dtype=np.uint8))
            (setting / '.done.json').write_text(json.dumps({'status': 'payload_validated', 'files': 22}))
    return records, backbone


class UpstreamReleaseBuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'OccStress'
        environment = dict(os.environ, OCCSTRESS_DATA_ROOT=str(self.root), PYTHONDONTWRITEBYTECODE='1')
        environment.pop('OCCSTRESS_EXTERNAL_ROOT', None)
        self.env = patch.dict(os.environ, environment, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def shared_command(self, dataset='nuscenes', source='alocc', corruption='clean', mode='current'):
        export = self.root / 'occ/upstream' / dataset_name(dataset) / 'camera_only' / source
        args = ['build', '--dataset', dataset, '--occstress-root', str(self.root),
                '--subtrack', 'camera_only', '--source-model', source,
                '--corruption', corruption, '--clean-input-root', str(export / 'clean')]
        if corruption != 'clean':
            args += ['--severity', 'hard', '--frame-protocol', mode,
                     '--corrupted-input-root', str(export / corruption / 'hard')]
        return args

    def call_shared(self, args):
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
            shared.main()

    def quiet_children(self):
        run = subprocess.run

        def quiet(*args, **kwargs):
            return run(*args, **dict(kwargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        return patch.object(subprocess, 'run', side_effect=quiet)

    def assert_rows(self, path, reference, external=None):
        rows = load_records(path, trusted_pickle=True)
        for row, original in zip(rows, reference):
            for key in ('target', 'future_targets', 'pose', 'command', 'future_trajectory'):
                self.assertEqual(row[key], original[key])
            for frame in [*row['history'], row['current_input'], row['target'], *row['future_targets']]:
                self.assertFalse(Path(frame['occ_path']).is_absolute())
                resolved = resolve_occstress_path(frame['occ_path'], occstress_root=self.root,
                                                 dataset=row['dataset'], external_root=external)
                with np.load(resolved, allow_pickle=False) as payload:
                    self.assertEqual(payload['semantics'].shape, (2, 2, 1))
        return rows

    def test_shared_builder_all_patterns_preserve_gt_controls_and_portability(self):
        reference, _ = fixture(self.root, 'nuscenes', [('camera_only', 'alocc', ['Snow'], ['hard'])])
        for corruption, mode, expected in [('clean', '', [0] * 5), ('Snow', 'current', [0, 0, 0, 0, 1]),
                                           ('Snow', 'history_k1', [0, 0, 0, 1, 1]), ('Snow', 'all_frame', [1, 1, 1, 1, 0])]:
            self.call_shared(self.shared_command(corruption=corruption, mode=mode))
            relative = f'clean/{BACKBONE}.pkl' if corruption == 'clean' else f'Snow/hard/{mode}_{BACKBONE}.pkl'
            output = self.root / 'protocols/upstream/OccStress-nuScenes/camera_only/alocc' / relative
            rows = self.assert_rows(output, reference)
            self.assertEqual([int(f['prediction_variant'] != 'clean') for f in [*rows[0]['history'], rows[0]['current_input']]], expected)
        moved = self.root.with_name('moved')
        shutil.move(self.root, moved)
        self.root = moved
        self.assert_rows(moved / 'protocols/upstream/OccStress-nuScenes/camera_only/alocc/clean' / (BACKBONE + '.pkl'), reference)

    def test_explicit_root_beats_environment(self):
        reference, _ = fixture(self.root, 'carla', [('camera_only', 'flashocc_carla', [], [])])
        args = self.shared_command('carla', 'flashocc_carla')
        with patch.dict(os.environ, {'OCCSTRESS_DATA_ROOT': str(self.root / 'wrong')}):
            self.call_shared(args)
        self.assert_rows(self.root / 'protocols/upstream/OccStress-CARLA/camera_only/flashocc_carla/clean' / (BACKBONE + '.pkl'), reference)
        self.assertFalse((self.root / 'wrong').exists())

    def test_external_mount_and_missing_gt_fail_before_write(self):
        external = self.root.parent / 'external-assets'
        reference, _ = fixture(self.root, 'carla', [('camera_only', 'flashocc_carla', [], [])], external=external)
        args = self.shared_command('carla', 'flashocc_carla') + ['--external-root', str(external)]
        self.call_shared(args)
        output = self.root / 'protocols/upstream/OccStress-CARLA/camera_only/flashocc_carla/clean' / (BACKBONE + '.pkl')
        self.assert_rows(output, reference, external)
        before = output.read_bytes()
        missing = resolve_occstress_path(reference[0]['future_targets'][0]['occ_path'], occstress_root=self.root, external_root=external)
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            self.call_shared(args + ['--overwrite'])
        self.assertEqual(output.read_bytes(), before)

    def test_existing_output_is_not_overwritten_without_opt_in(self):
        fixture(self.root, 'nuscenes', [('camera_only', 'alocc', [], [])])
        args = self.shared_command()
        self.call_shared(args)
        with self.assertRaises(FileExistsError):
            self.call_shared(args)
        self.call_shared(args + ['--overwrite'])

    def test_nonclean_or_wrong_dataset_backbone_is_rejected(self):
        rows, path = fixture(self.root, 'nuscenes', [('camera_only', 'alocc', [], [])])
        original = copy.deepcopy(rows)
        rows[0]['corruption']['type'] = 'traffic'
        path.write_bytes(pickle.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'backbone must be clean'):
            self.call_shared(self.shared_command())
        original[0]['dataset'] = 'OccStress-CARLA'
        path.write_bytes(pickle.dumps(original))
        with self.assertRaisesRegex(ValueError, 'release-normalized backbone'):
            self.call_shared(self.shared_command())

    def test_alocc_full_73_builder_and_release_receipts(self):
        reference, _ = fixture(self.root, 'nuscenes', [('camera_only', 'alocc', alocc.CORRUPTIONS, alocc.SEVERITIES)])
        with patch.object(alocc, 'EXPECTED_ANCHORS', 2), patch.object(alocc, 'EXPECTED_FRAMES', 22), self.quiet_children():
            with patch.object(sys, 'argv', ['alocc', '--occstress-root', str(self.root)]), contextlib.redirect_stdout(io.StringIO()):
                alocc.main()
        outputs = list((self.root / 'protocols/upstream/OccStress-nuScenes/camera_only/alocc').rglob('*.pkl'))
        self.assertEqual(len(outputs), 73)
        for path in outputs:
            self.assert_rows(path, reference)
        manifest = json.loads((self.root / 'meta/OccStress-nuScenes/upstream/camera_only/alocc/manifest.json').read_text())
        self.assertFalse(manifest['checkpoint_provenance_available'])
        self.assertIsNone(manifest['checkpoint_sha256'])

    def test_alocc_native_receipts_and_partial_markers(self):
        fixture(self.root, 'nuscenes', [('camera_only', 'alocc', alocc.CORRUPTIONS, alocc.SEVERITIES)])
        root = self.root / 'occ/upstream/OccStress-nuScenes/camera_only/alocc'
        for marker in root.rglob('.done.json'):
            marker.write_text(json.dumps(dict(status='success', frame_count=6019, checkpoint_sha256='test')))
        self.assertEqual(len(alocc.require_exports(root)), 25)
        (root / 'clean/.done.json').write_text(json.dumps(dict(status='payload_validated', files=1)))
        with self.assertRaisesRegex(RuntimeError, 'incomplete export marker'):
            alocc.require_exports(root)

    def test_stcocc_shell_builds_73_and_fails_on_missing_setting(self):
        reference, _ = fixture(self.root, 'nuscenes', [('camera_only', 'stcocc', alocc.CORRUPTIONS, alocc.SEVERITIES)])
        script = ROOT / 'scripts/stcocc/build_occstress_upstream_protocols_from_nuscc.sh'
        env = dict(os.environ, PYTHON=sys.executable, ROOT=str(ROOT))
        command = ['bash', str(script)]
        subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outputs = list((self.root / 'protocols/upstream/OccStress-nuScenes/camera_only/stcocc').rglob('*.pkl'))
        self.assertEqual(len(outputs), 73)
        names = {path.name for path in outputs if path.parent.name == 'hard'}
        self.assertEqual(names, {f'{prefix}_{BACKBONE}.pkl' for prefix in
                                 ('current_only', 'recent_burst', 'history_only')})
        for path in outputs:
            self.assert_rows(path, reference)
        shutil.rmtree(self.root / 'occ/upstream/OccStress-nuScenes/camera_only/stcocc/Snow/hard')
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'missing export:', result.stderr)

    def test_carla_both_tracks_and_external_gt(self):
        external = self.root.parent / 'external-assets'
        specs = [carla.track_spec(track) for track in ('camera', 'lidar')]
        reference, _ = fixture(self.root, 'carla', specs, external=external)
        with patch.object(carla, 'EXPECTED_ANCHORS', 2), self.quiet_children():
            with patch.object(sys, 'argv', ['carla', '--occstress-root', str(self.root), '--external-root', str(external), '--jobs', '2']), contextlib.redirect_stdout(io.StringIO()):
                carla.main()
        outputs = list((self.root / 'protocols/upstream/OccStress-CARLA').rglob('*.pkl'))
        self.assertEqual(len(outputs), 146)
        for path in outputs:
            self.assert_rows(path, reference, external)
        self.assertTrue((self.root / 'meta/OccStress-CARLA/upstream/protocol_validation_both.json').is_file())

    def test_carla_fusion_camera_keeps_existing_source_identity(self):
        subtrack, source, _, _ = carla.track_spec('fusion_camera')
        self.assertEqual((subtrack, source), ('pointcloud_fusion', 'effocc_carla'))
