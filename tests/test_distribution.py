import copy
import json
from pathlib import Path
import tempfile
import unittest

from occstress.distribution import (
    component_for_path, load_distribution, placeholder_text, public_source_issues,
    require_source, source_status, withheld,
)
from tools.build_public_release import build, copy_source, omit
from tools.run_formal import command_plan
from tools.install_environment import install_plan

ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def fixture(self, root, profile='public_candidate'):
        manifest = copy.deepcopy(load_distribution(ROOT))
        manifest['profile'] = profile
        (root / 'configs').mkdir()
        (root / 'configs/distribution.json').write_text(json.dumps(manifest))
        for component in manifest['pending_components']:
            directory = root / component['directory']
            directory.mkdir(parents=True)
            for name in ('README.md', 'README_OCCSTRESS.md'):
                (directory / name).write_text(placeholder_text(component))
        return manifest

    def test_only_five_pending_and_no_fictitious_contact(self):
        manifest = load_distribution(ROOT)
        self.assertEqual({c['id'] for c in manifest['pending_components']},
                         {'sparseworld-tc', 'cvtocc', 'fusionocc', 'sdgocc', 'occfusion'})
        self.assertTrue(all(c['contact_status'] == 'not_contacted' for c in manifest['pending_components']))

    def test_placeholder_links_to_source_packaging(self):
        component = load_distribution(ROOT)['pending_components'][0]
        text = placeholder_text(component)
        self.assertIn('Source not included in this package profile.', text)
        self.assertIn('docs/CODE_LAYOUT.md#source-packaging', text)
        self.assertNotIn('PENDING_PERMISSIONS.md', text)
        self.assertNotIn('Pending upstream permission', text)
        self.assertTrue((ROOT / 'docs/CODE_LAYOUT.md').is_file())

    def test_five_placeholders_accept_but_source_and_patch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            self.assertEqual(public_source_issues(root), [])
            for component in manifest['pending_components']:
                path = root / component['directory'] / 'model.py'
                path.write_text('secret = 1\n')
                self.assertTrue(public_source_issues(root), component['id'])
                path.unlink()
            path = root / 'scripts/waymo/patches/cvtocc-iiworld-registry-compat.patch'
            path.parent.mkdir(parents=True)
            path.write_text('diff')
            self.assertTrue(public_source_issues(root))

    def test_placeholders_are_not_upstream_readmes_or_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            path = root / manifest['pending_components'][0]['directory'] / 'README.md'
            path.write_text('original upstream text')
            self.assertTrue(public_source_issues(root))
            path.unlink()
            path.symlink_to('README_OCCSTRESS.md')
            self.assertTrue(public_source_issues(root))

    def test_public_guard_does_not_affect_retained_or_maintainer_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            for component in manifest['pending_components']:
                with self.assertRaisesRegex(ValueError, 'Pending upstream permission'):
                    require_source(root, component['directory'])
            require_source(root, 'EXIST/4D/GenieDrive/occ_gen')
            manifest['profile'] = 'maintainer'
            (root / 'configs/distribution.json').write_text(json.dumps(manifest))
            require_source(root, 'EXIST/4D/SparseWorld')
            self.assertTrue(public_source_issues(root))

    def test_copy_excludes_source_helpers_patch_history_and_mounts(self):
        manifest = load_distribution(ROOT)
        manifest['profile'] = 'public_candidate'
        for relative in ('EXIST/3D/SDGOCC/model.py', 'scripts/fusionocc/dump_occstress_upstream.py',
                         'scripts/waymo/patches/cvtocc-iiworld-registry-compat.patch',
                         '.git/objects/pack/source.pack', 'data/OccStress/labels.npz'):
            self.assertTrue(omit(relative, manifest), relative)
        # This shared operator is Robo3D-derived, not SDGOcc model source.
        self.assertFalse(omit('scripts/waymo/waymo_sdgocc_corruptions.py', manifest))
        self.assertFalse(omit('data/README.md', manifest))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            output.mkdir()
            pending = source / 'EXIST/4D/SparseWorld/model.py'
            pending.parent.mkdir(parents=True)
            pending.write_text('do not distribute')
            (source / 'core.py').write_text('x = 1')
            data = source / 'data'
            data.mkdir()
            (data / 'README.md').write_text('mount instructions')
            (source / 'core-data').mkdir()
            (source / 'core-data/data').symlink_to('../data')
            copy_source(source, output, manifest)
            self.assertFalse((output / 'EXIST/4D/SparseWorld').exists())
            self.assertEqual((output / 'core.py').read_text(), 'x = 1')
            self.assertTrue(pending.is_file())
            self.assertTrue((output / 'core-data/data').is_symlink())
            self.assertTrue((output / 'data/README.md').is_file())

    def test_alias_to_withheld_source_fails_closed(self):
        manifest = load_distribution(ROOT)
        manifest['profile'] = 'public_candidate'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            output.mkdir()
            pending = source / 'EXIST/3D/FusionOcc/model.py'
            pending.parent.mkdir(parents=True)
            pending.touch()
            (source / 'alias.py').symlink_to('EXIST/3D/FusionOcc/model.py')
            with self.assertRaisesRegex(ValueError, 'aliases omitted content'):
                copy_source(source, output, manifest)

    def test_no_overwrite_or_nested_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            self.fixture(source, 'maintainer')
            with self.assertRaisesRegex(ValueError, 'non-nested'):
                build(source, source / 'public')
            output = root / 'existing'
            output.mkdir()
            (output / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                build(source, output)
            self.assertEqual((output / 'keep.txt').read_text(), 'keep')

    def test_launch_and_installer_fail_before_native_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'configs/evaluation_contract.json').write_text(
                (ROOT / 'configs/evaluation_contract.json').read_text())
            (root / 'environments').mkdir()
            (root / 'environments/profiles.json').write_text(
                (ROOT / 'environments/profiles.json').read_text())
            with self.assertRaisesRegex(ValueError, 'Pending upstream permission'):
                command_plan('sparseworld-tc', 'waymo', 'python', [], root=root, environ={})
            for model in ('sparseworld-tc', 'cvtocc', 'fusionocc', 'sdgocc'):
                with self.subTest(model=model), self.assertRaisesRegex(ValueError, 'Pending upstream permission'):
                    install_plan(model, Path('/new/env'), Path('/cuda'), '8.0', root=root)

    def test_component_prefix_is_not_a_substring_match(self):
        manifest = load_distribution(ROOT)
        self.assertIsNone(component_for_path('EXIST/3D/FusionOcc-other/model.py', manifest))
        self.assertIsNotNone(component_for_path('EXIST/3D/FusionOcc/model.py', manifest))

    def test_full_source_preserves_permission_status_and_allows_native_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, 'full_source_candidate')
            self.assertTrue(public_source_issues(root))
            for component in manifest['pending_components']:
                directory = root / component['directory']
                (directory / 'model.py').write_text('x = 1\n')
                for name in ('README.md', 'README_OCCSTRESS.md'):
                    (directory / name).write_text('Original attribution retained.\n')
                for relative in component['excluded_paths']:
                    path = root / relative
                    if path == directory:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix:
                        path.write_text('adapter\n')
                    else:
                        path.mkdir(exist_ok=True)
                require_source(root, component['directory'])
                self.assertFalse(withheld(root, component['directory']))
                self.assertEqual(source_status(root, component['directory']),
                                 'included_permission_unconfirmed')
                self.assertFalse(omit(component['directory'] + '/model.py', manifest))
            self.assertEqual(public_source_issues(root), [])
            manifest['public_redistribution_ready'] = True
            self.assertTrue(public_source_issues(root, manifest))

    def test_full_copy_still_excludes_runtime_assets_and_git(self):
        manifest = load_distribution(ROOT)
        manifest['profile'] = 'full_source_candidate'
        for relative in ('.git/config', 'model/__pycache__/model.pyc',
                         'data/OccStress/labels.npz', 'model/checkpoint.pth'):
            self.assertTrue(omit(relative, manifest), relative)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'output'
            path = source / 'EXIST/4D/SparseWorld/model.py'
            path.parent.mkdir(parents=True)
            path.write_text('x = 1\n')
            copy_source(source, output, manifest)
            self.assertEqual((output / path.relative_to(source)).read_text(), 'x = 1\n')

    def test_filtered_tree_cannot_be_mislabeled_as_full_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            self.fixture(source)
            with self.assertRaisesRegex(ValueError, 'Missing|placeholder'):
                build(source, root / 'full', 'full_source_candidate')
            self.assertFalse((root / 'full').exists())

    def test_default_build_requires_full_source_instead_of_silently_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            self.fixture(source, 'maintainer')
            with self.assertRaisesRegex(ValueError, 'Missing|placeholder'):
                build(source, root / 'full')
            self.assertFalse((root / 'full').exists())
