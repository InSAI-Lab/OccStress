import ast
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_public_release import copy_source, omit
from tools.check_documentation import issues as documentation_issues, local_links
from tools.check_release import scan
from occstress.distribution import load_distribution

ROOT = Path(__file__).resolve().parents[1]


class ReleasePortabilityTests(unittest.TestCase):
    def test_private_artifacts_are_excluded_without_deleting_originals(self):
        manifest = load_distribution(ROOT)
        paths = ('docs/md/old.md', 'docs/archive/old.md',
                 'environments/observed-cluster.json', 'environments/observed-local.json',
                 'docs/code-layout-migration.json', 'ANONYMIZATION_NOTES.md')
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source', Path(tmp) / 'output'
            for relative in (*paths, 'environments/profiles.json'):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            copy_source(source, output, manifest)
            for relative in paths:
                self.assertTrue(omit(relative, manifest))
                self.assertTrue((source / relative).exists())
                self.assertFalse((output / relative).exists())
            self.assertTrue((output / 'environments/profiles.json').exists())

    def test_public_scan_rejects_cluster_names_in_files_and_contents(self):
        site = 'hore' + 'ka'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / (site + '-runtime.json')).write_text('{}')
            (root / 'note.md').write_text(site.upper())
            (root / 'environments').mkdir()
            (root / 'environments/observed-cluster.json').write_text('{}')
            found = scan(root, public=True)
            self.assertTrue(any('forbidden name' in error for error in found))
            self.assertTrue(any('forbidden text' in error for error in found))
            self.assertTrue(any('private release artifact' in error for error in found))

    def test_public_scan_rejects_mounted_data_and_dangling_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/OccStress').mkdir(parents=True)
            (root / 'missing').symlink_to('does-not-exist')
            found = scan(root, public=True)
            self.assertTrue(any('data asset or mount' in error for error in found))
            self.assertTrue(any('broken symlink' in error for error in found))

    def test_public_scan_rejects_old_dataset_names(self):
        for dataset in ('nuScenes', 'nuscenes', 'Waymo', 'CARLA'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                name = dataset + '-' + 'S' + 'C'
                (root / (name + '.json')).write_text('{}')
                (root / 'metadata.json').write_text(json.dumps({'benchmark_name': name}))
                found = scan(root, public=True)
                self.assertTrue(any('forbidden name' in error for error in found))
                self.assertTrue(any('forbidden text' in error for error in found))

    def test_public_scan_accepts_canonical_names_and_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ('nuScenes', 'Waymo', 'CARLA'):
                name = 'OccStress-' + dataset
                (root / (name + '.json')).write_text(json.dumps({'dataset': name}))
            (root / 'waymo_occstress_world_dataset.py').write_text('class OccStressWaymoWorldDataset: pass\n')
            self.assertEqual(scan(root, public=True, syntax=True), [])

    def test_public_scan_rejects_obsolete_benchmark_identifiers(self):
        suffix = 's' + 'c'
        examples = ['waymo_' + suffix + '_world_dataset',
                    suffix + '_protocol', 'WAYMO_' + suffix.upper() + '_PROTOCOL',
                    'NuScenes' + suffix.upper() + 'WorldDataset',
                    'IISceneTokenizer' + suffix.upper(), 'LoadStreamOcc3D' + suffix.upper(),
                    '--' + suffix + '-root']
        for identifier in examples:
            with self.subTest(identifier=identifier), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'adapter.py').write_text('value = ' + repr(identifier))
                self.assertTrue(any('forbidden text' in error for error in scan(root, public=True)))
                (root / 'adapter.py').unlink()
                (root / (identifier + '.txt')).write_text('')
                self.assertTrue(any('forbidden name' in error for error in scan(root, public=True)))

    def test_native_scene_completion_metrics_are_not_benchmark_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'EXIST/3D/example/metrics.py'
            path.parent.mkdir(parents=True)
            suffix = 'S' + 'C'
            path.write_text('format_' + suffix + '_results = None\nsemkitti_' + suffix + '_IoU = None\n')
            self.assertEqual(scan(root, public=True, syntax=True), [])

    def test_model_augmentation_seed_namespace_is_stable(self):
        path = ROOT / 'EXIST/4D/II-World/mmdet3d/models/ii_world/world_model/ii_world_robust_finetune.py'
        tree = ast.parse(path.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'AUGMENTATION_SEED_NAMESPACE'
                                  for target in node.targets))
        namespace = eval(compile(ast.Expression(assignment.value), str(path), 'eval'), {'bytes': bytes})
        self.assertEqual(namespace, 'latent_' + 's' + 'c' + '_aug')

    def test_environment_profiles_do_not_depend_on_private_inventories(self):
        profiles = json.loads((ROOT / 'environments/profiles.json').read_text())
        for recipe in profiles['recipes'].values():
            self.assertNotIn('observed', recipe)
            self.assertTrue((ROOT / recipe['requirements']).is_file())

    def test_links_include_html_reference_style_and_escaped_spaces(self):
        text = '[guide](guide.md) <img src="image.png">\n[x]: <file name.md>\n'
        text += '[web](https://example.com) [anchor](#part)\n'
        text += '```text\n[not-a-link](absent.md)\n```\n'
        self.assertEqual(set(local_links(text)), {'guide.md', 'image.png', 'file name.md'})

    def test_links_reject_missing_files_and_repo_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'README.md').write_text('[bad](missing.md) [escape](../outside.md)')
            self.assertEqual(len(documentation_issues(root)), 2)

    def test_public_documentation_links(self):
        self.assertEqual(documentation_issues(ROOT), [])

    def test_geniedrive_training_defers_checkpoint_loading_to_runner(self):
        source = ROOT / 'EXIST/4D/GenieDrive/occ_gen/tools/train.py'
        tree = ast.parse(source.read_text())
        hardcoded_loads = [node for node in ast.walk(tree)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == 'load_checkpoint']
        self.assertEqual(hardcoded_loads, [])
        runner = (source.parents[1] / 'mmdet3d/apis/train.py').read_text()
        self.assertIn('runner.load_checkpoint(cfg.load_from', runner)
