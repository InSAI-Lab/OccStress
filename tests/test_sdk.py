import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.adapters.native_results import extract_counts
from occstress.datasets.labels import remap_labels
from occstress.metrics.occupancy import (confusion_from_labels, from_confusion,
                                        score_counts, validate_counts)
from occstress.protocols.suites import conditions
from occstress.protocols.validation import load_records, validate_records
from occstress.results.io import content_hash, load_json, sha256_file, write_json
from occstress.results.schema import build_result, validate_result
from occstress.results.summary import merge_shards, summarize_suite
from tools.run_manifest import can_resume, run_manifest, task_plan
from tools.summarize import normalize


def histograms(correct=1, wrong=0):
    semantic = np.zeros((6, 18, 18), dtype=np.int64)
    binary = np.zeros((6, 2, 2), dtype=np.int64)
    semantic[:, 0, 0] = correct
    semantic[:, 0, 17] = wrong
    binary[:, 1, 1] = correct
    binary[:, 1, 0] = wrong
    return semantic, binary


def result(anchor='a', protocol='clean', correct=1, wrong=0):
    identity = dict(model='test-model', dataset='waymo', track='manual', source='gt',
                    protocol_id=protocol, protocol_sha256=content_hash(protocol),
                    code_revision='test-only', checkpoint_sha256={'model': content_hash('ckpt')},
                    class_map_sha256=content_hash('map'), base_info_sha256=content_hash('base'),
                    input_offsets_seconds=[-1.5, -1, -0.5, 0], control_policy='GT',
                    voxel_mask='all-valid', seed=0)
    return build_result(identity, [['s', anchor]], from_confusion(*histograms(correct, wrong)))


def suite():
    return dict(schema_version=1, id='test', dataset='waymo', track='manual', source='gt',
                expected_anchors=1, conditions=[dict(id='clean', kind='clean'),
                                              dict(id='hard', kind='corrupted'),
                                              dict(id='traffic', kind='traffic')])


class MetricsTests(unittest.TestCase):
    def test_zero_iou_is_not_excluded(self):
        scores = result(correct=0, wrong=5)['scores']
        self.assertEqual(scores['paper_1_2_3s']['miou'], 0)
        self.assertEqual(scores['horizons'][0]['present_classes'], [0])

    def test_absent_gt_class_excluded(self):
        semantic, binary = histograms()
        semantic[:, 17, 4] = 1
        binary[:, 0, 1] = 1
        scores = score_counts(from_confusion(semantic, binary))
        self.assertEqual(scores['paper_1_2_3s'], {'miou': 100., 'iou': 50.})

    def test_paper_average_uses_1_2_3_seconds(self):
        semantic, binary = histograms()
        semantic[::2] = 0
        binary[::2] = 0
        semantic[::2, 0, 17] = 1
        binary[::2, 1, 0] = 1
        scores = score_counts(from_confusion(semantic, binary))
        self.assertEqual(scores['paper_1_2_3s']['miou'], 100.)
        self.assertEqual(scores['all_six']['miou'], 50.)

    def test_empty_occupied_gt_is_null_not_nan(self):
        scores = result(correct=0)['scores']
        self.assertIsNone(scores['paper_1_2_3s']['miou'])
        json.dumps(scores, allow_nan=False)

    def test_histogram_orientation_and_ignore(self):
        matrix = confusion_from_labels(np.array([2, 255]), np.array([1, 255]))
        self.assertEqual(matrix[1, 2], 1)
        self.assertEqual(matrix.sum(), 1)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            confusion_from_labels(np.array([18]), np.array([0]))
        with self.assertRaises(ValueError):
            confusion_from_labels(np.array([1.5]), np.array([0]))

    def test_invalid_counts_rejected(self):
        for bad in (-1, 1.5, float('nan'), float('inf'), 10**16):
            value = result()['counts']
            value['semantic_tp'][0][0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate_counts(value)

    def test_binary_mask_disagreement_rejected(self):
        semantic, binary = histograms()
        binary[:, 1, 1] = 2
        with self.assertRaises(ValueError):
            from_confusion(semantic, binary)

    def test_six_native_formats(self):
        semantic, binary = (x.tolist() for x in histograms())
        counts = result()['counts']
        come_counts = copy.deepcopy(counts)
        for key in ('semantic_pred', 'semantic_gt', 'semantic_tp'):
            come_counts[key] = [row + [0] for row in counts[key]]
        variants = [dict(aggregate_counts=dict(semantic_confusion=semantic, binary_confusion=binary)),
                    dict(metrics={'_aggregate_counts': dict(semantic_confusion=semantic, binary_confusion=binary)}),
                    dict(counts=come_counts), dict(counts=counts),
                    dict(raw_confusion=dict(semantic=semantic, binary=binary)),
                    dict(semantic_confusion=semantic, occupancy_confusion=binary)]
        for value in variants:
            self.assertEqual(score_counts(extract_counts(value))['paper_1_2_3s']['miou'], 100.)

    def test_metrics_only_or_failed_native_rejected(self):
        for value in ({'metrics': {'miou': 50}}, {'status': 'failed', 'counts': result()['counts']}):
            with self.assertRaises(ValueError):
                extract_counts(value)


class ResultsTests(unittest.TestCase):
    def test_merge_counts_not_mean_of_shard_scores(self):
        merged = merge_shards([result('a', correct=100), result('b', correct=0, wrong=1)])
        self.assertAlmostEqual(merged['scores']['paper_1_2_3s']['miou'], 10000 / 101)
        self.assertEqual(merged['evaluated_records'], 2)

    def test_duplicate_anchors_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            merge_shards([result(), result()])

    def test_mixed_identity_rejected(self):
        for field, value in [('seed', 1), ('control_policy', 'predicted'), ('code_revision', 'different')]:
            right = result('b')
            right['identity'][field] = value
            with self.assertRaisesRegex(ValueError, 'identities'):
                merge_shards([result(), right])

    def test_stale_scores_rejected(self):
        value = result()
        value['scores']['paper_1_2_3s']['miou'] = 50
        with self.assertRaises(ValueError):
            validate_result(value)

    def test_atomic_json_forbids_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'a.json'
            write_json(path, {'value': 1})
            with self.assertRaises(ValueError):
                write_json(path, {'value': float('nan')})
            self.assertEqual(load_json(path), {'value': 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_suite_excludes_traffic(self):
        summary = summarize_suite(suite(), [result(), result(protocol='hard', correct=1, wrong=1),
                                            result(protocol='traffic', correct=0, wrong=1)])
        self.assertEqual(summary['robust_mean_excluding_traffic']['miou'], 50)
        self.assertEqual(summary['coverage'], '3/3')

    def test_incomplete_suite_never_claims_full_rs(self):
        with self.assertRaises(ValueError):
            summarize_suite(suite(), [result()])
        summary = summarize_suite(suite(), [result()], allow_partial=True)
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['robust_mean_excluding_traffic']['miou'])

    def test_mismatched_cohort_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cohorts'):
            summarize_suite(suite(), [result(), result('b', 'hard'), result(protocol='traffic')])

    def test_unknown_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Undeclared'):
            summarize_suite(suite(), [result(protocol='exploratory')], True)

    def test_normalize_requires_correct_count_and_horizons(self):
        canonical = result()
        meta = {key: canonical[key] for key in ('identity', 'anchor_ids', 'future_horizons_seconds')}
        native = dict(counts=canonical['counts'], evaluated_records=1)
        self.assertEqual(normalize(native, meta, 'digest')['evaluated_records'], 1)
        native['evaluated_records'] = 2
        with self.assertRaises(ValueError):
            normalize(native, meta, 'digest')
        native['evaluated_records'] = 1
        meta['future_horizons_seconds'] = [0, .5, 1, 1.5, 2, 2.5]
        with self.assertRaises(ValueError):
            normalize(native, meta, 'digest')


class StructureTests(unittest.TestCase):
    def test_lightweight_import(self):
        subprocess.run([sys.executable, '-c',
                        "import sys, occstress; import occstress.metrics.occupancy; "
                        "assert not any(x in sys.modules for x in ('torch','mmcv','scipy','PIL'))"],
                       cwd=ROOT, check=True)

    def test_sparseworld_namespace_preserved(self):
        from occstress.distribution import withheld
        if withheld(ROOT, 'EXIST/4D/SparseWorld'):
            self.skipTest('SparseWorld source withheld pending permission')
        code = "import sys, importlib.util; sys.path.insert(0, 'EXIST/4D/SparseWorld'); import occstress; "
        code += "assert importlib.util.find_spec('occstress.waymo_dataset'); assert importlib.util.find_spec('occstress.carla_dataset')"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_legacy_generator_is_same_module(self):
        legacy = importlib.import_module('scripts.generate_semantic_noise_subset')
        actual = importlib.import_module('occstress.corruptions.semantic')
        self.assertIs(legacy, actual)
        saved = legacy.REALIZATION_SEED
        try:
            legacy.REALIZATION_SEED = 19
            self.assertEqual(actual.REALIZATION_SEED, 19)
        finally:
            legacy.REALIZATION_SEED = saved

    def test_manual_suite_sizes_and_paths(self):
        for dataset in ('nuscenes', 'waymo', 'carla'):
            rows = conditions(load_json(ROOT / f'configs/suites/{dataset}.json'))
            self.assertEqual(len(rows), 38)
            self.assertEqual(sum(x['kind'] == 'corrupted' for x in rows), 36)
            self.assertEqual(len({x['protocol_path'] for x in rows}), 38)

    def test_fixture_and_duplicate_tokens(self):
        records = load_records(ROOT / 'examples/fixtures/protocol.json')
        self.assertEqual(validate_records(records, 1)['anchors'], 1)
        records[0]['future_targets'][0]['token'] = 'a'
        with self.assertRaises(ValueError):
            validate_records(records)

    def test_pickle_requires_explicit_trust(self):
        with self.assertRaisesRegex(ValueError, 'trust'):
            load_records('/not-opened.pkl')

    def test_unknown_labels_are_not_silently_free(self):
        self.assertEqual(remap_labels(np.array([23, 4, 255]), {23: 17, 4: 2}).tolist(), [17, 2, 255])
        with self.assertRaises(ValueError):
            remap_labels(np.array([24]), {23: 17})


class ResumeTests(unittest.TestCase):
    def task(self, directory):
        path = directory / 'input.json'
        write_json(path, {'fixture': True})
        return dict(id='test-task', model='geniedrive', dataset='waymo', python=sys.executable,
                    native_args=[], native_output=str(directory / 'out.json'), expected_records=1,
                    inputs={'fixture': str(path)})

    def test_dry_run_starts_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            task = self.task(directory)
            path = directory / 'manifest.json'
            write_json(path, dict(schema_version=1, tasks=[task]))
            with patch('tools.run_manifest.subprocess.run') as run:
                run_manifest(path, ROOT)
            run.assert_not_called()
            self.assertFalse(Path(task['native_output']).exists())

    def test_execution_receipt_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            task = self.task(directory)
            path = directory / 'manifest.json'
            write_json(path, dict(schema_version=1, tasks=[task]))

            def fake_model(*args, **kwargs):
                write_json(task['native_output'], {'counts': result()['counts'], 'evaluated_records': 1})

            with patch('tools.run_manifest.subprocess.run', side_effect=fake_model) as run:
                run_manifest(path, ROOT, True)
                run_manifest(path, ROOT, True)
                self.assertEqual(run.call_count, 1)
            write_json(task['inputs']['fixture'], {'fixture': 'changed'})
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                run_manifest(path, ROOT, True)
            self.assertFalse((directory / 'out.json.lock').exists())

    def test_tampered_result_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output, receipt = Path(directory) / 'out.json', Path(directory) / 'receipt.json'
            write_json(output, {'counts': result()['counts'], 'evaluated_records': 1})
            write_json(receipt, {'status': 'success', 'fingerprint': 'x', 'native_sha256': sha256_file(output)})
            self.assertTrue(can_resume(output, receipt, 'x', 1))
            write_json(output, {'counts': result()['counts'], 'evaluated_records': 2})
            self.assertFalse(can_resume(output, receipt, 'x', 1))

    def test_duplicate_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            task = self.task(directory)
            other = {**task, 'id': 'other'}
            path = directory / 'manifest.json'
            write_json(path, dict(schema_version=1, tasks=[task, other]))
            with self.assertRaisesRegex(ValueError, 'unique'):
                run_manifest(path, ROOT)


if __name__ == '__main__':
    unittest.main()
