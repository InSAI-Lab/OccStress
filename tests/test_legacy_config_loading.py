import ast
from pathlib import Path
import unittest


class LegacyConfigLoadingTest(unittest.TestCase):
    def test_native_configs_use_legacy_parser(self):
        root = Path(__file__).resolve().parents[1]
        for relative, expected in (
            ('EXIST/4D/COME/tools/test_diffusion_control.py', 3),
            ('EXIST/4D/OccWorld/eval_metric_stp3.py', 1),
        ):
            with self.subTest(file=relative):
                tree = ast.parse((root / relative).read_text())
                calls = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == 'Config' and node.func.attr == 'fromfile']
                self.assertEqual(len(calls), expected)
                for call in calls:
                    option = next((kw.value for kw in call.keywords if kw.arg == 'lazy_import'), None)
                    self.assertIsInstance(option, ast.Constant)
                    self.assertIs(option.value, False)
