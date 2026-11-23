from pathlib import Path
import ast
import hashlib
import json
import tempfile
import unittest

import numpy as np
import benchmark_engine as engine
from run_experiments import DEFAULT_DATA, validate_data
from scripts.validate_results import validate_run

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_recorded_data_and_label_independence(self):
        if not DEFAULT_DATA.is_file():
            self.skipTest("Benchmark data is supplied separately")
        self.assertEqual(validate_data()["rows"], 9216)
        df = engine.load_dataset(DEFAULT_DATA)
        original, eligible = engine.prepare_features(df, engine.DEFAULT_CONFIG)
        df["rul_minutes"] = 0
        df["observed_lifetime_minutes"] = 0
        altered, _ = engine.prepare_features(df, engine.DEFAULT_CONFIG)
        for name in original:
            np.testing.assert_array_equal(original[name], altered[name])
            self.assertTrue(np.isfinite(original[name]).all().all())
        self.assertEqual(int(eligible.sum()), 9066)

    def test_corrupted_data_is_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("corrupted data")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_data(path)

    def test_notebook_is_safe_to_open_and_syntax_valid(self):
        notebooks = list(ROOT.rglob("*.ipynb"))
        self.assertEqual(len(notebooks), 1)
        nb = json.loads(notebooks[0].read_text(encoding="utf-8"))
        sources = []
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                sources.append(source)
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertIn("START_TRAINING = False", "\n".join(sources))
        from run_experiments import EXPECTED_DATA_SHA256
        self.assertIn(EXPECTED_DATA_SHA256, "\n".join(sources))
        definitions = lambda source: {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        root_definitions = definitions((ROOT / "benchmark_engine.py").read_text(encoding="utf-8"))
        embedded = {}
        for source in sources:
            embedded.update(definitions(source))
        for name, definition in root_definitions.items():
            self.assertEqual(embedded[name], definition, f"Notebook engine drift: {name}")

    def test_reference_metrics_reproduce(self):
        if not (ROOT / "artifacts" / "reference_run" / "completed.json").is_file():
            self.skipTest("Recorded results are supplied separately")
        self.assertTrue(validate_run(ROOT / "artifacts" / "reference_run")["passed"])

    def test_small_model_checkpoint_recovery(self):
        import pandas as pd
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"bearing_id": ["synthetic"] * 120, "run": [1] * 120,
                           "rul_minutes": np.linspace(120, 1, 120)})
        variants = {"basic": pd.DataFrame(rng.normal(size=(120, 9)), columns=[f"x{i}" for i in range(9)])}
        eligible = pd.Series(True, index=df.index)
        with tempfile.TemporaryDirectory() as directory:
            engine.verify_resume(df, variants, eligible, directory, engine.DEFAULT_CONFIG)
            result = json.loads((Path(directory) / "results" / "resume_verification.json").read_text())
            self.assertTrue(all(row["passed"] for row in result))


if __name__ == "__main__":
    unittest.main()
