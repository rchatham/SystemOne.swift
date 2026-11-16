"""Model-free checks of tools/jev_golden_capture.py (no torch required)."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_capture():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("jev_golden_capture", REPO_ROOT / "tools" / "jev_golden_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JevGoldenCaptureTests(unittest.TestCase):
    def test_import_is_lazy(self):
        load_capture()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("peft", sys.modules)

    def test_bare_v1_matches_the_model_card(self):
        capture = load_capture()
        self.assertEqual(
            capture.bare_v1("choice", "S", "Q?", ["x", "y"]),
            "[kind] choice\n[state] S\n[question] Q?\n[options]\nA) x\nB) y\n[decision]:",
        )
        self.assertEqual(
            capture.bare_v1("noul", "S", "Q?", ["false", "true"]),
            "[kind] noul\n[state] S\n[question] Q?\n[options]\nfalse\ntrue\n[decision]:",
        )

    def test_tiny_fixture_is_consistent_with_its_model_folder(self):
        capture = load_capture()
        fixture = json.loads(capture.TINY_FIXTURE.read_text())
        judge = json.loads((capture.TINY_DIR / "judge_config.json").read_text())
        calibration = json.loads((capture.TINY_DIR / "calibration.json").read_text())
        self.assertEqual(judge["slots"], capture.SLOTS)
        self.assertEqual(calibration["per_kind"], fixture["temperatures"])
        for case in fixture["cases"]:
            self.assertEqual(case["probe_rows"], capture.probe_rows(len(case["ids"])))
            for kind, count in [("noul", 2), ("score", 6), ("choice", fixture["choice_options"])]:
                expected = capture.distribution(case["head_logits"], kind, count, fixture["temperatures"])
                for actual, wanted in zip(case["probabilities"][kind], expected):
                    self.assertAlmostEqual(actual, wanted, places=12)


if __name__ == "__main__":
    unittest.main()
