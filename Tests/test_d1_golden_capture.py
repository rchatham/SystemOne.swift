"""Model-free checks of tools/d1_golden_capture.py (no torch required)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_capture():
    spec = importlib.util.spec_from_file_location("d1_golden_capture", REPO_ROOT / "tools" / "d1_golden_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class D1GoldenCaptureTests(unittest.TestCase):
    def test_import_is_lazy(self):
        load_capture()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_synthetic_inputs_match_fixture(self):
        capture = load_capture()
        fixture = json.loads((REPO_ROOT / "testdata" / "d1_golden_v1.json").read_text())
        for clip in fixture["audio"]["clips"]:
            samples = capture.synthetic_audio(clip["samples"], clip["seed"])
            data = b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples)
            self.assertEqual(hashlib.sha256(data).hexdigest(), clip["clip_sha256"])
        pixels = capture.synthetic_image(4, 3, 1)
        self.assertEqual(len(pixels), 4 * 3 * 3)
        self.assertTrue(all(0 <= p < 256 for p in pixels))

    def test_compare_json_tolerates_only_float_drift(self):
        capture = load_capture()
        self.assertEqual(capture.compare_json({"a": [1.0, "x"]}, {"a": [1.0 + 1e-7, "x"]}), [])
        self.assertTrue(capture.compare_json({"a": [1.0]}, {"a": [1.1]}))
        self.assertTrue(capture.compare_json({"b": 1}, {"a": 1}))


if __name__ == "__main__":
    unittest.main()
