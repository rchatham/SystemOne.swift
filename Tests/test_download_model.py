"""Model-free tests for the pinned checkpoint downloader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "download_model.py"
SPEC = importlib.util.spec_from_file_location("download_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)

MANIFEST_PATH = REPO_ROOT / "testdata" / "option_marker_weights_manifest_v1.json"


class DownloadModelTests(unittest.TestCase):
    def test_import_is_model_free(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("safetensors", sys.modules)

    def test_source_hash_matches_committed_manifest(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            DOWNLOADER.SOURCE_SHA256, manifest["source"]["sha256"]
        )

    def test_download_url_is_pinned_and_well_formed(self) -> None:
        self.assertTrue(DOWNLOADER.DOWNLOAD_URL.startswith("https://huggingface.co/"))
        self.assertIn(DOWNLOADER.HF_REPO, DOWNLOADER.DOWNLOAD_URL)
        self.assertIn(DOWNLOADER.HF_REVISION, DOWNLOADER.DOWNLOAD_URL)
        self.assertTrue(DOWNLOADER.DOWNLOAD_URL.endswith(DOWNLOADER.SOURCE_ARTIFACT_NAME))

    def test_sha256_file_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            payload = b"von downloader test payload\n" * 1000
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(DOWNLOADER.sha256_file(path), expected)

    def test_check_returns_one_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            self.assertEqual(DOWNLOADER.check(missing), 1)

    def test_check_returns_one_for_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            path.write_bytes(b"not the pinned checkpoint")
            self.assertEqual(DOWNLOADER.check(path), 1)

    def test_check_returns_zero_for_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "right.pt"
            path.write_bytes(b"placeholder bytes")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(DOWNLOADER, "SOURCE_SHA256", expected):
                self.assertEqual(DOWNLOADER.check(path), 0)


if __name__ == "__main__":
    unittest.main()
