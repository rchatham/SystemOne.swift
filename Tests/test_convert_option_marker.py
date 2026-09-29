"""Model-free schema/fixture tests plus optional tiny dependency round trips."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "convert_option_marker.py"
SPEC = importlib.util.spec_from_file_location("convert_option_marker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class OptionMarkerSchemaTests(unittest.TestCase):
    def test_import_is_model_free(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("safetensors", sys.modules)

    def test_expected_schema_is_exact_and_complete(self) -> None:
        schema = CONVERTER.EXPECTED_SCHEMA
        names = [spec.name for spec in schema]
        self.assertEqual(len(schema), 178)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(name.startswith("encoder.") for name in names), 170)
        self.assertEqual(sum(name.startswith("scorer.") for name in names), 8)
        self.assertEqual(sum(spec.byte_count for spec in schema), 1_581_240_324)

        expected_endpoints = {
            "encoder.embeddings.tok_embeddings.weight": (50368, 1024),
            "encoder.embeddings.norm.weight": (1024,),
            "encoder.final_norm.weight": (1024,),
            "scorer.input_norm.weight": (1024,),
            "scorer.input_norm.bias": (1024,),
            "scorer.dense.weight": (512, 1024),
            "scorer.dense.bias": (512,),
            "scorer.norm.weight": (512,),
            "scorer.norm.bias": (512,),
            "scorer.out_proj.weight": (1, 512),
            "scorer.out_proj.bias": (1,),
        }
        shapes = {spec.name: spec.shape for spec in schema}
        for name, shape in expected_endpoints.items():
            self.assertEqual(shapes[name], shape)

        for layer in range(28):
            prefix = f"encoder.layers.{layer}"
            expected = {
                f"{prefix}.attn.Wqkv.weight": (3072, 1024),
                f"{prefix}.attn.Wo.weight": (1024, 1024),
                f"{prefix}.mlp_norm.weight": (1024,),
                f"{prefix}.mlp.Wi.weight": (5248, 1024),
                f"{prefix}.mlp.Wo.weight": (1024, 2624),
            }
            if layer > 0:
                expected[f"{prefix}.attn_norm.weight"] = (1024,)
            else:
                self.assertNotIn(f"{prefix}.attn_norm.weight", shapes)
            for name, shape in expected.items():
                self.assertEqual(shapes[name], shape)

    def test_committed_fixtures_are_canonical_and_deeply_valid(self) -> None:
        metadata_bytes = CONVERTER.TENSOR_METADATA_PATH.read_bytes()
        manifest_bytes = CONVERTER.MANIFEST_PATH.read_bytes()
        metadata = CONVERTER.load_json_bytes(metadata_bytes)
        manifest = CONVERTER.load_json_bytes(manifest_bytes)

        self.assertEqual(metadata_bytes, CONVERTER.canonical_json_bytes(metadata))
        self.assertEqual(manifest_bytes, CONVERTER.canonical_json_bytes(manifest))
        CONVERTER.validate_tensor_metadata_fixture(metadata)
        CONVERTER.validate_manifest(manifest, metadata_bytes)
        self.assertEqual(
            manifest["tensor_metadata"]["sha256"],
            __import__("hashlib").sha256(metadata_bytes).hexdigest(),
        )
        self.assertEqual(manifest["output"]["payload_bytes"], 1_581_240_324)
        self.assertEqual(manifest["output"]["tensor_count"], 178)

    def test_fixtures_contain_no_machine_paths_or_time_fields(self) -> None:
        fixtures = [
            CONVERTER.load_json_bytes(CONVERTER.TENSOR_METADATA_PATH.read_bytes()),
            CONVERTER.load_json_bytes(CONVERTER.MANIFEST_PATH.read_bytes()),
        ]

        def visit(value: object, path: str = "$") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    lowered = key.lower()
                    self.assertNotIn("timestamp", lowered, path)
                    self.assertNotIn("created_at", lowered, path)
                    self.assertNotIn("absolute_path", lowered, path)
                    visit(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")
            elif isinstance(value, str):
                self.assertFalse(value.startswith(("/", "~", "file:")), f"machine path at {path}")
                self.assertNotIn(str(Path.home()), value, path)

        for fixture in fixtures:
            visit(fixture)

    def test_fixture_validators_reject_schema_hash_and_range_mutations(self) -> None:
        metadata_bytes = CONVERTER.TENSOR_METADATA_PATH.read_bytes()
        metadata = CONVERTER.load_json_bytes(metadata_bytes)
        manifest = CONVERTER.load_json_bytes(CONVERTER.MANIFEST_PATH.read_bytes())

        renamed = copy.deepcopy(metadata)
        renamed["tensors"][0]["name"] = "encoder.renamed.weight"
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.validate_tensor_metadata_fixture(renamed)

        wrong_shape = copy.deepcopy(metadata)
        wrong_shape["tensors"][0]["shape"][0] += 1
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.validate_tensor_metadata_fixture(wrong_shape)

        wrong_offset = copy.deepcopy(metadata)
        wrong_offset["tensors"][0]["data_offsets"][1] -= 4
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.validate_tensor_metadata_fixture(wrong_offset)

        wrong_hash = copy.deepcopy(manifest)
        wrong_hash["tensor_metadata"]["sha256"] = "0" * 64
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.validate_manifest(wrong_hash, metadata_bytes)

        wrong_output_hash = copy.deepcopy(manifest)
        wrong_output_hash["output"]["sha256"] = "not-a-sha256"
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.validate_manifest(wrong_output_hash, metadata_bytes)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.load_json_bytes(b'{"a":1,"a":2}')


def _tiny_source_case() -> None:
    import torch

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "tiny.pt"
        expected = torch.arange(4, dtype=torch.float32)
        torch.save(OrderedDict([("value", expected)]), source)
        with mock.patch.object(torch, "load") as mocked_load:
            try:
                CONVERTER.load_source_state(source, expected_sha256="0" * 64)
            except CONVERTER.ConversionError:
                pass
            else:
                raise AssertionError("wrong source hash was accepted")
            mocked_load.assert_not_called()

        digest = CONVERTER.sha256_file(source)
        state = CONVERTER.load_source_state(source, expected_sha256=digest)
        assert list(state) == ["value"]
        assert torch.equal(state["value"], expected)


def _tiny_round_trip_case() -> None:
    import torch

    schema = (
        CONVERTER.TensorSpec("encoder.tiny.weight", (2, 3)),
        CONVERTER.TensorSpec("scorer.tiny.bias", (2,)),
    )
    tensors = OrderedDict(
        [
            ("encoder.tiny.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            ("scorer.tiny.bias", torch.tensor([1.25, -2.5], dtype=torch.float32)),
        ]
    )
    validated = CONVERTER.validate_source_tensors(tensors, schema=schema)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first.safetensors"
        second = root / "second.safetensors"
        os.replace(CONVERTER._save_candidate(validated, first), first)
        os.replace(CONVERTER._save_candidate(validated, second), second)
        assert CONVERTER.sha256_file(first) == CONVERTER.sha256_file(second)
        assert first.stat().st_mode & 0o777 == 0o644

        inspection = CONVERTER.inspect_safetensors(first, schema=schema, source_tensors=validated)
        metadata = CONVERTER.build_tensor_metadata_fixture(inspection, schema=schema)
        metadata_bytes = CONVERTER.canonical_json_bytes(metadata)
        manifest = CONVERTER.build_manifest(inspection, metadata_bytes, schema=schema)
        metadata_path = root / "metadata.json"
        manifest_path = root / "manifest.json"
        metadata_path.write_bytes(metadata_bytes)
        manifest_path.write_bytes(CONVERTER.canonical_json_bytes(manifest))

        before = {path.name: path.read_bytes() for path in root.iterdir()}
        checked = CONVERTER.check_output(
            first,
            tensor_metadata_path=metadata_path,
            manifest_path=manifest_path,
            schema=schema,
        )
        assert checked == inspection
        assert before == {path.name: path.read_bytes() for path in root.iterdir()}

        bad_manifest = copy.deepcopy(manifest)
        bad_manifest["output"]["sha256"] = "0" * 64
        manifest_path.write_bytes(CONVERTER.canonical_json_bytes(bad_manifest))
        files_before_failure = {path.name: path.read_bytes() for path in root.iterdir()}
        try:
            CONVERTER.check_output(
                first,
                tensor_metadata_path=metadata_path,
                manifest_path=manifest_path,
                schema=schema,
            )
        except CONVERTER.ConversionError:
            pass
        else:
            raise AssertionError("wrong output hash was accepted")
        assert files_before_failure == {path.name: path.read_bytes() for path in root.iterdir()}


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
    "tiny round-trip requires the repo .venv conversion dependencies",
)
class OptionMarkerTinyDependencyTests(unittest.TestCase):
    def _run_isolated(self, case: str) -> None:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), case], check=True)

    def test_source_hash_is_checked_before_torch_load(self) -> None:
        self._run_isolated("tiny-source")

    def test_tiny_safetensors_round_trip_is_deterministic_and_check_is_read_only(self) -> None:
        self._run_isolated("tiny-round-trip")


if __name__ == "__main__":
    if sys.argv[1:] == ["tiny-source"]:
        _tiny_source_case()
    elif sys.argv[1:] == ["tiny-round-trip"]:
        _tiny_round_trip_case()
    else:
        unittest.main()
