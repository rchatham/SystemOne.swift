"""Model-free stdlib validation for the committed Python-von golden fixture."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "golden_capture.py"
SPEC = importlib.util.spec_from_file_location("golden_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)


class GoldenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_bytes = GOLDEN.FIXTURE_PATH.read_bytes()
        cls.fixture = GOLDEN.load_json_bytes(cls.fixture_bytes)

    def test_import_is_model_free(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("von", sys.modules)

    def test_fixture_is_canonical_compact_and_deeply_valid(self) -> None:
        self.assertEqual(self.fixture_bytes, GOLDEN.canonical_json_bytes(self.fixture))
        self.assertLess(len(self.fixture_bytes), 512 * 1024)
        GOLDEN.validate_fixture(self.fixture)

    def test_fixture_has_exact_capture_shape(self) -> None:
        self.assertEqual([case["id"] for case in self.fixture["cases"]], GOLDEN.EXPECTED_CASE_IDS)
        self.assertEqual(
            self.fixture["metadata"]["transformers_hidden_state_entries"],
            GOLDEN.TRANSFORMERS_HIDDEN_STATE_ENTRIES,
        )
        self.assertEqual(len(GOLDEN.TRANSFORMERS_HIDDEN_STATE_ENTRIES), 29)
        self.assertEqual(GOLDEN.TRANSFORMERS_HIDDEN_STATE_ENTRIES[-2:], ["layer_26", "final_norm"])
        self.assertEqual(
            [(forward["case_id"], forward["role"]) for forward in self.fixture["forwards"]],
            GOLDEN.EXPECTED_FORWARD_ROLES,
        )
        self.assertEqual(len(self.fixture["forwards"]), 5)
        for forward in self.fixture["forwards"]:
            option_count = len(forward["mask_positions"])
            self.assertEqual([entry["name"] for entry in forward["backbone"]], GOLDEN.BACKBONE_ENTRIES)
            self.assertEqual(len(forward["backbone"]), 30)
            self.assertNotEqual(
                forward["backbone"][-2]["mask_state_probes"],
                forward["backbone"][-1]["mask_state_probes"],
            )
            self.assertEqual(len(forward["final_mask_states"]), option_count)
            self.assertTrue(all(len(row) == GOLDEN.HIDDEN_SIZE for row in forward["final_mask_states"]))
            for entry in forward["backbone"]:
                self.assertEqual(len(entry["mask_state_probes"]), option_count)
                self.assertTrue(
                    all(len(row) == len(GOLDEN.PROBE_DIMENSIONS) for row in entry["mask_state_probes"])
                )

    def test_cases_preserve_order_and_production_answers(self) -> None:
        choice, default_noul, explicit_noul, score = self.fixture["cases"]
        self.assertGreater(len(self.fixture["forwards"][0]["input_ids"]), 128)
        self.assertIn("東京", self.fixture["forwards"][0]["packed_text"])
        self.assertEqual(default_noul["forward_indices"], [1, 2])
        self.assertTrue(default_noul["postprocess"]["debias_applied"])
        self.assertEqual(default_noul["postprocess"]["debias_factor"], 0.7)
        self.assertEqual(explicit_noul["forward_indices"], [3])
        self.assertFalse(explicit_noul["postprocess"]["debias_applied"])
        self.assertEqual(score["forward_indices"], [4])
        self.assertEqual(
            [option["key"] for option in choice["ordered_options"]],
            ["infrastructure", "billing", "feature_request", "security"],
        )
        self.assertEqual(
            [option["description"] for option in score["ordered_options"]],
            [
                "No user impact Examples: cosmetic typo, internal-only warning",
                "Limited degradation Examples: one slow shard, brief retry spike",
                "Major incident Examples: multi-region errors, sustained outage",
                "Critical outage Examples: all requests fail, data loss is occurring",
            ],
        )
        for case in self.fixture["cases"]:
            serialized = case["answer_model_dump_json"]
            answer = json.loads(serialized)
            self.assertEqual(serialized, json.dumps(answer, ensure_ascii=False, separators=(",", ":")))
            self.assertEqual(answer["type"], case["primitive"])

    def test_metadata_has_exact_pins_and_no_machine_data(self) -> None:
        metadata = self.fixture["metadata"]
        self.assertEqual(metadata["hf_revision"], GOLDEN.HF_REVISION)
        self.assertEqual(metadata["von_source_revision"], GOLDEN.VON_SOURCE_REVISION)
        self.assertEqual(metadata["von_model_id"], GOLDEN.VON_MODEL_ID)
        self.assertEqual(metadata["runtime_versions"], GOLDEN.EXPECTED_RUNTIME_VERSIONS)
        self.assertEqual(
            metadata["artifacts"],
            [{"name": name, "sha256": digest} for name, digest in GOLDEN.ARTIFACT_HASHES.items()],
        )
        serialized = json.dumps(metadata)
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn("timestamp", serialized.lower())

    def test_deep_validation_ties_answers_and_final_probes_to_math(self) -> None:
        changed_answer = copy.deepcopy(self.fixture)
        changed_answer["cases"][1]["answer_model_dump_json"] = '{"type":"noul","noul":0.1234}'
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.validate_fixture(changed_answer)

        changed_probe = copy.deepcopy(self.fixture)
        changed_probe["forwards"][0]["backbone"][-1]["mask_state_probes"][0][0] += 0.001
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.validate_fixture(changed_probe)

        changed_choice_logit = copy.deepcopy(self.fixture)
        changed_choice_logit["forwards"][0]["raw_logits"][0] += 0.1
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.validate_fixture(changed_choice_logit)

        changed_confidence = copy.deepcopy(self.fixture)
        changed_confidence["cases"][3]["postprocess"]["confidence_unrounded"] += 0.1
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.validate_fixture(changed_confidence)

        changed_noul_primary = copy.deepcopy(self.fixture)
        changed_noul_primary["cases"][1]["postprocess"]["primary_raw_logits"][0] += 0.1
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.validate_fixture(changed_noul_primary)

        changed_adjusted_logit = copy.deepcopy(self.fixture)
        changed_adjusted_logit["cases"][1]["postprocess"]["adjusted_logits"][0] += 0.1
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.validate_fixture(changed_adjusted_logit)

    def test_validator_rejects_token_boundary_mask_and_separator_mutations(self) -> None:
        extra_mask = copy.deepcopy(self.fixture)
        extra_mask_forward = extra_mask["forwards"][0]
        ordinary_position = next(
            index
            for index, token_id in enumerate(extra_mask_forward["input_ids"])
            if token_id not in (GOLDEN.CLS_TOKEN_ID, GOLDEN.SEP_TOKEN_ID, GOLDEN.MASK_TOKEN_ID)
        )
        extra_mask_forward["input_ids"][ordinary_position] = GOLDEN.MASK_TOKEN_ID

        bad_first = copy.deepcopy(self.fixture)
        bad_first["forwards"][0]["input_ids"][0] = 0

        bad_final = copy.deepcopy(self.fixture)
        bad_final["forwards"][0]["input_ids"][-1] = 0

        bad_separator_token = copy.deepcopy(self.fixture)
        separator_positions = [
            index
            for index, token_id in enumerate(bad_separator_token["forwards"][0]["input_ids"])
            if token_id == GOLDEN.SEP_TOKEN_ID
        ]
        bad_separator_token["forwards"][0]["input_ids"][separator_positions[0]] = 0

        bad_packed_separator = copy.deepcopy(self.fixture)
        bad_packed_separator["forwards"][0]["packed_text"] = bad_packed_separator["forwards"][0][
            "packed_text"
        ].replace(" [SEP] ", " SEP ", 1)

        mutations = {
            "extra unlisted mask": extra_mask,
            "bad first token": bad_first,
            "bad final token": bad_final,
            "bad encoded separator": bad_separator_token,
            "bad packed separator": bad_packed_separator,
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(GOLDEN.FixtureError):
                    GOLDEN.validate_fixture(mutation)

    def test_check_is_bytes_first_tolerant_only_for_floats_and_never_rewrites(self) -> None:
        original_path = GOLDEN.FIXTURE_PATH
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "fixture.json"
            fixture_path.write_bytes(self.fixture_bytes)
            GOLDEN.FIXTURE_PATH = fixture_path
            try:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    GOLDEN.check_fixture(self.fixture_bytes)
                self.assertIn("byte-identical", output.getvalue())
                self.assertEqual(fixture_path.read_bytes(), self.fixture_bytes)

                tolerated = copy.deepcopy(self.fixture)
                tolerated["forwards"][0]["backbone"][0]["mask_state_probes"][0][0] += (
                    GOLDEN.FLOAT_ABS_TOLERANCE / 2
                )
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    GOLDEN.check_fixture(GOLDEN.canonical_json_bytes(tolerated))
                self.assertIn("finite float leaves", output.getvalue())
                self.assertEqual(fixture_path.read_bytes(), self.fixture_bytes)

                with self.assertRaises(GOLDEN.FixtureError):
                    GOLDEN.check_fixture(b"\n" + self.fixture_bytes)
                self.assertEqual(fixture_path.read_bytes(), self.fixture_bytes)

                structural = copy.deepcopy(self.fixture)
                structural["cases"][0]["q_id"] = "changed"
                with self.assertRaises(GOLDEN.FixtureError):
                    GOLDEN.check_fixture(GOLDEN.canonical_json_bytes(structural))
                self.assertEqual(fixture_path.read_bytes(), self.fixture_bytes)
            finally:
                GOLDEN.FIXTURE_PATH = original_path

    def test_comparator_accepts_only_tolerated_float_changes(self) -> None:
        tolerated = GOLDEN.compare_json(
            {"a": [1.0, {"b": 2}]},
            {"a": [1.0 + GOLDEN.FLOAT_ABS_TOLERANCE / 2, {"b": 2}]},
        )
        self.assertEqual(tolerated, ["a[0]"])
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.compare_json({"a": 1.0}, {"a": 1.1})
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.compare_json({"a": float("inf")}, {"a": float("inf")})

    def test_comparator_rejects_structure_order_types_and_nonfloats(self) -> None:
        differing_pairs = [
            ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
            ([1, 2], [2, 1]),
            ([1], [1, 2]),
            ({"a": 1}, {"a": 1.0}),
            ({"a": True}, {"a": 1}),
            ({"a": 1}, {"a": 2}),
            ({"a": "x"}, {"a": "y"}),
        ]
        for expected, actual in differing_pairs:
            with self.subTest(expected=expected, actual=actual):
                with self.assertRaises(GOLDEN.FixtureError):
                    GOLDEN.compare_json(expected, actual)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaises(GOLDEN.FixtureError):
            GOLDEN.load_json_bytes(b'{"a":1,"a":2}')


if __name__ == "__main__":
    unittest.main()
