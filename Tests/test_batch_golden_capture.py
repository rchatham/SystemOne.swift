"""Model-free stdlib validation for the Python-von batch golden fixture."""

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
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "batch_golden_capture.py"
SPEC = importlib.util.spec_from_file_location("batch_golden_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)

EXPECTED_ANSWERS = [
    '{"type":"choice","choice":"infrastructure","probabilities":{"infrastructure":0.6274,"billing":0.1173,"feature_request":0.1301,"security":0.1252},"confidence":0.497}',
    '{"type":"noul","noul":0.3125}',
    '{"type":"noul","noul":0.2703}',
    '{"type":"score","score":1.96,"confidence":0.753,"legend":{"0":"No user impact Examples: cosmetic typo, internal-only warning","1":"Limited degradation Examples: one slow shard, brief retry spike","2":"Major incident Examples: multi-region errors, sustained outage","3":"Critical outage Examples: all requests fail, data loss is occurring"},"probabilities":{"0":0.0265,"1":0.0622,"2":0.8323,"3":0.079}}',
]


class BatchGoldenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_bytes = BATCH.FIXTURE_PATH.read_bytes()
        cls.fixture = BATCH.load_json_bytes(cls.fixture_bytes)

    def test_import_is_model_free(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("von", sys.modules)

    def test_fixture_is_canonical_modest_and_deeply_valid(self) -> None:
        self.assertEqual(self.fixture_bytes, BATCH.canonical_json_bytes(self.fixture))
        self.assertLess(len(self.fixture_bytes), 16 * 1024)
        BATCH.validate_fixture(self.fixture)

    def test_batch_metadata_and_padding_contract_are_exact(self) -> None:
        metadata = self.fixture["metadata"]
        self.assertEqual(metadata["source_fixture"]["sha256"], BATCH.SOURCE_FIXTURE_SHA256)
        self.assertEqual(metadata["pad_token_id"], 50283)
        self.assertEqual(metadata["padding_side"], "right")
        self.assertEqual(metadata["sequence_lengths"], [541, 42, 27, 53, 87])
        self.assertEqual(self.fixture["batch_capture"]["padded_shape"], [5, 541])
        rows = self.fixture["batch_capture"]["rows"]
        self.assertEqual(
            [(row["case_id"], row["role"]) for row in rows],
            BATCH.EXPECTED_ROW_ORDER,
        )
        self.assertEqual([row["padding_length"] for row in rows], [0, 499, 514, 488, 454])
        self.assertEqual(
            [row["mask_positions"] for row in rows],
            [[485, 504, 517, 528], [27, 34], [12, 19], [28, 40], [32, 46, 59, 72]],
        )

    def test_batch_drift_is_finite_nonzero_and_within_declared_limits(self) -> None:
        capture = self.fixture["batch_capture"]
        self.assertGreater(capture["max_absolute_error"], 0.0)
        self.assertGreater(capture["max_relative_error"], 0.0)
        self.assertLessEqual(capture["max_absolute_error"], BATCH.BATCH_ABS_TOLERANCE)
        self.assertLessEqual(capture["max_relative_error"], BATCH.BATCH_REL_TOLERANCE)
        for row in capture["rows"]:
            for key in ("sequential_raw_logits", "batched_raw_logits"):
                self.assertTrue(all(isinstance(value, float) for value in row[key]))

    def test_heterogeneous_request_has_ordered_primitives_and_exact_answers(self) -> None:
        request = self.fixture["heterogeneous_request"]
        self.assertEqual(request["state"], BATCH.SHARED_STATE)
        self.assertEqual(request["question_ids"], BATCH.EXPECTED_QUESTION_IDS)
        self.assertEqual([question["type"] for question in request["questions"]], BATCH.EXPECTED_PRIMITIVES)
        self.assertEqual(
            [entry["key"] for entry in request["questions"][0]["ordered_criteria"]],
            ["infrastructure", "billing", "feature_request", "security"],
        )
        self.assertIsNone(request["questions"][1]["criteria"])
        self.assertEqual(set(request["questions"][2]["criteria"]), {"true", "false"})
        self.assertEqual(
            [answer["question_id"] for answer in request["answers"]],
            BATCH.EXPECTED_QUESTION_IDS,
        )
        self.assertEqual(
            [answer["answer_model_dump_json"] for answer in request["answers"]],
            EXPECTED_ANSWERS,
        )

    def test_fixture_contains_no_machine_path_or_time_metadata(self) -> None:
        serialized = self.fixture_bytes.decode("utf-8")
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn("timestamp", serialized.lower())
        self.assertNotIn("created_at", serialized.lower())
        self.assertNotIn("generated_at", serialized.lower())

    def test_validator_rejects_batch_contract_mutations(self) -> None:
        mutations: dict[str, dict[str, object]] = {}

        changed_hash = copy.deepcopy(self.fixture)
        changed_hash["metadata"]["source_fixture"]["sha256"] = "0" * 64
        mutations["source hash"] = changed_hash

        changed_order = copy.deepcopy(self.fixture)
        changed_order["batch_capture"]["rows"][0], changed_order["batch_capture"]["rows"][1] = (
            changed_order["batch_capture"]["rows"][1],
            changed_order["batch_capture"]["rows"][0],
        )
        mutations["row order"] = changed_order

        changed_length = copy.deepcopy(self.fixture)
        changed_length["batch_capture"]["rows"][1]["sequence_length"] += 1
        mutations["sequence length"] = changed_length

        changed_mask = copy.deepcopy(self.fixture)
        changed_mask["batch_capture"]["rows"][2]["mask_positions"][0] += 1
        mutations["mask position"] = changed_mask

        changed_logit = copy.deepcopy(self.fixture)
        changed_logit["batch_capture"]["rows"][4]["batched_raw_logits"][0] += 0.01
        mutations["logit drift"] = changed_logit

        changed_error = copy.deepcopy(self.fixture)
        changed_error["batch_capture"]["max_absolute_error"] = 0.0
        mutations["aggregate error"] = changed_error

        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(BATCH.FixtureError):
                    BATCH.validate_fixture(mutation)

    def test_validator_rejects_request_mutations(self) -> None:
        mutations = []
        changed_ids = copy.deepcopy(self.fixture)
        changed_ids["heterogeneous_request"]["question_ids"][0] = "changed"
        mutations.append(changed_ids)

        changed_default = copy.deepcopy(self.fixture)
        changed_default["heterogeneous_request"]["questions"][1]["criteria"] = {}
        mutations.append(changed_default)

        changed_explicit = copy.deepcopy(self.fixture)
        changed_explicit["heterogeneous_request"]["questions"][2]["criteria"] = None
        mutations.append(changed_explicit)

        changed_answer = copy.deepcopy(self.fixture)
        changed_answer["heterogeneous_request"]["answers"][3]["answer_model_dump_json"] = '{"type":"noul","noul":0.5}'
        mutations.append(changed_answer)

        for mutation in mutations:
            with self.assertRaises(BATCH.FixtureError):
                BATCH.validate_fixture(mutation)

    def test_duplicate_keys_and_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaises(BATCH.FixtureError):
            BATCH.load_json_bytes(b'{"a":1,"a":2}')
        nonfinite = copy.deepcopy(self.fixture)
        nonfinite["batch_capture"]["rows"][0]["batched_raw_logits"][0] = float("inf")
        with self.assertRaises(BATCH.FixtureError):
            BATCH.validate_fixture(nonfinite)
        with self.assertRaises(ValueError):
            BATCH.canonical_json_bytes(nonfinite)

    def test_comparator_tolerates_only_small_finite_float_changes(self) -> None:
        tolerated = BATCH.compare_json(
            {"value": 1.0},
            {"value": 1.0 + BATCH.CHECK_ABS_TOLERANCE / 2},
        )
        self.assertEqual(tolerated, ["value"])
        with self.assertRaises(BATCH.FixtureError):
            BATCH.compare_json({"value": 1.0}, {"value": 1.1})
        with self.assertRaises(BATCH.FixtureError):
            BATCH.compare_json({"value": 1}, {"value": 1.0})
        with self.assertRaises(BATCH.FixtureError):
            BATCH.compare_json({"value": float("nan")}, {"value": float("nan")})

    def test_check_and_cli_check_never_rewrite(self) -> None:
        original_path = BATCH.FIXTURE_PATH
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "fixture.json"
            fixture_path.write_bytes(self.fixture_bytes)
            BATCH.FIXTURE_PATH = fixture_path
            try:
                before = fixture_path.read_bytes()
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    BATCH.check_fixture(self.fixture_bytes)
                self.assertIn("byte-identical", output.getvalue())
                self.assertEqual(fixture_path.read_bytes(), before)

                with mock.patch.object(BATCH, "generate_fixture", return_value=self.fixture):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(BATCH.main(["--check"]), 0)
                self.assertEqual(fixture_path.read_bytes(), before)

                structural = copy.deepcopy(self.fixture)
                structural["heterogeneous_request"]["question_ids"][0] = "changed"
                with self.assertRaises(BATCH.FixtureError):
                    BATCH.check_fixture(BATCH.canonical_json_bytes(structural))
                self.assertEqual(fixture_path.read_bytes(), before)
            finally:
                BATCH.FIXTURE_PATH = original_path


if __name__ == "__main__":
    unittest.main()
