#!/usr/bin/env python3
"""Generate and verify deterministic Python-von golden vectors.

Heavy oracle imports are intentionally lazy so the fixture validator and JSON
comparator can be imported by model-free stdlib tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
HF_REPO = "wfzyx/von"
HF_REVISION = "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd"
VON_SOURCE_REVISION = "6c3c095d03ce78a546cb197df490be9a56c1ad3e"
VON_MODEL_ID = "von-1.1.0"
ARTIFACT_HASHES = {
    "option_marker.pt": "52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2",
    "model.safetensors": "349c9b1bf3d5e3217a78c8039606e07069ffae4383640998f99a96bdb6b90c8d",
    "tokenizer.json": "3703d011b3ee259e00b59b60772f8fd54998d79e8fa5a1388d507eff28f8f09e",
    "tokenizer_config.json": "57170b72a72c60407eec1f7894000c567309a8e41c5d0504b946c624cb069984",
    "config.json": "968db6072229e54691be4d535487a5c8d8132cc4893b6a3fdfa1e605457a7ec3",
    "marker_calibration.json": "c3b5345c89be9ee802a5587dcda4e530e69a9bb5965dd839b877b7e8cfff048f",
}
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.8",
    "von_sdk_distribution": "1.1.1",
    "von_module": "1.0.1",
    "torch": "2.14.0",
    "transformers": "5.17.0",
    "tokenizers": "0.23.2",
    "huggingface_hub": "1.32.0",
    "pydantic": "2.13.5",
    "pydantic_core": "2.46.5",
    "safetensors": "0.8.0",
}
PROBE_DIMENSIONS = [0, 1, 2, 3, 63, 127, 191, 255, 383, 511, 639, 767, 895, 1021, 1022, 1023]
BACKBONE_ENTRIES = ["embeddings", *[f"layer_{index}" for index in range(28)], "final_norm"]
TRANSFORMERS_HIDDEN_STATE_ENTRIES = [
    "embeddings",
    *[f"layer_{index}" for index in range(27)],
    "final_norm",
]
FLOAT_ABS_TOLERANCE = 1e-5
FLOAT_REL_TOLERANCE = 1e-6
CLS_TOKEN_ID = 50281
SEP_TOKEN_ID = 50282
MASK_TOKEN_ID = 50284
HIDDEN_SIZE = 1024
CALIBRATION_COEFFICIENTS = {
    "bias": 0.2056,
    "entropy": -3.255,
    "log_tokens": 20.2391,
    "n_options": -5.2263,
}
CALIBRATION_LOWER = 0.3
CALIBRATION_UPPER = 12.0
INTERNAL_MATH_TOLERANCE = 2e-6
EXPECTED_CASE_IDS = ["unicode_long_choice", "default_noul", "explicit_noul", "score_examples"]
EXPECTED_FORWARD_ROLES = [
    ("unicode_long_choice", "primary"),
    ("default_noul", "primary"),
    ("default_noul", "null"),
    ("explicit_noul", "primary"),
    ("score_examples", "primary"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "testdata" / "von_golden_v1.json"


class FixtureError(ValueError):
    """Raised when a fixture is invalid or differs beyond allowed tolerance."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize compact canonical UTF-8 JSON with one final newline."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def load_json_bytes(data: bytes) -> Any:
    """Parse JSON while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FixtureError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"invalid UTF-8 JSON: {exc}") from exc


def _path_child(path: str, child: str | int) -> str:
    if isinstance(child, int):
        return f"{path}[{child}]"
    return f"{path}.{child}" if path else child


def compare_json(
    expected: Any,
    actual: Any,
    *,
    abs_tolerance: float = FLOAT_ABS_TOLERANCE,
    rel_tolerance: float = FLOAT_REL_TOLERANCE,
) -> list[str]:
    """Compare JSON values strictly, except finite float leaves within tolerance.

    Object key order, array order and length, scalar JSON types, and every
    non-float value must be exact. The returned paths identify tolerated float
    differences; a non-tolerated difference raises ``FixtureError``.
    """
    tolerated: list[str] = []

    def visit(want: Any, got: Any, path: str) -> None:
        if isinstance(want, Mapping) and isinstance(got, Mapping):
            want_keys = list(want.keys())
            got_keys = list(got.keys())
            if want_keys != got_keys:
                raise FixtureError(f"{path or '$'}: object keys/order differ: {want_keys!r} != {got_keys!r}")
            for key in want_keys:
                visit(want[key], got[key], _path_child(path, key))
            return

        if (
            isinstance(want, Sequence)
            and not isinstance(want, (str, bytes, bytearray))
            and isinstance(got, Sequence)
            and not isinstance(got, (str, bytes, bytearray))
        ):
            if len(want) != len(got):
                raise FixtureError(f"{path or '$'}: array lengths differ: {len(want)} != {len(got)}")
            for index, (want_item, got_item) in enumerate(zip(want, got)):
                visit(want_item, got_item, _path_child(path, index))
            return

        if type(want) is not type(got):
            raise FixtureError(
                f"{path or '$'}: scalar types differ: {type(want).__name__} != {type(got).__name__}"
            )
        if isinstance(want, float):
            if not math.isfinite(want) or not math.isfinite(got):
                raise FixtureError(f"{path or '$'}: non-finite floats are forbidden")
            if want != got:
                if not math.isclose(want, got, rel_tol=rel_tolerance, abs_tol=abs_tolerance):
                    raise FixtureError(f"{path or '$'}: floats differ beyond tolerance: {want!r} != {got!r}")
                tolerated.append(path or "$")
            return
        if want != got:
            raise FixtureError(f"{path or '$'}: values differ: {want!r} != {got!r}")

    visit(expected, actual, "")
    return tolerated


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def _require_finite_float(value: Any, path: str) -> None:
    _require(type(value) is float and math.isfinite(value), f"{path}: expected finite float")


def _require_float_close(actual: Any, expected: float, path: str) -> None:
    _require_finite_float(actual, path)
    _require(
        math.isclose(actual, expected, rel_tol=INTERNAL_MATH_TOLERANCE, abs_tol=INTERNAL_MATH_TOLERANCE),
        f"{path}: inconsistent math: {actual!r} != {expected!r}",
    )


def _stable_softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def validate_fixture(fixture: Any) -> None:
    """Deeply validate the committed fixture using stdlib only."""
    _require(isinstance(fixture, dict), "fixture root must be an object")
    _require(fixture.get("schema_version") == SCHEMA_VERSION, "unexpected schema_version")

    metadata = fixture.get("metadata")
    _require(isinstance(metadata, dict), "metadata must be an object")
    _require(metadata.get("hf_repo") == HF_REPO, "metadata hf_repo mismatch")
    _require(metadata.get("hf_revision") == HF_REVISION, "metadata hf_revision mismatch")
    _require(metadata.get("von_source_revision") == VON_SOURCE_REVISION, "metadata von_source_revision mismatch")
    _require(metadata.get("von_model_id") == VON_MODEL_ID, "metadata von_model_id mismatch")
    _require(metadata.get("runtime_versions") == EXPECTED_RUNTIME_VERSIONS, "runtime version pins mismatch")
    _require(metadata.get("device") == "cpu", "fixture must be captured on CPU")
    _require(metadata.get("dtype") == "float32", "fixture must use float32")
    _require(metadata.get("generation_command") == ".venv/bin/python tools/golden_capture.py", "generation command mismatch")
    _require(metadata.get("probe_dimensions") == PROBE_DIMENSIONS, "probe dimensions mismatch")
    _require(metadata.get("backbone_entries") == BACKBONE_ENTRIES, "backbone entry list mismatch")
    _require(
        metadata.get("transformers_hidden_state_entries") == TRANSFORMERS_HIDDEN_STATE_ENTRIES,
        "Transformers hidden-state entry list mismatch",
    )
    tolerance = metadata.get("check_float_tolerance")
    _require(
        tolerance == {"absolute": FLOAT_ABS_TOLERANCE, "relative": FLOAT_REL_TOLERANCE},
        "float tolerance metadata mismatch",
    )

    artifacts = metadata.get("artifacts")
    _require(isinstance(artifacts, list), "metadata artifacts must be an array")
    expected_artifacts = [
        {"name": name, "sha256": digest} for name, digest in ARTIFACT_HASHES.items()
    ]
    _require(artifacts == expected_artifacts, "artifact names, order, or hashes differ")

    forbidden_key_fragments = ("timestamp", "hostname", "machine_path", "absolute_path", "home_directory")
    for key, value in metadata.items():
        lowered = key.lower()
        _require(
            not any(fragment in lowered for fragment in forbidden_key_fragments),
            f"metadata contains machine-specific key {key!r}",
        )
        if isinstance(value, str):
            _require(not value.startswith(("/", "~", "file:")), f"metadata contains machine path in {key!r}")

    cases = fixture.get("cases")
    forwards = fixture.get("forwards")
    _require(isinstance(cases, list) and len(cases) == 4, "fixture must contain four cases")
    _require(isinstance(forwards, list) and len(forwards) == 5, "fixture must contain five forwards")
    _require([case.get("id") for case in cases] == EXPECTED_CASE_IDS, "case order/ids mismatch")
    _require(
        [(forward.get("case_id"), forward.get("role")) for forward in forwards] == EXPECTED_FORWARD_ROLES,
        "forward case/role order mismatch",
    )

    forward_by_index: dict[int, dict[str, Any]] = {}
    for expected_index, forward in enumerate(forwards):
        path = f"forwards[{expected_index}]"
        _require(isinstance(forward, dict), f"{path} must be an object")
        _require(forward.get("index") == expected_index, f"{path}.index mismatch")
        forward_by_index[expected_index] = forward
        packed_text = forward.get("packed_text")
        input_ids = forward.get("input_ids")
        attention_mask = forward.get("attention_mask")
        mask_positions = forward.get("mask_positions")
        raw_logits = forward.get("raw_logits")
        backbone = forward.get("backbone")
        final_states = forward.get("final_mask_states")
        _require(isinstance(packed_text, str), f"{path}.packed_text must be a string")
        _require(isinstance(input_ids, list) and input_ids, f"{path}.input_ids must be nonempty")
        _require(all(type(item) is int for item in input_ids), f"{path}.input_ids must contain integers")
        _require(isinstance(attention_mask, list), f"{path}.attention_mask must be an array")
        _require(len(attention_mask) == len(input_ids), f"{path} token/mask lengths differ")
        _require(all(type(item) is int and item in (0, 1) for item in attention_mask), f"{path} invalid attention mask")
        _require(input_ids[0] == CLS_TOKEN_ID, f"{path}.input_ids must begin with CLS")
        _require(input_ids[-1] == SEP_TOKEN_ID, f"{path}.input_ids must end with SEP")
        _require(
            [index for index, token_id in enumerate(input_ids) if token_id == CLS_TOKEN_ID] == [0],
            f"{path}.input_ids must contain exactly one leading CLS",
        )
        separator_positions = [index for index, token_id in enumerate(input_ids) if token_id == SEP_TOKEN_ID]
        _require(
            len(separator_positions) == 2 and separator_positions[-1] == len(input_ids) - 1,
            f"{path}.input_ids must contain one packed separator and one final SEP",
        )
        _require(isinstance(mask_positions, list) and mask_positions, f"{path}.mask_positions must be nonempty")
        _require(all(type(item) is int for item in mask_positions), f"{path}.mask_positions must be integers")
        _require(mask_positions == sorted(mask_positions), f"{path}.mask_positions must be ordered")
        actual_mask_positions = [
            index for index, token_id in enumerate(input_ids) if token_id == MASK_TOKEN_ID
        ]
        _require(
            mask_positions == actual_mask_positions,
            f"{path}.mask_positions must list every mask token exactly",
        )
        _require(
            0 < separator_positions[0] < mask_positions[0],
            f"{path} packed separator must occur between CLS and the first mask",
        )
        _require(
            packed_text.count(" [SEP] ") == 1 and packed_text.count("[SEP]") == 1,
            f"{path}.packed_text must contain exactly one spaced [SEP] separator",
        )
        packed_prefix, packed_options = packed_text.split(" [SEP] ", maxsplit=1)
        _require(packed_prefix != "", f"{path}.packed_text prefix must be nonempty")
        _require(packed_options.startswith("[MASK] "), f"{path}.packed_text options must begin with [MASK]")
        option_descriptions = packed_options[len("[MASK] ") :].split(" [MASK] ")
        _require(
            len(option_descriptions) == len(mask_positions)
            and all(description != "" for description in option_descriptions),
            f"{path}.packed_text must contain one nonempty [MASK] segment per mask position",
        )
        option_count = len(mask_positions)
        _require(isinstance(raw_logits, list) and len(raw_logits) == option_count, f"{path}.raw_logits length mismatch")
        for index, value in enumerate(raw_logits):
            _require_finite_float(value, f"{path}.raw_logits[{index}]")
        _require(isinstance(backbone, list) and len(backbone) == 30, f"{path} must contain 30 backbone entries")
        _require([entry.get("name") for entry in backbone] == BACKBONE_ENTRIES, f"{path} backbone order mismatch")
        for entry_index, entry in enumerate(backbone):
            probes = entry.get("mask_state_probes")
            probe_path = f"{path}.backbone[{entry_index}].mask_state_probes"
            _require(isinstance(probes, list) and len(probes) == option_count, f"{probe_path} option count mismatch")
            for option_index, row in enumerate(probes):
                _require(isinstance(row, list) and len(row) == len(PROBE_DIMENSIONS), f"{probe_path}[{option_index}] width mismatch")
                for dim_index, value in enumerate(row):
                    _require_finite_float(value, f"{probe_path}[{option_index}][{dim_index}]")
        _require(
            backbone[-2]["mask_state_probes"] != backbone[-1]["mask_state_probes"],
            f"{path} layer_27 and final_norm probes must be distinct",
        )
        _require(isinstance(final_states, list) and len(final_states) == option_count, f"{path}.final_mask_states option count mismatch")
        for option_index, row in enumerate(final_states):
            _require(isinstance(row, list) and len(row) == HIDDEN_SIZE, f"{path}.final_mask_states[{option_index}] width mismatch")
            for dim_index, value in enumerate(row):
                _require_finite_float(value, f"{path}.final_mask_states[{option_index}][{dim_index}]")
            expected_final_probe = [row[dimension] for dimension in PROBE_DIMENSIONS]
            _require(
                backbone[-1]["mask_state_probes"][option_index] == expected_final_probe,
                f"{path}.backbone final_norm probes do not match complete final state",
            )

    for case_index, case in enumerate(cases):
        path = f"cases[{case_index}]"
        _require(isinstance(case, dict), f"{path} must be an object")
        ordered_options = case.get("ordered_options")
        _require(isinstance(ordered_options, list) and ordered_options, f"{path}.ordered_options must be a nonempty array")
        _require(
            all(
                isinstance(option, dict)
                and set(option.keys()) == {"description", "key"}
                and isinstance(option.get("key"), str)
                and isinstance(option.get("description"), str)
                for option in ordered_options
            ),
            f"{path}.ordered_options entries are invalid",
        )
        indices = case.get("forward_indices")
        _require(isinstance(indices, list) and indices, f"{path}.forward_indices must be nonempty")
        _require(all(type(index) is int and index in forward_by_index for index in indices), f"{path} invalid forward index")
        _require(all(forward_by_index[index]["case_id"] == case["id"] for index in indices), f"{path} forward ownership mismatch")
        answer_json = case.get("answer_model_dump_json")
        _require(isinstance(answer_json, str), f"{path}.answer_model_dump_json must be a string")
        answer = load_json_bytes(answer_json.encode("utf-8"))
        _require(isinstance(answer, dict), f"{path} serialized answer must be an object")
        calibration = case.get("calibration")
        _require(isinstance(calibration, dict), f"{path}.calibration must be an object")
        for key in (
            "input_logits",
            "unscaled_probabilities",
            "scaled_logits",
            "probabilities",
        ):
            values = calibration.get(key)
            _require(isinstance(values, list) and values, f"{path}.calibration.{key} must be nonempty")
            for index, value in enumerate(values):
                _require_finite_float(value, f"{path}.calibration.{key}[{index}]")
        for key in (
            "entropy_numerator",
            "entropy_normalizer",
            "normalized_entropy",
            "raw_temperature",
            "effective_temperature",
        ):
            _require_finite_float(calibration.get(key), f"{path}.calibration.{key}")
        option_count = len(ordered_options)
        _require(
            all(len(calibration[key]) == option_count for key in ("input_logits", "unscaled_probabilities", "scaled_logits", "probabilities")),
            f"{path}.calibration option counts differ",
        )
        _require(calibration.get("n_options") == option_count, f"{path}.calibration n_options mismatch")
        state_token_count = calibration.get("state_token_count")
        _require(type(state_token_count) is int and state_token_count >= 1, f"{path}.calibration state_token_count invalid")

        expected_unscaled = _stable_softmax(calibration["input_logits"])
        for index, expected in enumerate(expected_unscaled):
            _require_float_close(
                calibration["unscaled_probabilities"][index],
                expected,
                f"{path}.calibration.unscaled_probabilities[{index}]",
            )
        expected_entropy_numerator = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in calibration["unscaled_probabilities"]
        )
        expected_entropy_normalizer = math.log(option_count) if option_count > 1 else 1.0
        expected_normalized_entropy = (
            expected_entropy_numerator / expected_entropy_normalizer if option_count > 1 else 0.0
        )
        _require_float_close(
            calibration["entropy_numerator"],
            expected_entropy_numerator,
            f"{path}.calibration.entropy_numerator",
        )
        _require_float_close(
            calibration["entropy_normalizer"],
            expected_entropy_normalizer,
            f"{path}.calibration.entropy_normalizer",
        )
        _require_float_close(
            calibration["normalized_entropy"],
            expected_normalized_entropy,
            f"{path}.calibration.normalized_entropy",
        )

        expected_feature_values = {
            "bias": 1.0,
            "entropy": calibration["normalized_entropy"],
            "log_tokens": math.log10(state_token_count) / 4.0,
            "n_options": option_count / 8.0,
        }
        features = calibration.get("features")
        _require(isinstance(features, list) and len(features) == 4, f"{path}.calibration.features invalid")
        _require([feature.get("name") for feature in features] == list(CALIBRATION_COEFFICIENTS), f"{path}.calibration feature order mismatch")
        for feature_index, feature in enumerate(features):
            feature_path = f"{path}.calibration.features[{feature_index}]"
            name = feature["name"]
            _require_float_close(feature.get("value"), expected_feature_values[name], f"{feature_path}.value")
            _require(feature.get("coefficient") == CALIBRATION_COEFFICIENTS[name], f"{feature_path}.coefficient mismatch")
            _require_float_close(
                feature.get("contribution"),
                feature["coefficient"] * feature["value"],
                f"{feature_path}.contribution",
            )
        expected_raw_temperature = sum(feature["contribution"] for feature in features)
        _require_float_close(calibration["raw_temperature"], expected_raw_temperature, f"{path}.calibration.raw_temperature")
        _require(calibration.get("clamp_lower") == CALIBRATION_LOWER, f"{path}.calibration clamp_lower mismatch")
        _require(calibration.get("clamp_upper") == CALIBRATION_UPPER, f"{path}.calibration clamp_upper mismatch")
        expected_temperature = min(CALIBRATION_UPPER, max(CALIBRATION_LOWER, calibration["raw_temperature"]))
        _require_float_close(
            calibration["effective_temperature"],
            expected_temperature,
            f"{path}.calibration.effective_temperature",
        )
        expected_denominator = max(calibration["effective_temperature"], 1e-4)
        _require_float_close(
            calibration.get("temperature_denominator"),
            expected_denominator,
            f"{path}.calibration.temperature_denominator",
        )
        for index, input_logit in enumerate(calibration["input_logits"]):
            _require_float_close(
                calibration["scaled_logits"][index],
                input_logit / expected_denominator,
                f"{path}.calibration.scaled_logits[{index}]",
            )
        expected_probabilities = _stable_softmax(calibration["scaled_logits"])
        for index, expected in enumerate(expected_probabilities):
            _require_float_close(
                calibration["probabilities"][index],
                expected,
                f"{path}.calibration.probabilities[{index}]",
            )
        _require_float_close(sum(calibration["probabilities"]), 1.0, f"{path}.calibration.probability_sum")

        postprocess = case.get("postprocess")
        _require(isinstance(postprocess, dict), f"{path}.postprocess must be an object")
        primitive = case.get("primitive")
        option_keys = [option["key"] for option in ordered_options]
        rounded_probabilities = {
            key: round(probability, 4)
            for key, probability in zip(option_keys, calibration["probabilities"])
        }
        primary_raw_logits = forward_by_index[indices[0]]["raw_logits"]
        sorted_probabilities = sorted(calibration["probabilities"], reverse=True)
        confidence_unrounded = max(0.0, min(1.0, sorted_probabilities[0] - sorted_probabilities[1]))
        if primitive == "choice":
            _require(calibration["input_logits"] == primary_raw_logits, f"{path} Choice calibration/raw logits mismatch")
            _require(answer.get("type") == "choice", f"{path} answer type mismatch")
            selected_index = max(range(option_count), key=primary_raw_logits.__getitem__)
            _require(postprocess.get("selected_option_index") == selected_index, f"{path} selected option index mismatch")
            _require(postprocess.get("sorted_probabilities") == sorted_probabilities, f"{path} sorted probabilities mismatch")
            _require_float_close(postprocess.get("confidence_unrounded"), confidence_unrounded, f"{path}.postprocess.confidence_unrounded")
            _require(answer.get("choice") == option_keys[selected_index], f"{path} answer choice mismatch")
            _require(answer.get("probabilities") == rounded_probabilities, f"{path} answer probabilities mismatch")
            _require(answer.get("confidence") == round(confidence_unrounded, 3), f"{path} answer confidence mismatch")
        elif primitive == "noul":
            _require(answer.get("type") == "noul", f"{path} answer type mismatch")
            probability_true = postprocess.get("probability_true_unrounded")
            _require_finite_float(probability_true, f"{path}.postprocess.probability_true_unrounded")
            _require(answer.get("noul") == round(max(0.0, min(1.0, probability_true)), 4), f"{path} answer Noul mismatch")
            _require(calibration["probabilities"][0] == probability_true, f"{path} Noul probability mismatch")
            if postprocess.get("debias_applied"):
                _require(len(indices) == 2, f"{path} default Noul must have a null forward")
                _require(postprocess.get("debias_factor") == 0.7, f"{path} default Noul debias factor mismatch")
                null_logits = postprocess.get("null_raw_logits")
                stored_primary = postprocess.get("primary_raw_logits")
                actual_null = forward_by_index[indices[1]]["raw_logits"]
                _require(stored_primary == primary_raw_logits, f"{path} stored primary logits differ from forward")
                _require(null_logits == actual_null, f"{path} stored null logits differ from forward")
                _require(isinstance(null_logits, list) and len(null_logits) == 2, f"{path} null logits invalid")
                _require(isinstance(stored_primary, list) and len(stored_primary) == 2, f"{path} primary logits invalid")
                expected_null_bias = null_logits[0] - null_logits[1]
                _require_float_close(postprocess.get("null_bias"), expected_null_bias, f"{path}.postprocess.null_bias")
                expected_adjustment = 0.7 * postprocess["null_bias"]
                _require_float_close(
                    postprocess.get("debias_adjustment"),
                    expected_adjustment,
                    f"{path}.postprocess.debias_adjustment",
                )
                adjusted_logits = postprocess.get("adjusted_logits")
                _require(isinstance(adjusted_logits, list) and len(adjusted_logits) == 2, f"{path} adjusted logits invalid")
                _require_float_close(
                    adjusted_logits[0],
                    primary_raw_logits[0] - postprocess["debias_adjustment"],
                    f"{path}.postprocess.adjusted_logits[0]",
                )
                _require_float_close(
                    adjusted_logits[1],
                    primary_raw_logits[1],
                    f"{path}.postprocess.adjusted_logits[1]",
                )
                _require(adjusted_logits == calibration["input_logits"], f"{path} adjusted/calibration logits mismatch")
            else:
                _require(len(indices) == 1, f"{path} explicit Noul unexpectedly has a null forward")
                _require(calibration["input_logits"] == primary_raw_logits, f"{path} explicit Noul calibration/raw logits mismatch")
        elif primitive == "score":
            _require(calibration["input_logits"] == primary_raw_logits, f"{path} Score calibration/raw logits mismatch")
            _require(answer.get("type") == "score", f"{path} answer type mismatch")
            _require(answer.get("probabilities") == rounded_probabilities, f"{path} answer probabilities mismatch")
            _require(
                answer.get("legend") == {option["key"]: option["description"] for option in ordered_options},
                f"{path} answer legend mismatch",
            )
            _require(postprocess.get("sorted_probabilities") == sorted_probabilities, f"{path} sorted probabilities mismatch")
            _require_float_close(postprocess.get("confidence_unrounded"), confidence_unrounded, f"{path}.postprocess.confidence_unrounded")
            _require(answer.get("confidence") == round(confidence_unrounded, 3), f"{path} answer confidence mismatch")
            weighted_terms = postprocess.get("weighted_terms")
            _require(
                weighted_terms == [float(index * probability) for index, probability in enumerate(calibration["probabilities"])],
                f"{path} weighted score terms mismatch",
            )
            expected_score = sum(weighted_terms)
            _require_float_close(postprocess.get("score_unrounded"), expected_score, f"{path}.postprocess.score_unrounded")
            _require(answer.get("score") == round(expected_score, 2), f"{path} answer score mismatch")
        else:
            raise FixtureError(f"{path}: unknown primitive {primitive!r}")

    _require(len(forwards[0]["input_ids"]) > 128, "Unicode Choice packed sequence must exceed 128 tokens")
    _require(cases[0]["forward_indices"] == [0], "Choice must use one forward")
    _require(cases[1]["forward_indices"] == [1, 2], "default Noul must use primary and null forwards")
    _require(cases[2]["forward_indices"] == [3], "explicit Noul must use one forward")
    _require(cases[3]["forward_indices"] == [4], "Score must use one forward")
    _require(cases[1]["postprocess"].get("debias_factor") == 0.7, "default Noul must use 0.7 debias")
    _require(cases[2]["postprocess"].get("debias_applied") is False, "explicit Noul must not be debiased")
    _require(
        all(" Examples: " in item["description"] for item in cases[3]["ordered_options"]),
        "Score options must preserve production example rendering",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path() -> Path:
    if cache_override := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hub_cache = Path(cache_override).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
        hub_cache = hf_home / "hub"
    return hub_cache / "models--wfzyx--von" / "snapshots" / HF_REVISION


def _von_source_path() -> Path:
    override = os.environ.get("VON_SOURCE_DIR")
    return Path(override).expanduser() if override else REPO_ROOT.parent / "von"


def validate_local_inputs(snapshot: Path, source: Path) -> None:
    _require(snapshot.is_dir(), f"pinned local Hugging Face snapshot is missing: {HF_REVISION}")
    for name, expected in ARTIFACT_HASHES.items():
        path = snapshot / name
        _require(path.is_file(), f"pinned artifact is missing: {name}")
        actual = _sha256(path)
        _require(actual == expected, f"SHA256 mismatch for {name}: {actual} != {expected}")

    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FixtureError("could not inspect the local von source clone") from exc
    _require(revision == VON_SOURCE_REVISION, f"von source revision mismatch: {revision} != {VON_SOURCE_REVISION}")


def validate_runtime_sources(source: Path) -> None:
    """Tie imported production modules to blobs at the pinned source commit."""
    import von.backends.option_marker_backend as backend_module
    import von.models.option_marker as model_module
    import von.types as types_module

    runtime_and_git_paths = [
        (Path(backend_module.__file__).resolve(), "src/von/backends/option_marker_backend.py"),
        (Path(model_module.__file__).resolve(), "src/von/models/option_marker.py"),
        (Path(types_module.__file__).resolve(), "src/von/types.py"),
    ]
    for runtime_path, git_path in runtime_and_git_paths:
        try:
            committed_bytes = subprocess.run(
                ["git", "-C", str(source), "show", f"{VON_SOURCE_REVISION}:{git_path}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FixtureError(f"could not read pinned von source blob: {git_path}") from exc
        runtime_digest = hashlib.sha256(runtime_path.read_bytes()).digest()
        committed_digest = hashlib.sha256(committed_bytes).digest()
        _require(
            runtime_digest == committed_digest,
            f"runtime von module does not match pinned source commit: {git_path}",
        )


def _tensor_rows(tensor: Any, mask_positions: list[int], dimensions: list[int] | None = None) -> list[list[float]]:
    selected = tensor[0, mask_positions, :]
    if dimensions is not None:
        selected = selected[:, dimensions]
    return [[float(value) for value in row] for row in selected.detach().cpu().tolist()]


class ForwardRecorder:
    """Capture the single production model's real forward calls and hooks."""

    def __init__(self, model: Any):
        self.model = model
        self.forwards: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._pending_packed: list[str] = []
        self._current: dict[str, Any] | None = None
        self._case_id: str | None = None
        self._roles: list[str] = []
        self._role_index = 0
        self._original_pack = model.pack_sequence
        self._original_output_hidden_states = model.encoder.config.output_hidden_states

    @contextmanager
    def installed(self) -> Iterator["ForwardRecorder"]:
        def pack_sequence(state: str, question: str, options: list[str]) -> str:
            packed = self._original_pack(state, question, options)
            self._pending_packed.append(packed)
            return packed

        self.model.pack_sequence = pack_sequence
        self.model.encoder.config.output_hidden_states = True
        self._handles.append(self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        self._handles.append(self.model.register_forward_hook(self._model_post_hook, with_kwargs=True))
        self._handles.append(self.model.encoder.register_forward_hook(self._encoder_output_hook))
        self._handles.append(self.model.encoder.embeddings.register_forward_hook(self._hidden_hook("embeddings")))
        for index in range(28):
            self._handles.append(self.model.encoder.layers[index].register_forward_hook(self._hidden_hook(f"layer_{index}")))
        self._handles.append(self.model.encoder.final_norm.register_forward_hook(self._hidden_hook("final_norm", final=True)))
        try:
            yield self
        finally:
            self.model.pack_sequence = self._original_pack
            self.model.encoder.config.output_hidden_states = self._original_output_hidden_states
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._current = None

    @contextmanager
    def case(self, case_id: str, roles: list[str]) -> Iterator[None]:
        _require(self._case_id is None, "nested capture cases are not supported")
        self._case_id = case_id
        self._roles = roles
        self._role_index = 0
        try:
            yield
            _require(self._role_index == len(roles), f"{case_id}: expected {len(roles)} forwards, saw {self._role_index}")
        finally:
            self._case_id = None
            self._roles = []
            self._role_index = 0

    def _model_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        _require(self._current is None, "overlapping model forwards are not supported")
        _require(self._case_id is not None, "model forward occurred outside a capture case")
        _require(self._role_index < len(self._roles), f"unexpected extra forward for {self._case_id}")
        _require(bool(self._pending_packed), "model forward has no recorded packed text")
        input_ids = kwargs["input_ids"].detach().cpu()
        attention_mask = kwargs["attention_mask"].detach().cpu()
        mask_positions = [int(position) for position in kwargs["mask_positions"][0]]
        _require(tuple(input_ids.shape[:1]) == (1,), "golden capture expects batch size one")
        self._current = {
            "index": len(self.forwards),
            "case_id": self._case_id,
            "role": self._roles[self._role_index],
            "packed_text": self._pending_packed.pop(0),
            "input_ids": [int(value) for value in input_ids[0].tolist()],
            "attention_mask": [int(value) for value in attention_mask[0].tolist()],
            "mask_positions": mask_positions,
            "raw_logits": [],
            "backbone": [],
            "final_mask_states": [],
        }
        self._role_index += 1

    def _hidden_hook(self, name: str, *, final: bool = False):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            _require(self._current is not None, f"{name} hook ran outside a model forward")
            tensor = output[0] if isinstance(output, tuple) else output
            positions = self._current["mask_positions"]
            self._current["backbone"].append(
                {
                    "name": name,
                    "mask_state_probes": _tensor_rows(tensor, positions, PROBE_DIMENSIONS),
                }
            )
            if final:
                self._current["final_mask_states"] = _tensor_rows(tensor, positions)

        return hook

    def _encoder_output_hook(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        _require(self._current is not None, "encoder output hook ran outside a model forward")
        hidden_states = output.hidden_states
        _require(hidden_states is not None, "Transformers did not return public hidden states")
        _require(
            len(hidden_states) == len(TRANSFORMERS_HIDDEN_STATE_ENTRIES),
            "unexpected Transformers public hidden-state count",
        )
        expected_entries = [*self._current["backbone"][:28], self._current["backbone"][-1]]
        _require(
            [entry["name"] for entry in expected_entries] == TRANSFORMERS_HIDDEN_STATE_ENTRIES,
            "internal public hidden-state mapping is invalid",
        )
        positions = self._current["mask_positions"]
        for index, (hidden_state, expected) in enumerate(zip(hidden_states, expected_entries)):
            actual_probes = _tensor_rows(hidden_state, positions, PROBE_DIMENSIONS)
            _require(
                actual_probes == expected["mask_state_probes"],
                f"Transformers public hidden state {index} does not match {expected['name']}",
            )

    def _model_post_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        _require(self._current is not None, "model post-hook ran without a current forward")
        _require(len(output) == 1, "golden capture expects batch size one output")
        self._current["raw_logits"] = [float(value) for value in output[0].detach().cpu().tolist()]
        _require(
            [entry["name"] for entry in self._current["backbone"]] == BACKBONE_ENTRIES,
            "backbone hook order/count mismatch",
        )
        self.forwards.append(self._current)
        self._current = None


class CalibrationRecorder:
    """Record the production temperature call and expose its unrounded math."""

    def __init__(self, backend: Any, torch_module: Any):
        self.backend = backend
        self.torch = torch_module
        self.records: dict[str, dict[str, Any]] = {}
        self._case_id: str | None = None
        self._original = backend._effective_temperature

    @contextmanager
    def installed(self) -> Iterator["CalibrationRecorder"]:
        def effective_temperature(
            logits: Any,
            state_text: str,
            n_options: int,
            tokenizer: Any,
            override: float | None = None,
        ) -> float:
            _require(self._case_id is not None, "calibration occurred outside a capture case")
            _require(self._case_id not in self.records, f"duplicate calibration for {self._case_id}")
            result = self._original(logits, state_text, n_options, tokenizer, override)
            probs = self.torch.softmax(logits.float(), dim=-1)
            count = max(probs.numel(), 1)
            if count > 1:
                entropy_numerator = -(probs * self.torch.log(probs.clamp_min(1e-12))).sum().item()
                entropy_normalizer = math.log(count)
                normalized_entropy = entropy_numerator / entropy_normalizer
            else:
                entropy_numerator = 0.0
                entropy_normalizer = 1.0
                normalized_entropy = 0.0
            state_token_count = max(len(tokenizer.encode(state_text, add_special_tokens=False)), 1)
            values = {
                "bias": 1.0,
                "entropy": normalized_entropy,
                "log_tokens": math.log10(state_token_count) / 4.0,
                "n_options": n_options / 8.0,
            }
            params = self.backend._calib_map
            _require(params is not None, "expected the pinned input-conditioned calibration map")
            features = []
            for name, value in values.items():
                coefficient = params.get(name, 0.0)
                features.append(
                    {
                        "name": name,
                        "value": float(value),
                        "coefficient": float(coefficient),
                        "contribution": float(coefficient * value),
                    }
                )
            raw_temperature = sum(item["contribution"] for item in features)
            clamped = min(params["hi"], max(params["lo"], raw_temperature)) if override is None else override
            _require(result == clamped, f"production temperature math mismatch for {self._case_id}")
            denominator = max(result, 1e-4)
            scaled = logits / denominator
            calibrated_probs = self.torch.softmax(scaled, dim=-1)
            self.records[self._case_id] = {
                "input_logits": [float(value) for value in logits.detach().cpu().tolist()],
                "unscaled_probabilities": [float(value) for value in probs.detach().cpu().tolist()],
                "entropy_numerator": float(entropy_numerator),
                "entropy_normalizer": float(entropy_normalizer),
                "normalized_entropy": float(normalized_entropy),
                "state_token_count": int(state_token_count),
                "n_options": int(n_options),
                "features": features,
                "raw_temperature": float(raw_temperature),
                "clamp_lower": float(params["lo"]),
                "clamp_upper": float(params["hi"]),
                "effective_temperature": float(result),
                "temperature_denominator": float(denominator),
                "scaled_logits": [float(value) for value in scaled.detach().cpu().tolist()],
                "probabilities": [float(value) for value in calibrated_probs.detach().cpu().tolist()],
            }
            return result

        self.backend._effective_temperature = effective_temperature
        try:
            yield self
        finally:
            self.backend._effective_temperature = self._original
            self._case_id = None

    @contextmanager
    def case(self, case_id: str) -> Iterator[None]:
        _require(self._case_id is None, "nested calibration cases are not supported")
        self._case_id = case_id
        try:
            yield
            _require(case_id in self.records, f"{case_id}: production calibration was not called")
        finally:
            self._case_id = None


def _case_specs(types_module: Any) -> list[dict[str, Any]]:
    Choice = types_module.Choice
    Noul = types_module.Noul
    Score = types_module.Score

    telemetry = " ".join(
        f"Shard {index:02d}: réplica latency={120 + index * 7}ms, Δ={index - 9}, status=degradé;"
        for index in range(20)
    )
    choice_state = (
        "Incident résumé 🚨: café checkout failures affect 東京 and São Paulo; "
        "the decomposed label is cafe\u0301. " + telemetry
    )
    return [
        {
            "id": "unicode_long_choice",
            "q_id": "root_cause",
            "primitive": "choice",
            "state": choice_state,
            "question": Choice(
                instructions="Classify the primary root-cause domain; preserve multilingual evidence.",
                criteria={
                    "infrastructure": "Database, hardware, réseau, 東京 region, or server failures",
                    "billing": "Invoices, payments, refunds, or subscription queries",
                    "feature_request": "Requests for new platform capabilities or UX changes",
                    "security": "Authentication, abuse, credential, or privacy incidents",
                },
            ),
            "roles": ["primary"],
        },
        {
            "id": "default_noul",
            "q_id": "urgent_payment_failure",
            "primitive": "noul",
            "state": "Payment API errors have persisted for 18 minutes and checkout retries are exhausted.",
            "question": Noul(instructions="Is this an urgent or critical payment failure?"),
            "roles": ["primary", "null"],
        },
        {
            "id": "explicit_noul",
            "q_id": "rollback_required",
            "primitive": "noul",
            "state": "Error rate returned to baseline after the canary was disabled; no active alerts remain.",
            "question": Noul(
                instructions="Should the release be rolled back now?",
                criteria={
                    "true": "Rollback is still required to stop active customer impact.",
                    "false": "Rollback is not required because customer impact has ended.",
                },
            ),
            "roles": ["primary"],
        },
        {
            "id": "score_examples",
            "q_id": "incident_severity",
            "primitive": "score",
            "state": "Two regions return HTTP 500 for most writes; reads remain available and no data loss is observed.",
            "question": Score(
                instructions="Rate the current incident severity on this ordered scale.",
                criteria=[
                    {"what": "No user impact", "examples": ["cosmetic typo", "internal-only warning"]},
                    {"what": "Limited degradation", "examples": ["one slow shard", "brief retry spike"]},
                    {"what": "Major incident", "examples": ["multi-region errors", "sustained outage"]},
                    {"what": "Critical outage", "examples": ["all requests fail", "data loss is occurring"]},
                ],
            ),
            "roles": ["primary"],
        },
    ]


def _ordered_options(spec: dict[str, Any]) -> list[dict[str, str]]:
    question = spec["question"]
    if spec["primitive"] == "choice":
        return [
            {"key": key, "description": value.strip() if value else key.strip()}
            for key, value in question.criteria.items()
        ]
    if spec["primitive"] == "noul":
        criteria = question.criteria or {}
        positive = criteria.get("true") or "Yes, condition holds true."
        negative = criteria.get("false") or "No, condition is false."
        return [
            {"key": "true", "description": positive},
            {"key": "false", "description": negative},
        ]

    rendered: list[dict[str, str]] = []
    for index, item in enumerate(question.criteria):
        if isinstance(item, dict):
            examples = item.get("examples", [])
            suffix = f" Examples: {', '.join(examples)}" if examples else ""
            description = f"{item.get('what', '')}{suffix}".strip()
        else:
            description = str(item).strip()
        rendered.append({"key": str(index), "description": description})
    return rendered


def _postprocess(
    spec: dict[str, Any],
    forward_indices: list[int],
    forwards: list[dict[str, Any]],
    calibration: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    primitive = spec["primitive"]
    probabilities = calibration["probabilities"]
    if primitive == "choice":
        sorted_probabilities = sorted(probabilities, reverse=True)
        confidence = max(0.0, min(1.0, sorted_probabilities[0] - sorted_probabilities[1]))
        raw_logits = forwards[forward_indices[0]]["raw_logits"]
        return {
            "selected_option_index": int(max(range(len(raw_logits)), key=raw_logits.__getitem__)),
            "sorted_probabilities": sorted_probabilities,
            "confidence_unrounded": float(confidence),
        }
    if primitive == "score":
        weighted_terms = [float(index * probability) for index, probability in enumerate(probabilities)]
        sorted_probabilities = sorted(probabilities, reverse=True)
        confidence = max(0.0, min(1.0, sorted_probabilities[0] - sorted_probabilities[1]))
        return {
            "weighted_terms": weighted_terms,
            "score_unrounded": float(sum(weighted_terms)),
            "sorted_probabilities": sorted_probabilities,
            "confidence_unrounded": float(confidence),
        }
    if len(forward_indices) == 1:
        return {
            "debias_applied": False,
            "probability_true_unrounded": float(probabilities[0]),
        }

    primary = torch_module.tensor(forwards[forward_indices[0]]["raw_logits"], dtype=torch_module.float32)
    null = torch_module.tensor(forwards[forward_indices[1]]["raw_logits"], dtype=torch_module.float32)
    null_bias = null[0] - null[1]
    debias_adjustment = torch_module.tensor(0.7, dtype=torch_module.float32) * null_bias
    adjusted = torch_module.stack([primary[0] - debias_adjustment, primary[1]])
    adjusted_values = [float(value) for value in adjusted.tolist()]
    _require(adjusted_values == calibration["input_logits"], "captured default Noul debias does not match production logits")
    return {
        "debias_applied": True,
        "debias_factor": 0.7,
        "primary_raw_logits": [float(value) for value in primary.tolist()],
        "null_raw_logits": [float(value) for value in null.tolist()],
        "null_bias": float(null_bias.item()),
        "debias_adjustment": float(debias_adjustment.item()),
        "adjusted_logits": adjusted_values,
        "probability_true_unrounded": float(probabilities[0]),
    }


def generate_fixture() -> dict[str, Any]:
    """Load the pinned oracle once and capture all Phase 1 golden vectors."""
    expected_prefix = REPO_ROOT / ".venv"
    _require(
        Path(sys.prefix).resolve() == expected_prefix.resolve(),
        "run the capture with the repository interpreter: .venv/bin/python",
    )
    snapshot = _snapshot_path()
    source = _von_source_path()
    validate_local_inputs(snapshot, source)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import pydantic
    import safetensors
    import torch
    import transformers
    import von
    import von.types as types_module
    from von.backends.option_marker_backend import OptionMarkerBackend, VON_MODEL_ID as runtime_model_id

    runtime_versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "von_sdk_distribution": package_version("von-sdk"),
        "von_module": str(von.__version__),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "tokenizers": package_version("tokenizers"),
        "huggingface_hub": package_version("huggingface-hub"),
        "pydantic": package_version("pydantic"),
        "pydantic_core": package_version("pydantic-core"),
        "safetensors": package_version("safetensors"),
    }
    _require(runtime_versions == EXPECTED_RUNTIME_VERSIONS, f"runtime version mismatch: {runtime_versions!r}")
    _require(runtime_model_id == VON_MODEL_ID, f"von model id mismatch: {runtime_model_id!r}")
    _require(str(torch.__version__) == runtime_versions["torch"], "Torch module/distribution versions differ")
    _require(
        str(transformers.__version__) == runtime_versions["transformers"],
        "Transformers module/distribution versions differ",
    )
    _require(str(pydantic.__version__) == runtime_versions["pydantic"], "Pydantic module/distribution versions differ")
    _require(
        str(safetensors.__version__) == runtime_versions["safetensors"],
        "safetensors module/distribution versions differ",
    )
    validate_runtime_sources(source)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    backend = OptionMarkerBackend(checkpoint_dir=str(snapshot), device="cpu")
    model = backend._get_model()
    model_identity = id(model)
    _require(model.training is False, "production model must be in eval mode")
    for name, parameter in model.named_parameters():
        _require(parameter.device.type == "cpu", f"parameter {name} is not on CPU")
        if parameter.is_floating_point():
            _require(parameter.dtype == torch.float32, f"parameter {name} is not float32")

    cases: list[dict[str, Any]] = []
    recorder = ForwardRecorder(model)
    calibration_recorder = CalibrationRecorder(backend, torch)
    specs = _case_specs(types_module)

    with recorder.installed(), calibration_recorder.installed(), torch.inference_mode():
        for spec in specs:
            start = len(recorder.forwards)
            with recorder.case(spec["id"], spec["roles"]), calibration_recorder.case(spec["id"]):
                if spec["primitive"] == "choice":
                    answer = backend.evaluate_choice(spec["q_id"], spec["state"], spec["question"])
                elif spec["primitive"] == "noul":
                    answer = backend.evaluate_noul(spec["q_id"], spec["state"], spec["question"])
                else:
                    answer = backend.evaluate_score(spec["q_id"], spec["state"], spec["question"])
            stop = len(recorder.forwards)
            forward_indices = list(range(start, stop))
            calibration = calibration_recorder.records[spec["id"]]
            ordered_options = _ordered_options(spec)
            cases.append(
                {
                    "id": spec["id"],
                    "q_id": spec["q_id"],
                    "primitive": spec["primitive"],
                    "state": spec["state"],
                    "instructions": spec["question"].instructions,
                    "ordered_options": ordered_options,
                    "forward_indices": forward_indices,
                    "calibration": calibration,
                    "postprocess": _postprocess(spec, forward_indices, recorder.forwards, calibration, torch),
                    "answer_model_dump_json": answer.model_dump_json(),
                }
            )

    _require(id(backend._get_model()) == model_identity, "backend replaced the single loaded model")
    _require(not recorder._pending_packed, "unconsumed packed text remains after capture")
    _require(len(recorder.forwards[0]["input_ids"]) > 128, "Unicode Choice did not exceed 128 packed tokens")
    _require(
        [item["description"] for item in cases[3]["ordered_options"]]
        == [
            "No user impact Examples: cosmetic typo, internal-only warning",
            "Limited degradation Examples: one slow shard, brief retry spike",
            "Major incident Examples: multi-region errors, sustained outage",
            "Critical outage Examples: all requests fail, data loss is occurring",
        ],
        "Score production rendering changed",
    )

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "generator": "tools/golden_capture.py",
            "generation_command": ".venv/bin/python tools/golden_capture.py",
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "torch_num_threads": 1,
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "von_source_revision": VON_SOURCE_REVISION,
            "von_model_id": VON_MODEL_ID,
            "runtime_versions": runtime_versions,
            "artifacts": [
                {"name": name, "sha256": digest} for name, digest in ARTIFACT_HASHES.items()
            ],
            "probe_dimensions": PROBE_DIMENSIONS,
            "backbone_entries": BACKBONE_ENTRIES,
            "transformers_hidden_state_entries": TRANSFORMERS_HIDDEN_STATE_ENTRIES,
            "check_float_tolerance": {
                "absolute": FLOAT_ABS_TOLERANCE,
                "relative": FLOAT_REL_TOLERANCE,
            },
        },
        "cases": cases,
        "forwards": recorder.forwards,
    }
    validate_fixture(fixture)
    return fixture


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_fixture(data: bytes) -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIXTURE_PATH.with_suffix(FIXTURE_PATH.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(FIXTURE_PATH)


def check_fixture(generated: bytes) -> None:
    fixture_label = _display_path(FIXTURE_PATH)
    _require(FIXTURE_PATH.is_file(), f"fixture does not exist: {fixture_label}")
    committed = FIXTURE_PATH.read_bytes()
    if committed == generated:
        print(f"golden fixture is byte-identical: {fixture_label}")
        return

    expected_value = load_json_bytes(committed)
    generated_value = load_json_bytes(generated)
    validate_fixture(expected_value)
    validate_fixture(generated_value)
    tolerated = compare_json(expected_value, generated_value)
    _require(bool(tolerated), "fixture bytes differ despite identical parsed JSON; regenerate canonical JSON")
    print(
        "golden fixture differs only at "
        f"{len(tolerated)} finite float leaves within abs={FLOAT_ABS_TOLERANCE:g}, "
        f"rel={FLOAT_REL_TOLERANCE:g}; fixture was not rewritten"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="capture in memory and compare without rewriting the committed fixture",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        fixture = generate_fixture()
        data = canonical_json_bytes(fixture)
        if args.check:
            check_fixture(data)
        else:
            write_fixture(data)
            print(f"wrote deterministic golden fixture: {FIXTURE_PATH.relative_to(REPO_ROOT)}")
        return 0
    except (FixtureError, OSError, RuntimeError) as exc:
        print(f"golden capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
