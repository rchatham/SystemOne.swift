#!/usr/bin/env python3
"""Capture and verify Python-von heterogeneous batch oracle vectors.

Heavy oracle imports are intentionally lazy so fixture validation and tests use
only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PAD_TOKEN_ID = 50283
EXPECTED_SEQUENCE_LENGTHS = [541, 42, 27, 53, 87]
EXPECTED_ROW_ORDER = [
    ("unicode_long_choice", "primary"),
    ("default_noul", "primary"),
    ("default_noul", "null"),
    ("explicit_noul", "primary"),
    ("score_examples", "primary"),
]
EXPECTED_QUESTION_IDS = [
    "root_cause",
    "urgent_payment_failure",
    "rollback_required",
    "incident_severity",
]
EXPECTED_PRIMITIVES = ["choice", "noul", "noul", "score"]
SOURCE_FIXTURE_SHA256 = "8c364c01e27d01244f10f03caac55cfdde39465fc1320b403803100c32a817e7"
BATCH_ABS_TOLERANCE = 1e-4
BATCH_REL_TOLERANCE = 1e-4
CHECK_ABS_TOLERANCE = 1e-5
CHECK_REL_TOLERANCE = 1e-6
RELATIVE_ERROR_FLOOR = 1e-12

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "testdata" / "von_batch_golden_v1.json"
SOURCE_FIXTURE_PATH = REPO_ROOT / "testdata" / "von_golden_v1.json"
GOLDEN_CAPTURE_PATH = REPO_ROOT / "tools" / "golden_capture.py"

SHARED_STATE = (
    "Checkout writes fail in two regions after a release; retries are exhausted, "
    "reads remain available, and disabling the canary restored the error rate."
)
QUESTION_DEFINITIONS = [
    {
        "id": "root_cause",
        "type": "choice",
        "instructions": "Classify the primary root-cause domain.",
        "criteria": {
            "infrastructure": "Database, hardware, network, region, or server failures",
            "billing": "Invoices, payments, refunds, or subscription queries",
            "feature_request": "Requests for new platform capabilities or UX changes",
            "security": "Authentication, abuse, credential, or privacy incidents",
        },
        "ordered_criteria": [
            {
                "key": "infrastructure",
                "description": "Database, hardware, network, region, or server failures",
            },
            {
                "key": "billing",
                "description": "Invoices, payments, refunds, or subscription queries",
            },
            {
                "key": "feature_request",
                "description": "Requests for new platform capabilities or UX changes",
            },
            {
                "key": "security",
                "description": "Authentication, abuse, credential, or privacy incidents",
            },
        ],
    },
    {
        "id": "urgent_payment_failure",
        "type": "noul",
        "instructions": "Is this an urgent or critical payment failure?",
        "criteria": None,
    },
    {
        "id": "rollback_required",
        "type": "noul",
        "instructions": "Should the release be rolled back now?",
        "criteria": {
            "true": "Rollback is still required to stop active customer impact.",
            "false": "Rollback is not required because customer impact has ended.",
        },
    },
    {
        "id": "incident_severity",
        "type": "score",
        "instructions": "Rate the current incident severity on this ordered scale.",
        "criteria": [
            {"what": "No user impact", "examples": ["cosmetic typo", "internal-only warning"]},
            {"what": "Limited degradation", "examples": ["one slow shard", "brief retry spike"]},
            {"what": "Major incident", "examples": ["multi-region errors", "sustained outage"]},
            {"what": "Critical outage", "examples": ["all requests fail", "data loss is occurring"]},
        ],
    },
]


class FixtureError(ValueError):
    """Raised when the batch fixture is invalid or out of tolerance."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def _load_golden_module() -> Any:
    spec = importlib.util.spec_from_file_location("_von_golden_capture", GOLDEN_CAPTURE_PATH)
    _require(spec is not None and spec.loader is not None, "could not load golden capture helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize compact canonical UTF-8 JSON with one final newline."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_json_bytes(data: bytes) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate object keys."""

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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_fixture() -> dict[str, Any]:
    try:
        data = SOURCE_FIXTURE_PATH.read_bytes()
    except OSError as exc:
        raise FixtureError("source fixture is unavailable") from exc
    digest = _sha256_bytes(data)
    _require(digest == SOURCE_FIXTURE_SHA256, f"source fixture SHA256 mismatch: {digest}")
    value = load_json_bytes(data)
    _require(isinstance(value, dict), "source fixture root must be an object")
    return value


def _finite_float(value: Any, path: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{path}: expected finite float")
    return value


def _error_metrics(sequential: list[float], batched: list[float]) -> tuple[float, float]:
    _require(len(sequential) == len(batched) and bool(sequential), "logit arrays must have equal nonzero length")
    absolute = [abs(left - right) for left, right in zip(sequential, batched)]
    relative = [error / max(abs(left), RELATIVE_ERROR_FLOOR) for left, error in zip(sequential, absolute)]
    return max(absolute), max(relative)


def _close(actual: float, expected: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def validate_fixture(fixture: Any) -> None:
    """Deeply validate the committed fixture without model dependencies."""
    _require(isinstance(fixture, dict), "fixture root must be an object")
    _require(
        set(fixture) == {"schema_version", "metadata", "batch_capture", "heterogeneous_request"},
        "fixture root keys mismatch",
    )
    _require(fixture["schema_version"] == SCHEMA_VERSION, "unexpected schema_version")

    source = _source_fixture()
    source_forwards = source.get("forwards")
    _require(isinstance(source_forwards, list) and len(source_forwards) == 5, "source forwards mismatch")

    metadata = fixture["metadata"]
    _require(isinstance(metadata, dict), "metadata must be an object")
    golden = _load_golden_module()
    expected_metadata = {
        "generator": "batch_golden_capture.py",
        "generation_command": ".venv/bin/python tools/batch_golden_capture.py",
        "device": "cpu",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "torch_num_threads": 1,
        "hf_repo": golden.HF_REPO,
        "hf_revision": golden.HF_REVISION,
        "von_source_revision": golden.VON_SOURCE_REVISION,
        "von_model_id": golden.VON_MODEL_ID,
        "runtime_versions": golden.EXPECTED_RUNTIME_VERSIONS,
        "artifacts": [{"name": name, "sha256": digest} for name, digest in golden.ARTIFACT_HASHES.items()],
        "source_fixture": {"name": "von_golden_v1.json", "sha256": SOURCE_FIXTURE_SHA256},
        "pad_token_id": PAD_TOKEN_ID,
        "padding_side": "right",
        "batch_size": 5,
        "maximum_sequence_length": 541,
        "sequence_lengths": EXPECTED_SEQUENCE_LENGTHS,
        "batch_drift_tolerance": {"absolute": BATCH_ABS_TOLERANCE, "relative": BATCH_REL_TOLERANCE},
        "check_float_tolerance": {"absolute": CHECK_ABS_TOLERANCE, "relative": CHECK_REL_TOLERANCE},
    }
    _require(metadata == expected_metadata, "metadata mismatch")
    serialized_metadata = json.dumps(metadata, ensure_ascii=False)
    _require(str(Path.home()) not in serialized_metadata, "metadata contains a home path")
    _require("timestamp" not in serialized_metadata.lower(), "metadata contains a timestamp")

    capture = fixture["batch_capture"]
    _require(isinstance(capture, dict), "batch_capture must be an object")
    _require(capture.get("padded_shape") == [5, 541], "padded shape mismatch")
    rows = capture.get("rows")
    _require(isinstance(rows, list) and len(rows) == 5, "batch rows mismatch")

    row_absolute_errors: list[float] = []
    row_relative_errors: list[float] = []
    for index, (row, source_forward, identity, length) in enumerate(
        zip(rows, source_forwards, EXPECTED_ROW_ORDER, EXPECTED_SEQUENCE_LENGTHS)
    ):
        path = f"batch_capture.rows[{index}]"
        _require(isinstance(row, dict), f"{path} must be an object")
        case_id, role = identity
        _require(row.get("index") == index, f"{path}.index mismatch")
        _require(row.get("case_id") == case_id and row.get("role") == role, f"{path} identity mismatch")
        _require(source_forward.get("case_id") == case_id and source_forward.get("role") == role, f"{path} source identity mismatch")
        _require(row.get("sequence_length") == length, f"{path}.sequence_length mismatch")
        _require(len(source_forward.get("input_ids", [])) == length, f"{path} source sequence length mismatch")
        _require(source_forward.get("attention_mask") == [1] * length, f"{path} source attention mask mismatch")
        _require(row.get("padding_length") == 541 - length, f"{path}.padding_length mismatch")
        _require(row.get("mask_positions") == source_forward.get("mask_positions"), f"{path}.mask_positions mismatch")

        sequential = row.get("sequential_raw_logits")
        batched = row.get("batched_raw_logits")
        option_count = len(source_forward.get("mask_positions", []))
        _require(isinstance(sequential, list) and len(sequential) == option_count, f"{path} sequential logits mismatch")
        _require(isinstance(batched, list) and len(batched) == option_count, f"{path} batched logits mismatch")
        for logit_index, value in enumerate(sequential):
            _finite_float(value, f"{path}.sequential_raw_logits[{logit_index}]")
        for logit_index, value in enumerate(batched):
            _finite_float(value, f"{path}.batched_raw_logits[{logit_index}]")

        absolute_error, relative_error = _error_metrics(sequential, batched)
        stored_absolute = _finite_float(row.get("max_absolute_error"), f"{path}.max_absolute_error")
        stored_relative = _finite_float(row.get("max_relative_error"), f"{path}.max_relative_error")
        _require(_close(stored_absolute, absolute_error), f"{path}.max_absolute_error inconsistent")
        _require(_close(stored_relative, relative_error), f"{path}.max_relative_error inconsistent")
        _require(absolute_error <= BATCH_ABS_TOLERANCE, f"{path} absolute drift exceeds tolerance")
        _require(relative_error <= BATCH_REL_TOLERANCE, f"{path} relative drift exceeds tolerance")
        row_absolute_errors.append(absolute_error)
        row_relative_errors.append(relative_error)

    maximum_absolute = _finite_float(capture.get("max_absolute_error"), "batch_capture.max_absolute_error")
    maximum_relative = _finite_float(capture.get("max_relative_error"), "batch_capture.max_relative_error")
    _require(_close(maximum_absolute, max(row_absolute_errors)), "aggregate max_absolute_error inconsistent")
    _require(_close(maximum_relative, max(row_relative_errors)), "aggregate max_relative_error inconsistent")

    request = fixture["heterogeneous_request"]
    _require(isinstance(request, dict), "heterogeneous_request must be an object")
    _require(request.get("state") == SHARED_STATE, "heterogeneous state mismatch")
    _require(request.get("question_ids") == EXPECTED_QUESTION_IDS, "question order mismatch")
    _require(request.get("questions") == QUESTION_DEFINITIONS, "question definitions mismatch")
    _require(request.get("model") == golden.VON_MODEL_ID, "response model mismatch")
    answers = request.get("answers")
    _require(isinstance(answers, list) and len(answers) == 4, "heterogeneous answers mismatch")
    for index, (answer, question_id, primitive) in enumerate(zip(answers, EXPECTED_QUESTION_IDS, EXPECTED_PRIMITIVES)):
        path = f"heterogeneous_request.answers[{index}]"
        _require(isinstance(answer, dict), f"{path} must be an object")
        _require(answer.get("question_id") == question_id, f"{path} question id mismatch")
        serialized = answer.get("answer_model_dump_json")
        _require(isinstance(serialized, str), f"{path} serialized answer missing")
        parsed = load_json_bytes(serialized.encode("utf-8"))
        _require(isinstance(parsed, dict) and parsed.get("type") == primitive, f"{path} primitive mismatch")
        _require(
            serialized == json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
            f"{path} answer JSON is not compact",
        )
    usage = request.get("usage")
    _require(isinstance(usage, dict), "heterogeneous usage must be an object")
    expected_input_tokens = max(1, len(SHARED_STATE) // 4) + max(
        1, sum(len(question["instructions"]) for question in QUESTION_DEFINITIONS) // 4
    )
    _require(usage == {"input_tokens": expected_input_tokens, "output_tokens": 4}, "usage mismatch")


def _path_child(path: str, child: str | int) -> str:
    return f"{path}[{child}]" if isinstance(child, int) else (f"{path}.{child}" if path else child)


def compare_json(
    expected: Any,
    actual: Any,
    *,
    abs_tolerance: float = CHECK_ABS_TOLERANCE,
    rel_tolerance: float = CHECK_REL_TOLERANCE,
) -> list[str]:
    """Compare structure exactly and finite float leaves within tolerance."""
    tolerated: list[str] = []

    def visit(want: Any, got: Any, path: str) -> None:
        if isinstance(want, Mapping) and isinstance(got, Mapping):
            _require(list(want) == list(got), f"{path or '$'}: object keys/order differ")
            for key in want:
                visit(want[key], got[key], _path_child(path, key))
            return
        if (
            isinstance(want, Sequence)
            and not isinstance(want, (str, bytes, bytearray))
            and isinstance(got, Sequence)
            and not isinstance(got, (str, bytes, bytearray))
        ):
            _require(len(want) == len(got), f"{path or '$'}: array lengths differ")
            for index, (want_item, got_item) in enumerate(zip(want, got)):
                visit(want_item, got_item, _path_child(path, index))
            return
        _require(type(want) is type(got), f"{path or '$'}: scalar types differ")
        if isinstance(want, float):
            _require(math.isfinite(want) and math.isfinite(got), f"{path or '$'}: non-finite float")
            if want != got:
                _require(
                    math.isclose(want, got, abs_tol=abs_tolerance, rel_tol=rel_tolerance),
                    f"{path or '$'}: floats differ beyond tolerance",
                )
                tolerated.append(path or "$")
            return
        _require(want == got, f"{path or '$'}: values differ")

    visit(expected, actual, "")
    return tolerated


def _question_objects(types_module: Any) -> dict[str, Any]:
    constructors = {
        "choice": types_module.Choice,
        "noul": types_module.Noul,
        "score": types_module.Score,
    }
    questions: dict[str, Any] = {}
    for definition in QUESTION_DEFINITIONS:
        values = {key: value for key, value in definition.items() if key != "id"}
        questions[definition["id"]] = constructors[definition["type"]](**values)
    return questions


def generate_fixture() -> dict[str, Any]:
    """Load the pinned production model and capture batch and request oracles."""
    golden = _load_golden_module()
    expected_prefix = REPO_ROOT / ".venv"
    _require(Path(sys.prefix).resolve() == expected_prefix.resolve(), "run with .venv/bin/python")
    snapshot = golden._snapshot_path()
    source_path = golden._von_source_path()
    golden.validate_local_inputs(snapshot, source_path)
    source_fixture = _source_fixture()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import von
    import von.types as types_module
    from von.backends.option_marker_backend import OptionMarkerBackend

    runtime_versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "von_sdk_distribution": golden.package_version("von-sdk"),
        "von_module": str(von.__version__),
        "torch": golden.package_version("torch"),
        "transformers": golden.package_version("transformers"),
        "tokenizers": golden.package_version("tokenizers"),
        "huggingface_hub": golden.package_version("huggingface-hub"),
        "pydantic": golden.package_version("pydantic"),
        "pydantic_core": golden.package_version("pydantic-core"),
        "safetensors": golden.package_version("safetensors"),
    }
    _require(runtime_versions == golden.EXPECTED_RUNTIME_VERSIONS, f"runtime version mismatch: {runtime_versions!r}")
    golden.validate_runtime_sources(source_path)

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    backend = OptionMarkerBackend(checkpoint_dir=str(snapshot), device="cpu")
    model = backend._get_model()
    _require(model.training is False, "production model must be in eval mode")
    _require(model.tokenizer.padding_side == "right", "production tokenizer is not right-padding")
    _require(model.tokenizer.pad_token_id == PAD_TOKEN_ID, "production PAD token mismatch")
    for name, parameter in model.named_parameters():
        _require(parameter.device.type == "cpu", f"parameter {name} is not on CPU")
        if parameter.is_floating_point():
            _require(parameter.dtype == torch.float32, f"parameter {name} is not float32")

    forwards = source_fixture["forwards"]
    input_ids = torch.full((5, 541), PAD_TOKEN_ID, dtype=torch.long, device="cpu")
    attention_mask = torch.zeros((5, 541), dtype=torch.long, device="cpu")
    mask_positions: list[list[int]] = []
    for index, (forward, length) in enumerate(zip(forwards, EXPECTED_SEQUENCE_LENGTHS)):
        tokens = forward["input_ids"]
        _require(len(tokens) == length, f"source row {index} length changed")
        input_ids[index, :length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[index, :length] = 1
        mask_positions.append(forward["mask_positions"])
        _require(input_ids[index, length:].eq(PAD_TOKEN_ID).all().item(), f"row {index} is not right padded")
        _require(attention_mask[index, length:].eq(0).all().item(), f"row {index} padded mask is not zero")

    sequential_logits: list[list[float]] = []
    with torch.inference_mode():
        for index, length in enumerate(EXPECTED_SEQUENCE_LENGTHS):
            outputs = model(
                input_ids=input_ids[index : index + 1, :length],
                attention_mask=attention_mask[index : index + 1, :length],
                mask_positions=[mask_positions[index]],
            )
            sequential_logits.append([float(value) for value in outputs[0].detach().cpu().tolist()])
        batch_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
        )
    batched_logits = [[float(value) for value in output.detach().cpu().tolist()] for output in batch_outputs]

    rows: list[dict[str, Any]] = []
    for index, ((case_id, role), length, positions, sequential, batched) in enumerate(
        zip(EXPECTED_ROW_ORDER, EXPECTED_SEQUENCE_LENGTHS, mask_positions, sequential_logits, batched_logits)
    ):
        absolute_error, relative_error = _error_metrics(sequential, batched)
        rows.append(
            {
                "index": index,
                "case_id": case_id,
                "role": role,
                "sequence_length": length,
                "padding_length": 541 - length,
                "mask_positions": positions,
                "sequential_raw_logits": sequential,
                "batched_raw_logits": batched,
                "max_absolute_error": absolute_error,
                "max_relative_error": relative_error,
            }
        )

    response = backend.evaluate(SHARED_STATE, _question_objects(types_module), model=golden.VON_MODEL_ID)
    _require(list(response.answers) == EXPECTED_QUESTION_IDS, "backend changed heterogeneous answer order")
    answers = [
        {"question_id": question_id, "answer_model_dump_json": response.answers[question_id].model_dump_json()}
        for question_id in EXPECTED_QUESTION_IDS
    ]

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "generator": "batch_golden_capture.py",
            "generation_command": ".venv/bin/python tools/batch_golden_capture.py",
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "torch_num_threads": 1,
            "hf_repo": golden.HF_REPO,
            "hf_revision": golden.HF_REVISION,
            "von_source_revision": golden.VON_SOURCE_REVISION,
            "von_model_id": golden.VON_MODEL_ID,
            "runtime_versions": runtime_versions,
            "artifacts": [{"name": name, "sha256": digest} for name, digest in golden.ARTIFACT_HASHES.items()],
            "source_fixture": {"name": "von_golden_v1.json", "sha256": SOURCE_FIXTURE_SHA256},
            "pad_token_id": PAD_TOKEN_ID,
            "padding_side": "right",
            "batch_size": 5,
            "maximum_sequence_length": 541,
            "sequence_lengths": EXPECTED_SEQUENCE_LENGTHS,
            "batch_drift_tolerance": {"absolute": BATCH_ABS_TOLERANCE, "relative": BATCH_REL_TOLERANCE},
            "check_float_tolerance": {"absolute": CHECK_ABS_TOLERANCE, "relative": CHECK_REL_TOLERANCE},
        },
        "batch_capture": {
            "padded_shape": [5, 541],
            "rows": rows,
            "max_absolute_error": max(row["max_absolute_error"] for row in rows),
            "max_relative_error": max(row["max_relative_error"] for row in rows),
        },
        "heterogeneous_request": {
            "state": SHARED_STATE,
            "question_ids": EXPECTED_QUESTION_IDS,
            "questions": QUESTION_DEFINITIONS,
            "model": response.model,
            "answers": answers,
            "usage": response.usage.model_dump(),
        },
    }
    validate_fixture(fixture)
    return fixture


def write_fixture(data: bytes) -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIXTURE_PATH.with_suffix(FIXTURE_PATH.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(FIXTURE_PATH)


def check_fixture(generated: bytes) -> None:
    _require(FIXTURE_PATH.is_file(), "batch fixture does not exist")
    committed = FIXTURE_PATH.read_bytes()
    if committed == generated:
        print("batch golden fixture is byte-identical: testdata/von_batch_golden_v1.json")
        return
    expected_value = load_json_bytes(committed)
    generated_value = load_json_bytes(generated)
    validate_fixture(expected_value)
    validate_fixture(generated_value)
    tolerated = compare_json(expected_value, generated_value)
    _require(bool(tolerated), "fixture bytes differ despite identical parsed JSON; regenerate canonical JSON")
    print(
        f"batch golden fixture differs only at {len(tolerated)} finite float leaves within "
        f"abs={CHECK_ABS_TOLERANCE:g}, rel={CHECK_REL_TOLERANCE:g}; fixture was not rewritten"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="capture and compare without rewriting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = canonical_json_bytes(generate_fixture())
        if args.check:
            check_fixture(data)
        else:
            write_fixture(data)
            print("wrote deterministic batch golden fixture: testdata/von_batch_golden_v1.json")
        return 0
    except (FixtureError, OSError, RuntimeError) as exc:
        print(f"batch golden capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
