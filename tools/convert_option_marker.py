#!/usr/bin/env python3
"""Deterministically convert the pinned von OptionMarker checkpoint.

PyTorch and safetensors imports are intentionally lazy. Importing this module,
validating the committed JSON fixtures, and inspecting the expected tensor
schema therefore require only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version as package_version
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 1
HF_REPO = "wfzyx/von"
HF_REVISION = "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd"
SOURCE_ARTIFACT_NAME = "option_marker.pt"
SOURCE_SHA256 = "52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2"
OUTPUT_ARTIFACT_NAME = "option_marker.safetensors"
TENSOR_METADATA_ARTIFACT_NAME = "option_marker_tensor_metadata_v1.json"
EXPECTED_PAYLOAD_BYTES = 1_581_240_324
EXPECTED_TENSOR_COUNT = 178
EXPECTED_ENCODER_TENSOR_COUNT = 170
EXPECTED_SCORER_TENSOR_COUNT = 8
DTYPE = "F32"
DTYPE_BYTES = 4
EXPECTED_RUNTIME_VERSIONS = {"safetensors": "0.8.0", "torch": "2.14.0"}
# safetensors 0.8.0 stores user metadata through a Rust HashMap whose JSON key
# order varies between writes. Keep provenance in the canonical external
# manifest instead so identical tensors always produce identical bytes.
EMBEDDED_METADATA: None = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "weights" / OUTPUT_ARTIFACT_NAME
TENSOR_METADATA_PATH = REPO_ROOT / "testdata" / TENSOR_METADATA_ARTIFACT_NAME
MANIFEST_PATH = REPO_ROOT / "testdata" / "option_marker_weights_manifest_v1.json"


class ConversionError(ValueError):
    """Raised when source, output, or fixture validation fails."""


class TensorSpec(NamedTuple):
    name: str
    shape: tuple[int, ...]

    @property
    def byte_count(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * DTYPE_BYTES


class OutputInspection(NamedTuple):
    sha256: str
    file_size: int
    header_length: int
    tensor_records: list[dict[str, Any]]


def _expected_schema() -> tuple[TensorSpec, ...]:
    specs = [
        TensorSpec("encoder.embeddings.tok_embeddings.weight", (50368, 1024)),
        TensorSpec("encoder.embeddings.norm.weight", (1024,)),
    ]
    for layer in range(28):
        prefix = f"encoder.layers.{layer}"
        if layer > 0:
            specs.append(TensorSpec(f"{prefix}.attn_norm.weight", (1024,)))
        specs.extend(
            [
                TensorSpec(f"{prefix}.attn.Wqkv.weight", (3072, 1024)),
                TensorSpec(f"{prefix}.attn.Wo.weight", (1024, 1024)),
                TensorSpec(f"{prefix}.mlp_norm.weight", (1024,)),
                TensorSpec(f"{prefix}.mlp.Wi.weight", (5248, 1024)),
                TensorSpec(f"{prefix}.mlp.Wo.weight", (1024, 2624)),
            ]
        )
    specs.extend(
        [
            TensorSpec("encoder.final_norm.weight", (1024,)),
            TensorSpec("scorer.input_norm.weight", (1024,)),
            TensorSpec("scorer.input_norm.bias", (1024,)),
            TensorSpec("scorer.dense.weight", (512, 1024)),
            TensorSpec("scorer.dense.bias", (512,)),
            TensorSpec("scorer.norm.weight", (512,)),
            TensorSpec("scorer.norm.bias", (512,)),
            TensorSpec("scorer.out_proj.weight", (1, 512)),
            TensorSpec("scorer.out_proj.bias", (1,)),
        ]
    )
    return tuple(specs)


EXPECTED_SCHEMA = _expected_schema()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def _validate_expected_schema() -> None:
    names = [spec.name for spec in EXPECTED_SCHEMA]
    _require(len(names) == EXPECTED_TENSOR_COUNT, "internal tensor count is invalid")
    _require(len(names) == len(set(names)), "internal tensor names are not unique")
    _require(
        sum(name.startswith("encoder.") for name in names) == EXPECTED_ENCODER_TENSOR_COUNT,
        "internal encoder tensor count is invalid",
    )
    _require(
        sum(name.startswith("scorer.") for name in names) == EXPECTED_SCORER_TENSOR_COUNT,
        "internal scorer tensor count is invalid",
    )
    _require(
        sum(spec.byte_count for spec in EXPECTED_SCHEMA) == EXPECTED_PAYLOAD_BYTES,
        "internal payload byte count is invalid",
    )


_validate_expected_schema()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize compact, canonical UTF-8 JSON with one final newline."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def load_json_bytes(data: bytes) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConversionError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid UTF-8 JSON: {exc}") from exc


def sha256_file(path: Path, *, offset: int = 0, length: int | None = None) -> str:
    """Hash an entire file or an exact byte range without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if offset:
            handle.seek(offset)
        remaining = length
        while remaining is None or remaining > 0:
            chunk_size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if length is not None:
        _require(remaining == 0, f"file ended before the requested {length}-byte range")
    return digest.hexdigest()


def validate_runtime_versions() -> None:
    """Require the conversion libraries that produced the canonical bytes."""
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        try:
            actual = package_version(distribution)
        except PackageNotFoundError as exc:
            raise ConversionError(f"required conversion dependency is missing: {distribution}") from exc
        _require(actual == expected, f"{distribution} version mismatch: {actual} != {expected}")


def default_source_path() -> Path:
    """Return the pinned local Hugging Face snapshot path without networking."""
    if cache_override := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hub_cache = Path(cache_override).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
        hub_cache = hf_home / "hub"
    return hub_cache / "models--wfzyx--von" / "snapshots" / HF_REVISION / SOURCE_ARTIFACT_NAME


def load_source_state(source: Path, *, expected_sha256: str = SOURCE_SHA256) -> Mapping[str, Any]:
    """Verify the source bytes before invoking the restricted PyTorch loader."""
    _require(source.is_file(), f"source checkpoint is missing: {source}")
    actual_sha256 = sha256_file(source)
    _require(
        actual_sha256 == expected_sha256,
        f"source SHA256 mismatch: {actual_sha256} != {expected_sha256}",
    )

    validate_runtime_versions()
    import torch

    state = torch.load(source, weights_only=True, map_location="cpu", mmap=True)
    _require(isinstance(state, Mapping), "source checkpoint must contain a state-dict mapping")
    return state


def validate_source_tensors(
    state: Mapping[str, Any],
    *,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
) -> OrderedDict[str, Any]:
    """Reject every state-dict deviation and return tensors in schema order."""
    import torch

    expected_names = [spec.name for spec in schema]
    actual_names = list(state.keys())
    missing = [name for name in expected_names if name not in state]
    extra = [name for name in actual_names if name not in set(expected_names)]
    _require(not missing, f"source checkpoint is missing tensors: {missing}")
    _require(not extra, f"source checkpoint has unexpected tensors: {extra}")
    _require(len(actual_names) == len(expected_names), "source checkpoint tensor count mismatch")

    ordered: OrderedDict[str, Any] = OrderedDict()
    for spec in schema:
        tensor = state[spec.name]
        _require(torch.is_tensor(tensor), f"{spec.name}: value is not a tensor")
        _require(tensor.device.type == "cpu", f"{spec.name}: tensor is not on CPU")
        _require(tensor.dtype == torch.float32, f"{spec.name}: expected torch.float32, got {tensor.dtype}")
        _require(tuple(tensor.shape) == spec.shape, f"{spec.name}: shape {tuple(tensor.shape)} != {spec.shape}")
        _require(tensor.is_contiguous(), f"{spec.name}: tensor is not contiguous")
        _require(
            tensor.numel() * tensor.element_size() == spec.byte_count,
            f"{spec.name}: tensor byte count mismatch",
        )
        ordered[spec.name] = tensor
    return ordered


def _parse_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    _require(path.is_file(), f"safetensors artifact is missing: {path}")
    file_size = path.stat().st_size
    _require(file_size >= 8, "safetensors artifact is shorter than its length prefix")
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little", signed=False)
        _require(header_length > 0, "safetensors header length must be positive")
        _require(8 + header_length <= file_size, "safetensors header extends beyond the file")
        header_bytes = handle.read(header_length)
    header = load_json_bytes(header_bytes)
    _require(isinstance(header, dict), "safetensors header must be an object")
    return header_length, header


def _header_tensor_entries(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        _require(isinstance(name, str) and isinstance(value, dict), "invalid safetensors tensor header entry")
        entries[name] = value
    return entries


def _validate_header(
    path: Path,
    schema: Sequence[TensorSpec],
    expected_metadata: Mapping[str, str] | None,
) -> tuple[int, dict[str, dict[str, Any]]]:
    header_length, header = _parse_safetensors_header(path)
    if expected_metadata is None:
        _require("__metadata__" not in header, "safetensors must not contain nondeterministic embedded metadata")
    else:
        _require(header.get("__metadata__") == dict(expected_metadata), "embedded safetensors metadata mismatch")
    entries = _header_tensor_entries(header)
    expected_names = [spec.name for spec in schema]
    _require(set(entries) == set(expected_names), "safetensors tensor names do not match the expected schema")

    intervals: list[tuple[int, int, str]] = []
    for spec in schema:
        entry = entries[spec.name]
        _require(set(entry) == {"data_offsets", "dtype", "shape"}, f"{spec.name}: unexpected header fields")
        _require(entry.get("dtype") == DTYPE, f"{spec.name}: expected safetensors dtype {DTYPE}")
        _require(entry.get("shape") == list(spec.shape), f"{spec.name}: safetensors shape mismatch")
        offsets = entry.get("data_offsets")
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(value) is int for value in offsets),
            f"{spec.name}: invalid data_offsets",
        )
        start, end = offsets
        _require(0 <= start <= end, f"{spec.name}: invalid data offset range")
        _require(end - start == spec.byte_count, f"{spec.name}: data range byte count mismatch")
        intervals.append((start, end, spec.name))

    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        _require(start == cursor, f"{name}: tensor payload has a gap or overlap at offset {cursor}")
        cursor = end
    expected_payload = sum(spec.byte_count for spec in schema)
    _require(cursor == expected_payload, "safetensors payload byte count mismatch")
    _require(
        path.stat().st_size == 8 + header_length + expected_payload,
        "safetensors file size does not equal header plus payload",
    )
    return header_length, entries


def inspect_safetensors(
    path: Path,
    *,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
    expected_metadata: Mapping[str, str] | None = EMBEDDED_METADATA,
    source_tensors: Mapping[str, Any] | None = None,
) -> OutputInspection:
    """Validate layout and tensor data through safe_open, then record hashes."""
    validate_runtime_versions()
    import torch
    from safetensors import safe_open

    header_length, entries = _validate_header(path, schema, expected_metadata)
    records: list[dict[str, Any]] = []
    data_start = 8 + header_length

    with safe_open(path, framework="pt", device="cpu") as handle:
        _require(set(handle.keys()) == {spec.name for spec in schema}, "safe_open tensor names mismatch")
        safe_metadata = None if expected_metadata is None else dict(expected_metadata)
        _require(handle.metadata() == safe_metadata, "safe_open metadata mismatch")
        for spec in schema:
            tensor = handle.get_tensor(spec.name)
            _require(tensor.dtype == torch.float32, f"{spec.name}: safe_open dtype mismatch")
            _require(tuple(tensor.shape) == spec.shape, f"{spec.name}: safe_open shape mismatch")
            _require(tensor.is_contiguous(), f"{spec.name}: safe_open tensor is not contiguous")
            if source_tensors is not None:
                _require(torch.equal(tensor, source_tensors[spec.name]), f"{spec.name}: saved tensor data differs")
            start, end = entries[spec.name]["data_offsets"]
            records.append(
                {
                    "byte_count": spec.byte_count,
                    "data_offsets": [start, end],
                    "dtype": DTYPE,
                    "name": spec.name,
                    "sha256": sha256_file(path, offset=data_start + start, length=end - start),
                    "shape": list(spec.shape),
                }
            )

    return OutputInspection(
        sha256=sha256_file(path),
        file_size=path.stat().st_size,
        header_length=header_length,
        tensor_records=records,
    )


def build_tensor_metadata_fixture(
    inspection: OutputInspection,
    *,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
) -> dict[str, Any]:
    names = [spec.name for spec in schema]
    encoder_count = sum(name.startswith("encoder.") for name in names)
    scorer_count = sum(name.startswith("scorer.") for name in names)
    return {
        "artifact_name": OUTPUT_ARTIFACT_NAME,
        "dtype": DTYPE,
        "encoder_tensor_count": encoder_count,
        "payload_bytes": sum(spec.byte_count for spec in schema),
        "schema_version": SCHEMA_VERSION,
        "scorer_tensor_count": scorer_count,
        "tensor_count": len(schema),
        "tensors": inspection.tensor_records,
    }


def build_manifest(
    inspection: OutputInspection,
    tensor_metadata_bytes: bytes,
    *,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
) -> dict[str, Any]:
    payload_bytes = sum(spec.byte_count for spec in schema)
    return {
        "conversion": {
            "command": ".venv/bin/python tools/convert_option_marker.py --update-fixtures",
            "embedded_metadata": EMBEDDED_METADATA,
            "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
            "tool": "tools/convert_option_marker.py",
            "torch_load": {"map_location": "cpu", "mmap": True, "weights_only": True},
            "writer": "safetensors.torch.save_file",
        },
        "output": {
            "artifact_name": OUTPUT_ARTIFACT_NAME,
            "data_start": 8 + inspection.header_length,
            "dtype": DTYPE,
            "file_size": inspection.file_size,
            "format": "safetensors",
            "header_length": inspection.header_length,
            "payload_bytes": payload_bytes,
            "sha256": inspection.sha256,
            "tensor_count": len(schema),
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "artifact_name": SOURCE_ARTIFACT_NAME,
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "sha256": SOURCE_SHA256,
        },
        "tensor_metadata": {
            "artifact_name": TENSOR_METADATA_ARTIFACT_NAME,
            "sha256": hashlib.sha256(tensor_metadata_bytes).hexdigest(),
        },
    }


def validate_tensor_metadata_fixture(
    fixture: Any,
    *,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
) -> None:
    _require(isinstance(fixture, dict), "tensor metadata fixture must be an object")
    expected_names = [spec.name for spec in schema]
    _require(fixture.get("schema_version") == SCHEMA_VERSION, "tensor metadata schema version mismatch")
    _require(fixture.get("artifact_name") == OUTPUT_ARTIFACT_NAME, "tensor metadata artifact name mismatch")
    _require(fixture.get("dtype") == DTYPE, "tensor metadata dtype mismatch")
    _require(fixture.get("tensor_count") == len(schema), "tensor metadata count mismatch")
    _require(
        fixture.get("encoder_tensor_count") == sum(name.startswith("encoder.") for name in expected_names),
        "tensor metadata encoder count mismatch",
    )
    _require(
        fixture.get("scorer_tensor_count") == sum(name.startswith("scorer.") for name in expected_names),
        "tensor metadata scorer count mismatch",
    )
    _require(
        fixture.get("payload_bytes") == sum(spec.byte_count for spec in schema),
        "tensor metadata payload byte count mismatch",
    )
    records = fixture.get("tensors")
    _require(isinstance(records, list) and len(records) == len(schema), "tensor metadata records mismatch")
    _require(
        [record.get("name") for record in records if isinstance(record, dict)] == expected_names,
        "tensor metadata names or order mismatch",
    )

    intervals: list[tuple[int, int, str]] = []
    for spec, record in zip(schema, records):
        _require(isinstance(record, dict), f"{spec.name}: tensor metadata record must be an object")
        _require(
            set(record) == {"byte_count", "data_offsets", "dtype", "name", "sha256", "shape"},
            f"{spec.name}: tensor metadata fields mismatch",
        )
        _require(record["name"] == spec.name, f"{spec.name}: tensor metadata name mismatch")
        _require(record["dtype"] == DTYPE, f"{spec.name}: tensor metadata dtype mismatch")
        _require(record["shape"] == list(spec.shape), f"{spec.name}: tensor metadata shape mismatch")
        _require(record["byte_count"] == spec.byte_count, f"{spec.name}: tensor metadata byte count mismatch")
        digest = record["sha256"]
        _require(
            isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            f"{spec.name}: invalid tensor SHA256",
        )
        offsets = record["data_offsets"]
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(value) is int for value in offsets),
            f"{spec.name}: invalid tensor metadata offsets",
        )
        start, end = offsets
        _require(end - start == spec.byte_count, f"{spec.name}: tensor metadata range mismatch")
        intervals.append((start, end, spec.name))

    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        _require(start == cursor, f"{name}: tensor metadata ranges have a gap or overlap")
        cursor = end
    _require(cursor == sum(spec.byte_count for spec in schema), "tensor metadata ranges do not cover payload")


def validate_manifest(
    manifest: Any,
    tensor_metadata_bytes: bytes,
    *,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
) -> None:
    _require(isinstance(manifest, dict), "weights manifest must be an object")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "weights manifest schema version mismatch")
    _require(
        manifest.get("source")
        == {
            "artifact_name": SOURCE_ARTIFACT_NAME,
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "sha256": SOURCE_SHA256,
        },
        "weights manifest source provenance mismatch",
    )
    expected_conversion = {
        "command": ".venv/bin/python tools/convert_option_marker.py --update-fixtures",
        "embedded_metadata": EMBEDDED_METADATA,
        "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
        "tool": "tools/convert_option_marker.py",
        "torch_load": {"map_location": "cpu", "mmap": True, "weights_only": True},
        "writer": "safetensors.torch.save_file",
    }
    _require(manifest.get("conversion") == expected_conversion, "weights manifest conversion metadata mismatch")
    _require(
        manifest.get("tensor_metadata")
        == {
            "artifact_name": TENSOR_METADATA_ARTIFACT_NAME,
            "sha256": hashlib.sha256(tensor_metadata_bytes).hexdigest(),
        },
        "weights manifest tensor metadata hash mismatch",
    )
    output = manifest.get("output")
    _require(isinstance(output, dict), "weights manifest output must be an object")
    _require(
        set(output)
        == {
            "artifact_name",
            "data_start",
            "dtype",
            "file_size",
            "format",
            "header_length",
            "payload_bytes",
            "sha256",
            "tensor_count",
        },
        "weights manifest output fields mismatch",
    )
    _require(output["artifact_name"] == OUTPUT_ARTIFACT_NAME, "weights manifest output name mismatch")
    _require(output["format"] == "safetensors", "weights manifest output format mismatch")
    _require(output["dtype"] == DTYPE, "weights manifest output dtype mismatch")
    _require(output["tensor_count"] == len(schema), "weights manifest output tensor count mismatch")
    _require(
        output["payload_bytes"] == sum(spec.byte_count for spec in schema),
        "weights manifest output payload mismatch",
    )
    _require(
        type(output["header_length"]) is int and output["header_length"] > 0,
        "weights manifest header length is invalid",
    )
    _require(output["data_start"] == 8 + output["header_length"], "weights manifest data start mismatch")
    _require(
        output["file_size"] == output["data_start"] + output["payload_bytes"],
        "weights manifest file size mismatch",
    )
    digest = output["sha256"]
    _require(
        isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
        "weights manifest output SHA256 is invalid",
    )


def load_and_validate_fixtures(
    *,
    tensor_metadata_path: Path = TENSOR_METADATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    _require(tensor_metadata_path.is_file(), f"tensor metadata fixture is missing: {tensor_metadata_path}")
    _require(manifest_path.is_file(), f"weights manifest is missing: {manifest_path}")
    tensor_metadata_bytes = tensor_metadata_path.read_bytes()
    tensor_metadata = load_json_bytes(tensor_metadata_bytes)
    manifest = load_json_bytes(manifest_path.read_bytes())
    _require(
        tensor_metadata_bytes == canonical_json_bytes(tensor_metadata),
        "tensor metadata fixture is not canonical JSON",
    )
    _require(manifest_path.read_bytes() == canonical_json_bytes(manifest), "weights manifest is not canonical JSON")
    validate_tensor_metadata_fixture(tensor_metadata, schema=schema)
    validate_manifest(manifest, tensor_metadata_bytes, schema=schema)
    return tensor_metadata, tensor_metadata_bytes, manifest


def _validate_inspection_against_fixtures(
    inspection: OutputInspection,
    tensor_metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    output = manifest["output"]
    _require(inspection.sha256 == output["sha256"], "output SHA256 does not match the committed manifest")
    _require(inspection.file_size == output["file_size"], "output file size does not match the committed manifest")
    _require(inspection.header_length == output["header_length"], "output header length does not match the manifest")
    _require(inspection.tensor_records == tensor_metadata["tensors"], "output tensor metadata does not match fixture")


def check_output(
    output: Path,
    *,
    tensor_metadata_path: Path = TENSOR_METADATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    schema: Sequence[TensorSpec] = EXPECTED_SCHEMA,
    expected_metadata: Mapping[str, str] | None = EMBEDDED_METADATA,
) -> OutputInspection:
    """Validate an existing artifact without modifying any file."""
    tensor_metadata, _, manifest = load_and_validate_fixtures(
        tensor_metadata_path=tensor_metadata_path,
        manifest_path=manifest_path,
        schema=schema,
    )
    _require(output.is_file(), f"converted output is missing: {output}")
    actual_sha256 = sha256_file(output)
    _require(
        actual_sha256 == manifest["output"]["sha256"],
        f"output SHA256 mismatch: {actual_sha256} != {manifest['output']['sha256']}",
    )
    inspection = inspect_safetensors(output, schema=schema, expected_metadata=expected_metadata)
    _validate_inspection_against_fixtures(inspection, tensor_metadata, manifest)
    return inspection


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_candidate(tensors: Mapping[str, Any], output: Path) -> Path:
    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.fchmod(descriptor, 0o644)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary)
        temporary.chmod(0o644)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def convert(
    source: Path,
    output: Path,
    *,
    update_fixtures: bool = False,
    tensor_metadata_path: Path = TENSOR_METADATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> OutputInspection:
    """Convert, fully validate, and atomically install the pinned artifact."""
    state = load_source_state(source)
    tensors = validate_source_tensors(state)
    candidate = _save_candidate(tensors, output)
    try:
        inspection = inspect_safetensors(candidate, source_tensors=tensors)
        tensor_metadata = build_tensor_metadata_fixture(inspection)
        tensor_metadata_bytes = canonical_json_bytes(tensor_metadata)
        manifest = build_manifest(inspection, tensor_metadata_bytes)
        manifest_bytes = canonical_json_bytes(manifest)
        validate_tensor_metadata_fixture(tensor_metadata)
        validate_manifest(manifest, tensor_metadata_bytes)

        if not update_fixtures:
            committed_metadata, _, committed_manifest = load_and_validate_fixtures(
                tensor_metadata_path=tensor_metadata_path,
                manifest_path=manifest_path,
            )
            _validate_inspection_against_fixtures(inspection, committed_metadata, committed_manifest)

        os.replace(candidate, output)
        if update_fixtures:
            _atomic_write(tensor_metadata_path, tensor_metadata_bytes)
            _atomic_write(manifest_path, manifest_bytes)
        return inspection
    finally:
        candidate.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source_path(),
        help="pinned option_marker.pt path (default: local Hugging Face snapshot)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="destination safetensors path (default: weights/option_marker.safetensors)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate an existing output against committed fixtures; write nothing",
    )
    parser.add_argument(
        "--update-fixtures",
        action="store_true",
        help="replace the canonical metadata fixtures after a successful pinned conversion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.check and args.update_fixtures:
        parser.error("--check and --update-fixtures are mutually exclusive")

    try:
        if args.check:
            inspection = check_output(args.output)
            action = "verified"
        else:
            inspection = convert(args.source, args.output, update_fixtures=args.update_fixtures)
            action = "converted"
    except (ConversionError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{action} {OUTPUT_ARTIFACT_NAME}: "
        f"sha256={inspection.sha256} size={inspection.file_size} "
        f"tensors={len(inspection.tensor_records)} payload={sum(item['byte_count'] for item in inspection.tensor_records)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
