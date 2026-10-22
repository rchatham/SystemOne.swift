# Changelog

All notable changes to SystemOne (formerly VonSwift) are documented here. The project follows
[Semantic Versioning](https://semver.org/) and
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- `SystemOneD1`: LiquidAI d1-omni-600M (text, images, and speech) as a
  `DecisionBackend`, verified against the pinned snapshot's own code with
  `tools/d1_golden_capture.py` and `testdata/d1_golden_v1.json`.
- `DecisionBackend`, `PlannedRow`, and `DecisionPlan`: the extension point for
  System One models; `SystemOneEngine.d1(modelFolder:)` and
  `SystemOneEngine.von(tokenizerFolder:verifiedWeightsURL:)` factories.
- `SystemOneMedia`, `RGBImage` (including `init?(cgImage:)`), and `AudioClip`.
- `ModelProfile`, `VonModelProfile`, and `D1ModelProfile` for model-agnostic
  HTTP admission and service metadata.
- Shared `WeightSpecification`, `PinnedArtifact`, and `WeightsError`; I64
  safetensors tensors; torch-exact `AntialiasedResize` and Slaney
  `MelFilterbank`.
- `system-one serve --model-folder` serves d1.
- iOS 17 platform support for the library.
- `Examples/SystemOneHarness` (formerly `VonHarness`), a SwiftUI app that loads the verified weights on an
  iPhone or iPad and reports `systemOne` answers, latency, and memory.
- CI build of the harness for generic iOS devices.

### Changed

- **Breaking:** the package is now `SystemOne`. Modules: `SystemOne` (core),
  `SystemOneVon`, `SystemOneD1`, `SystemOneHTTP`; the executable is
  `system-one`. `VonEngine` → `SystemOneEngine`, `VonState` →
  `SystemOneState`, `VonStateField` → `StateField`, `VonBatchingStrategy` →
  `BatchingStrategy`, `VonEngineError` → `SystemOneEngineError`,
  `VonHTTPServer` → `SystemOneHTTPServer`, `VonWireCodec` →
  `SystemOneWireCodec` (now takes a `profile:`), `VonCORSConfiguration` →
  `CORSConfiguration`, `VonWeightsError` → `WeightsError`.
- **Breaking:** `VON_API_KEY` / `VON_CORS_ORIGINS` are now `SYSTEMONE_API_KEY`
  / `SYSTEMONE_CORS_ORIGINS`.
- The engine resolves each question as soon as its rows finish, for every model.

## [0.1.0] - 2026-10-20

Initial release. A native Swift + MLX implementation of the
[von](https://github.com/wfzyx/von) "System One" decision model, verified
against the Python reference at every stage.

### Added

- Pinned byte-level BPE tokenizer (via `swift-transformers` 1.1.6) with exact
  Python parity, including the 541-token Unicode input.
- Deterministic safetensors conversion of the pinned `option_marker.pt`
  checkpoint into a 1,581,259,300-byte, 178-tensor artifact with whole-file and
  per-tensor SHA-256 verification.
- ModernBERT backbone (embeddings, RoPE, sliding/full attention, gated MLP) on
  Apple Silicon CPU/GPU via `mlx-swift`.
- Decision head (OptionMarker scorer) with input-conditioned temperature
  calibration and float32/double boundary parity.
- Typed public API: `decide`, `judge`, `rate`, and `systemOne` over ordered,
  `Codable & Sendable` state, question, and answer models.
- Security-hardened SwiftNIO HTTP server with `/v1/systemone` parity, bounded
  framing/concurrency/deadlines, Bearer auth, and an explicit CORS allowlist.
- Order-preserving, resource-bounded JSON wire codec (backed by the released
  `rchatham/JSON.swift` 1.1.0).
- Bounded multi-question microbatch fan-out with `.sequential` / `.automatic` /
  `.batched` strategies and a stable `B × L²` memory guard.
- Reproducible Release Xcode and Python MPS benchmark harnesses.
- Model-free Swift and Python test suites plus opt-in CPU/GPU parity tests.
- macOS GitHub Actions CI with a manual production-weight parity job.

### Changed

- The ordered bounded JSON parser is now sourced from the published
  `rchatham/JSON.swift` package rather than a vendored copy.

### Licensing

- Apache-2.0 with attribution in `THIRD_PARTY_NOTICES.md`. Model weights are
  user-generated via `tools/download_model.py` + `tools/convert_option_marker.py`.

[0.1.0]: https://github.com/rchatham/von.swift/releases/tag/0.1.0
