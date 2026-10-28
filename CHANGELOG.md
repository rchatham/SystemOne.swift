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
- `D1Precision`: an opt-in float16 mode for d1 (`precision:` on
  `SystemOneEngine.d1`, `D1Backend`, and `D1Weights`) that halves weight
  memory (2.24 GB to 1.12 GB in MLX) so d1-omni-600M runs on 4 GB iOS
  devices (verified on an iPhone 13: 1.78 GB MLX peak, 1.07 GB to spare). Weights convert to float16 on the CPU at load,
  and a weight beyond float16's range fails the load instead of becoming
  infinity; the mel front end, logits, and calibration stay float32.
  The iOS harness picks float16 when `os_proc_available_memory()` is too small
  for float32 and shows the loaded precision.
- d1 head-only training spike (internal): `D1HeadTraining` computes a masked
  cross-entropy over the decision head's raw logits, with the trunk frozen and
  every head weight passed to `valueAndGrad`. `tools/d1_training_capture.py`
  records PyTorch's loss, gradients, and one gradient-descent step in
  `testdata/d1_training_golden_v1.json`; on the GPU, Swift matches all 31
  head gradients to 1e-5 relative in norm, and the step's loss (3.8168 to
  3.4239) to 1e-5. See `docs/FEDERATED.md`.
- `SystemOneMedia`, `RGBImage` (including `init?(cgImage:)`), and `AudioClip`.
- `ModelProfile`, `VonModelProfile`, and `D1ModelProfile` for model-agnostic
  HTTP admission and service metadata.
- Shared `WeightSpecification`, `PinnedArtifact`, and `WeightsError`; I64
  safetensors tensors; torch-exact `AntialiasedResize` and Slaney
  `MelFilterbank`.
- `system-one serve --model-folder` serves d1.
- The iOS harness runs d1 on device with photo-library, camera, and
  microphone input, plus a `-selfTest` launch mode; verified on an iPhone 14 Pro.
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

### Licensing

- `SystemOneD1` is a port of d1-omni's reference code and is distributed under
  the LFM Open License v1.0 (`LICENSES/LFM-Open-License-1.0.txt`), which limits
  commercial use by entities with $10M or more in annual revenue. All other
  modules remain Apache-2.0.
- The repository moved to `rchatham/SystemOne.swift`.

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

[0.1.0]: https://github.com/rchatham/SystemOne.swift/releases/tag/0.1.0
