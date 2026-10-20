# Changelog

All notable changes to VonSwift are documented here. The project follows
[Semantic Versioning](https://semver.org/) and
[Conventional Commits](https://www.conventionalcommits.org/).

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
