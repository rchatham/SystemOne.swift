# Changelog

All notable changes to SystemOne (formerly VonSwift) are documented here. The project follows
[Semantic Versioning](https://semver.org/) and
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- Weight formats for the next model families: safetensors float16, bfloat16,
  and integer tensors (`SafetensorsDataType`); sharded checkpoints
  (`SafetensorsCheckpoint`, over a shared `TensorStore` protocol); and MLX
  affine-quantized weights at 2–8 bits, loaded as `QuantizedLinear` or a
  packed quantized embedding, with group size and bits read from the stored
  shapes and checked against `config.json` (`MLXQuantizationConfiguration`).
  A pure-Swift `AffineQuantization` reference and
  `testdata/mlx_affine_quantization_v1.json` (`tools/mlx_quantization_fixture.py`,
  mlx 0.32.4) pin MLX's packing.
- CI runs MLX checks from the Xcode test bundle (`SYSTEMONE_MLX_TESTS=1`) on
  tiny fixtures.
- `SystemOneD1` loads LiquidAI's mlx-vlm exports (`d1-omni-600M-MLX-fp16`,
  `-MLX-8bit`) as well as the PyTorch checkpoint: `D1WeightLayout`, an
  export schema matching the published tensor names, and quantized linears and
  token embedding. `D1Weights.init(modelFolder:configuration:artifact:precision:)`
  opens either; `D1Backend` uses it.
- `SystemOneEncoders` (Apache-2.0): `ModernBERTEncoder`, a config-driven
  ModernBERT / mmBERT / Ettin encoder, and `EncoderAttention` for padded rows,
  explicit position ids, and per-row visibility (tree layouts with the sliding
  window on position ids). Checked against transformers' `ModernBertModel` with
  `testdata/modernbert_tiny/` (`tools/modernbert_fixture.py`). See
  [docs/ENCODERS.md](docs/ENCODERS.md).
- Encoder decision backends in `SystemOneEncoders`: `DecisionTuneBackend`
  (DecisionTune 1.0), `MarkerHeadBackend` (Laya and Julia-1), and
  `TreeDeciderBackend` (tasksource-decider-nano, with its tree attention). The
  prompts match each reference's packed ids, and the heads match the reference
  heads on a tiny encoder (`testdata/encoder_decisions/`,
  `tools/encoder_decisions_fixture.py`).
- `SystemOneDecoders` (Apache-2.0): `Qwen35Trunk`, the Qwen3.5 text trunk
  (Gated DeltaNet and gated full attention) returning final-norm hidden states
  for decision readouts. It loads transformers checkpoints and mlx-lm exports,
  dense or quantized, under any published prefix. `DecoderAttention` gives
  causal, bidirectional (d3-style), or per-row visibility plans with explicit
  positions; `DecoderCache` holds a one-row prefix that many rows can read; and
  `readoutRows(_:)` gathers output-embedding rows without full-vocabulary
  logits. The gated delta rule runs in chunks as transformers does. Checked
  against transformers and mlx-lm with `testdata/qwen35_tiny/`
  (`tools/qwen35_fixture.py`). See [docs/DECODERS.md](docs/DECODERS.md).
- `D3Backend`: the d3 family (vllm-sr d3, -lite, -mini, -flash, -edge) on the
  Qwen3.5 trunk, with the reference's prompt (`QwenChat` ChatML, thinking off),
  `noncausal_full_attention`, the 255-way answer-code readout, and its choice,
  noul, and score answers. `HubDecoderTokenizer` reads the folder's tokenizer.
  Prompt goldens and tiny-readout answers in `testdata/d3_tiny/`
  (`tools/d3_fixture.py`).
- `LFM2Trunk` (`SystemOneDecoders`): the causal LFM2 / LFM2.5 trunk (gated
  short convolutions and GQA) with the shared plans, prefix caches, and
  readout rows. Checked against
  transformers with `testdata/lfm2_tiny/` (`tools/lfm2_fixture.py`).
- `D1ChatBackend` (`SystemOneD1`): d1-3B's text decisions, with the
  reference's prompt, single-token option codes, max-pooled next-token
  readout, and answers; a request's questions share one prefilled state.
  Goldens in `testdata/d1_3b_tiny/` (`tools/d1_3b_fixture.py`).
- `LFM3Trunk` (`SystemOneDecoders`): d1-2B's LFM3 backbone (7-tap gated
  convolutions, gated GQA with LFM3's partial RoPE, sliding attention, and
  sandwich norms), checked against mlx-vlm with `testdata/lfm3_tiny/`
  (`tools/lfm3_fixture.py`).
- `D1SetBackend` (`SystemOneD1`): d1-2B's noul, choice, and score answers
  under set attention, with escaped caller text, `d1_codes` option codes,
  calibrated temperatures, and median score levels. Rows and answers are
  checked against mlx-vlm's `d1_lfm3_vl` with `testdata/d1_2b_tiny/`
  (`tools/d1_2b_fixture.py`).
- `SystemOneD1`: LiquidAI d1-omni-600M (text, images, and speech) as a
  `DecisionBackend`, verified against the pinned snapshot's own code with
  `tools/d1_golden_capture.py` and `testdata/d1_golden_v1.json`. Image and
  audio requests run the media prefix through the trunk once and reuse its
  keys, values, and conv state for every question (a three-question photo on
  an iPhone 14 Pro: 5.0 s to 2.9 s).
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
  head gradients to 3e-5 relative in norm, and the step's loss (3.8168 to
  3.4239) to 2e-5. See `docs/FEDERATED.md`.
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

- Safetensors headers with `"__metadata__": null`, which `mx.save_safetensors`
  writes when given no metadata, now load with no metadata, as the reference
  implementation reads them. Other non-object metadata is still rejected.
- **Breaking:** `D1Weights.safetensors` is now `store` (`any TensorStore`), and
  `D1Weights.init(safetensors:configuration:precision:)` is
  `init(store:configuration:precision:quantization:)`.

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
