# Handoff

Single place to resume work in a fresh session. Keep current; update at the
end of every working session.

## Project

**SystemOne** (formerly VonSwift) — Swift + MLX runtime for System One decision models; it began as a port of [wfzyx/von](https://github.com/wfzyx/von),
the open-source non-autoregressive "System One" decision model
(choice / noul / score in a single forward pass). Goal: run the released von
checkpoint natively in Swift on Apple Silicon, verified step-by-step against
Python von as a reference oracle.

Read these first, in order:

1. [PLAN.md](PLAN.md) — phases, decisions, current status checkboxes
2. [ARCHITECTURE.md](ARCHITECTURE.md) — how von actually works (verified map)

## Current state (update after each session)

| Phase | Status |
|---|---|
| 0 — Recon & oracle setup | ✅ done |
| 1 — Golden test vector capture | ✅ done |
| 2 — Tokenizer in Swift | ✅ done |
| 3 — Weight conversion and loading (safetensors) | ✅ done |
| 4 — ModernBERT backbone (MLX Swift) | ✅ done |
| 5 — Decision head + calibration | ✅ done |
| 6 — Public API | ✅ done |
| 7 — HTTP server `/v1/systemone` | ✅ done |
| 8 — Performance | ✅ done |
| 9 — SystemOne core + d1-omni (text, vision, audio) | ✅ done |

d1 lives in `SystemOneD1`; see [D1.md](D1.md). Its oracle is a separate venv,
`.venv-d1` (torch 2.14.0, torchvision 0.29.1, transformers 5.17.0, tokenizers
0.23.2, Pillow, soundfile), and the pinned snapshot is
`~/.cache/huggingface/hub/models--LiquidAI--d1-omni-600M/snapshots/02b55d7076f15129e59ab3f94783f32c4b088674`.
MLX tests that touch weights need the Xcode test bundle (SwiftPM bundles and
CLI builds lack the Metal library, even for the CPU device).

## Environment facts

- Repo: `~/Developer/von-swift` (branch `main`)
- Machine: Apple M4 Max, 64 GB RAM, macOS 15.6, Swift 6.2.4
- Python oracle: `.venv/` in repo (gitignored), `von-sdk==1.1.1`
  - Run: `.venv/bin/python` (no activate needed)
- Von source clone (read-only reference): `~/Developer/von`, pinned revision
  `6c3c095d03ce78a546cb197df490be9a56c1ad3e`
- Model cache: `~/.cache/huggingface/hub/models--wfzyx--von/`, pinned local
  revision `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd`
  (`model.safetensors` 1.58 GB, `option_marker.pt`, `marker_calibration.json`,
  `tokenizer.json`, `tokenizer_config.json`, `config.json`); the generator
  validates all six hashes and exact oracle dependency versions: CPython
  3.12.8, von-sdk 1.1.1 (module 1.0.1), Torch 2.14.0, Transformers 5.17.0,
  Tokenizers 0.23.2, huggingface-hub 1.32.0, Pydantic 2.13.5,
  pydantic-core 2.46.5, and safetensors 0.8.0
- Key von sources:
  - `~/Developer/von/src/von/models/option_marker.py` (pack_sequence, scorer)
  - `~/Developer/von/src/von/backends/option_marker_backend.py` (all decision
    logic, calibration, noul debiasing)
  - `~/Developer/von/src/von/engine.py` (orchestrator)

## Phase 1 oracle fixture

`testdata/von_golden_v1.json` is generated deterministically on CPU float32 by
one production backend/model load. It covers four cases and five model forwards
(Choice; default Noul primary + null pass; explicit Noul; Score), including all
packed inputs/logits, 30 backbone probe entries (embeddings, layers 0...27,
then final norm), complete final K×1024 mask states, unrounded calibration
math, and exact serialized production answers.
Metadata contains pinned revisions and artifact hashes, never local paths or
timestamps.

Regenerate only when intentionally changing the oracle fixture:

```bash
.venv/bin/python tools/golden_capture.py
```

## Phase 2 tokenizer

`VonTokenizer` is a SystemOne-owned local-only wrapper around the `Tokenizers`
product from `huggingface/swift-transformers` 1.1.6. It loads only an explicit
model folder via `AutoTokenizer.from(modelFolder:)`; the public API exposes
special-token-aware encoding, Python-compatible state token counts (including
the empty-state clamp to one), exact `pack_sequence`, and packed text/ids/
attention masks/mask positions.

The exact pinned `tokenizer.json` and `tokenizer_config.json` are test resources
under `Tests/SystemOneVonTests/Resources/Tokenizer/`. `source.json` records the
`wfzyx/von` revision, Apache-2.0 license, and hashes; `LICENSE.md` carries the
license. Swift parity tests cover all five golden forwards (including the
541-token Unicode input), all four state counts, whitespace, `[MASK]` lstrip,
NFC, and empty input. Tests read these checked-in files directly and never use
a user Hugging Face cache.

## Phase 3 production weights

`tools/convert_option_marker.py` deterministically converts the pinned
`option_marker.pt` into `weights/option_marker.safetensors`: 178 float32 tensors
(170 encoder + 8 scorer), 1,581,259,300 bytes, SHA-256
`3cadf5c5f001e859489da2cb035aa015b5b74b8d28a384abedb9d86290419e06`.
`testdata/option_marker_weights_manifest_v1.json` records both the pinned source
hash and output hash; `testdata/option_marker_tensor_metadata_v1.json` records
canonical names, shapes, offsets, sizes, and per-tensor payload hashes. Neither
the `.pt` source nor generated safetensors payload is committed.

`SafetensorsFile` strictly checks the float32 header and gap-free payload layout
and provides retained zero-copy views over a read-only private mmap.
`VonWeights` then checks the exact tensor names and shapes and exposes typed
encoder/scorer groups. `VonWeights.verifiedProduction(contentsOf:)`
additionally checks the pinned whole-file size and SHA-256 before parsing the
same mapping. The mapped
inode must not be mutated or truncated while the file or any view is alive.

## Phase 4 ModernBERT backbone

`VonModernBERT` is the exact 28-layer production encoder implemented with
`mlx-swift` 0.31.4 (`MLX` + `MLXNN`, macOS 14 minimum). Initialization copies
all float32 encoder tensors out of the Phase 3 mmap into MLX-owned arrays. The
forward path implements the layer-zero norm exception, `[B,L,3,16,64]` QKV
layout, nontraditional full-head RoPE at the two production bases, full
attention every third layer, inclusive radius-64 sliding attention with
key-only additive padding masks, 0.125-scaled SDPA, gated exact-GELU MLPs, and
the final norm.

The parity test is enabled with `VON_MODERNBERT_PARITY=1` plus
`VON_PRODUCTION_WEIGHTS_PATH`; it uses `VonWeights.verifiedProduction`, covers
all five golden forwards and all 30 boundaries, and checks complete final
K×1024 mask vectors. Measured global max absolute errors are `1.8692017e-4` on MLX CPU and
`1.66893e-4` on MLX GPU, both below the fixed `< 1e-3` target.

MLX Swift's command-line SwiftPM build does not package Metal shaders. Default
offline tests remain pure Swift and pass with `swift test`; runtime CPU/GPU
parity was run from the Xcode-built test bundle after `xcodebuild` produced
`default.metallib`. The shell parity variables are not propagated by the
auto-generated Xcode package scheme, so the built bundle was invoked directly
with `xcrun xctest`. The full Xcode suite also passes after using
`Bundle.module.resourceURL` for tokenizer test resources.

## Phase 5 decision head and calibration

`VonOptionMarkerScorer` copies all eight validated scorer tensors into
MLX-owned float32 arrays and implements affine LayerNorm(ε=1e-5), biased dense
1024→512, exact GELU, affine LayerNorm(512), and biased 512→1 projection.
Training dropout is intentionally absent. `VonOptionMarkerModel` composes the
scorer with `VonModernBERT` and gathers final hidden states at validated mask
token positions.

`VonDecisionMath` is the request-independent Choice/Noul/Score math layer. It
embeds the pinned `marker_calibration.json` coefficients with provenance,
reproduces Torch float32 softmax and entropy reduction before applying the map
in double precision, uses float32 0.7 default-Noul debiasing, first-maximum tie
selection, and nearest-even output rounding. Phase 6 request types and state
formatting are intentionally not present yet.

All five isolated scorer goldens, composed raw logits, all four calibrated
probability vectors, and guarded exact rounded answers pass on both devices.
Measured global max absolute errors were:

| Path | MLX CPU | MLX GPU |
|---|---:|---:|
| Isolated scorer logits | `9.536743e-7` | `2.861023e-6` |
| Composed backbone + scorer logits | `6.1035156e-5` | `5.9604645e-5` |
| Four decision probability vectors | `4.991889e-6` | `4.068017e-6` |

## Phase 6 public API

`SystemOneEngine` is a public actor that loads a tokenizer only from an explicit
local folder and weights only through `VonWeights.verifiedProduction`. It owns
all MLX state, evaluates lazy output inside actor isolation, and converts to raw
`[Float]` only after `eval`. `systemOne` uses a two-phase plan-then-execute
pipeline with bounded, stable microbatch fan-out (see Phase 8 and
[PERFORMANCE.md](PERFORMANCE.md)). Phase 7 wraps this actor in the
order-preserving HTTP adapter described below.

Public request/response models are `Codable & Sendable`. Inference-facing state
fields, Choice options, questions, answers, probabilities, and legends use
ordered entry arrays rather than dictionaries. Codable request and response
objects keep Python's keyed object forms, plus `state_order`, `question_order`,
`criteria_order`, `answer_order`, `probability_order`, and `legend_order`
metadata so decoding never relies on keyed-container order; Python's Pydantic
models ignore those extra fields. Encoding rejects duplicate keyed entries
before creating their keyed containers, so ordered arrays cannot silently
collapse values. The engine reports `von-1.1.0` and Python-compatible
approximate usage.

Rendering preserves Python behavior: Choice falls back to a stripped key only
for `nil`/empty descriptions (whitespace-only is truthy, then strips empty);
either non-empty Noul side disables null debiasing, while each missing or empty
side independently receives its default; Score supports text and `what` plus
`examples`. Swift intentionally rejects empty Choice/Score inputs, duplicate
state or Choice keys, and duplicate question IDs instead of producing ambiguous
or synthetic results. Packed input must contain exactly one tokenizer mask per
rendered option; a user-supplied literal `[MASK]` rejects that affected packed
input before it is passed to the injected provider or production model.

Offline tests inject ordered logits and reproduce the fixture's four cases and
five packed forwards exactly. The public production API also passed exact
rounded golden outputs through the Xcode-built bundle on both MLX CPU and GPU.

## Phase 7 HTTP server and CLI

`VonHTTP` provides an order-preserving parser/serializer, a metadata-free
`VonWireCodec`, and a security-hardened SwiftNIO HTTP/1.1 server. The wire path
never uses Swift `Codable` dictionary iteration. It preserves Python request
order and emits Python field/object order for successful responses. GET `/`,
`/health`, and `/v1/models` match pinned Python bytes; POST `/v1/systemone`
uses the public `SystemOneEngine` actor.

The server bounds raw and decoded headers/trailers, body bytes, connections,
admitted/queued inference, semantic inference work, worst-case escaped response
size, request/header/body/idle deadlines, and shutdown drain time. Empty Choice
or Score criteria and requests whose successful response could exceed 1 MiB are
rejected as 422 before inference. `Expect: 100-continue` receives 100 only after
authentication, declared-length/size, and request-admission checks; unsupported
expectations receive 417. Authentication comparisons are timing-safe, and
non-loopback binds fail closed without a key. CORS is disabled by default and
accepts only `SYSTEMONE_CORS_ORIGINS` explicit origins; unlike Python's wildcard
default, `*` is intentionally rejected. Swift also intentionally rejects a
strict resource/syntax subset of Python inputs; error parity is therefore by
HTTP status category rather than error text.

`von serve` requires all of `--port`, `--tokenizer`, and `--weights`, defaults
only the host to `127.0.0.1`, validates local artifacts before model loading,
and handles SIGINT/SIGTERM through bounded server shutdown. It never assigns
port 8000. Example:

```bash
SYSTEMONE_API_KEY=replace-me SYSTEMONE_CORS_ORIGINS=https://app.example \
  swift run system-one serve --host 127.0.0.1 --port 8015 \
  --tokenizer Tests/SystemOneVonTests/Resources/Tokenizer \
  --weights weights/option_marker.safetensors
```

`Tests/SystemOneHTTPTests/PythonServerParityTests.swift` is opt-in with
`VON_HTTP_PARITY=1`, an external loopback `VON_PYTHON_SERVER_URL`, production
weights, tokenizer resources, and a shared API key. It starts only Swift on
`127.0.0.1:0`. `tools/run_http_parity.sh` safely supplies the external pinned
Python server: it avoids assigned ports 8000/8014, verifies 8015 is clear,
inventories Docker, traps cleanup, waits for health, and runs the Xcode bundle.
Observed local verification on 2026-10-01 (not durable CI evidence) used the
exact command `tools/run_http_parity.sh` against the repo-local pinned
`von-sdk==1.1.1`. The Xcode build reported `TEST BUILD SUCCEEDED`, and the
selected `PythonServerParityTests` run reported 1 test executed with 0 failures
in 5.405 seconds. That local run covered ordered Choice, default/explicit Noul,
Score, reordered state/criteria/questions, exact successful raw JSON/GET
payloads, and auth/error status categories. Re-run the command rather than
assuming the observation remains valid on another machine or revision.

## Build & test

Resolve dependencies, then run the default checks:

```bash
swift package resolve
swift build
swift test
.venv/bin/python tools/golden_capture.py --check
.venv/bin/python -m unittest discover -s Tests -p 'test_*.py'
```

Create and verify the local converted artifact, then opt into the production
Swift test with:

```bash
.venv/bin/python tools/convert_option_marker.py
.venv/bin/python tools/convert_option_marker.py --check
VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" swift test
```

Run numeric parity on the Xcode-built test bundle. The Phase 5 command covers
the isolated scorer, composed raw logits, calibrated probabilities, and rounded
answers; run it once with each device:

```bash
xcodebuild build-for-testing -scheme SystemOne-Package -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
VON_OPTION_MARKER_PARITY=1 VON_OPTION_MARKER_DEVICE=cpu \
  VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" \
  xcrun xctest -XCTest SystemOneVonTests.VonOptionMarkerModelTests/testProductionScorerAndComposedParityWhenConfigured \
  "$PWD/.build/xcode/Build/Products/Debug/SystemOneVonTests.xctest"
# Repeat with VON_OPTION_MARKER_DEVICE=gpu.
```

Run public API parity once per device:

```bash
VON_PUBLIC_API_PARITY=1 VON_PUBLIC_API_DEVICE=cpu \
  VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" \
  xcrun xctest -XCTest SystemOneVonTests.VonEngineTests/testProductionPublicAPIParityWhenConfigured \
  "$PWD/.build/xcode/Build/Products/Debug/SystemOneVonTests.xctest"
# Repeat with VON_PUBLIC_API_DEVICE=gpu.
```

The Phase 4 backbone-only parity command remains available (set device to
`cpu` for the CPU oracle comparison):

```bash
xcodebuild build-for-testing -scheme SystemOne-Package -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
VON_MODERNBERT_PARITY=1 VON_MODERNBERT_DEVICE=gpu \
  VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" \
  xcrun xctest -XCTest SystemOneVonTests.VonModernBERTTests/testProductionBackboneParityWhenConfigured \
  "$PWD/.build/xcode/Build/Products/Debug/SystemOneVonTests.xctest"
```

The default Swift tests use checked-in resources and require no model cache or
network. Run `.venv/bin/python tools/convert_option_marker.py --update-fixtures`
only when intentionally replacing the canonical metadata/manifest fixtures. Both
converter `--check` and golden-capture `--check` are non-rewriting validation.
Golden capture permits only finite float-leaf drift within absolute `1e-5` /
relative `1e-6`; structure, order, types, and non-floats remain exact.

## Working agreement

- One phase per session where feasible; commit atomically per logical change.
- Every phase lands with a parity test against Python von goldens.
- Never commit weights, `.venv`, or `.build`. Golden and weight
  metadata/manifest fixtures (small JSON) are committed under `testdata/`.
- Conventional Commits; see `~/.pi/agent/skills/git-conventions/SKILL.md`.

## Phase 8 performance

`BatchingStrategy` (`.sequential` / `.automatic` / `.batched`) is a method
parameter on `systemOne` (default `.automatic`). It is not serialized or on the
HTTP wire. `SystemOneEngine.systemOne` pre-plans all inference rows and fans out via
bounded greedy microbatches under an absolute B×L² ≤ 1,048,576 guard and
amplification ceilings (auto 1.5×, batched 2.0×).

`VonOptionMarkerModel.forward` accepts ragged `[VonPackedInput]` rows: right-pad
to max length, flatten marker positions, run one backbone+scorer forward, return
row-major logits with `rowOffsets`. Padded queries in sliding-attention layers
see only themselves.

Release Xcode GPU benchmarks on Apple M4 Max (see [PERFORMANCE.md](PERFORMANCE.md)
for full tables):

| Strategy | Heterogeneous 4-question | Homogeneous n=4 | n=16 |
|---|---|---|---|
| `.sequential` | 67.0 ms | 40.6 ms | 159.5 ms |
| `.automatic` | 55.0 ms | 16.1 ms | 46.3 ms |
| `.batched` | 46.5 ms | — | — |

GPU batch model parity: max abs 7.05719×10⁻⁵ vs Python batch and
4.386902×10⁻⁵ vs Swift sequential. Public engine rounded parity GPU passed.

CPU benchmarks and CPU batch parity can trigger macOS's Metal watchdog;
CPU is diagnostic only. GPU (Release Xcode, Metal) is the acceptance target.

## Open questions / risks

- `mlx-swift` runtime testing requires an Xcode-produced Metal library even for
  an explicitly selected CPU device; plain `swift test` cannot run opt-in model
  parity directly with the upstream 0.31.4 package layout.
- The model and decision parity paths are float32-first. Precision/performance
  changes must retain the golden path and its mixed float32/double calibration
  boundary.
- Phase 6 Codable remains a typed Swift interchange contract, not the raw HTTP
  decoder. Phase 7's `OrderedJSON`/`VonWireCodec` handles metadata-free wire
  order and a documented, bounded subset of Python/Pydantic coercions.
- CPU benchmarks can hit the Metal watchdog; always use Release Xcode GPU for
  acceptance.
- The ordered bounded JSON parser was upstreamed locally on JSON.swift feature
  branch `dddbe8b` but SystemOne retains its current parser until a remote
  release/tag exists.
