# VonSwift

A Swift implementation of the [von](https://github.com/wfzyx/von) "System One"
decision model — non-autoregressive, calibrated choice / noul / score
inference running locally on Apple Silicon.

## Goal

Run the von decision model natively in Swift on Apple Silicon, using the
released von checkpoint. Built incrementally, each step verified against the
Python von package as a reference oracle (golden test vectors).

## Stack

- **Swift** owns the shell: tokenizer, safetensors loading, decision head,
  public API, HTTP server
- **swift-transformers** `Tokenizers` 1.1.6 provides the pinned byte-level BPE
  engine behind VonSwift's local-only tokenizer wrapper
- **MLX Swift** (`ml-explore/mlx-swift`) 0.31.4 owns the tensor math:
  ModernBERT backbone on Apple Silicon CPU/GPU
- **Python von** (repo-local `.venv`) is the reference oracle generating
  golden test vectors at every step

## Docs

- [docs/PLAN.md](docs/PLAN.md) — phased plan with status
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how von actually works
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — Phase 8 benchmarks and batching architecture
- [docs/HANDOFF.md](docs/HANDOFF.md) — resume-here state for fresh sessions
- [Examples/VonHarness](Examples/VonHarness/README.md) — iOS app harness for running the engine on an iPhone

## Status

Phases 0–8 are complete. The engine supports bounded, stable microbatch fan-out
via `VonBatchingStrategy` (`.sequential` / `.automatic` / `.batched`). On an
Apple M4 Max, the heterogeneous four-question fixture runs at 67.0 ms sequential,
55.0 ms automatic (default), and 46.5 ms batched; homogeneous rows scale
sub-linearly (4 rows: 16.1 ms; 16 rows: 46.3 ms). See
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) for full benchmarks.

The public API is a local-only `VonEngine` actor with ordered, typed
`Codable & Sendable` state, question, and answer models. Phase 7 adds a
security-hardened SwiftNIO server and order-preserving raw JSON wire codec. The pinned CPU float32
Python oracle fixture is at `testdata/von_golden_v1.json`. The Swift tokenizer
matches all five golden forwards exactly, including the 541-token Unicode
input, and the MLX ModernBERT backbone matches every one of the 30 probe
boundaries plus all complete final mask vectors. Measured backbone global max
absolute error is `1.8692017e-4` on CPU and `1.66893e-4` on GPU.

The scorer copies the eight production float32 tensors and implements both
affine norms, biased projections, exact GELU, and no inference dropout.
Composed raw-logit global max error is `6.1035156e-5` on CPU and
`5.9604645e-5` on GPU. Pure Swift decision math preserves Torch float32
softmax/entropy and default-Noul debiasing, then applies the pinned calibration
map in double precision; all four probability vectors match within
`4.991889e-6` on CPU and `4.068017e-6` on GPU, with exact guarded rounded
answers.

Phase 3 deterministically converts the pinned `option_marker.pt` into one
1,581,259,300-byte safetensors artifact containing the exact 178 production
float32 tensors (170 encoder + 8 scorer). The source and output SHA-256 values
are recorded in `testdata/option_marker_weights_manifest_v1.json`; canonical
per-tensor metadata and payload hashes are in
`testdata/option_marker_tensor_metadata_v1.json`. Weight payloads are not
committed.

The exact Apache-2.0 `wfzyx/von` tokenizer files are checked in as Swift test
resources with revision and SHA-256 provenance in
`Tests/SystemOneVonTests/Resources/Tokenizer/source.json`. Runtime tokenization
loads only an explicitly supplied local folder and has no model-name/network
API.

## Public API

```swift
let engine = try await VonEngine(
    tokenizerFolder: tokenizerFolder,
    verifiedWeightsURL: weightsURL
)

let answer = try await engine.decide(
    state: .fields([
        VonStateField(key: "status", value: "degraded"),
        VonStateField(key: "region", value: "Tokyo"),
    ]),
    choices: [
        ChoiceOption(key: "infrastructure", description: "Server or region failure"),
        ChoiceOption(key: "billing", description: "Payment or invoice issue"),
    ]
)
```

`systemOne` accepts ordered `[QuestionEntry]` and an optional
`batchingStrategy` parameter (default `.automatic`). Rows are pre-planned,
sorted by length, and greedily packed into bounded microbatches before
inference. Choice options and key/value state are also
ordered arrays so model semantics never depend on `Dictionary` iteration.
Codable values retain Python's keyed objects and add explicit order metadata
(`state_order`, `question_order`, `criteria_order`, `answer_order`,
`probability_order`, and `legend_order`) where needed; Python Pydantic ignores
these extra fields, while Swift decoding uses them to preserve order.

Python-compatible edge behavior is intentional: either non-empty Noul side
(including whitespace) disables null debiasing, each absent/empty side receives
its own default, and a Choice description falls back to its stripped key only
when it is `nil` or exactly empty. Swift safely rejects zero Choice options,
zero Score levels, duplicate state/Choice keys, and duplicate question IDs,
which Python's dictionary API cannot represent consistently.

## HTTP server

The CLI only loads explicit local artifacts and requires an explicit port (it
never defaults to the workspace-assigned port 8000):

```bash
VON_API_KEY='replace-me' \
VON_CORS_ORIGINS='https://app.example,https://admin.example' \
  swift run system-one serve \
    --host 127.0.0.1 \
    --port 8015 \
    --tokenizer Tests/SystemOneVonTests/Resources/Tokenizer \
    --weights weights/option_marker.safetensors
```

`VON_API_KEY` enables Bearer authentication for `/v1/systemone` and is required
for every non-loopback bind. `VON_CORS_ORIGINS` is an optional comma-separated
explicit allowlist; missing means CORS is disabled and `*` is rejected. SIGINT
and SIGTERM stop admission and run bounded graceful shutdown.

The HTTP adapter preserves raw JSON order for state fields, questions, Choice
criteria, answers, probabilities, and legends without relying on Swift
`Codable` dictionary order or `_order` metadata. Swift deliberately accepts a
strict, resource-bounded subset where Python permits ambiguous or unbounded
inputs. Empty Choice/Score criteria and worst-case escaped responses over 1 MiB
are rejected before inference. `Expect: 100-continue` is acknowledged only
after auth, size, and admission checks; unsupported expectations receive 417.
The server also differs from Python's default wildcard CORS policy by disabling
CORS until an explicit allowlist is configured.

Live parity is opt-in because it loads both production models. The script checks
that loopback port 8015 is clear (8014 remains assigned), inventories Docker,
uses a cleanup trap, waits for Python health, starts Swift on `127.0.0.1:0`, and
runs the Metal-enabled Xcode test bundle:

```bash
tools/run_http_parity.sh
```

It compares ordered raw successful JSON for Choice, default and explicit Noul,
Score, reordered state/criteria/questions, and GET routes; error cases compare
HTTP status categories, including Bearer auth. Neither weights nor server
processes are committed or launched by the default test suite.

Resolve dependencies, build, and run the default checks with:

```bash
swift package resolve
swift build
swift test
.venv/bin/python tools/golden_capture.py --check
.venv/bin/python -m unittest discover -s Tests -p 'test_*.py'
```

## Model installation

The inference weights are intentionally not committed. Install them locally from
the pinned, Apache-2.0 `wfzyx/von` checkpoint in three steps.

1. Create a Python environment with the pinned conversion dependencies
   (`torch==2.14.0`, `safetensors==0.8.0`):

   ```bash
   uv venv .venv
   uv pip install --python .venv/bin/python torch==2.14.0 safetensors==0.8.0
   ```

   (A plain `python3 -m venv .venv` plus
   `pip install torch==2.14.0 safetensors==0.8.0` works equally well for
   conversion alone; the reference-oracle `von-sdk==1.1.1` is only needed to
   regenerate golden vectors.)

2. Download the pinned `option_marker.pt` checkpoint and verify its SHA-256:

   ```bash
   python3 tools/download_model.py
   python3 tools/download_model.py --check
   ```

3. Convert it to the verified `option_marker.safetensors` artifact and confirm:

   ```bash
   .venv/bin/python tools/convert_option_marker.py --source weights/option_marker.pt
   .venv/bin/python tools/convert_option_marker.py --check
   VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" swift test
   ```

Both the download and conversion steps are checksum-verified and fail closed on
any deviation from the committed manifest. The converted artifact is derived
from an Apache-2.0 checkpoint and may be redistributed under that license, but
VonSwift defaults to this user-generated, verified local install.

For CPU/GPU public-API parity, use Xcode's Metal-enabled test bundle (the plain
SwiftPM test bundle lacks the Metal library):

```bash
xcodebuild build-for-testing -scheme SystemOne-Package -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
VON_PUBLIC_API_PARITY=1 VON_PUBLIC_API_DEVICE=cpu \
  VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" \
  xcrun xctest -XCTest SystemOneVonTests.VonEngineTests/testProductionPublicAPIParityWhenConfigured \
  "$PWD/.build/xcode/Build/Products/Debug/SystemOneVonTests.xctest"
# Repeat with VON_PUBLIC_API_DEVICE=gpu.
```

For lower-level CPU/GPU scorer, composed-model, and decision parity:

```bash
xcodebuild build-for-testing -scheme SystemOne-Package -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
VON_OPTION_MARKER_PARITY=1 VON_OPTION_MARKER_DEVICE=cpu \
  VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" \
  xcrun xctest -XCTest SystemOneVonTests.VonOptionMarkerModelTests/testProductionScorerAndComposedParityWhenConfigured \
  "$PWD/.build/xcode/Build/Products/Debug/SystemOneVonTests.xctest"
# Repeat with VON_OPTION_MARKER_DEVICE=gpu.
```

Use `.venv/bin/python tools/convert_option_marker.py --update-fixtures` only
when intentionally regenerating the canonical conversion fixtures. The default
Swift tests use checked-in resources and require no model cache or network; the
environment variable opts into the mmap, production hash, schema, and
per-tensor payload-hash integration test.

## License

VonSwift is licensed under the Apache License 2.0 (see `LICENSE`). It is an
independent Swift + MLX port of the [von](https://github.com/wfzyx/von)
System One decision model, which is also Apache-2.0. Attribution for the
reference model, the redistributed tokenizer assets, and third-party
Swift dependencies is recorded in `THIRD_PARTY_NOTICES.md`.

Model weights are intentionally not committed. The converted
`option_marker.safetensors` artifact is derived from the Apache-2.0
`option_marker.pt` checkpoint and may be redistributed under that license;
VonSwift defaults to a user-generated, checksum-verified artifact.