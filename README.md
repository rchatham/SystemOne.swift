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
- [docs/HANDOFF.md](docs/HANDOFF.md) — resume-here state for fresh sessions

## Status

Phases 0–5 are done; Phase 6 (the public API) is next. The pinned CPU float32
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
`Tests/VonSwiftTests/Resources/Tokenizer/source.json`. Runtime tokenization
loads only an explicitly supplied local folder and has no model-name/network
API.

Resolve dependencies, build, and run the default checks with:

```bash
swift package resolve
swift build
swift test
.venv/bin/python tools/golden_capture.py --check
.venv/bin/python -m unittest discover -s Tests -p 'test_*.py'
```

Create and verify the local production artifact separately:

```bash
.venv/bin/python tools/convert_option_marker.py
.venv/bin/python tools/convert_option_marker.py --check
VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" swift test
```

For CPU/GPU scorer, composed-model, and decision parity, use Xcode's
Metal-enabled test bundle (the plain SwiftPM test bundle lacks the Metal
library):

```bash
xcodebuild build-for-testing -scheme VonSwift -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
VON_OPTION_MARKER_PARITY=1 VON_OPTION_MARKER_DEVICE=cpu \
  VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" \
  xcrun xctest -XCTest VonSwiftTests.VonOptionMarkerModelTests/testProductionScorerAndComposedParityWhenConfigured \
  "$PWD/.build/xcode/Build/Products/Debug/VonSwiftTests.xctest"
# Repeat with VON_OPTION_MARKER_DEVICE=gpu.
```

Use `.venv/bin/python tools/convert_option_marker.py --update-fixtures` only
when intentionally regenerating the canonical conversion fixtures. The default
Swift tests use checked-in resources and require no model cache or network; the
environment variable opts into the mmap, production hash, schema, and
per-tensor payload-hash integration test.

## License

TBD (will mirror Apache-2.0 implications of the reference implementation).