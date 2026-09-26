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
- **MLX Swift** (`ml-explore/mlx-swift`) owns the tensor math: ModernBERT
  backbone on Apple Silicon GPU
- **Python von** (repo-local `.venv`) is the reference oracle generating
  golden test vectors at every step

## Docs

- [docs/PLAN.md](docs/PLAN.md) — phased plan with status
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how von actually works
- [docs/HANDOFF.md](docs/HANDOFF.md) — resume-here state for fresh sessions

## Status

Phases 0–2 are done. The pinned CPU float32 Python oracle fixture is at
`testdata/von_golden_v1.json`; the Swift tokenizer matches all five golden
forwards exactly, including the 541-token Unicode input. Phase 3 (deterministic
weight conversion and loading) is next.

The exact Apache-2.0 `wfzyx/von` tokenizer files are checked in as Swift test
resources with revision and SHA-256 provenance in
`Tests/VonSwiftTests/Resources/Tokenizer/source.json`. Runtime tokenization
loads only an explicitly supplied local folder and has no model-name/network
API.

Build and verify with:

```bash
swift package resolve
swift build
swift test
.venv/bin/python tools/golden_capture.py --check
python3 -m unittest discover -s Tests -p 'test_*.py'
```

Golden generation validates the pinned von source revision, local Hugging Face
model revision, exact oracle dependency versions, and all six consumed artifact
hashes before loading one production backend and model. `--check` does not
rewrite the fixture.

## License

TBD (will mirror Apache-2.0 implications of the reference implementation).