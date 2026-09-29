# Handoff

Single place to resume work in a fresh session. Keep current; update at the
end of every working session.

## Project

**VonSwift** — Swift + MLX port of [wfzyx/von](https://github.com/wfzyx/von),
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
| 4 — ModernBERT backbone (MLX Swift) | ⬅️ **next** |
| 5 — Decision head + calibration | ⬜ |
| 6 — Public API | ⬜ |
| 7 — HTTP server `/v1/systemone` | ⬜ |
| 8 — Performance | ⬜ |

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

`VonTokenizer` is a VonSwift-owned local-only wrapper around the `Tokenizers`
product from `huggingface/swift-transformers` 1.1.6. It loads only an explicit
model folder via `AutoTokenizer.from(modelFolder:)`; the public API exposes
special-token-aware encoding, Python-compatible state token counts (including
the empty-state clamp to one), exact `pack_sequence`, and packed text/ids/
attention masks/mask positions.

The exact pinned `tokenizer.json` and `tokenizer_config.json` are test resources
under `Tests/VonSwiftTests/Resources/Tokenizer/`. `source.json` records the
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

## Open questions / risks

- ModernBERT has no learned absolute position embedding; RoPE uses absolute
  sequence positions. The CPU oracle's sliding mask has inclusive radius 64
  (up to 129 positions) for `local_attention=128`; the module's separate 65 is
  only a FlashAttention adapter detail.
- float32 parity first; performance precision changes later.
