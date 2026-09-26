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
| 3 — Weight loading (safetensors) | ⬅️ **next** |
| 4 — ModernBERT backbone (MLX Swift) | ⬜ |
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

## Build & test

```bash
swift package resolve
swift build
swift test
.venv/bin/python tools/golden_capture.py --check
python3 -m unittest discover -s Tests -p 'test_*.py'
```

`--check` captures in memory, compares canonical bytes first, and never rewrites
the fixture. If bytes differ, only finite float leaves within absolute `1e-5` /
relative `1e-6` may differ; all structure, order, types, and non-floats are exact.

## Working agreement

- One phase per session where feasible; commit atomically per logical change.
- Every phase lands with a parity test against Python von goldens.
- Never commit weights, `.venv`, or `.build`. Golden fixtures (small JSON) are
  committed under `testdata/`.
- Conventional Commits; see `~/.pi/agent/skills/git-conventions/SKILL.md`.

## Open questions / risks

- Phase 3 deterministic conversion of the production `option_marker.pt` is
  next; tokenizer parity is complete.
- ModernBERT has no learned absolute position embedding; RoPE uses absolute
  sequence positions. The CPU oracle's sliding mask has inclusive radius 64
  (up to 129 positions) for `local_attention=128`; the module's separate 65 is
  only a FlashAttention adapter detail.
- The released `model.safetensors` has no custom scorer and its encoder values
  differ from the production overlay. Phase 3 must deterministically convert
  all 170 `encoder.*` and eight `scorer.*` tensors from `option_marker.pt` to a
  validated Swift-loadable safetensors artifact.
- float32 parity first; performance precision changes later.
