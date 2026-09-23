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
| 1 — Golden test vector capture | ⬅️ **next** |
| 2 — Tokenizer in Swift | ⬜ |
| 3 — Weight loading (safetensors) | ⬜ |
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
- Von source clone (read-only reference): `~/Developer/von`
- Model cache: `~/.cache/huggingface/hub/models--wfzyx--von/`
  (`model.safetensors` 1.58 GB, `option_marker.pt`, `marker_calibration.json`,
  `tokenizer.json`, `config.json`)
- Key von sources:
  - `~/Developer/von/src/von/models/option_marker.py` (pack_sequence, scorer)
  - `~/Developer/von/src/von/backends/option_marker_backend.py` (all decision
    logic, calibration, noul debiasing)
  - `~/Developer/von/src/von/engine.py` (orchestrator)

## Verified smoke test (works today)

```bash
cd ~/Developer/von-swift
.venv/bin/python /tmp/golden_decide.py   # scratch; may not survive reboots
```

Result: choice=infrastructure, confidence=0.865,
probs ≈ {0.9119, 0.0407, 0.0474}. Any change to the pipeline can be
re-verified against this.

## Build & test

```bash
cd ~/Developer/von-swift
swift build && swift test
```

## Working agreement

- One phase per session where feasible; commit atomically per logical change.
- Every phase lands with a parity test against Python von goldens.
- Never commit weights, `.venv`, or `.build`. Golden fixtures (small JSON) are
  committed under `testdata/`.
- Conventional Commits; see `~/.pi/agent/skills/git-conventions/SKILL.md`.

## Open questions / risks

- Tokenizer byte-level BPE parity (highest risk — do first).
- Sliding-window attention + RoPE position conventions must match
  `transformers`' ModernBERT exactly; consult its source when implementing.
- float32 parity first; performance precision changes later.