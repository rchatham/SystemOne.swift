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
- **MLX Swift** (`ml-explore/mlx-swift`) owns the tensor math: ModernBERT
  backbone on Apple Silicon GPU
- **Python von** (repo-local `.venv`) is the reference oracle generating
  golden test vectors at every step

## Docs

- [docs/PLAN.md](docs/PLAN.md) — phased plan with status
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how von actually works
- [docs/HANDOFF.md](docs/HANDOFF.md) — resume-here state for fresh sessions

## Status

Phase 0 (recon + oracle setup) done; Phase 1 (golden capture) next.
See commits and the plan for progress.

## License

TBD (will mirror Apache-2.0 implications of the reference implementation).