# VonSwift Plan

Step-by-step plan for porting [wfzyx/von](https://github.com/wfzyx/von) (the
"System One" non-autoregressive decision model) to Swift + MLX, verified at
every step against Python von as a reference oracle.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how von actually works and
[HANDOFF.md](HANDOFF.md) for cross-session state.

## Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Language split | Swift owns shell (tokenizer, loaders, head, API, server); MLX Swift owns tensor math | MLX gives Apple Silicon GPU matmuls without leaving Swift |
| ML backend | `ml-explore/mlx-swift` | Native Apple Silicon, Swift-native API |
| Weights | von's released `wfzyx/von` checkpoint (1.58 GB float32) | Inference port first; training is out of scope for now |
| Parity strategy | Golden test vectors generated from Python von | Hard numeric targets for tokenizer, backbone, head, end-to-end |
| Reference oracle | Python von in `.venv` (repo-local, gitignored) + source clone at `~/Developer/von` | Lets us run and diff against the real thing at any step |

## Phases

### Phase 0 — Recon & oracle setup ✅ DONE
- [x] Scaffold repo, SPM package builds, tests pass
- [x] Install Python von (`uv venv .venv`, `von-sdk==1.1.1`)
- [x] Clone von source to `~/Developer/von` for reference
- [x] Run first real inference (golden decide() example works)
- [x] Read full inference path: `engine.py`, `option_marker_backend.py`, `option_marker.py`
- [x] Inspect checkpoint files + `config.json` + `marker_calibration.json`

### Phase 1 — Golden test vector capture
- [ ] Write `tools/golden_capture.py` (Python, uses `.venv` von)
- [ ] Tokenizer goldens: packed string → `input_ids`, mask positions
- [ ] Backbone goldens: hidden states at mask positions (per-layer, at least final)
- [ ] Head goldens: raw K logits pre-calibration
- [ ] End-to-end goldens: choice / noul / score cases incl. noul with and without
      explicit criteria (exercises null-state debiasing)
- [ ] Commit goldens as JSON fixtures under `testdata/`

### Phase 2 — Tokenizer in Swift
- [ ] Parse `tokenizer.json` (HF tokenizer format)
- [ ] Byte-level BPE encode matching ModernBERT's tokenizer exactly
- [ ] Special tokens: CLS 50281, SEP 50282, PAD 50283, mask 50284, BOS/EOS
- [ ] Parity test: Swift token ids == golden ids for all fixtures

### Phase 3 — Weight loading
- [ ] Safetensors header parser (Swift, no deps — format is simple JSON header)
- [ ] Load `model.safetensors`; map tensor names to typed Swift structs
- [ ] Memory-map the 1.58 GB file; report param counts per tensor group

### Phase 4 — ModernBERT backbone in MLX Swift
- [ ] Embeddings: token + position embeddings, layer norms
- [ ] Attention: RoPE (θ=160000 full / 10000 sliding), sliding window 128,
      full attention every 3rd layer, 16 heads, no attention bias
- [ ] MLP: Linear(1024→2624) → GELU → Linear(2624→1024), no biases
- [ ] Layer parity vs backbone goldens (tolerance target: < 1e-3 on mask-position
      hidden states)
- [ ] End-to-end logits within tolerance of head goldens

### Phase 5 — Decision head + calibration
- [ ] `OptionMarkerScorer` head: LayerNorm → Linear(1024→512) → GELU → LayerNorm
      → Linear(512→1)
- [ ] Input-conditioned temperature: T = bias + entropy·H_norm +
      log_tokens·log10(n)/4 + n_options·K/8, clamped [0.3, 12.0]
- [ ] Choice: softmax(logits/T), argmax on raw logits, confidence = p₁ − p₂
- [ ] Noul: 2 options + null-state debiasing (subtract 0.7·bias from true logit
      when criteria not explicit)
- [ ] Score: expected value Σ i·pᵢ

### Phase 6 — Public API
- [ ] `decide`, `judge`, `rate`, `systemOne` mirroring von's Python API
- [ ] State formatting: dict → `"key: value"` lines
- [ ] Question types: Choice / Noul / Score structs

### Phase 7 — HTTP server, `/v1/systemone` parity
- [ ] SwiftNIO or similar minimal HTTP server
- [ ] Wire protocol per von's `server.py`
- [ ] Round-trip tests against Python `von serve`

### Phase 8 — Performance
- [ ] Latency benchmark vs Python von on MPS
- [ ] Single-pass multi-question fan-out (`system_one` semantics)

## Acceptance criteria (overall)

1. Every phase has a parity test against Python von goldens before merge.
2. No committed weights or venv; goldens are small JSON fixtures only.
3. `swift test` green at every commit.
4. Final: end-to-end probabilities match Python von within ~1e-2 on the
   golden suite, latency competitive with Python-on-MPS.