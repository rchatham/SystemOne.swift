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
| Tokenizer | `huggingface/swift-transformers` `Tokenizers` product pinned to 1.1.6, wrapped by VonSwift | Exact parity is verified; local-folder loading avoids a network API |
| Reference oracle | Python von in `.venv` (repo-local, gitignored) + source clone at `~/Developer/von` | Lets us run and diff against the real thing at any step |

## Phases

### Phase 0 — Recon & oracle setup ✅ DONE
- [x] Scaffold repo, SPM package builds, tests pass
- [x] Install Python von (`uv venv .venv`, `von-sdk==1.1.1`)
- [x] Clone von source to `~/Developer/von` for reference
- [x] Run first real inference (golden decide() example works)
- [x] Read full inference path: `engine.py`, `option_marker_backend.py`, `option_marker.py`
- [x] Inspect checkpoint files + `config.json` + `marker_calibration.json`

### Phase 1 — Golden test vector capture ✅ DONE
- [x] Write `tools/golden_capture.py` (Python, uses `.venv` von)
- [x] Pin and validate the von source revision, local HF revision, oracle
      dependency versions, and six consumed artifact SHA-256 hashes before loading
- [x] Tokenizer goldens: exact packed string → `input_ids`, attention mask,
      mask positions (including Unicode and a >128-token sequence)
- [x] Backbone goldens: 30 compact per-mask probes (embeddings, every layer
      0...27, and final norm) plus complete final K×1024 mask states for every
      production model forward
- [x] Head goldens: raw K logits pre-calibration for all five forwards
- [x] End-to-end goldens: choice / default Noul + null pass / explicit Noul /
      score, with unrounded calibration math and exact serialized answers
- [x] Canonical fixture at `testdata/von_golden_v1.json`, model-free stdlib
      validation, and non-rewriting tolerant `--check`

Regenerate and verify with:

```bash
.venv/bin/python tools/golden_capture.py
.venv/bin/python tools/golden_capture.py --check
python3 -m unittest discover -s Tests -p 'test_*.py'
```

### Phase 2 — Tokenizer in Swift ✅ DONE
- [x] Pin `swift-transformers` 1.1.6 and load its `Tokenizers` implementation
      only from an explicit local model folder
- [x] Add a VonSwift-owned tokenizer wrapper for encode, Python-compatible
      state token counts, exact `pack_sequence`, and packed model inputs
- [x] Special tokens: tokenizer CLS 50281, SEP 50282, PAD 50283, mask 50284;
      tokenizer BOS/EOS are `None` (only model config ids alias CLS/SEP)
- [x] Check in the exact 3.6 MB tokenizer JSON/config from pinned `wfzyx/von`,
      with source revision, SHA-256 metadata, and Apache-2.0 license
- [x] Verify exact packed text, ids, attention masks, and mask positions for all
      five forwards, all four state counts, Python empty-state semantics,
      whitespace/special-token behavior, NFC, and the 541-token Unicode input

### Phase 3 — Weight conversion and loading ⬅️ NEXT
- [ ] Add a deterministic pinned Python conversion tool that extracts all 170
      production `encoder.*` tensors and eight `scorer.*` tensors from
      `option_marker.pt` into a Swift-loadable safetensors artifact
- [ ] Record and validate the pinned `option_marker.pt` source hash and generated
      safetensors hash; reject missing, extra, renamed, shape-mismatched, or
      non-float32 tensors
- [ ] Add a safetensors header parser in Swift, memory-map the converted artifact,
      and map its 178 tensors into reviewed encoder/scorer groups
- [ ] Do not use released `model.safetensors` for parity: it has different
      pre-OptionMarker encoder values and only an unrelated classifier head

### Phase 4 — ModernBERT backbone in MLX Swift
- [ ] Embeddings: token embeddings → bias-free LayerNorm → dropout; no learned
      absolute position embeddings
- [ ] Attention: RoPE over absolute positions (θ=160000 full / 10000 sliding),
      inclusive local radius 64 (up to 129 tokens), full attention every 3rd
      layer, 16 heads, no bias
- [ ] Gated MLP: bias-free Wi(1024→5248), split 2624/2624,
      GELU(input)×gate, then bias-free Wo(2624→1024)
- [ ] Layer parity vs backbone goldens (tolerance target: < 1e-3 on mask-position
      hidden states)
- [ ] End-to-end logits within tolerance of head goldens

### Phase 5 — Decision head + calibration
- [ ] Load the eight converted `scorer.*` tensors from the Phase 3 safetensors
      artifact (the released `model.safetensors` contains none of them)
- [ ] `OptionMarkerScorer`: affine LayerNorm(1024) → biased Linear(1024→512) →
      exact GELU → affine LayerNorm(512) → biased Linear(512→1)
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
2. No committed weights or venv; only reviewable oracle fixtures and pinned
   tokenizer metadata are checked in.
3. `swift test` green at every commit.
4. Final: end-to-end probabilities match Python von within ~1e-2 on the
   golden suite, latency competitive with Python-on-MPS.