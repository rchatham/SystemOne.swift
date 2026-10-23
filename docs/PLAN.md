# SystemOne Plan

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
| Tokenizer | `huggingface/swift-transformers` `Tokenizers` product pinned to 1.1.6, wrapped by SystemOne | Exact parity is verified; local-folder loading avoids a network API |
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
- [x] Add a SystemOne-owned tokenizer wrapper for encode, Python-compatible
      state token counts, exact `pack_sequence`, and packed model inputs
- [x] Special tokens: tokenizer CLS 50281, SEP 50282, PAD 50283, mask 50284;
      tokenizer BOS/EOS are `None` (only model config ids alias CLS/SEP)
- [x] Check in the exact 3.6 MB tokenizer JSON/config from pinned `wfzyx/von`,
      with source revision, SHA-256 metadata, and Apache-2.0 license
- [x] Verify exact packed text, ids, attention masks, and mask positions for all
      five forwards, all four state counts, Python empty-state semantics,
      whitespace/special-token behavior, NFC, and the 541-token Unicode input

### Phase 3 — Weight conversion and loading ✅ DONE
- [x] Add a deterministic pinned Python conversion tool that extracts all 170
      production `encoder.*` tensors and eight `scorer.*` tensors from
      `option_marker.pt` into a 1,581,259,300-byte float32 safetensors artifact
- [x] Commit canonical manifest and per-tensor metadata fixtures under
      `testdata/`; the manifest records the pinned source and deterministic
      output hashes, while no source or converted weight payload is committed
- [x] Reject missing, extra, renamed, shape-mismatched, non-float32, malformed,
      overlapping, gapped, or hash-mismatched tensors/artifacts
- [x] Add a strict safetensors header/layout parser and read-only mmap-backed
      tensor views, then map the exact 178-tensor schema into typed encoder and
      scorer groups
- [x] Separate checked float32 format/layout and Von schema loading from
      `VonWeights.verifiedProduction(contentsOf:)`, which additionally verifies
      the pinned whole-file size and SHA-256 on the same mapping
- [x] Document the immutable-inode contract—callers must not mutate or truncate
      the mapped inode while a file or retained tensor view is alive—and test
      retained view lifetime independently of the pathname
- [x] Keep the released `model.safetensors` out of the parity path: it has
      different pre-OptionMarker encoder values and only an unrelated classifier
      head

Fixture refresh is intentional-only; normal conversion validates against the
fixtures, while `--check` is read-only:

```bash
.venv/bin/python tools/convert_option_marker.py
.venv/bin/python tools/convert_option_marker.py --check
# Intentional canonical fixture replacement only:
.venv/bin/python tools/convert_option_marker.py --update-fixtures
VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" swift test
```

Default Swift tests remain offline; the environment variable opts into the
production artifact hash, schema, and per-tensor payload-hash integration test.

### Phase 4 — ModernBERT backbone in MLX Swift ✅ DONE
- [x] Embeddings: token embeddings → bias-free LayerNorm → dropout; no learned
      absolute position embeddings
- [x] Attention: RoPE over absolute positions (θ=160000 full / 10000 sliding),
      inclusive local radius 64 (up to 129 tokens), full attention every 3rd
      layer, 16 heads, no bias
- [x] Gated MLP: bias-free Wi(1024→5248), split 2624/2624,
      GELU(input)×gate, then bias-free Wo(2624→1024)
- [x] All five golden forwards match at all 30 mask-position probe boundaries
      and across every complete final K×1024 vector below the `< 1e-3` target
      (CPU global max `1.8692017e-4`; GPU global max `1.66893e-4`)

### Phase 5 — Decision head + calibration ✅ DONE
- [x] Load and copy the eight converted float32 `scorer.*` tensors from the
      Phase 3 artifact (the released `model.safetensors` contains none of them)
- [x] `OptionMarkerScorer`: affine LayerNorm(1024) → biased Linear(1024→512) →
      exact GELU → affine LayerNorm(512) → biased Linear(512→1), with inference
      dropout omitted
- [x] Gather final backbone states at marker positions and match all five raw
      logit goldens on MLX CPU/GPU below `< 1e-3`
- [x] Input-conditioned temperature: T = bias + entropy·H_norm +
      log_tokens·log10(n)/4 + n_options·K/8, clamped [0.3, 12.0], preserving
      Torch's float32 softmax/entropy reduction before double map coefficients
- [x] Choice: softmax(logits/T), first-max argmax on raw logits, confidence = p₁ − p₂
- [x] Noul: 2 options + float32 null-state debiasing (subtract 0.7·bias from
      true logit when criteria are not explicit)
- [x] Score: expected value Σ i·pᵢ with nearest-even production rounding

### Phase 6 — Public API ✅ DONE
- [x] Public local-only `SystemOneEngine` actor with async verified tokenizer/weight initialization
- [x] Sequential `decide`, `judge`, `rate`, and `systemOne` over ordered state fields, options, and questions
- [x] Codable + Sendable Choice / Noul / Score requests and typed Python-keyed answers
- [x] Exact Python rendering/truthiness, including partial Noul criteria and Choice fallback edge cases
- [x] Offline injected-logit four-case/five-forward goldens and Xcode CPU/GPU public-API parity

### Phase 7 — HTTP server, `/v1/systemone` parity ✅ DONE
- [x] Security-hardened SwiftNIO HTTP/1.1 server with bounded framing, bodies,
      connections, inference concurrency, deadlines, and graceful shutdown
- [x] Order-preserving raw JSON adapter for Python's metadata-free wire format,
      with GET routes, Bearer auth, and an explicit CORS allowlist
- [x] Local-only `von serve` CLI requiring explicit port, tokenizer folder, and
      verified safetensors paths; non-loopback binds fail closed without a key
- [x] Opt-in live round-trip parity against pinned Python `von serve` for ordered
      Choice, default/explicit Noul, Score, reordered inputs, GET routes, auth,
      and error status categories

### Phase 8 — Performance ✅ DONE
- [x] Latency benchmark vs Python von on MPS (see [PERFORMANCE.md](PERFORMANCE.md))
- [x] Single-pass multi-question fan-out (`system_one` semantics) with bounded, stable microbatch batching
- [x] `BatchingStrategy` (`.sequential` / `.automatic` / `.batched`), method parameter only, not HTTP wire
- [x] Absolute B×L² ≤ 1,048,576 guard and amplification ceilings (auto 1.5×, batched 2.0×)
- [x] Right-padded batch forward with flattened variable marker scorer
- [x] Padded query self-attention isolation (padded queries see only themselves)
- [x] GPU batch model parity: max abs 7.05719×10⁻⁵ vs Python batch and 4.386902×10⁻⁵ vs Swift sequential
- [x] Public engine rounded parity GPU passed
- [x] Cancellation support interspersed at batch boundaries
- [x] Release Xcode benchmark harness (diagnostic only; results not committed)

### Phase 9 — SystemOne: multi-model core + d1-omni ✅ DONE

- [x] Rename the package to SystemOne; split `SystemOne` (core), `SystemOneVon`,
      `SystemOneHTTP`, and the `system-one` CLI
- [x] `DecisionBackend` extension point; von behind `VonBackend` with all
      existing tests passing
- [x] Shared weight schemas and pins, I64 safetensors, `ModelProfile` for the
      HTTP boundary, `SystemOneMedia`
- [x] `tools/d1_golden_capture.py` oracle for LiquidAI/d1-omni-600M
- [x] d1 tokenizer, prompt encoding, LFM2.5 trunk, decision head (exact ids;
      logits within 5.8e-5 on GPU)
- [x] d1 vision: torch-exact crops, SigLIP2 tower, projector
- [x] d1 audio: mel front end, FastConformer, adapter
- [x] `system-one serve --model-folder` serves d1 over HTTP

Open follow-ups: d1's own microbatch policy (flat token budget), media over
HTTP, the iOS harness on d1, d1 benchmarks, and training/fine-tuning support
(`D1Backend` already accepts unpinned checkpoints with `artifact: nil`).

## Acceptance criteria (overall)

1. Every phase has a parity test against Python von goldens before merge.
2. No committed weights or venv; only reviewable oracle fixtures and pinned
   tokenizer metadata are checked in.
3. `swift test` green at every commit.
4. Final: end-to-end probabilities match Python von within ~1e-2 on the
   golden suite, latency competitive with Python-on-MPS.