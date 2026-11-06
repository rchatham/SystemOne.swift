# JEV-9B (reference map for the Swift port)

[autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) is an open,
Apache-2.0 student of TypeSafe's closed, hosted **Jev 1.13** System One model:
a frozen Qwen3.5-9B text backbone, a LoRA (r = 16, α = 32) that adapts it for
decisions, and a 24-slot fp32 decision head. Its model card reports a mean KL
of ≈ 0.019 to Jev 1.13's own distributions. `SystemOneJev` runs its System 1
path (typed decisions) natively; System 2 (text generation) is out of scope.

Verified against revision `7ac669b8fb476ba926491e3d1c39f9330d68dbed` with
transformers 5.17.0 (`modeling_qwen3_5.py`), peft 0.21.0 and torch 2.14.0. The
pinned files are in `JevArtifacts.jev9B`: five backbone shards (17.9 GB),
`adapter/adapter_model.safetensors` and `head.safetensors`.

TypeSafe Jev itself has no public weights, so it cannot run here. The von
profile's `/v1/models` lists a `jev-latest` alias for byte parity with the
upstream von server; it is a name only, and requests' `model` field never
selects a backend.

## What differs from d1 and von

| | d1-omni | JEV-9B |
|---|---|---|
| Backbone | LFM2.5-Encoder, 16 layers, bidirectional | Qwen3.5-9B text: 24 Gated DeltaNet + 8 gated attention layers, causal |
| Prompt | special-token delimiters, one marker per option | plain `bare-v1` text, no special tokens |
| Readout | head over marker positions | last position → linear head, 24 fixed slots |
| Answer sets | any options and levels | noul `false/true`, score `0–5`, choice 2–16 options |
| Calibration | per type and option-count bucket | one temperature per kind (`calibration.json`) |
| Weights | one float32 file | five bf16 shards + fp32 LoRA merged at load + fp32 head |
| Media | images, audio | none (`unsupportedMedia`) |

Answers are unrounded softmaxes; confidence is the top probability, as d1's.

## Prompt (`JevPrompt`)

```
[kind] choice
[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
[question] Supplier response for this scenario.
[options]
A) issue_warning
B) renegotiate
[decision]:
```

- The whole string is tokenised once with `add_special_tokens=False`; there is
  no chat template. Nothing is escaped, as in the reference, so
  `JevModelProfile.markerText` is `nil`: the answer is read at the final
  position, not at markers a state could forge.
- A text state passes through; ordered fields serialize like
  `json.dumps(state, ensure_ascii=False)`.
- Choice options read `key: description`, or the key alone (d1's convention),
  labelled `A)` … `P)`. More than 16 options are rejected; one option is
  answered with probability 1 and runs no forward.
- Noul options are always `false`, `true`, and score options `0` … `5`; a score
  must have exactly six levels. What a noul's sides or a score's levels mean
  is folded into the `[question]` line by `JevPrompt.foldedQuestion`.
- A prompt longer than `maxLength` tokens (default 4,096) keeps the first 60 %
  and last 40 % of its state's tokens, as JEV's own server truncates.

## Backbone (`JevModel`)

Every layer is `x + mixer(norm(x))` then `x + SwiGLU(norm(x))`, with
Qwen3.5's zero-centred RMSNorm (`norm(x) · (1 + w)`, computed in float32).

- **Gated attention** (layers 3, 7, …): 16 query heads and 4 KV heads of
  dimension 256. `q_proj` emits each head's query and output gate side by
  side; q and k get per-head RMSNorms; RoPE (θ = 1e7) rotates the first 25 %
  (64 dimensions) of each head. For text, every mRoPE axis carries the same
  position, so mRoPE is plain RoPE. The output is gated by `sigmoid(gate)`.
- **Gated DeltaNet** (the other 24): `in_proj_qkv` feeds a causal depthwise
  convolution (4 taps) and SiLU, then splits into 16 query/key heads and 32
  value heads of 128; keys and queries repeat-interleave to 32 heads.
  `β = σ(in_proj_b x)`, `g = −exp(A_log) · softplus(in_proj_a x + dt_bias)`.
  The gated delta rule (`JevDeltaRule`) runs in float32 in chunks of 64, the
  unit-triangular system solved by forward substitution (the reference's
  export path; mlx-swift's `solveTriangular` is CPU-only). The output passes a
  gated RMSNorm, `norm(o) · w · SiLU(z)`, and `out_proj`.
- Rows are right-padded and read at their own last position. Both mixers are
  causal, so padding never reaches a real position.

## Weights and readout (`JevWeights`)

- `ShardedSafetensors` opens the five shards behind
  `model.safetensors.index.json`, pinning each one. Everything is bf16 except
  `A_log` and the gated-norm weights, which stay float32 as released.
  `lm_head` (2 GB) is validated but never loaded.
- The LoRA merges at load exactly as peft's `merge_and_unload` does onto a
  bf16 base: `bf16(f32(W) + (α / r) · B @ A)`.
- The head is `proj.weight` `[24, 4096]` and `proj.bias` `[24]` in float32.
  Slots (from `judge_config.json`): noul 0–1, score 2–7, choice 8–23. The
  reference's vLLM path reads log-probabilities of 24 verbalizer tokens; the
  log-softmax normaliser cancels within a kind, so the head alone is exact.
- `JevPrecision.bfloat16` (the default) computes as released;
  `.float32` computes over the same bf16 weights in float32, matching the
  golden fixtures.

## Verification

Two fixtures, both from `tools/jev_golden_capture.py` running transformers'
own Qwen3.5 classes and peft on CPU float32:

- `tiny` (committed, 435 KB): a seeded random Qwen3.5 model in JEV-9B's exact
  layout. It checks every layer, the LoRA merge, the head, padded batching and
  the delta rule without downloading anything.
- `full`: the pinned snapshot. Tokenizer ids, prompts, a per-layer trunk probe,
  and decision probabilities for the model card's examples.

```bash
swift test --filter SystemOneJevTests     # prompts, planning, validation (no MLX)

# MLX parity on the tiny model, via the Xcode test bundle
xcodebuild build-for-testing -scheme SystemOne-Package -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
JEV_DEVICE=gpu xcrun xctest -XCTest SystemOneJevTests "$PWD/.build/xcode/Build/Products/Debug/SystemOneJevTests.xctest"

# Full model (~18 GB): download, then tokenizer, trunk and decision parity
hf download autotrust/JEV-9B --revision 7ac669b8fb476ba926491e3d1c39f9330d68dbed \
  --include "*.json" "*.jinja" "model-*.safetensors" "head.safetensors" "adapter/*" --local-dir weights/JEV-9B
JEV_MODEL_FOLDER="$PWD/weights/JEV-9B" JEV_MODEL_PARITY=1 JEV_DEVICE=gpu \
  xcrun xctest -XCTest SystemOneJevTests.JevFullModelTests "$PWD/.build/xcode/Build/Products/Debug/SystemOneJevTests.xctest"

# Regenerate or check the fixtures (venv: torch 2.14.0, transformers 5.17.0, peft 0.21.0)
.venv-jev/bin/python tools/jev_golden_capture.py tiny --check
.venv-jev/bin/python tools/jev_golden_capture.py full --check --model-folder weights/JEV-9B
```

Full-model parity needs about 16 GB of unified memory for the weights, so it
runs locally rather than on hosted CI runners.
