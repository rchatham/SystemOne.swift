# System One model survey

This survey covers System One model architectures and source models that
SystemOne could support next, as of 2026-11-03. It answers four questions:

- Where do the two supported models stand?
- Which architectures has the field moved to?
- What does the library need to run them?
- How has the wire protocol moved?

Model claims cite the file they came from. Leaderboard numbers come from one
pinned board file unless marked *self-reported*.

## Summary

- **Both supported models now rank near the bottom of the main leaderboard.**
  - On the Decision Index 0.4.1 board (248 entries, generated 2026-10-29),
    d1-omni-600M scores 11.75 (#228) and von 1.3 scores 11.65 (#229).
  - The leading Apache-2.0 encoders under 500M score 17–31.
  - The leading decoder decision models of 0.8–4B score 26–56.
- **One request shape, nine readouts.** Every model below answers the same
  `choice` / `noul` / `score` request with zero generated tokens, but they read
  the answer in nine distinct ways ([taxonomy](#answer-mechanisms)). Most fit
  `DecisionBackend` unchanged, because it already expresses rows, multi-row
  questions, and rows shared by several questions.
- **Three backbones unlock most of the field.**
  - A generalized ModernBERT-family encoder: about 20 encoder models, including
    Laya, Julia-1, DecisionTune, Amber and the Ettin-based decision models.
  - A Qwen3.5 hybrid decoder trunk (Gated DeltaNet plus gated attention):
    nearly every open decoder model, including d3, Perplexity Decider,
    KnowLine, Mapika decider, Kev, JEV, OneJev and OpenThai.
  - The existing LFM2 trunk in `SystemOneD1`, made causal: d1-3B.
- **The first blocker is the weight loader, not the network code.** It accepts
  only F32 and I64 safetensors. Nearly every new checkpoint is BF16, F16 or
  MLX affine-quantized (U32 plus scales and biases), and several are sharded.
- **LiquidAI's d1 line has moved on.** d1-3B (LFM2.5-VL, causal, answers read
  from tied embedding rows) and d1-2B / d1-Omni-2B (LFM3, "set" attention,
  new vision and audio towers, eight question types) are new architectures.
  Only d1-omni-600M's MLX exports drop into the current module.
- **The protocol grew past noul, choice and score.**
  - TypeSafe's Jev 1.14 adds `box` questions, image `assets`, optional
    `instructions`, and zstd request bodies.
  - d1-2B adds multi-label choice, extraction and estimates.
  - Gateways are moving to OpenAI's `POST /v1/decisions`, which has a
    `refusal` answer.
- **Treating any LLM as a System One model is now mainstream.** vLLM, SGLang
  and the Decision Index kit read next-token label logits with prefix
  caching. Ollama's MLX runner ships native readouts for d1, d3, Kev, Laya,
  Clef and Perplexity Decider. Both are reference implementations for a
  Swift port.
- **Recommended order** ([roadmap](#recommended-roadmap)):
  1. Loader work, verified on d1-omni-600M's MLX exports.
  2. A generalized ModernBERT encoder module.
  3. A Qwen3.5 decoder module with row-gather readouts.
  4. d1-3B, then d1-2B.
  5. Wire and schema extensions.

## Where the supported models stand

**The board.** `hf://spaces/multimodalart/jev-decision-index/data/v03.json`,
labelled "Decision Index 0.4.1", generated 2026-10-29T14:04:16Z, 248 models.
- The Full score `v` = 0.6 × same-skills + 0.4 × new-domains. Both parts are
  run privately by the maintainers.
- The public index (`pub`) has weight 0 in this edition, so public-index
  numbers on model cards are not comparable to `v`.
- Models slower than 1,000 ms on one RTX PRO 6000 are excluded.

| Model | Params | `v` (rank) | `pub` |
|---|---|---|---|
| Jev 1.14 (closed, TypeSafe) | — | 60.24 (#49) | 60.70 |
| LiquidAI d1-3B | 3B | 31.12 (#169) | 48.99 |
| **LiquidAI d1-omni-600M** (supported) | 0.6B | **11.75 (#228)** | 17.90 |
| **Von 1.3** (supported) | 0.4B | **11.65 (#229)** | 13.86 |

d1-2B and d1-Omni-2B are not on the board. Their cards self-report Decision
Index v0.3 scores of 51.1 and 49.0, which is a different edition and scale.

**Other benchmarks.** On JevBench v1.6.1 the leaders are Gemma-4-31B decoders:
Quyet-1.0-Large scores 81.69 Capability, and Jev 1.13 scores 77.13. Models
under 500M score 31–43. On jabr classifier-benchmark v2 (49 tasks, 866 cases),
von 1.1/1.2 reached 0.724 micro accuracy against 0.967 for hosted Jev 1.13.
Von therefore holds up on short routing tasks better than its Decision Index
rank suggests.

**The von port is one generation behind von.** It targets von-sdk 1.1.1
(`docs/ARCHITECTURE.md`).
- Von 1.2 introduced order-invariant scoring (`independent_options: true`):
  each option attends only to the premise and itself, and position ids
  restart.
- Per the model card, 49.5% of answers flipped under option reordering in
  1.1, against 0 in 1.2 (`hf://models/wfzyx/von/README.md`).
- `Sources/SystemOneVon` has no such mask, so von 1.2/1.3 weights would run
  in the wrong attention mode.
- Von is archived upstream, which lowers the priority. But the Bekko and
  tasksource tree-mask models need the same mask.

## Answer mechanisms

Every model takes a state and typed questions and returns a distribution over
each question's options. They differ in where the logits come from.
"Rows/question" is the `DecisionPlan` shape.

| # | Mechanism | How option logits are read | Rows/question | Models |
|---|---|---|---|---|
| 1 | **Option markers + MLP** (encoder) | marker token before each option, MLP on its hidden state, softmax over the question's markers | 1 | von ≤1.1, Crivo, Dinah-1, JEV Ettin 150M, DecisionTune (single Linear), PICK |
| 2 | **Option markers + transformer head** (encoder) | type embedding, 2 pre-norm encoder layers, then an MLP at the markers | 1 | d1-omni-600M, Laya (English / multilingual / typed), Julia-1 |
| 3 | **Order-invariant tree mask** (encoder) | each option sees the shared prefix and itself only; positions restart after the prefix; marker MLP or mean-pool + Linear | 1 | von 1.2/1.3, Bekko v0, tasksource-decider-nano |
| 4 | **Label tokens / schema markers** (encoder) | labels as tokens in one sequence, scored against pooled text (GLiClass), at `[L]` markers (GLiNER2), or by cosine to a pooled state plus a marker MLP (vela3) | 1 | gliclass-v4-edge, Verdict 1.4, GLiNER2.5-Decide, vela3-fast |
| 5 | **Pair scoring** (encoder or decoder) | (state, option) scored per option by entailment, yes−no, a scalar head, or a learned scorer over separately encoded vectors; softmax across options | K or 1 + K | NLI zero-shot (kenning, raz-systemone-nli, MoritzLaurer), Open-Jev, pngwn scorer, Amber (`MLP([c; o; c⊙o])` over mean-pooled context and option encodings) |
| 6 | **Row-gather at a slot** (decoder) | hidden state at one answer position × a small matrix: LM-head rows of label codes, a separate readout file, or a K-slot head | 1 (or shared) | d3 (255 codes), Perplexity Decider, KnowLine, StartLux, Mapika decider, OneJev, JEV (24-slot head), OpenThai-SystemOne (256 slots; slot 255 = abstain), Jev-Omni, d1-3B (tied embedding rows, max over aliases), any chat LLM through next-token labels |
| 7 | **Yes − no per option** (decoder) | `h · (w_yes − w_no)` at a slot after each option, with all options in one sequence | 1 | Jev-Style 2B v4 |
| 8 | **Pointer head** (decoder) | `(W_k h_optEnd_k + b_k) · (W_q h_query + b_q) / 16` between each option's closing token and a query token | 1 | Kev 0.8B–27B |
| 9 | **Joint schema head** (decoder) | a separate network over all hidden states (Clef: 121.8M, cross-attention plus decoder layers) giving one logit per option for every question at once | shared | Cloudflare clef / clef-flash, Jeva-4B |

The second axis is attention. A decoder decision model is often *not* run
causally, so a causal engine silently serves a different model:

| Attention plan | Where it applies | Models |
|---|---|---|
| Bidirectional encoder | all layers | ModernBERT, LFM2.5-Encoder, DeBERTa and UltraBERT models |
| Tree mask (options independent) | all layers | von 1.2+, Bekko, tasksource nano |
| Causal | all layers | KnowLine, Mapika decider, JEV, Kev, OpenThai, d1-3B, generic chat LLMs |
| Non-causal full attention | full-attention layers only; DeltaNet and conv layers stay causal | d3 (`"attention_mode": "noncausal_full_attention"`), Perplexity Decider |
| Prompt-bidirectional | full-attention layers only | StartLux (`"prompt_attention": "bidirectional"`), AutoTrust-JEV-27B v1.1 |
| Block-causal (≤2,048-token blocks) | full-attention layers only | Jev-Style 2B v4 |
| "Set": state ↔ all questions, each question ↔ itself | full-attention layers only; sliding and conv layers stay causal | d1-2B, d1-Omni-2B (`"d1_attention": "set"`, up to 16 questions) |

**Prefix caching is exact only for causal plans.** For causal models, the
backend can prefill the state once, then fork the KV cache (plus the DeltaNet
and conv state) for each question. Every row comes out bit-identical to an
uncached row; the Core AI port of Kev demonstrates this. Under non-causal or
prompt-bidirectional plans, state tokens attend to question tokens. The
backend must then re-read the state for every row or pack the questions into
one sequence. vLLM, for example, disables prefix caching for d3.

**Multi-question rows already work.** `SystemOneEngine.run` gathers each
decision's logits by `rowIndices`, and nothing stops two decisions from naming
the same row. A backend that packs several questions into one row gives each
decision a `resolve` closure that slices its own options out of the row's
logits. This covers d1-2B "set", Mapika decider, Perplexity joint mode and
Clef.

## Backbones and what they unlock

| Backbone | Status in repo | Unlocks |
|---|---|---|
| ModernBERT family: ModernBERT-base and -large, mmBERT, Ettin | `VonModernBERT` hard-codes ModernBERT-large: vocab 50,368, 28 layers, RoPE `offset: 0`, key-padding and sliding masks only | Laya English (config identical to von's trunk: `hf://models/convaiinnovations/laya/encoder/config.json`); Amber and DecisionTune (unchanged trunk); Laya multilingual and Julia-1 (mmBERT: 22 layers, RoPE θ 160k on sliding layers, 256k Gemma-style vocab); Dinah-1, Bekko, tasksource-decider-nano, JEV Ettin, Crivo, Mola, PICK, GLiNER2.5-Decide-1B, ModernBERT NLI models, von 1.2/1.3 |
| LFM2 / LFM2.5 (conv + GQA) | `D1Trunk`: bidirectional, config-driven `layer_types` | d1-3B, once made causal (left-padded conv, causal mask, `rope_parameters`, no FFN auto-adjust); prism-system-one (first 11 layers of LFM2.5-Encoder-230M) |
| LFM3 (kernel-7 conv, sliding attention, partial-RoPE full attention, sigmoid output gate, sandwich norms) | none | d1-2B, d1-Omni-2B |
| Qwen3.5 / Qwen3.8 hybrid (Gated DeltaNet and gated full attention, 3:1) | none | d3, Perplexity Decider, KnowLine 4B/9B, StartLux, Mapika decider, OneJev, Kev, JEV, OpenThai-SystemOne, Jev-Style, Clef, GLiDE |
| Gemma 4 | none | KnowLine 31B, Sieve, Quyet, Jev-Omni, GEV-26B-Decide, Mapika decider 12b/31b |
| DeBERTa-v3 / mDeBERTa (disentangled attention) | none | gliclass-v4-edge, GLiNER2.5-Decide and multi-Decide, NLI zero-shot models |
| UltraBERT (Llama-style encoder blocks: RMSNorm, SiLU GLU, 128-token local window on 2 of 3 layers, RoPE θ per layer) | none | vela3-fast, vela3-fast-vision (`hf://models/dmis-lab/UltraBERT-380M/config.json`) |

**mlx-swift-lm is not a drop-in dependency.** At commit b8d4935,
`ml-explore/mlx-swift-lm` implements `qwen3_5`, `qwen3_8`, `gemma4`, `lfm2`,
`lfm2_vl` and `llama`, but it has two gaps for this library:
- Hidden states and mask construction are internal. The public API returns
  only LM-head logits, so a slot readout or a non-causal plan cannot be
  expressed without forking it.
- It requires mlx-swift 0.33, while this package pins 0.31.4 exactly.

Its `GatedDelta` Metal kernel, cache `copy()` and bidirectional-mask helper are
still useful references. Ollama's MLX runner (`mlxrunner/model/{d1, d3, kev,
laya, clef, pplxdecider, decisionqwen, lfm3}`) is the closest existing
MLX implementation of these readouts.

## Candidate models

Board `v` is from Decision Index 0.4.1 unless marked otherwise. Licenses are
from each model card. "NC" means non-commercial (CC-BY-NC-4.0).

### On-device (≤1B, iPhone class)

| Model | HF id | Params | Mechanism | License | `v` | Fit and notes |
|---|---|---|---|---|---|---|
| GLiClass v4 edge | `knowledgator/gliclass-v4-edge` | 0.44B | 4, plus bge-small label model and calibrator | Apache-2.0 | 31.21 | Needs DeBERTa-v3. Best sub-1B model on the board. |
| OneJev1.6-0.8B | `OmniJev/OneJev1.6-0.8B` | 0.8B | 6 (readout not yet read) | Apache-2.0 | 29.58 | Qwen3.5-0.8B full fine-tune; images and video. |
| vela3-fast | `vllm-sr/vela3-fast` | 0.38B (decision backbone) | 4: cosine(option + question, pooled state) / τ + MLP(option marker) | Apache-2.0 | 29.27 | Needs UltraBERT. Decisions use only `model.safetensors` (1.52 GB fp32); span and router adapters are optional. No score type. |
| d3-lite / d3-edge | `vllm-sr/d3-lite`, `d3-edge` | 0.8B / 0.6B | 6: 255-code readout, non-causal full attention | Apache-2.0 | 26.16 / 17.69 | Qwen3.5 trunk. |
| Amber 0.4B | `MLukauskas/amber-0.4b` | 0.4B | 5: 1 context row + K option rows | NC | 26.00 | Frozen ModernBERT-large plus a 20.7 MB LoRA and scorer. Runs on `VonModernBERT` after merging the LoRA; the scorer must stay fp32. |
| tasksource Decider nano v5 | `tasksource/tasksource-decider-nano` | 150M | 3 | Apache-2.0 | 19.85 | Ettin, BF16. Choice and noul only. |
| JEV Ettin 150M | `Elfsong/jev-ettin-150m-v16` | 150M | 1 | Apache-2.0 | 18.52 | Ettin. |
| Bekko v0 400M / 68M / 17M | `hotchpotch/bekko-system-one-v0-*` | 400M / 68M / 17M | 3 | not yet assigned | 17.98 / 12.40 / 7.66 | The 17M is a 29 MB ONNX. Wait for a license. |
| DecisionTune 1.0 | `decision-tune/decisiontune-1.0` | 395M | 1 (Linear head) | Apache-2.0 | 17.22 | **Trunk identical to von's**; the cheapest model to add. |
| Dinah-1 | `Lukitaduarte/dinah-1` | ModernBERT-base | 1 | Apache-2.0 | 16.36 | English and Portuguese. |
| Kev 0.8B | `jaredpalmer/kev-0.8b` | 0.8B | 8 | Apache-2.0 | 16.30 | Qwen3.5. Causal; the state is shared across questions. |
| Crivo 1B … 17M | `tardellirs/crivo-*` | 1B–17M | 1 | NC | 29.25 … 11.65 | Best quality per size; the 17M ties von 1.3. |
| Laya multilingual / Julia-1 | `convaiinnovations/laya` (`multilingual/`), `SupersonicLabs/Julia-1` | 322M / 144M | 2 | Apache-2.0 | 9.21 (Laya English) / 4.75 | mmBERT, 100+ languages. Self-reported typed-decisions 73–77%. Needs fp32 compute because of massive activations. |
| OpenThai-SystemOne | `iapp/OpenThai-SystemOne` | 0.8B | 6: 256-slot head with abstain | Apache-2.0 | not on board | Thai and English. Official MLX builds. |
| prism-system-one | `paulprt/prism-system-one` | ~0.2B | 1, several questions per sequence; noul by sigmoid | LFM Open License | not on board | Reuses `D1Trunk`. Preview release. |

### iPad / Mac (2–4B)

| Model | HF id | Mechanism | License | `v` | Notes |
|---|---|---|---|---|---|
| KnowLine 4B Gen5.5 | `PelaAI/KnowLine-4B-Gen5.5` | 6, causal; T = 1.5 folded into the final norm | Apache-2.0 | 55.99 | Simplest contract; prefix cache valid. |
| StartLux-Decision v1.1 4B | `startlux-models/StartLux-Decision-v1.1-4B-Instant` | 6, prompt-bidirectional | NC | 54.80 | Official MLX builds. |
| Perplexity Decider Flash v1.2 | `perplexity-ai/pplx-decider-flash-v1.2` | 6, non-causal; joint inline readout for all questions | Apache-2.0 | 54.29 | Shares readout code with d3. |
| d3-mini / d3-nano | `vllm-sr/d3-mini`, `d3-nano` | 6, non-causal; 255 codes (`decision_config.json`, `readout.safetensors`) | Apache-2.0 | 49.97 / 34.30 | One d3 format covers 0.6B–27B. |
| Mapika decider 4b / 2b / 0.8b / 2b-vision | `Mapika/decider-*` | 6, causal; every question in one sequence | Apache-2.0 | 36.30 / 23.65 / — | The schema-first layout caches the questions across states. |
| Kev 4B | `jaredpalmer/kev-4b` | 8 | Apache-2.0 | 34.93 | |
| Jev-Style 2B v4 | `chaoliangUNSW/Jev-Style-2B-Decision-v4` | 7, block-causal | Apache-2.0 | 34.53 | |
| LiquidAI d1-3B | `LiquidAI/d1-3B` | 6: tied embedding rows; no calibration | LFM Open License | 31.12 | LFM2 trunk made causal; SigLIP2 so400m vision features inlined in the text. |
| LiquidAI d1-2B / d1-Omni-2B | `LiquidAI/d1-2B`, `d1-Omni-2B` | 6 at answer slots under "set" attention | LFM Open License | *self-reported DI v0.3 51.1 / 49.0* | New LFM3 trunk, vision tower and audio encoder. Adds multi-label, extraction, estimates and boxes. |

### High-memory Mac (9B and up)

| Model | HF id | `v` | Notes |
|---|---|---|---|
| KnowLine 9B / 31B | `PelaAI/KnowLine-9B-Gen1`, `-31B-Gen1` | 58.44 / 71.86 (#1) | Apache-2.0. The 31B is Gemma 4. |
| Perplexity Decider v1.2 (27B) | `perplexity-ai/pplx-decider-v1.2` | 67.98 | Apache-2.0; same code as Flash. |
| d3 / d3-ultra (27B), d3-flash (9B) | `vllm-sr/d3*` | 67.15 / 70.18, 54.27 | Apache-2.0. |
| AutoTrust-JEV-27B v1.1 | `autotrust/AutoTrust-JEV-27B-v1.1` | 67.95 | Apache-2.0; prompt-bidirectional. |
| JEV-9B / 27B / 35B-A3B | `autotrust/JEV-*` | 42.15 / 57.35 / 57.61 | Apache-2.0. Distilled from Jev 1.13; not TypeSafe weights. |
| Fastino GLiDE 1.2 | `fastino/GLiDE` | 71.52 (#2) | About 26B Qwen3.5 hybrid; 52 GB bf16. Gated, `fastino-community` license. Also a hosted `/v1/systemone` API. |

## What the library needs

Ordered by how many models each item unlocks.

### 1. Weight formats (`SystemOne`)

- Add BF16, F16 and U32 to `SafetensorsDataType`. Only F32 and I64 are
  accepted today (`Sources/SystemOne/Safetensors.swift`).
- Read sharded checkpoints through `model.safetensors.index.json`.
- Load MLX affine quantization:
  - Read `quantization {group_size, bits}` from `config.json`.
  - Treat a tensor `X` with `X.scales` and `X.biases` beside it as quantized.
  - Load into `QuantizedLinear` / `QuantizedEmbedding`.
- Add per-format tensor name maps. The MLX exports rename and split PyTorch
  tensors; for example, the head's fused `in_proj` becomes `query/key/value_proj`.
- Extend the `Float16Overflow` guard to BF16 → F16.
- Merge LoRA adapters at conversion time (Amber, JEV).

### 2. Config polymorphism

- `rope_parameters` per layer type.
- `head_dim`, `block_auto_adjust_ff_dim`, `sliding_window` and sandwich norms.
- `model_type` dispatch.
- Model-specific files and keys: `d1_*`, `decision_config.json`,
  `calibration.json`, `readout_config.json`.

### 3. A generalized bidirectional encoder

- Config-driven ModernBERT dimensions.
- RoPE θ per layer type.
- **Explicit position ids**, replacing `MLXFast.RoPE(offset: 0)`.
- **Arbitrary block masks**, with the sliding window computed from position ids.
- A Gemma-style tokenizer path for mmBERT.

### 4. A decoder trunk protocol and readouts

- **Prefill:** returns final-norm hidden states at chosen positions.
- **Cache:** `makeCache` and `copy` for prefix forks.
- **Attention plan:** applied to full-attention layers only.
- **Readout kinds:**
  - `rows(matrix, ids, bias)` covers mechanism 6 and never computes
    full-vocabulary logits.
  - `yesMinusNo` and `pointer` cover mechanisms 7 and 8.
  - `plugin` covers joint heads.
- **Generic next-token-label mode** for any supported chat model:
  - Labels: A–Z for choice, yes/no for noul, 0–9 for score.
  - Report label mass, the probability the full vocabulary puts on the labels.
  - Calibrate: on JevBench, raw Qwen3-4B label logits scored 35.2 on
    Calibration against 90.6 for Jev 1.13.

### 5. Shared prefixes

- A backend-owned cache keyed by prefix tokens.
- It overrides `microbatches(for:strategy:)` to group rows by prefix.
- It applies only when the attention plan is causal.
- d1's media prefix is the existing precedent.

### 6. Answers and limits

- **Abstain probability.** Optional; from OpenThai slot 255, StartLux's hidden
  "none of the above", and OpenAI's `refusal`.
- **Uncalibrated flag.** A way to say an answer is uncalibrated (d1-3B; d3,
  where T = 1).
- **Option cap.** Raise `maximumOptionsPerQuestion` (default 64) to the
  model's own limit from `ModelProfile`: 255 for most decoders and for the
  TypeSafe and Decision Index schemas.

### 7. Calibration as data

One canonical JSON per model, pinned by SHA-256, holding:
- Scalar T, T per type, T per (type, option-count bucket), or slot bias.
- A "folded" flag (KnowLine).
- The score rule: mean, median or argmax.
- The noul option order.

Pickled `head.pt` files (Kev, Open-Jev, Jev-Omni) need a Python conversion
tool like `tools/convert_option_marker.py`.

### 8. Parity fixtures

- Keep the per-model Python golden capture.
- The Core AI zoo publishes fixture files usable as a second oracle, not a
  replacement:

  | Model | Fixture size |
  |---|---|
  | Mapika decider | 44 rows |
  | Kev | 434 questions |
  | Clef | 216 records |
  | APUS | 48 rows |
  | decider-2b-vision | 157 slots |
  | OpenThai | 50 questions |
  | Laya multilingual | 201 rows |
  | Julia-1 | publisher logits |

### 9. Licensing boundaries

- `SystemOneD1` is LFM-licensed. d1-3B and d1-2B belong there or in a sibling
  LFM-licensed module.
- Laya and Julia's head has the same structure as d1's (type embedding, two
  pre-norm layers, scorer). An Apache module for them needs a clean-room head
  written from the Laya reference, not a copy of `D1DecisionHead`.
- Non-commercial models (Crivo, StartLux, Amber, Mola, PICK, the pngwn scorer)
  need explicit labelling if supported at all.

## Protocol and runtimes

### Schema diff

Sources:
- TypeSafe: the official Python SDK `typesafe-sdk` 0.7.5, generated from
  `api.typesafe.ai/openapi.json`.
- Decision Index: `apolinario/decision-index` `docs/format.md`.
- SystemOne.swift: `Sources/SystemOneHTTP/SystemOneWireCodec.swift`.

| Feature | TypeSafe Jev 1.14 | Decision Index 0.4 | SystemOne.swift |
|---|---|---|---|
| Question types | `noul`, `choice`, `score`, `box` (image grounding) | `choice`, `noul` (no `score`) | `noul`, `choice`, `score` |
| `instructions` | optional on every type (noul then needs criteria) | string or object | **required**, string only |
| Missing `type` | required | — | defaults to `choice` (von compatibility) |
| Criterion values | string, object, array or null | string | string or null; score also `{what, examples}` |
| Choice options | non-empty; 255 enforced server-side | 2–255 | ≤64 by default over HTTP |
| `state` | string, object or array | string or object | string or ordered object; 4,096 UTF-8 bytes by default over HTTP |
| Images | `assets` map, referenced from state, instructions or criteria (images as options); multipart bodies | vision suite: `images` (1–4) | library only (`SystemOneMedia`); text-only over HTTP |
| Request body encoding | the SDK sends `Content-Encoding: zstd` above 1 KB when zstd is available | — | not handled |
| `GET /v1/models` | `{"models":[{name, description, release_date}]}` | — | the von profile emits both shapes; the default profile (d1) emits only OpenAI-style `data` |
| Abstention | none | `abstained` status, scored as wrong | none |

Beyond these, other providers have extended the schema:

- **d1-2B** adds `choice` with `multiple: true` (one P(yes) per option).
  - `text` / `number` / `date` reads return `{value, span, found}`.
  - `number` with `range`/`step` returns a median and a p10/p90 interval.
  - `box` / `boxes` locate regions in an image.
  - `score` returns the median level, not the expected value.
- **OpenAI `POST /v1/decisions`** (served by vLLM, SGLang and llmgateway):
  - Questions: `predicate`, `choice` and `score`, up to 200 per request.
  - Answers come as an array in question order, with a `{"type": "refusal"}`
    answer.
  - llmgateway marks `/v1/systemone` "deprecated soon" in favour of it.

Neither TypeSafe nor any runtime surveyed exposes `independent_options` on the
wire; it is a property of von's weights.

### Runtimes

| Runtime | Generic LLMs? | How answers are read |
|---|---|---|
| vLLM | yes: `NextTokenStrategy` (Qwen3, Qwen3.5, Gemma 4) | Single-token labels at the first reply token; softmax over label logprobs; `diagnostics.label_mass`; prefix caching prefills the state once. Native heads: Clef, d3, Liquid d1. Also serves `/v1/decisions`. |
| SGLang | yes | One prefill per question through `/v1/score`; request `temperature` divides the logits. |
| Decision Index `transformers` engine | yes | One pass over the prompt plus all option tokens with a 4D mask and restarted position ids; KV reused across questions. |
| llama.cpp `llama-server` | no ("Only native decision models") | Per-family readouts: label token, pplx codes, Kev pointer, Laya and d1-omni markers, Clef joint rows, d3 / d1 non-causal codes. Returns 501 otherwise. |
| Ollama | no | `/v1/systemone`, choice 2–26. The MLX runner has decision code for d1, d3, Kev, Laya, Clef and Perplexity Decider. |
| coreai-kit (Swift, Core AI on iOS/macOS 27) | via Foundation Models (one-hot, no probabilities) | 13 decision models across marker, slot, pointer, tied-embedding, letter and joint readouts; `TypedDecisions` prefills a state once for N questions. |

**What this means for SystemOne.swift:**

- **Generic-LLM backends are established.** The recipe is consistent across
  runtimes: single-token labels, thinking off, read the answer position,
  softmax over the labels, report label mass, cache the state prefix. It
  belongs in the Phase 12 decoder module.
- **A remote backend is cheap and useful.** Many hosted endpoints serve
  `/v1/systemone`: TypeSafe, Fastino GLiDE, OpenRouter, Microsoft Foundry,
  Cloudflare, Liquid. A remote backend is a request-level service rather than
  a row-level `DecisionBackend`, so it needs its own small protocol in
  `SystemOne`.
- **Core AI targets the Neural Engine but needs OS 27.** This package targets
  iOS 17 and macOS 14. Keep MLX as the runtime and use coreai-kit's fixtures
  and readout contracts as references.
- **Two test suites can drive the existing server directly.** The jabr TOML
  suites (CC0, hash-locked) and the Decision Index `http` engine both talk to
  `POST /v1/systemone` with no adapter.

## Recommended roadmap

| Phase | Work | Proves or unlocks | Parity oracle |
|---|---|---|---|
| 10 | Loader: BF16/F16/U32, shards, MLX affine quantization, name maps | d1-omni-600M MLX fp16 (1.17 GB download) and 8-bit (0.73 GB; per its card, 175/175 top answers kept with max Δp 0.060), through the existing `SystemOneD1` | existing `testdata/d1_golden_v1.json` |
| 11 | Generalized ModernBERT module (Apache): config, position ids, block masks | DecisionTune (no new network code), then Laya English / multilingual and Julia-1 (clean-room head), tasksource nano, JEV Ettin, Dinah-1; von 1.3's order-invariant mode | each model's Python reference; Core AI fixtures for Laya and Julia |
| 12 | Qwen3.5 decoder module: trunk, attention plans, row-gather readout, causal prefix cache, generic next-token labels | d3 (0.6B–9B), Perplexity Decider Flash, KnowLine 4B, OneJev 0.8B, Mapika decider, OpenThai; then Kev's pointer readout | `decision_config.json` codes plus Python reference; Core AI fixtures |
| 13 | d1-3B: causal LFM2 trunk, tied embedding readout, ChatML prompt, so400m vision with bicubic resize and inline image tokens | LiquidAI's 3B model, reusing Phase 12's readout and Phase 10's loader | `hf://models/LiquidAI/d1-3B` code |
| 14 | d1-2B / d1-Omni-2B: LFM3 trunk, "set" mask, new vision and audio towers; noul / choice / score first, as llama.cpp does | LiquidAI's current flagship | `hf://models/LiquidAI/d1-2B` code (transformers ≥5.19) |
| 15 | Wire and schema: optional `instructions`; object and array criteria; 255 options; abstain; TypeSafe-shaped `/v1/models` for every profile; reject `Content-Encoding: zstd` with 415 until supported; then multi-label, `box`, extraction, `assets` | TypeSafe SDK and Decision Index clients unchanged; d1-2B's full API | TypeSafe OpenAPI schema; `tools/run_http_parity.sh` pattern |
| — | DeBERTa-v3 and UltraBERT encoders | gliclass-v4-edge and vela3-fast, the two strongest Apache sub-500M models; GLiNER2.5-Decide | Python references on the Hub |
| — | Evaluation harness: jabr v2 TOML suites and the Decision Index `http` engine against `system-one serve` | comparable numbers for every backend added | — |

Phases 11 and 12 are independent and can run in parallel once Phase 10 lands.
Phase 10 comes first because every later model needs it, and d1-omni-600M's
existing fixture can verify it without new oracle work. The quick wire fixes
in Phase 15 (optional `instructions`, the option cap, `/v1/models`, the zstd
rejection) are small and can land at any time.

## Not verified

- **Unread readout code:** OneJev's readout, KnowLine's layout file, and
  d3's handling of DeltaNet layers under non-causal attention. The tables
  infer d3's from vLLM's and Perplexity's descriptions.
- **Card-only figures:** model sizes and latencies quoted from cards were not
  measured here.
- **Search-summary sources:** pages that the session network could not reach
  (docs.typesafe.ai, llmgateway.io, hackernoon.com, liquid.ai,
  huggingface.co over HTTP). Facts from them are limited to the TypeSafe
  version history.

## Sources

Model files were read through the Hugging Face Hub (`hf://` paths) and
GitHub. Key sources:

- **Board:** `hf://spaces/multimodalart/jev-decision-index/data/v03.json`
  (Decision Index 0.4.1, 2026-10-29).
- **d1:** `hf://models/LiquidAI/{d1-omni-600M,d1-3B,d1-2B,d1-Omni-2B}` and
  their `-MLX-*` exports (`config.json`, `modeling_*.py`, `prompt.py`,
  `runner.py`, `attention.py`, `calibration.json`, `README.md`).
- **von:** `hf://models/wfzyx/von/README.md`.
- **Encoders:**
  - `hf://models/convaiinnovations/laya`, `hf://models/SupersonicLabs/Julia-1`
  - `hf://models/fastino/GLiNER2.5-Decide`,
    `hf://models/knowledgator/gliclass-v4-edge`
  - `hf://models/hotchpotch/bekko-system-one-v0-68m`,
    `hf://models/tardellirs/crivo-150m`,
    `hf://models/tasksource/tasksource-decider-nano`
  - `hf://models/decision-tune/decisiontune-1.0`,
    `hf://models/MLukauskas/amber-0.4b`
  - `hf://models/vllm-sr/vela3-fast`, `hf://models/dmis-lab/UltraBERT-380M`
  - `hf://models/paulprt/prism-system-one`
- **Decoders:**
  - `hf://models/vllm-sr/d3-mini/decision_config.json`
  - `hf://models/perplexity-ai/pplx-decider-flash-v1.2`
  - `hf://models/PelaAI/KnowLine-4B-Gen5.5`
  - `hf://models/Mapika/decider-0.8b`
  - `hf://models/jaredpalmer/kev-0.8b`
  - `hf://models/autotrust/JEV-9B`
  - `hf://models/iapp/OpenThai-SystemOne`
  - `hf://models/OmniJev/OneJev1.6-0.8B`
  - `hf://models/chaoliangUNSW/Jev-Style-2B-Decision-v4`
  - `hf://models/startlux-models/StartLux-Decision-v1.1-4B-Instant`
  - `hf://models/fastino/GLiDE`
- **Core AI ports and readout contracts:**
  - `hf://models/mlboydaisuke/*-CoreAI`
  - [john-rocky/coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)
  - [coreai-kit docs/SYSTEM_ONE.md](https://github.com/john-rocky/coreai-kit/blob/main/docs/SYSTEM_ONE.md)
- **Protocol:**
  - [typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python)
    (`_core/question_types.py`, `_schemas/models.py`, `_core/compression.py`)
  - [decision-index docs/format.md](https://github.com/apolinario/decision-index/blob/HEAD/docs/format.md)
  - [llmgateway System One docs](https://github.com/theopenco/llmgateway/blob/HEAD/apps/docs/content/(gateway)/features/system-one.mdx)
- **Runtimes:**
  - [vLLM structured decisions](https://github.com/vllm-project/vllm/blob/HEAD/docs/models/structured_decisions_models.md)
  - [SGLang decision models](https://github.com/sgl-project/sglang/blob/HEAD/docs/docs/supported-models/decision_models.mdx)
  - [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/HEAD/tools/server/README.md)
  - [Ollama decision docs](https://github.com/ollama/ollama/blob/HEAD/docs/capabilities/decision.mdx)
    and [`mlxrunner/model`](https://github.com/ollama/ollama/tree/main/mlxrunner/model)
  - [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) at b8d4935
- **Benchmarks:**
  - [JevBench](https://github.com/fstandhartinger/jevbench)
  - [jabr classifier-benchmark](https://github.com/jabr/classifier-benchmark)
