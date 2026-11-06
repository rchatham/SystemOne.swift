# Decoders (`SystemOneDecoders`)

Decoder trunks for System One models that read decisions from a language
model's hidden states. Apache-2.0.

## Qwen3.5 (`Qwen35Trunk`)

Qwen3.5 interleaves two token mixers. The released models put three Gated
DeltaNet layers before each gated full-attention layer. Pruned models such as
d3-edge mix them freely, so `layer_types` is read per layer. One trunk covers:
- the d3 family (vllm-sr): d3-lite, -mini, -flash, and -edge;
- Perplexity Decider, KnowLine, OneJev, Kev, JEV (served by `JEVBackend`),
  and OpenThai-SystemOne;
- plain Qwen3.5 checkpoints, for generic next-token labels.

The forward pass matches transformers' `Qwen3_5TextModel`:
- **Token embedding.**
- **Each layer:** RMSNorm, then the token mixer, plus a residual; then RMSNorm
  and a SwiGLU MLP, plus a residual.
- **Gated DeltaNet:**
  - Projections: `in_proj_qkv`, gate `z`, `b` (β = sigmoid), and `a` (the decay
    `g = −exp(A_log)·softplus(a + dt_bias)`).
  - A causal depthwise convolution with kernel 4, then SiLU.
  - Queries and keys are L2-normalized (ε = 1e-6), and queries are scaled by
    `1/√d_k`. Key heads repeat up to the value-head count.
  - The gated delta rule runs over them.
  - The output goes through RMSNorm gated by SiLU(`z`), then `out_proj`.
- **Gated attention:**
  - `q_proj` gives each head's query and output gate.
  - RMSNorm on each query and key head.
  - Non-interleaved RoPE on the first `head_dim × partial_rotary_factor`
    features. For text, the multimodal sections all see the same position.
  - Grouped-query attention, scale `1/√head_dim`.
  - The output is multiplied by `sigmoid(gate)`, then `o_proj`.
- **A final RMSNorm.** Every RMSNorm except the gated one stores `w` and applies
  `1 + w`, computed in float32.

The gated delta rule follows transformers' `torch_chunk_gated_delta_rule` in
float32, in chunks of 64 tokens. Within a chunk, the updates combine through
the UT transform's unit lower-triangular system. Forward substitution solves
it, as transformers' export path does. Only the chunks run in sequence, so a
2,048-token prompt takes 32 sequential steps per layer, not 2,048. A Metal
kernel, as mlx-lm uses on the GPU, would be a later optimization.

### Weights

`Qwen35Trunk(modelFolder:)` opens `config.json` and either `model.safetensors`
or a sharded index. The config may nest `text_config` (vision-language
checkpoints) or not (text-only and mlx-lm exports).

The trunk's tensors are found under the first prefix that has
`embed_tokens.weight`:
- `model.language_model.` (`Qwen3_5ForConditionalGeneration`);
- `language_model.model.` (mlx-lm);
- `language_model.` (`Qwen3_5Model`, as the d3 checkpoints ship);
- `model.` (`Qwen3_5ForCausalLM`).

Vision tower and multi-token-prediction tensors are ignored.

Two layouts are told apart by the convolution's shape:

| Layout | `conv1d.weight` | RMSNorm weights | Quantized |
|---|---|---|---|
| `.transformers` | `[channels, 1, kernel]` | `w`, applied as `1 + w` | no |
| `.mlx` | `[channels, kernel, 1]` | `1 + w` already | any MLX affine quantization |

### Attention plans (`DecoderAttention`)

Rows are right-padded, and `lengths` marks the real tokens. A plan applies to
the full-attention layers only. The Gated DeltaNet layers are recurrent and
always causal.

| Plan | Full-attention layers see | Used by |
|---|---|---|
| `.causal` | earlier tokens and themselves | causal readouts, generic labels, prefix caching |
| `.bidirectional` | every real token of the row | d3, Perplexity Decider, StartLux (`noncausal_full_attention`) |
| `.visible(_:)` | a per-row `[query][key]` matrix | block-causal and tree layouts (Jev-Style) |

`positions` gives each real token an explicit RoPE position, for example
options that restart after a shared prefix. Default positions take MLX's fused
RoPE; explicit ones take an explicit rotation computed as transformers does.

### Prefix caches (`DecoderCache`)

`prefill(_:cache:)` extends a cache with one row of tokens under a causal
plan. The cache records, per layer:
- attention keys and values;
- or the convolution window and recurrent state.

`callAsFunction(_:attention:prefix:)` then runs any number of rows after that
prefix: the cached row is broadcast and the cache is not modified.
`copy()` forks a cache so it can be extended along another branch.

A shared state prefix is computed once for all of a request's questions. This
is only valid for models whose prefix tokens never see later tokens, that is
causal readouts. Bidirectional models (d3, Perplexity Decider) run whole rows.

### Readouts

The trunk never computes full-vocabulary logits. `readoutRows(_:)` returns the
output-embedding rows a readout needs:
- from `lm_head`, or from the input embeddings when they are tied;
- dequantized when stored quantized.

A label or code readout's logits are `hidden · rowsᵀ` at the answer position.

### Verification

`tools/qwen35_fixture.py` builds a tiny random Qwen3.5 in transformers 5.22.1:
6 layers, GQA, more value heads than key heads, and partial RoPE. It saves the
model in three forms:
- transformers' vision-language layout, with stray `mtp.` and `model.visual.`
  tensors;
- mlx-lm 0.32.2's float32 export;
- mlx-lm 0.32.2's 8-bit export.

It records hidden states for:
- **padded:** two right-padded causal rows of 150 and 70 tokens, which cross
  chunk boundaries;
- **bidirectional:** the bidirectional plan;
- **tree:** a block-causal row with restarting positions;
- **prefix:** a 37-token prefix, filled in two prefills, then two rows after
  it. These must match running each whole sequence alone.

It also records the readout rows and logits, and mlx-lm's own hidden states
for the 8-bit export.

The Swift tests check both float layouts against transformers, and the 8-bit
export against mlx-lm, within 2e-3. mlx-lm itself differs from transformers by
up to 6e-4 on this model, whose activations reach about 4.

## d3 (`D3Backend`)

The d3 models (vllm-sr, Apache-2.0) are Qwen3.5 checkpoints with a 255-way
answer-code readout, packaged with `decision_config.json`
(`d3-code-readout-v1`). The backend follows the reference's `d3_format.py`
and `d3_runtime.py`.

- **Prompt:** one per question. The system message is fixed. The user turn is:
  ```
  State:
  …

  Question:
  …

  Options:
  A: key: description
  …

  Reply with only the code of the best option.
  ```
  It goes through Qwen's ChatML template with thinking off (`QwenChat`). Fixed
  rules:
  - An empty text state reads `(empty)`.
  - Empty instructions read `Choose the best matching option.`
  - A choice option without a description shows its key alone.
  - A noul shows `false` then `true`, falling back to `No / false` and
    `Yes / true`.
  - A score is a choice over its 2–10 level texts.
- **Tokens:** the prompt is tokenized as rendered. No BOS is added, and
  special-token text becomes its token.
- **Attention:** under `noncausal_full_attention`, the full-attention layers
  see the whole prompt; the Gated DeltaNet layers stay causal.
- **Readout:**
  - The final-norm state at the prompt's last token (in float32) is multiplied
    by `readout.safetensors`' `weight`, `[255, hidden]`.
  - The first k codes are divided by the temperature and softmaxed.
- **Answers:**
  - Choice takes the first maximum, with confidence `1 − H / log k`.
  - Noul is `P(true)`.
  - Score is the expected level index, with the legend in compact sorted JSON.

The reference left-pads its batches and keeps the padding in the positions.
Its README notes that this can flip about 0.17% of near-tie decisions. This
backend right-pads, so each row reads as the reference reads it unpadded.

At load time, every answer code must encode to its listed single token. A
tokenizer that disagrees fails loudly instead of misreading prompts.

### Verification

`tools/d3_fixture.py` copies the reference's prompt and answer code. It
records:
- the full prompt text for three states × six questions, covering choices with
  null and empty descriptions, nouls with and without criteria, text and
  `{what, examples}` scores, and a one-option choice;
- tiny-readout answers through transformers on the tiny Qwen3.5 trunk, with a
  byte-level stub tokenizer.

Swift reproduces the prompts byte for byte. The MLX test runs each state's six
questions in one right-padded batch and matches the unbatched reference
answers within 5e-3. A Python MLX transliteration matches within 7e-6.

## JEV (`JEVBackend`)

[autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) (Apache-2.0) is an
open student of TypeSafe's closed, hosted Jev 1.13. Its model card reports a
mean KL of about 0.019 to Jev's own distributions. It ships three pieces:
- a frozen Qwen3.5-9B text backbone (`Qwen3_5ForCausalLM`, five bf16 shards);
- a peft LoRA (r = 16, α = 32) on every decoder projection;
- a 24-slot float32 head (`head.safetensors`).

The backend runs JEV's System 1 path (typed decisions), following the model
card's transformers + peft reference. TypeSafe Jev itself has no public
weights.

- **Prompt (`bare-v1`):** one per question, tokenized whole with no chat
  template and no special tokens:
  ```
  [kind] choice
  [state] …
  [question] …
  [options]
  A) key: description
  B) key
  [decision]:
  ```
  - A noul always lists `false`, `true`, and a score `0` … `5`. What their
    sides or levels mean travels in the `[question]` line
    (`JEVPrompt.foldedQuestion`).
  - A choice takes 1–16 options (`A)`–`P)`); a score must have exactly six levels.
  - A one-option choice is answered with probability 1 and runs no forward.
  - States render as d3's do.
  - A prompt over `maxLength` tokens (default 4,096) keeps the first 60 % and
    last 40 % of its state's tokens, as JEV's own server truncates.
- **Weights:**
  - `LoRAAdapter` merges the adapter into each targeted linear as
    `Qwen35Trunk` loads it, exactly as peft's `merge_and_unload` does onto
    the bf16 base: `bf16(f32(W) + (α / r) · B @ A)`. Every adapter tensor must
    be used.
  - `loadsReadout: false` skips the untied `lm_head` (2 GB).
  - `JEVArtifacts.jev9B` pins the shards, the adapter and the head at
    revision `7ac669b8fb476ba926491e3d1c39f9330d68dbed`.
- **Attention:** causal everywhere. Rows are right-padded and read at their
  own last token.
- **Readout:**
  - The final-norm state at the last token, in float32, goes through
    `proj.weight` `[24, hidden]` plus `proj.bias`.
  - Slots come from `judge_config.json`: noul 0–1, score 2–7, choice 8–23.
  - A kind's active slots are divided by its `calibration.json` temperature
    and softmaxed.
  - The reference's vLLM path reads the log-probabilities of 24 verbalizer
    tokens instead. The log-softmax normaliser cancels within a kind, so the
    head alone gives the same distribution.
- **Answers:** as d3's (`DecoderAnswers`): Jev's confidence `1 − H / log k`,
  a noul is `P(true)`, and a score is the expected level.

`system-one serve --backend jev --model-folder <snapshot>` serves it.

### Verification

`tools/jev_golden_capture.py` runs transformers' Qwen3.5 classes and peft on
CPU float32, as the oracle. Its reference is exact:
- a bf16 base;
- peft's merge (asserted against `bf16(f32(W) + s · B @ A)`);
- `A_log` and the gated-norm weights at their stored float32 values;
- float32 compute.

It writes two fixtures:
- **`tiny`** (committed, 435 KB): a seeded random Qwen3.5 in JEV-9B's exact
  layout, with two bf16 shards behind an index, an fp32 peft adapter, a
  24-slot head and the JSON configs. It regenerates byte for byte
  (`tiny --check`).
- **`full`** (needs the snapshot): tokenizer ids, prompts, a final-norm probe,
  head logits, and calibrated probabilities for the model card's examples.

| Check | Float32, GPU / CPU |
|---|---|
| Tiny final-norm states and head logits | 8.4e-6 / 1.1e-5 relative |
| Tiny merged weights | within one bf16 rounding |
| Tiny right-padded batch vs single rows | 1e-4 relative bound |
| Tiny bf16 vs float32 (noul probability) | 0.031 / 0.014 |

```bash
# Full model: download, write the fixture, then check tokenizer, prompts and weights
hf download autotrust/JEV-9B --revision 7ac669b8fb476ba926491e3d1c39f9330d68dbed \
  --include "*.json" "*.jinja" "model-*.safetensors" "head.safetensors" "adapter/*" --local-dir weights/JEV-9B
.venv-jev/bin/python tools/jev_golden_capture.py full --model-folder weights/JEV-9B
JEV_MODEL_FOLDER="$PWD/weights/JEV-9B" SYSTEMONE_MLX_TESTS=1 SYSTEMONE_MLX_DEVICE=gpu \
  xcrun xctest -XCTest SystemOneDecodersTests.JEVFullModelTests "$PWD/.build/xcode/Build/Products/Debug/SystemOneDecodersTests.xctest"
```

## LFM2 (`LFM2Trunk`)

`LFM2Trunk` is the causal LFM2 / LFM2.5 text trunk, following transformers'
`Lfm2Model`. It is used for d1-3B (`SystemOneD1`'s `D1ChatBackend`) and other
LFM2-VL checkpoints. Each layer is either a gated short convolution or an
attention layer, then a SwiGLU feed-forward block:
- **Gated short convolution:** `in_proj` gives B, C and x. Then
  `y = C · conv(B · x)`, a causal depthwise convolution with `conv_L_cache`
  taps, followed by `out_proj`.
- **Attention:** grouped-query attention, with query and key RMSNorm per head
  and full-width RoPE.
- **Feed-forward:** a SwiGLU block (`w1`, `w3`, `w2`). Its width is reduced
  when the config sets `block_auto_adjust_ff_dim`.

`embedding_norm` closes the stack. `LFM2Configuration` reads the LFM2-VL
config, which nests `text_config`. Layers come from `layer_types`, or from
`full_attn_idxs` in older configs.

Weights are found under `model.language_model.`, `language_model.model.`
(mlx-vlm) or `model.`. Convolution kernels may be stored as
`[channels, 1, kernel]` or `[channels, kernel, 1]`. The trunk takes the same
right-padded rows, `DecoderAttention` plans and `DecoderCache` prefixes as
`Qwen35Trunk`. `tools/lfm2_fixture.py` records transformers' hidden states for
padded rows and for rows after a prefix that is filled in two prefills. The
MLX tests match them within 1e-4; a Python MLX transliteration matches within
6e-6.

## LFM2-VL pictures (`LFM2VLVision`)

`LFM2VLVision` is LFM2-VL's image path, following transformers'
`Lfm2VlImageProcessor`, `Lfm2VlProcessor`, `Siglip2VisionModel` and
`Lfm2VlMultiModalProjector`. It is used by d1-3B.
- **Processor** (`processor_config.json`):
  - A smart resize to a multiple of 32 pixels per side, holding 64–256 image
    tokens.
  - Pictures above twice that budget are cut into a grid of 2–10 tiles of 512
    pixels, chosen by aspect ratio, plus a thumbnail.
  - Views are resized with torchvision's uint8 antialiased kernel at the
    configured `resample` (bicubic on d1-3B).
  - `markup` writes the `<image>` expansion; `featureCount` gives its token
    count.
- **Tower:** SigLIP2 NaFlex over each view's 16-pixel patches, with its learned
  16 × 16 position grid resampled to the view (antialiased bilinear in
  float32). Then pre-norm layers and a post layer norm. Views of one size
  (the tiles) run as one batch.
- **Projector:** a 2 × 2 pixel unshuffle, an optional layer norm, and two linear
  layers with GELU into the text trunk's width.
- **Weights:** transformers' `model.vision_tower.` (with or without
  `vision_model.`) and `model.multi_modal_projector.`, or mlx-vlm's
  `vision_tower.` and `multi_modal_projector.`. Quantized linears load as such.

`DecoderFeatures` puts the features into the trunk. Each row's placeholder
tokens take its feature rows in order, as transformers' `masked_scatter`
does. `LFM2Trunk` takes them in `callAsFunction` and `prefill`, and a row
whose placeholders and feature rows differ in number is rejected.

`AntialiasedResize` (`SystemOne`) reproduces the integer resizes bit for bit:
- torchvision's ATen kernel: int16 weights at a per-axis precision, width
  first;
- Pillow 13's `Image.resize`: 22-bit weights, with height first when the
  height shrinks by more than twice the width's shrink.

It has the triangle and Keys cubic (a = −0.5) filters.
`tools/resize_fixture.py` records both libraries' outputs, and
`tools/d1_3b_vision_fixture.py` the processor's views (see [D1](D1.md)).

## LFM3-VL pictures (`LFM3VLVision`)

`LFM3VLVision` is LFM3-VL's image path. It follows transformers' LFM3-VL code,
as vendored with d1-2B, and mlx-vlm's `lfm3_vl` port. It is used by d1-2B.
- **Processor:**
  - A smart resize to multiples of 32 pixels within
    `[min_pixels, max_pixels]`, with torchvision's uint8 bicubic kernel.
    Aspect ratios past 200 are refused.
  - 16-pixel patches in 2 × 2 merge-block order, each repeated over the
    temporal patch of 2.
  - `markup` writes `<|vision_start|>`, one `<|image|>` per merge block, and
    `<|vision_end|>`.
- **Tower:**
  - A linear patchifier and an RMSNorm.
  - Then layers that are either a gated depthwise 3 × 3 convolution over the
    raster patch grid (`out(C · conv(B · x))`) or bidirectional grouped-query
    attention. The attention has RMSNorm on query and key heads and axial RoPE
    over the patch's row and column.
  - A SwiGLU block follows each layer.
- **Projector:** a layer norm per patch, each merge block's four patches side by
  side, and two linear layers with GELU.
- **Weights:** transformers' `model.vision_tower.` and
  `model.multi_modal_projector.` with `[hidden, 1, k, k]` kernels, or
  mlx-vlm's names with `[hidden, k, k, 1]`.

- **Video** (`callAsFunction(video:layout:)`, `videoLayout`): the video
  processor samples frames at its rate, resizes them all to one size within
  the clip's pixel budget, and pairs them into temporal patches. Each pair
  runs through the tower and projector as one picture, and `videoMarkup`
  writes each pair's time and tokens.

`LFM3Trunk` takes the features through `DecoderFeatures`, as `LFM2Trunk`
does.

## LFM3-Omni speech (`LFM3AudioEncoder`)

`LFM3AudioEncoder` is LFM3-Omni's speech path. It follows transformers'
LFM3-Omni code, as vendored with d1-Omni-2B, and mlx-vlm's `lfm3_audio` port.
It is used by d1-Omni-2B.
- **Front end:**
  - The clip is resampled to the extractor's 16 kHz (`SincResample`, d1's
    windowed sinc).
  - It is right-padded to `n_fft + k · hop` samples. A periodic-Hann power
    spectrum is taken every 10 ms (400-point FFT), then 128 Slaney mel
    filters, `log10` floored at -12, and `(x + 4) / 4`. All of this runs in
    float32.
- **Encoder:**
  - Three stride-2 3 × 3 convolutions over the (mel, time) plane. The last two
    are depthwise, each followed by a 1 × 1 convolution, and each stage ends
    in SiLU.
  - A linear map of each step's channels × mel bins, channel-major.
  - An LFM2 stack without embeddings (`LFM2Trunk(store:configuration:prefix:)`,
    `callAsFunction(embeddings:window:)`): causal, its attention over the last
    `sliding_window` positions (32 on d1-Omni-2B), then its final norm.
  - One position per 8 frames.
- **Projector:** a layer norm and two linear layers with GELU into the text
  trunk's width.
- **Markup:** `<|vision_start|>`, one `<|audio|>` per position,
  `<|vision_end|>`.
- **Weights:** transformers' `model.audio_encoder.` and `model.audio_projector.`
  with torch's `[out, in, kh, kw]` kernels (`lin_out` a linear layer or a
  width-1 `Conv1d`), or mlx-vlm's `audio_tower.` and `audio_projector.` with
  MLX's `[out, kh, kw, in]`.

`DecoderFeatures` fills several placeholder tokens, each row with its own
features per token. One forward can mix rows with pictures, rows with a clip
and text rows, and one row can hold a picture state and spoken questions.

## LFM3 (`LFM3Trunk`)

`LFM3Trunk` is d1-2B's backbone. It follows mlx-vlm's Apache-2.0 `lfm3` port
of the LFM3 reference; transformers 5.22.1 has no LFM3. Three layer types sit
between sandwich norms, then a final `embedding_norm`:
- **Gated short convolutions:** as in LFM2, with 7 taps.
- **Full-attention layers:**
  - GQA with q/k RMSNorm.
  - A per-head sigmoid output gate (`gate_proj`).
  - LFM3's partial RoPE: it pairs coordinate `i` with `i + d/2` and rotates
    only the first `angles` pairs of each half, with frequencies over the whole
    head. On d1-2B, θ is 1e6 and half the pairs are rotated.
- **Sliding-attention layers:** the same, over the last `sliding_window`
  tokens (256), with full RoPE at θ 1e4.

A `DecoderAttention` plan governs the full-attention layers. Sliding layers and
convolutions stay causal, which is how d1-2B's set attention runs them.
`tools/lfm3_fixture.py` records mlx-vlm's hidden states for:
- padded rows with an 8-token window, so the window applies;
- a d1-2B set-attention row;
- rows after a prefix that is filled in two prefills.

A Python MLX transliteration of the Swift code matches all of them exactly.
