# Decoders (`SystemOneDecoders`)

Decoder trunks for System One models that read decisions from a language
model's hidden states. Apache-2.0.

## Qwen3.5 (`Qwen35Trunk`)

Qwen3.5 interleaves two token mixers. The released models put three Gated
DeltaNet layers before each gated full-attention layer. Pruned models such as
d3-edge mix them freely, so `layer_types` is read per layer. One trunk covers:
- the d3 family (vllm-sr): d3-lite, -mini, -flash, and -edge;
- Perplexity Decider, KnowLine, OneJev, Kev, JEV, and OpenThai-SystemOne;
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
