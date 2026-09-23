# Von Architecture (reference map for the Swift port)

Everything in this doc is verified against the von source clone at
`~/Developer/von` (commit of `master`, Apache-2.0) and the cached checkpoint at
`~/.cache/huggingface/hub/models--wfzyx--von/`.

## What von is

A non-autoregressive decision model. Instead of generating text, it packs a
question, the observed state, and all K candidate options into **one sequence**
and reads K logits off dedicated marker positions in a single forward pass.
Three decision primitives:

| Primitive | What it computes | Output |
|---|---|---|
| Choice | categorical distribution over K options | argmax (raw logits), probabilities, confidence = p₁ − p₂ |
| Noul | calibrated P(condition is true) | single probability in [0,1] |
| Score | expected ordinal level over K ordered levels | Σ i·pᵢ on [0, K−1] |

## Inference pipeline (the path we port)

```
state + question + options
        │
        ▼
pack_sequence ──► "{question} {state} [SEP] {option1} {option2} ..."
        │         (each option prefixed by a  mask token)
        ▼
ModernBERT-large encoder ──► last_hidden_state (1, seq_len, 1024)
        │
        ▼
gather hidden states at mask positions ──► (K, 1024)
        │
        ▼
OptionMarkerScorer head ──► K raw logits
        │
        ▼
temperature T (input-conditioned calibration map)
        │
        ▼
softmax(logits / T) ──► per-primitive answer
```

### 1. Sequence packing (`src/von/models/option_marker.py`)

```python
prefix = f"{question} {state}".strip()          # empty question -> just state
opts_packed = " ".join(f"{mask} {opt}" for opt in options)
return f"{prefix} {sep} {opts_packed}"
```

Key point: all options attend to the state **and to each other** via full
bidirectional attention — one pass instead of K cross-encoder passes.

### 2. Backbone — ModernBERT-large (`config.json` in `wfzyx/von`)

| Config | Value |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| intermediate_size | 2624 |
| vocab_size | 50368 |
| max_position_embeddings | 2048 (code default claims 8192, shipped config says 2048) |
| local_attention window | 128 |
| global_attn_every_n_layers | 3 |
| layer_types | layer 0,3,6,... full_attention; others sliding_attention |
| RoPE θ | 160000.0 (full attn), 10000.0 (sliding attn) |
| position_embedding_type | absolute (learned position embeddings + RoPE hybrid) |
| norm_bias / mlp_bias / attention_bias | all **false** |
| hidden_activation | gelu |
| dtype | float32 |
| tie_word_embeddings | true |

ModernBERT specifics to replicate carefully:

- **Hybrid attention**: sliding-window (128) layers + full-attention layers
  every third layer (indices 0, 3, 6, …, 27).
- **RoPE on every attention layer**, with different θ for the two layer types.
- **Learned absolute position embeddings** (token_pos embeddings) in the
  embedding layer, in addition to RoPE.
- **Bias-free**: LayerNorm without bias (`norm_bias: false`), attention and MLP
  projections without biases.
- `norm_eps` = 1e-05, `layer_norm_eps` = 1e-05.

### 3. Scorer head (`OptionMarkerScorer`)

```
x (K, 1024)
  → LayerNorm(1024)
  → Linear(1024 → 512), no bias
  → GELU
  → LayerNorm(512)
  → Linear(512 → 1), no bias
  → squeeze → (K,) logits
```

### 4. Calibration (`marker_calibration.json` shipped in the repo)

Scalar default `temperature: 2.2` plus an input-conditioned map:

```
feats = {
  bias: 1.0,
  entropy: H_norm,            # normalized entropy of unscaled softmax probs
  log_tokens: log10(n)/4.0,   # n = real tokenizer count of state text
  n_options: K/8.0,
}
T_raw = 0.2056 − 3.255·H_norm + 20.2391·(log10(n)/4) − 5.2263·(K/8)
T = clamp(T_raw, 0.3, 12.0)
```

Notes:

- Temperature is monotonic in the logits — never moves the argmax, only
  sharpness of reported confidence.
- Entropy uses unscaled probs; n_tokens uses the **real tokenizer** on the
  state text (`add_special_tokens=False`), not a character proxy.
- The calibration map was fitted on a JevBench subset; see
  `marker_calibration.json` fields for provenance.

### 5. Per-primitive behavior (`src/von/backends/option_marker_backend.py`)

**Choice**
- probabilities = softmax(logits / T)
- argmax taken on **raw** logits (same answer either way — monotonic)
- confidence = clamp(p₁ − p₂, 0, 1), rounded to 3 decimals; probs rounded to 4

**Noul**
- two options: `criteria["true"]` / `criteria["false"]`; defaults
  "Yes, condition holds true." / "No, condition is false."
- **null-state debiasing** when neither explicit criterion is given:
  1. pack `pack_sequence("", instructions, descriptions)` (empty state)
  2. run model → null logits
  3. bias = null_logit[0] − null_logit[1]
  4. logits = [logits[0] − 0.7·bias, logits[1]]
- answer = probs[0] (probability of "true")

**Score**
- levels: each item is a string, or dict with `what` + optional `examples`
  (rendered as `"{what} Examples: {', '.join(examples)}"`)
- expected value = Σ i·pᵢ, rounded to 2 decimals
- confidence = p₁ − p₂ like Choice

**State formatting** (`_format_state`): str passes through; dict →
`"key: value"` lines joined with `\n`.

## Checkpoint files (all cached locally under the HF hub cache)

| File | Size | Role |
|---|---|---|
| `model.safetensors` | 1.58 GB | ModernBERT-large encoder (+ unused classifier/head keys) |
| `option_marker.pt` | 1.58 GB | Full OptionMarkerModel state_dict (encoder + scorer), loaded strict |
| `marker_calibration.json` | <1 KB | temperature + calibration_map |
| `tokenizer.json` | 3.6 MB | HF fast tokenizer (byte-level BPE) |
| `config.json` | 6 KB | ModernBERT config (table above) |

Special token ids: CLS/BOS = 50281, SEP/EOS = 50282, PAD = 50283,
MASK = 50284 (from tokenizer config).

## Porting notes / risk areas

1. **Tokenizer parity** is the highest-risk item — byte-level BPE with
   ModernBERT's pre-tokenization regex must match exactly, or everything
   downstream diverges. Verify token ids against goldens before anything else.
2. **Sliding-window attention semantics**: window 128 around each token, with
   RoPE positions relative to the window — check `transformers`' ModernBERT
   implementation for the exact masking/position convention.
3. **RoPE + absolute position embeddings coexisting** — ModernBERT quirk.
4. **Null-state debiasing** is easy to miss; it materially changes noul outputs
   when no explicit criteria are given.
5. float32 everywhere first; consider fp16/bf16 later for speed, but parity
   tests must run in the same precision as the goldens.