# Encoders (`SystemOneEncoders`)

Shared bidirectional encoders for the decision models built on them. Apache-2.0.

## ModernBERT family (`ModernBERTEncoder`)

ModernBERT, mmBERT, and the Ettin encoders are one architecture with different
`config.json` values, so one implementation serves them all:

| Model family | Layers × width | Sliding-layer RoPE θ | Vocabulary |
|---|---|---|---|
| ModernBERT-large (von, DecisionTune, Laya English, Amber) | 28 × 1024 | 10k | 50,368 |
| mmBERT-base / -small (Laya multilingual, Julia-1) | 22 × 768 / 384 | 160k | 256,000 |
| Ettin 17M–1B (Bekko, Crivo, tasksource-decider-nano, JEV Ettin) | 7–28 × 256–1792 | 160k | 50,368+ |

`ModernBERTConfiguration` reads both config forms: transformers 5's
`layer_types` and per-type `rope_parameters`, and the original
`global_attn_every_n_layers`, `global_rope_theta`, and `local_rope_theta`.

The forward pass matches transformers' `ModernBertModel`:
- token embedding, then LayerNorm;
- per layer: pre-norm attention (layer 0 has no attention norm) with fused QKV,
  non-interleaved RoPE, and scale `1/√d_head`; full attention on `full_attention`
  layers, and on sliding layers only keys within `local_attention / 2` positions
  (inclusive);
- a gated MLP, `Wo(act(x) · gate)`, with exact GELU (or the tanh form when the
  config asks for it);
- a final LayerNorm.

Weights load from any `TensorStore` under a prefix with the Hugging Face names,
dense or MLX-quantized. `ModernBERTEncoder.schema(for:prefix:)` lists them for a
model's full schema.

### Attention layouts (`EncoderAttention`)

Right-padded rows with `lengths` give a plain bidirectional encoder. Decision
models that score options independently also give:

- `positions`: per-token position ids. Options restart after the shared prefix,
  so every option sits at the same positions.
- `visible`: per-row `[query][key]` visibility. An option sees the prefix and
  itself; the prefix sees only the prefix.

The sliding window is measured on the position ids, as von 1.2 does. Default
positions take MLX's fused RoPE kernel; explicit ones take an explicit rotation
computed as transformers computes it.

### Verification

`tools/modernbert_fixture.py` writes `testdata/modernbert_tiny/`: two tiny random
encoders, one with ModernBERT-large's RoPE and one with mmBERT/Ettin's. For each
it records transformers 5.22.1's hidden states for a padded batch and for a tree
layout passed as custom per-type 4-D masks with explicit position ids. Swift
matches both within 1e-4 on the CPU in CI. Shifting the positions by a constant
leaves the output unchanged, which ties the explicit rotation to the fused kernel.

## Decision models

Three open encoder decision models run as `DecisionBackend`s. Each follows its
reference code's prompt and head. The tokenizer comes from the model folder
(`HubEncoderTokenizer`, swift-transformers, nothing downloaded).

| Backend | Models | Prompt per question | Head | Answers |
|---|---|---|---|---|
| `DecisionTuneBackend` | DecisionTune 1.0 | `[CLS] instructions [SEP] ([MASK] option)… [SEP] state [SEP]` | `Linear(d, 1)` at each `[MASK]` | plain softmax; noul options are `yes`/`no`, `noul = P(yes)`; confidence is the top probability |
| `MarkerHeadBackend` | Laya (English, multilingual), Julia-1 | `[CLS] "<type> question: …" [SEP] ([MASK] " option")… [SEP] state [SEP]`, options capped at 48 tokens and shrunk to fit the head budget | type embedding, PyTorch `TransformerEncoderLayer`s, then LayerNorm → Linear → GELU → Linear at each `[MASK]` | Laya: per-bucket temperature, confidence `1 − H/log k`; Julia: no temperature, top probability; `noul = P(true)` |
| `TreeDeciderBackend` | tasksource-decider-nano | one row per request: `[CLS] state [SEP]`, each `"Question: q\nOption:"`, each `" option [SEP]"` | `Linear(d, 1)` over each option block's mean state | plain softmax; choice and noul only |

Option texts follow each reference. Laya renders `key: description`,
`level i: …`, and `false: …`/`true: …` with default descriptions. Julia reads
the descriptions alone and rejects a choice option without one. The tasksource
decider drops placeholder keys (`option_1`, `a`, `(b)`). Score levels in the
`{what, examples}` form are rendered as Python prints the dict.

The tasksource decider's tree layout comes from `decider.py`'s "shared" layout,
"opt" pooling, and "cached" attention:
- every token sees the state;
- question and option tokens see their question block;
- option tokens see their own option.

Positions follow the virtual `(state, question, option)` pair. Rows hold at most
`max(2P, P + 2048)` tokens for a `P`-token state, and the state is repeated in
each row. A question whose options spill over repeats its block in the next row.

`SystemOneEngine.decisionTune(modelFolder:)`, `.markerHead(modelFolder:)`, and
`.treeDecider(modelFolder:)` build engines from local folders.

### Verification

`tools/encoder_decisions_fixture.py` writes `testdata/encoder_decisions/`. Each
model's rendering code is copied from its reference, and the script records the
packed ids and markers for a text state and a fields state across choice, noul,
and score questions, with the ModernBERT tokenizer. It also records a 12-option
Laya prompt that hits the head budget, and the tasksource tree rows. Swift
matches these exactly in `swift test`.

For numerics, the script runs each reference head with random weights on the
tiny ModernBERT of `testdata/modernbert_tiny/large-rope`:
- DecisionTune's linear head;
- Laya's two-layer `TransformerEncoder` head with the type embedding;
- the tasksource scorer on a tree row with custom masks and position ids.

The MLX tests match each within 1e-4 on the CPU in CI.
