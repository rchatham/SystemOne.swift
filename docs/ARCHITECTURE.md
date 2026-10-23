# Architecture

## The shared engine

`SystemOne` is model-agnostic. A model plugs in as a `DecisionBackend`:

```swift
public protocol DecisionBackend: AnyObject {
    associatedtype Row: PlannedRow            // sequenceLength, logitCount
    var modelID: String { get }
    func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<Row>
    func microbatches(for rows: [Row], strategy: BatchingStrategy) -> [[Int]]  // default provided
    func logits(for rows: [Row]) async throws -> BatchLogitsOutput
}
```

`SystemOneEngine` (an actor) runs every request the same way:

1. Validate model-independent invariants (unique question ids, state keys,
   choice keys; non-empty criteria).
2. `plan`: the backend packs every forward pass up front and attaches to each
   question a `resolve` closure holding its calibration. Invalid input fails
   here, before any forward runs.
3. Group rows into microbatches (default: length-sorted with the bounded
   `B × L²` policy in [PERFORMANCE.md](PERFORMANCE.md)).
4. For each microbatch: check cancellation, run `logits`, check row offsets,
   then resolve every question whose rows are complete, in plan order. A
   postprocessing failure stops before later microbatches run.

Shared building blocks: `SafetensorsFile` (F32/I64, strict layout) with
`WeightSpecification` schemas and `PinnedArtifact` size + SHA-256 pins,
`SafetensorsTensorView.mlxArray()`, `PythonSemantics` (CPython `str.strip`,
`len`, `json.dumps` strings), `AntialiasedResize` (ATen's uint8 and float
antialiased bilinear), `MelFilterbank` (librosa Slaney), and `ModelProfile`,
which lets the HTTP codec bound a request's work before inference without
loading weights.

| Backend | Rows | Notes |
|---|---|---|
| `VonBackend` | 1 per question (+1 null pass for default Noul) | below |
| `D1Backend` | 1 per question, sharing one media prefix | [D1.md](D1.md) |

# von backend

Everything in this doc is verified against von source revision
`6c3c095d03ce78a546cb197df490be9a56c1ad3e` (Apache-2.0) and local
`wfzyx/von` checkpoint revision
`d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd`.

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
| intermediate_size | 2624 per gate branch (5248 combined `Wi` output) |
| vocab_size | 50368 |
| max_position_embeddings | 2048 in the artifact; overridden to 8192 by `OptionMarkerModel` before encoder construction |
| local_attention window | 128 (`config.sliding_window` radius 64; inclusive CPU mask covers up to 129 positions) |
| global_attn_every_n_layers | 3 |
| layer_types | layer 0,3,6,... full_attention; others sliding_attention |
| RoPE θ | 160000.0 (full attn), 10000.0 (sliding attn) |
| embedding implementation | token embedding → LayerNorm → dropout; **no learned absolute position embedding** |
| norm_bias / mlp_bias / attention_bias | all **false** in the backbone |
| hidden_activation | gelu |
| dtype | float32 |
| tie_word_embeddings | true |

ModernBERT specifics to replicate carefully:

- **Hybrid attention**: sliding-window (128) layers + full-attention layers
  every third layer (indices 0, 3, 6, …, 27).
- **RoPE on every attention layer**, with different θ for the two layer types.
  Position ids are absolute `arange(sequence_length)` positions, including in
  sliding-attention layers; they are not window-relative.
- **No learned absolute position embeddings**: despite the artifact's legacy
  `position_embedding_type: "absolute"` field, the installed ModernBERT
  embedding module contains only token embeddings, LayerNorm, and dropout.
- **Gated MLP**: bias-free `Wi: 1024 → 5248`, split into `input` and `gate`
  tensors of width 2624, then `GELU(input) * gate`, MLP dropout (0.0), and
  bias-free `Wo: 2624 → 1024`.
- **Bias-free backbone**: LayerNorm has no bias (`norm_bias: false`), and the
  attention and MLP projections have no biases. `norm_eps` is `1e-05`.

### 3. Scorer head (`OptionMarkerScorer`)

```
x (K, 1024)
  → LayerNorm(1024), affine weight + bias
  → Linear(1024 → 512), with bias
  → GELU
  → LayerNorm(512), affine weight + bias
  → Dropout(p=0.1; disabled by model.eval() during production inference)
  → Linear(512 → 1), with bias
  → squeeze → (K,) logits
```

The scorer is ordinary PyTorch `LayerNorm`/`Linear`, so its affine norms and
linear biases are present even though the ModernBERT backbone is configured
without norm/projection biases.

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
| `model.safetensors` | 1.58 GB | Base ModernBERT plus an unrelated classifier; contains no custom scorer tensors |
| `option_marker.pt` | 1.58 GB | Production OptionMarker state dict: 170 `encoder.*` + eight `scorer.*` tensors, loaded strict |
| `marker_calibration.json` | <1 KB | temperature + calibration_map |
| `tokenizer.json` | 3.6 MB | HF fast tokenizer (byte-level BPE) |
| `tokenizer_config.json` | <1 KB | Tokenizer class and special-token behavior |
| `config.json` | 6 KB | ModernBERT config (table above) |

Phase 1 validates the exact SHA-256 of all six consumed files before loading
them; the pinned values are recorded in `testdata/von_golden_v1.json`.
Production first constructs the base model, then strictly overlays the complete
encoder and scorer from `option_marker.pt`. The 170 corresponding encoder
values differ between the two files, so a parity implementation must convert
and load `option_marker.pt`; the released `model.safetensors` alone is not a
production-equivalent weight source.

### Phase 3 production weight artifact

`tools/convert_option_marker.py` validates the pinned source before loading it
and deterministically emits one float32 safetensors file with the exact 178
production tensors: 170 `encoder.*` and eight `scorer.*`. The output is
1,581,259,300 bytes with SHA-256
`3cadf5c5f001e859489da2cb035aa015b5b74b8d28a384abedb9d86290419e06`.
The source and output hashes are recorded in
`testdata/option_marker_weights_manifest_v1.json`; the companion
`testdata/option_marker_tensor_metadata_v1.json` fixes every tensor's name,
shape, dtype, offsets, byte count, and payload hash. These small fixtures are
committed, but neither source nor converted weight payload is.

The Swift validation layers are deliberately separate:

1. `SafetensorsFile` checks the float32 header and complete gap-free payload
   layout and exposes zero-copy tensor views over a read-only private mmap.
2. `VonWeights` checks exact names and shapes (and defensively confirms float32),
   then exposes typed encoder/scorer groups.
3. `VonWeights.verifiedProduction(contentsOf:)` first checks the pinned file
   size and whole-file SHA-256, then parses and schema-checks the same mapping.

The mmap is tied to the opened inode rather than later pathname contents. The
caller must not mutate or truncate that inode while the `SafetensorsFile` or any
retained tensor view is alive.

```bash
# Normal conversion; validates against the canonical fixtures
.venv/bin/python tools/convert_option_marker.py

# Read-only verification of an existing local output
.venv/bin/python tools/convert_option_marker.py --check

# Intentional canonical fixture regeneration only
.venv/bin/python tools/convert_option_marker.py --update-fixtures

# Optional production mmap/hash/schema/payload-hash Swift integration test
VON_PRODUCTION_WEIGHTS_PATH="$PWD/weights/option_marker.safetensors" swift test
```

The default Swift suite is offline and skips the production-artifact integration
test when `VON_PRODUCTION_WEIGHTS_PATH` is unset.

Tokenizer special-token ids are CLS = 50281, SEP = 50282, PAD = 50283,
and MASK = 50284. Its `bos_token` and `eos_token` are both `None`; only the
model config aliases `bos_token_id` to 50281 and `eos_token_id` to 50282.

## Phase 1 golden capture

`tools/golden_capture.py` invokes the production `evaluate_choice`,
`evaluate_noul`, and `evaluate_score` paths on one CPU float32 backend/model.
The four cases produce five real model forwards because default Noul also runs
a context-free null pass. Each forward records packed text, token ids,
attention mask, mask positions, and raw logits.

Backbone probes have 30 ordered entries: embeddings, post-layer outputs 0–27,
and the separate `final_norm` output. Installed Transformers 5.17's public
`hidden_states` tuple has 29 entries because it replaces raw post-layer 27 with
`last_hidden_state`; explicit hooks retain both boundaries. `final_norm` is the
post-layer-27 normalized `last_hidden_state` consumed by the scorer and stores
every complete K×1024 mask state; all 30 entries store the same 16 probe
dimensions at every mask position.

The fixture also preserves unrounded calibration/debias/postprocessing math
and exact Pydantic production-answer serialization. Output-affecting runtime
pins are CPython 3.12.8, von-sdk 1.1.1 (module version 1.0.1), Torch 2.14.0,
Transformers 5.17.0, Tokenizers 0.23.2, huggingface-hub 1.32.0, Pydantic 2.13.5,
pydantic-core 2.46.5, and safetensors 0.8.0. `--check` compares canonical bytes
first, then permits only finite float-leaf drift within absolute `1e-5` /
relative `1e-6`; structure, order, scalar types, and non-floats remain exact.

## Phase 5 Swift decision implementation

`VonOptionMarkerScorer` copies the eight validated `scorer.*` tensors into
MLX-owned float32 arrays. Its inference graph is exactly affine
LayerNorm(1024, ε=1e-5) → biased Linear(1024→512) → exact GELU → affine
LayerNorm(512, ε=1e-5) → biased Linear(512→1); the Python module's dropout is
training-only and is not represented in inference. `VonOptionMarkerModel`
composes this head with `VonModernBERT`, validates marker indices and token ids,
and gathers `[MASK]` states before scoring.

`VonDecisionMath` owns only request-independent numerical behavior; Phase 6
will supply request structs and state formatting. Calibration intentionally has
a mixed-precision boundary matching Python/Torch:

1. Stable softmax, `p * log(clamp_min(p, 1e-12))`, and entropy summation run in
   float32.
2. The resulting float32 entropy scalar is promoted to double; entropy
   normalization, `log10` token feature, JSON coefficients, feature sum, and
   temperature clamp run in double precision.
3. Dividing logits by temperature and the calibrated softmax return to
   float32. Default-Noul's `0.7 * null_bias` adjustment is also float32.
4. Choice ties retain the first raw-logit maximum. Serialized decision values
   use nearest-even rounding after float32 probabilities are promoted to
   double.

The embedded map values come from pinned `marker_calibration.json` SHA-256
`c3b5345c89be9ee802a5587dcda4e530e69a9bb5965dd839b877b7e8cfff048f`.
CPU/GPU Xcode-bundle parity covers five isolated scorer forwards, five composed
raw-logit forwards, four calibrated probability vectors, and exact rounded
answers guarded by each value's distance from a rounding boundary. Global max
absolute errors are `9.536743e-7` / `2.861023e-6` (isolated scorer CPU/GPU),
`6.1035156e-5` / `5.9604645e-5` (composed logits), and `4.991889e-6` /
`4.068017e-6` (decision probabilities).

## Phase 6 public Swift API

`VonBackend` is the only high-level production owner of the
`VonTokenizer` and `VonOptionMarkerModel`, driven by `SystemOneEngine`. Its async initializer accepts an
explicit tokenizer folder and a verified production safetensors URL; no model
name resolution, network fallback, or HTTP exists. `systemOne` uses a two-phase
plan-then-execute pipeline with bounded, stable microbatch fan-out (see
[PERFORMANCE.md](PERFORMANCE.md)). Each packed input is run and `eval`uated
under actor isolation before logits are materialized as `[Float]` and passed
to `VonDecisionMath`.

The public `Choice`, `Noul`, `Score`, `Question`, answer, request, response, and
usage models are typed `Codable & Sendable` values. All inference-significant
collections are ordered arrays: `StateField`, `ChoiceOption`, and
`QuestionEntry`. Codable request and response output retains Python's keyed
object shapes and adds explicit order metadata (`state_order`, `question_order`,
`criteria_order`, `answer_order`, `probability_order`, and `legend_order`);
decoding validates exact key coverage and reconstructs arrays from those fields
instead of relying on `KeyedDecodingContainer.allKeys` order. The extra fields
are ignored by Python's Pydantic defaults. Phase 7 supplies a separate
order-preserving parser for raw Python requests and responses that lack this
metadata.

`BatchingStrategy` is a method parameter on `systemOne` (`.sequential` /
`.automatic` / `.batched`). It is not serialized, not on the HTTP wire, and
not part of `SystemOneRequest`. The default is `.automatic`. See
[docs/PERFORMANCE.md](PERFORMANCE.md) for the microbatch policy and benchmarks.

Primitive rendering is shared with tokenizer packing and exactly preserves the
pinned backend's edge semantics:

- Choice uses the stripped key only when a description is `nil` or exactly
  empty. Whitespace-only descriptions are truthy first and then strip to empty.
- Noul computes explicitness before stripping. Either non-empty recognized side
  (including whitespace) disables null debiasing; each missing/empty side gets
  its own default independently.
- Score accepts ordered strings or `what`/`examples` values and renders examples
  with the exact `" Examples: "` and `", "` separators.
- Usage counts Python Unicode code points and applies the backend's approximate
  character-count formula rather than tokenizer counts.

Swift intentionally rejects empty Choice options, empty Score levels, duplicate
state/Choice keys, and duplicate question IDs. Python can return synthetic
empty answers and cannot express most duplicates through dictionaries; Swift
chooses safe, unambiguous inputs instead. Offline injected-logit tests reproduce
all four fixture answers and five packed forwards, and the public API exact
rounded outputs pass through the production model on Xcode MLX CPU and GPU.

## Phase 8 bounded microbatch fan-out

`SystemOneEngine.systemOne` pre-plans every inference row (primary + default-Noul null
passes) before any forward pass. Rows are sorted by descending sequence length
and greedily binned into microbatches by `SystemOneEngine.microbatchIndices`.

### Microbatch policy

Two guards, both checked with overflow-aware integer arithmetic:

- **Absolute**: `rows × max(L²) ≤ 1,048,576`
- **Relative** (amplification): `rows × max(L²) ≤ maxAmplification × ΣL²`

`maxAmplification` is `nil` (`.sequential` — each row alone), `1.5`
(`.automatic`), or `2.0` (`.batched`). The absolute bound always applies when
`maxAmplification` is non-nil.

### Right-padded batch forward

`VonOptionMarkerModel.forward` accepts ragged `[VonPackedInput]` rows:

1. Right-pad all `inputIDs` to the maximum row length with the tokenizer's PAD
   token id.
2. Extend attention masks: real tokens keep their original mask; padding
   positions get `0`.
3. Flatten marker positions from row-local to padded global indices.
4. Run a single `VonModernBERT.forward` on the batch.
5. Reshape hidden states to `(B×L, 1024)`, gather at flattened marker
   positions, score, and return row-major logits with `rowOffsets`.

Padded queries in sliding-attention layers see only themselves via
`row[query] == 0 ? query == key : …`, so padding never bleeds into real marker
representations.

### Cancellation

`Task.checkCancellation()` is called at each batch boundary, both in the
injected-provider and production paths.

### Default Noul null row

Default Noul's null-state pass (empty state) is pre-planned as a separate row
and may co-batch with the primary row if lengths are compatible.

## Phase 7 HTTP boundary

`SystemOneHTTP` separates raw wire compatibility from Phase 6's typed Codable
contract. `OrderedJSON` parses and serializes objects without losing insertion
order or permitting duplicate keys. `SystemOneWireCodec` converts metadata-free
Python requests into ordered Swift types and emits successful responses in the
same field/object order as pinned Python `von serve`; no `_order` metadata or
Swift keyed-container iteration participates in this path.

The codec deliberately implements a strict subset at the trust boundary. It
caps JSON depth/nodes/string/container sizes and inference-significant work
(question count, planned forwards and markers, options, score examples, and
rendered/packed bytes), rejects empty Choice/Score criteria, literal mask
injection, and ambiguous duplicate keys, and supports only the Python coercions
covered by parity tests. Before inference it also serializes a conservative
successful response shape so escaped request-controlled answer/Choice/
probability keys and Score legends cannot exceed the 1 MiB response cap after
work has already run. Successful
Choice, default/explicit Noul, Score, reordered state/criteria/questions, and
GET payloads match ordered Python response bytes. Rejected inputs compare by
HTTP status category because Swift sanitizes errors and intentionally accepts a
narrower input language.

`SystemOneHTTPServer` is a bounded SwiftNIO HTTP/1.1 shell: raw/decoded metadata and
body budgets, connection/request/inference admission caps, absolute header/body/
request deadlines, idle timeout, timing-safe Bearer auth, and bounded graceful
shutdown. It emits `100 Continue` only after authentication, declared body
length/size, and request-admission checks; unsupported expectations receive 417
without waiting for a body. Authentication is enabled by `SYSTEMONE_API_KEY`; a
non-loopback bind without a key fails closed. CORS is disabled unless `SYSTEMONE_CORS_ORIGINS` supplies
an explicit allowlist. This intentionally differs from Python's wildcard CORS
default, and wildcard origins are rejected.

The `system-one serve` executable requires explicit `--port`, `--tokenizer`, and
`--weights`, defaults only to loopback host `127.0.0.1`, validates local files,
and installs SIGINT/SIGTERM shutdown handling. Opt-in live parity uses an
external pinned Python server URL and starts Swift only on `127.0.0.1:0`;
`tools/run_http_parity.sh` performs the safe loopback Python lifecycle and
Metal-enabled Xcode test invocation without using ports 8000 or 8014.

## Porting notes / risk areas

1. **Tokenizer parity** is the highest-risk item — byte-level BPE with
   ModernBERT's pre-tokenization regex must match exactly, or everything
   downstream diverges. Verify token ids against goldens before anything else.
2. **Sliding-window attention semantics**: on the CPU oracle,
   `local_attention=128` produces an inclusive `abs(query-key) <= 64` mask, or
   up to 129 positions for an interior token. The module's separate value 65 is
   a FlashAttention adapter detail, not the effective CPU radius. RoPE still
   uses absolute sequence positions.
3. **Weight provenance**: use the deterministic Phase 3 safetensors conversion
   of production `option_marker.pt`; do not substitute the released base-model
   safetensors. Use `verifiedProduction(contentsOf:)` when byte-for-byte pinned
   artifact identity, rather than schema compatibility alone, is required.
4. **Null-state debiasing** is easy to miss; it materially changes Noul outputs
   only when neither explicit criterion description is truthy.
5. float32 everywhere first; consider fp16/bf16 later for speed, but parity
   tests must run in the same precision as the goldens.
6. **CPU benchmarks** can trigger macOS's Metal watchdog on large batches;
   CPU is diagnostic only. GPU (Release Xcode, Metal) is the acceptance target.
7. **Batch padding** in sliding-attention layers requires that padded queries
   see only themselves; the `row[query] == 0` guard prevents padding from
   bleeding into real marker representations.
