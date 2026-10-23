# SystemOne Performance (von)

Phase 8 latency benchmarks and batching architecture. All Swift numbers are
from the Release Xcode-built test bundle on an Apple M4 Max (64 GB RAM,
macOS 15.6, Swift 6.2), `VON_PRODUCTION_WEIGHTS_PATH` pointing at the
verified `weights/option_marker.safetensors`. Python MPS numbers use the same
machine, `von-sdk==1.1.1`, 30 iterations with 3 warmup.

## Benchmark fixture

The heterogeneous fixture is a single four-question `systemOne` request
against one shared incident-state paragraph:

| Question ID | Type | Options |
|---|---|---|
| `root_cause` | Choice | four root-cause domains |
| `urgent_payment_failure` | Noul | no explicit criteria (triggers null pass) |
| `rollback_required` | Noul | explicit true and false criteria |
| `incident_severity` | Score | four ordered severity levels |

Five inference rows are pre-planned (four primary + one null-state pass for
the default Noul). Their sequence lengths differ because instructions and
rendered criteria vary; the planner uses those exact token lengths when forming
microbatches.

The homogeneous fixture uses 4, 8, or 16 identical Noul questions with a
single explicit criterion and the same state text — all rows have identical
length.

## Swift MLX GPU results

### Heterogeneous fixture (4 questions, 5 rows)

| Strategy | Mean | p50 | p95 | σ |
|---|---|---|---|---|
| `.sequential` | 67.0 ms | 66.9 ms | 67.7 ms | 0.5 ms |
| `.automatic` (default) | 55.0 ms | 55.0 ms | 55.5 ms | 0.4 ms |
| `.batched` | 46.5 ms | 46.5 ms | 47.0 ms | 0.3 ms |

`.automatic` (max amplification 1.5×) clusters the two similar-length default
rows into one microbatch while keeping the shorter rows separate, trading some
padding overhead for lower peak memory. `.batched` (max amplification 2.0×)
packs all five rows into fewer microbatches at the cost of more right-padding.

### Homogeneous rows (n identical Noul questions)

| Rows | `.automatic` | `.sequential` |
|---|---|---|
| 4 | 16.1 ms | 40.6 ms |
| 8 | 25.8 ms | 79.4 ms |
| 16 | 46.3 ms | 159.5 ms |

Identical-length rows all fit in a single microbatch under automatic policy;
the batch forward amortises ModernBERT's quadratic attention cost.

## Python MPS results

Same fixture, 30 iterations, 3 warmup. Python runs on the same M4 Max via MPS.

| Path | Mean | p50 | p95 | σ |
|---|---|---|---|---|
| Production `system_one` (sequential API) | 77.7 ms | 76.8 ms | 81.8 ms | 3.0 ms |
| Direct batched model (bypasses API) | 48.5 ms | 48.7 ms | 51.8 ms | 2.3 ms |

Python's production `system_one` evaluates questions sequentially — it has no
batching API. The direct batched path constructs one contiguous tensor and
calls the underlying model once, but it bypasses `system_one`'s validation,
null-state debiasing, and calibration. It is a physical model latency
reference, not a supported API.

## Batching architecture

### Two-phase plan-then-execute

`SystemOneEngine.systemOne` pre-plans every inference row before any forward pass:

1. **Plan** — render state, options, and criteria; pack each question's primary
   row (and null-state row for default Noul); record expected logit counts.
2. **Execute** — sort rows by descending length, greedily bin into microbatches
   under the strategy's amplification ceiling, run one backbone+scorer forward
   per microbatch, then scatter logits back to their questions.

This avoids the Python production path's question-at-a-time sequence packing
and lets the model amortise attention over multiple rows.

### Microbatch policy

`SystemOneEngine.microbatchIndices` sorts by descending sequence length, then
greedily fills each batch while satisfying two guards:

- **Absolute**: `rows × max(L²) ≤ 1,048,576`. This caps the right-padded
  attention matrix regardless of input distribution.
- **Relative** (amplification): `rows × max(L²) ≤ maxAmplification × ΣL²`.
  The ceiling is 1.5× for `.automatic` and 2.0× for `.batched`.

`.sequential` sets `maxAmplification = nil`; each row is its own singleton
batch.

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

Padded queries in sliding-attention layers see only themselves (via
`row[query] == 0 ? query == key : …`), so padding never bleeds into real
marker representations.

### Strategy selection

`BatchingStrategy` is a **method parameter only** on `systemOne`:

```swift
let answer = try await engine.systemOne(
    request, batchingStrategy: .batched
)
```

It is not serialized, not on the HTTP wire, and not part of `SystemOneRequest`.
The default is `.automatic`. The HTTP server always uses `.automatic`; there
is no client-controlled batching mode.

## Parity validation

Batch model numeric parity was verified on GPU (Release Xcode, Metal) against
both Python's direct batched path and Swift's own sequential path:

| Comparison | GPU global max absolute error |
|---|---|
| Swift batch vs Python direct batch | 7.05719 × 10⁻⁵ |
| Swift batch vs Swift sequential | 4.386902 × 10⁻⁵ |

Public engine rounded-answer parity on GPU passed: batched `.automatic` and
`.batched` produce the same exact rounded Choice/Noul/Score answers as
`.sequential`. The calibration boundary (float32 softmax/entropy → double
coefficient application) is not affected by batching.

## CPU benchmark note

CPU benchmarks and CPU batch parity can trigger macOS's Metal watchdog on
large or many-row batches because MLX CPU evaluation still goes through Metal
command submission. CPU results are diagnostic only; GPU is the acceptance
target. All reported numbers above are GPU.

## Reproduction

Requires the verified production weights artifact and an Xcode-built Release
executable (the Metal library is not packaged by plain `swift run`):

```bash
xcodebuild build \
  -scheme SystemOneBenchmark \
  -destination 'platform=macOS,arch=arm64' \
  -configuration Release \
  -derivedDataPath "$PWD/.build/xcode-release"

# Heterogeneous benchmark (30 iterations, 3 warmup)
.build/xcode-release/Build/Products/Release/SystemOneBenchmark \
  --tokenizer Tests/SystemOneVonTests/Resources/Tokenizer \
  --weights weights/option_marker.safetensors \
  --device gpu --strategy automatic --workload fixture \
  --iterations 30 --warmup 3

# Run all Swift strategies plus both pinned Python MPS paths.
tools/run_phase8_benchmark.sh
```

Benchmark JSON and logs are written under the ignored `benchmark-results/`
directory and are not CI assertions. `tools/sweep_strategies.sh` additionally
measures homogeneous 4/8/16-question workloads.

## JSON.swift bounded parser note

The ordered bounded JSON parser used by `VonHTTP` was upstreamed locally on
the JSON.swift feature branch at commit
`dddbe8b12e72ab4a0efc0b537ad08b8574a23896`. SystemOne deliberately retains its
current parser until a remote release or tag exists; it does not depend on
that branch.