# On-device fine-tuning and federated aggregation (design study)

Status: proposal. Only the phase 1 head-only gradient step is implemented
(`D1HeadTraining`). Parameter counts are
estimated from `D1Weights.schema(for:)` with `hidden_size` 1024 and should be
re-derived from `config.json` before relying on them.

## Goal

1. Fine-tune d1-omni on a phone from the user's own interactions.
2. Optionally keep that fine-tune as a per-user personalization.
3. Periodically combine many users' fine-tunes into a better base checkpoint,
   in an offline post-processing step, without the raw states (text, photos,
   voice notes) ever leaving the device.

Only (3) needs a server, and the server only ever sees model updates, never
examples. "Anonymized data" is not enough on its own: a model update can leak
the text it was trained on (gradient inversion, memorization), so the design
below protects the updates themselves with clipping, noise, and secure
aggregation rather than relying on stripping identifiers.

## Why d1 is a good fit

A decision model's training signal is unusually clean. A training example is
`(state, question, media?, correct option)`, and the loss is cross-entropy
over the K marker logits the head already produces (`D1Model.logits(for:)`
returns `[B, K]`). There is no generated text, no sequence loss, and no
tokenizer change. Labels come from cheap user actions: correcting a wrong
answer, picking one of the options, or confirming a suggestion.

## What to train

| Part | Params (approx.) | Train on device? |
|---|---|---|
| Token embedding | 67M | no |
| Trunk, 16 layers (10 conv, 6 GQA) | 290M | LoRA only |
| Decision head (2 layers + scorer) | 26M | LoRA (or fully, for personalization only) |
| Vision tower, audio encoder | ~200M | no (frozen; prefixes are a function of the media alone) |

Full fine-tuning is out: Adam needs weights, gradients, and two moment
buffers, about 16 bytes per parameter, which is 9+ GB for the whole model.

**Recommendation: LoRA, rank 4–8, on every trunk and head `Linear`, base
frozen (float16 on memory-bound phones), adapters in float32.** That is about
3.3M trainable parameters, ~53 MB of optimizer state, and activation memory
for a single short row at a time.

A cheaper first tier is **head-only**: run the frozen trunk once per example
with no gradient, cache its `[T, D]` text output, and backpropagate through
the head alone. Training cost is then a fraction of one inference, and cached
trunk features can be replayed for several epochs. It cannot adapt how the
trunk reads states, so it is the right first spike, not the end state.

### Fitting LoRA into this codebase

The d1 classes hold weights as captured `MLXArray`s and `MLXNN.Linear`s, not
as a `Module` parameter tree, so `valueAndGrad(model:)` does not apply. The
functional form does:

```swift
// Frozen weights are captured constants; only the adapters are arguments,
// so MLX differentiates with respect to them alone.
let step = valueAndGrad { (adapters: [MLXArray]) -> [MLXArray] in
    let logits = model.logits(for: rows, adapters: D1Adapters(adapters))
    return [crossEntropy(logits: logits, targets: targets, rowMasks: optionMasks)]
}
```

Each frozen `linear(x)` becomes `linear(x) + scale * ((x ⊗ Aᵀ) ⊗ Bᵀ)` when an
adapter is present. Inference with no adapter stays byte-identical, so the
golden parity fixtures keep testing the port.

Checked in mlx-swift 0.31.4: `MLXOptimizers` (Adam/AdamW) ships with the
package, and `MLXFast` `rmsNorm`, `layerNorm`, and `scaledDotProductAttention`
all implement VJPs, so the existing forward pass is differentiable as written.
Padding marker slots (`D1DecisionHead` fills them with position 0) must be
masked before the loss; `D1HeadTraining` uses the reference head's -1e4.

## Federated aggregation

```
device                                         server (offline, per round)
──────                                         ───────────────────────────
base checkpoint (pinned SHA-256)               publish round r: base SHA, A seed,
  + A_r (from shared seed, frozen)               rank, clip C, local steps
local examples ─► train B for N steps
ΔB = B - B₀ ─► clip to ‖ΔB‖ ≤ C ─► ──────►  secure aggregation: Σ ΔB only
                (masked share)                 + Gaussian noise σ·C
                                               ÷ participants ─► ΔB̄
                                               merge W ← W + s·ΔB̄·A_r
                                               refit temperatures, evaluate,
                                               publish new pinned checkpoint
```

### Averaging LoRA correctly

Averaging `A` and `B` separately is wrong: `mean(Bᵢ)·mean(Aᵢ) ≠ mean(Bᵢ·Aᵢ)`.
Freeze `A` to a matrix every device regenerates from a seed the server
publishes per round, and train only `B` (FFA-LoRA, Sun et al., ICLR 2024).
Then `ΔW = ΔB·A` is linear in `ΔB`, so the server's average is the exact
average update, uploads halve (~1.6M parameters, ~3 MB at float16), and
clipping and noise act on one well-defined vector. A fresh `A` each round
keeps the cumulative update from being stuck in one rank-r subspace.

### Privacy: user-level DP-FedAvg with secure aggregation

- **Clip** each device's whole update to L2 norm `C`. This bounds any one
  user's influence, which both DP and poisoning defense need.
- **Secure aggregation** (Bonawitz et al., 2017): devices upload masked
  shares, and the server can decode only the sum over a cohort, never one
  device's update.
- **Gaussian noise** `N(0, σ²C²)` added to the sum (central DP, McMahan et
  al., 2018, the scheme behind Gboard). Track the ε spent across rounds with
  a privacy accountant.
- Local DP (noise on each device) avoids trusting the server's noise but needs
  far more participants for the same utility; start with central DP plus
  secure aggregation.

Utility depends on cohort size: with a few hundred users per round, the noise
needed for a meaningful ε swamps the signal. Until there are thousands of
participants, plan on opt-in donated adapters from testers, labelled as such,
and turn on DP rounds once cohorts are large enough.

### Integrity

- Each update carries the base checkpoint's SHA-256 (`PinnedArtifact`), the
  round id, and the `A` seed; mismatches are rejected, so stale devices
  cannot mix a delta for the wrong base into the sum.
- App Attest limits one participant to one real device.
- Clipping bounds a poisoned update's weight; secure aggregation hides
  individual updates, so robust aggregators (trimmed mean) need a design that
  runs inside it, or a smaller unencrypted tester cohort.

## The post-processing step

Per round, on the server:

1. Decode the aggregate, add noise, divide by participants.
2. Merge into the base: `W' = W + s·ΔB̄·A` for every adapted `Linear`, saved
   as a new `model.safetensors` with a new `PinnedArtifact` pin.
3. **Refit calibration.** d1 applies fixed per-type, per-option-count
   temperatures (`config.json` `temperatures`) to text questions; fine-tuning
   shifts the logit scale, so refit them by NLL on a held-out server eval set
   and ship them in the new `config.json`.
4. Gate the release: the new checkpoint must not regress on a fixed public
   eval suite (per question type and media type) and must keep float16 top
   answers matching float32 within the current tolerance.
5. Publish. Devices discard old personal adapters (they were trained against
   the old base) or re-train them from their kept examples.

The existing golden fixtures still pin the *port* against the upstream base
checkpoint; a merged checkpoint needs regression evals, not parity goldens.

## Device constraints

- **Memory.** The 4 GB iPhone 13 needed float16 just to run inference. Target
  6 GB+ devices first and bound training rows by length the way inference
  already bounds `B × L²`.
- **Background GPU.** iOS does not run Metal work for ordinary background
  tasks. Options: train in the foreground while charging, or iOS 26's
  `BGContinuedProcessingTask` with the GPU background entitlement where the
  device supports it (to verify on hardware). `BGProcessingTask` with
  `requiresExternalPower` fits the upload and bookkeeping, not the training.
- **Thermal and battery.** Check `ProcessInfo.thermalState`, pause on
  `.serious`, and cap steps per session.
- **Example store.** Keep examples encrypted at rest with file protection,
  bounded in count and age, deletable by the user, and never uploaded.

## Licensing

`SystemOneD1` and the weights are under the LFM Open License v1.0, which
permits preparing and distributing Derivative Works (§2, §4), so fine-tuned
and merged checkpoints are allowed, carrying the license and change notices.
The §5 commercial threshold applies to them as it does to the base model.

## Suggested phases

1. **Spike (Mac, then iPhone):** cross-entropy loss over `D1Model.logits`,
   head-only gradient step via `valueAndGrad`, compare one step's loss and
   gradients with PyTorch on a golden row. Measure memory and time.
   Head-only step done on the Mac GPU: `D1HeadTraining` and
   `D1HeadTrainingTests` against `tools/d1_training_capture.py` (three
   triage rows, padded to three options). All 31 head gradients match
   PyTorch to 8e-6 relative in norm and sampled values to 2e-4 of each
   tensor's max |g|; the loss goes 3.81682 to 3.42389 after one step at
   learning rate 1e-3 (PyTorch 3.81683 to 3.42389). The scorer's output bias
   gradient is zero by softmax shift invariance (MLX exactly 0, PyTorch
   -1.9e-9). Still to do: memory and time, then the same on an iPhone.
2. **LoRA in `SystemOneD1`:** optional adapters on trunk and head `Linear`s,
   zero-adapter output identical to today, adapter safetensors I/O.
3. **On-device trainer:** example store, training loop, harness UI for
   corrections, personalization on/off.
4. **Aggregation tooling (Python, in `tools/`):** round config, FFA-LoRA
   merge, temperature refit, eval gate. Simulate federated rounds with
   synthetic clients before any real upload.
5. **Transport:** secure aggregation, DP accountant, attestation.

## Open decisions

- Which user actions count as labels (an explicit correction only, or also an
  implicit pick among options), and how much to trust each.
- Whether personalization ships at all, or devices train only to contribute.
- Target ε per user per year, which sets cohort size and rounds.
