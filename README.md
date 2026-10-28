# SystemOne

Native Swift + MLX inference for **System One decision models**: models that
answer typed questions about a state (a yes/no `noul`, a `choice` among named
options, or an ordinal `score`) in a single forward pass, with zero output
tokens. Every answer is read off the model's distribution over the options.

| Model | Module | Inputs | Status |
|---|---|---|---|
| [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | `SystemOneD1` | text, images, speech | recommended |
| [wfzyx/von](https://huggingface.co/wfzyx/von) | `SystemOneVon` | text | deprecated upstream; kept for compatibility |

Both models share the Decision Index question schema, so one engine, one set of
public types, and one HTTP server serve either. Each port is verified layer by
layer against the model's own Python reference with golden test vectors.

## Layout

```
SystemOne        core: question/answer types, SystemOneEngine, DecisionBackend,
                 bounded microbatching, safetensors + weight schemas, ModelProfile,
                 torch-exact image resize, Slaney mel filterbank
SystemOneD1      d1-omni: LFM2.5 trunk, decision head, SigLIP2 vision, FastConformer audio
SystemOneVon     von: ModernBERT option-marker model and calibration
SystemOneHTTP    SwiftNIO server for POST /v1/systemone, model-agnostic
system-one       CLI: `system-one serve`
```

A model plugs in by conforming to `DecisionBackend`: it plans each request into
rows, runs microbatches of rows, and resolves each question's logits into an
answer. `SystemOneEngine` owns everything shared: validation, planning every
row before inference, bounded microbatch fan-out, cancellation, and fail-fast
answer resolution. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (d1)

Download the pinned snapshot (2.3 GB; LFM Open License v1.0):

```bash
hf download LiquidAI/d1-omni-600M --revision 02b55d7076f15129e59ab3f94783f32c4b088674 --local-dir weights/d1-omni-600M
```

```swift
import SystemOne
import SystemOneD1

let engine = try await SystemOneEngine.d1(modelFolder: modelFolder)  // verifies the pinned SHA-256

let response = try await engine.systemOne(
    state: .text("I was charged twice this month, please refund one of them."),
    questions: [
        QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Is the customer asking for a refund?"))),
        QuestionEntry(id: "team", question: .choice(Choice(
            instructions: "Which team should handle this?",
            criteria: [
                ChoiceOption(key: "billing", description: "Charges, refunds, invoices"),
                ChoiceOption(key: "technical", description: "App or site faults"),
            ]
        ))),
    ]
)

// Images (tiled like LFM2-VL) or one 16 kHz mono clip ride along as media.
let cats = try await engine.decide(
    state: .text(""),
    choices: [ChoiceOption(key: "one"), ChoiceOption(key: "two")],
    instructions: "How many cats are there?",
    media: .images([RGBImage(cgImage: photo)!])
)
let topic = try await engine.judge(
    state: .text("Voice note from a user."),
    instructions: "Is the speaker asking a question?",
    media: .audio(AudioClip(pcm16: samples))
)
```

d1 answers are the model's softmax, with the per-type temperatures from
`config.json` applied to text-only questions, exactly as `modeling_d1.py` does.
For audio-only requests pass `.text("")` (or `.fields([])`); both render as the
`{}` state the audio questions were trained on.

On memory-bound devices, `SystemOneEngine.d1(modelFolder:precision: .float16)`
halves weight memory (1.12 GB instead of 2.24 GB). It keeps the float32 top
answer on every fixture row, with probabilities within 6e-3; see
[docs/D1.md](docs/D1.md#float16-precision-float16).

## HTTP server

MLX needs its Metal library, which SwiftPM command-line builds do not bundle,
so build the CLI with Xcode:

```bash
xcodebuild build -scheme system-one -destination 'platform=macOS,arch=arm64' -derivedDataPath .build/xcode
.build/xcode/Build/Products/Debug/system-one serve --port 8016 --model-folder weights/d1-omni-600M
# von: --tokenizer <folder> --weights <option_marker.safetensors> instead of --model-folder
```

`POST /v1/systemone` takes the Decision Index JSON request
(`{"state": ..., "questions": {name: question}}`) and answers in order.
`SYSTEMONE_API_KEY` enables Bearer auth and is required for any non-loopback
bind; `SYSTEMONE_CORS_ORIGINS` is an explicit origin allowlist. The server
bounds framing, body sizes, concurrency, and deadlines, and the codec bounds the
work a request may plan using the served model's `ModelProfile` before any
inference runs. HTTP requests carry text states only.

## Verification

| | d1-omni-600M (GPU) | von (GPU) |
|---|---|---|
| Tokenizer ids, prompt encoding | exact | exact |
| Image crops (torchvision uint8 resize) | bit for bit | n/a |
| Backbone probes | 1.1e-5 relative | 1.7e-4 absolute |
| Head logits | 5.8e-5 absolute | 6.0e-5 absolute |
| Vision prefix / audio prefix | ~1e-5 / 2e-6 relative | n/a |

Golden vectors come from each model's pinned Python reference:
`tools/d1_golden_capture.py` → `testdata/d1_golden_v1.json`, and
`tools/golden_capture.py` → `testdata/von_golden_v1.json`.

```bash
swift build && swift test        # offline; weight-gated tests skip

# d1 tokenizer + prompt parity
D1_MODEL_FOLDER=weights/d1-omni-600M swift test --filter SystemOneD1Tests

# d1 weight parity (text, vision, audio) on the GPU via the Xcode test bundle
xcodebuild build-for-testing -scheme SystemOne-Package -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$PWD/.build/xcode"
D1_MODEL_FOLDER="$PWD/weights/d1-omni-600M" D1_MODEL_PARITY=1 D1_DEVICE=gpu \
  xcrun xctest -XCTest SystemOneD1Tests.D1TextTests,SystemOneD1Tests.D1VisionTests,SystemOneD1Tests.D1AudioTests,SystemOneD1Tests.D1HeadTrainingTests \
  "$PWD/.build/xcode/Build/Products/Debug/SystemOneD1Tests.xctest"

# Regenerate or check the d1 fixture (separate venv: torch 2.14, torchvision 0.29.1, transformers 5.17)
.venv-d1/bin/python tools/d1_golden_capture.py --check
.venv-d1/bin/python tools/d1_training_capture.py --check   # head gradients
```

von's installation, conversion, and parity commands are in
[docs/VON.md](docs/VON.md).

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the shared engine, and how von works
- [docs/D1.md](docs/D1.md): how d1-omni works and how the port matches it
- [docs/VON.md](docs/VON.md): von installation and verification record
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): microbatching and von benchmarks
- [docs/HANDOFF.md](docs/HANDOFF.md): resume-here state
- [Examples/SystemOneHarness](Examples/SystemOneHarness/README.md): iOS app running d1 on device with camera, photo, and microphone input

## License

SystemOne is licensed under the Apache License 2.0 (see `LICENSE`), except
`SystemOneD1`, a port of d1-omni's reference code, which is under the
[LFM Open License v1.0](LICENSES/LFM-Open-License-1.0.txt) and limits
commercial use by entities with $10M or more in annual revenue. Model weights
are never committed: von's are Apache-2.0, d1-omni's are under the LFM Open
License v1.0. Attribution is in `THIRD_PARTY_NOTICES.md`.
