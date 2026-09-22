# VonSwift

A Swift implementation of the [von](https://github.com/wfzyx/von) "System One"
decision model — non-autoregressive, calibrated choice / noul / score
inference running locally on Apple Silicon.

## Goal

Run the von decision model natively in Swift, without Python, using the
released von checkpoint. Built incrementally, each step verified against the
Python von package as a reference oracle (golden test vectors).

## Roadmap

- [ ] Step 0: Install von locally; capture golden reference outputs
- [ ] Step 1: Clone von source; map the model architecture
- [ ] Step 2: Tokenizer parity (HF `tokenizer.json` in Swift)
- [ ] Step 3: Weight loading (safetensors)
- [ ] Step 4: ModernBERT backbone in Swift
- [ ] Step 5: Decision head + calibrated softmax
- [ ] Step 6: Public API — `decide`, `judge`, `rate`, `systemOne`
- [ ] Step 7: HTTP server, `/v1/systemone` parity
- [ ] Step 8: Golden-test parity vs. Python von

## Status

Scaffold. See commits for progress.

## License

TBD (will mirror Apache-2.0 implications of the reference implementation).