# Third-Party Notices

SystemOne is an independent Swift + MLX implementation of System One decision
models. This project is licensed under the Apache License 2.0 (see `LICENSE`),
except the `SystemOneD1` module, which is licensed under the LFM Open License
v1.0 (see below). The following third-party works are incorporated, referenced, or distributed
with SystemOne.

## LiquidAI d1-omni-600M

`SystemOneD1` (`Sources/SystemOneD1/`) is a Swift port of the reference code
of [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
(pinned revision `02b55d7076f15129e59ab3f94783f32c4b088674`): `encoder.py`,
`vision.py`, `audio.py`, `prompt.py`, and `modeling_d1.py`. It is a Derivative
Work of that code and is distributed under the
[LFM Open License v1.0](LICENSES/LFM-Open-License-1.0.txt), not Apache-2.0.
Each of its files carries a notice that it was translated and modified from
the original Python.

- Licensor: Liquid AI, Inc.
- License: LFM Open License v1.0, which includes a Commercial Use limitation:
  commercial use by an entity with annual revenue of $10M or more is not
  licensed (Section 5). Read the full text before using `SystemOneD1`
  commercially.
- Source: https://huggingface.co/LiquidAI/d1-omni-600M

The other modules (`SystemOne`, `SystemOneVon`, `SystemOneHTTP`, the CLI, and
the benchmark) do not contain d1 code and remain Apache-2.0. A binary that
links `SystemOneD1` is subject to both licenses.

The d1 weights and tokenizer are not committed or redistributed here: users
download the snapshot themselves and accept its license.
`testdata/d1_golden_v1.json` contains token ids and numerical outputs generated
by running the reference on synthetic inputs, used only to verify this port.

## von reference model and tokenizer

SystemOne ports the "System One" decision model from the
[von](https://github.com/wfzyx/von) project and its published
`wfzyx/von` Hugging Face checkpoint (pinned revision
`d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd`).

- Copyright 2026 Victor Hugo Panisa
- License: Apache License 2.0
- Source: https://github.com/wfzyx/von
- Model: https://huggingface.co/wfzyx/von

The tokenizer `tokenizer.json` and `tokenizer_config.json` checked in under
`Tests/SystemOneVonTests/Resources/Tokenizer/` are redistributed from the same
`wfzyx/von` checkpoint under its Apache License 2.0. Provenance and SHA-256
hashes are recorded in `Tests/SystemOneVonTests/Resources/Tokenizer/source.json`.

The converted `option_marker.safetensors` inference weights are intentionally
not committed. They are derived from the Apache-2.0 `option_marker.pt`
checkpoint and may be redistributed under that license; SystemOne defaults to a
user-generated, checksum-verified artifact (see "Model installation" in
`README.md`).

## Swift package dependencies

Pinned by `Package.swift` / `Package.resolved`:

| Package | Version | License |
|---|---|---|
| ml-explore/mlx-swift | 0.31.4 | MIT |
| huggingface/swift-transformers | 1.1.6 | Apache-2.0 |
| apple/swift-nio | 2.84.0 | Apache-2.0 |
| apple/swift-atomics | 1.3.1 | Apache-2.0 |
| apple/swift-collections | 1.7.0 | Apache-2.0 |
| huggingface/swift-jinja | 2.5.1 | Apache-2.0 |
| apple/swift-numerics | 1.1.1 | Apache-2.0 |
| apple/swift-system | 1.8.1 | Apache-2.0 |
| swiftlang/swift-syntax | 510.0.3 | Apache-2.0 |
| rchatham/JSON.swift | 1.1.0 | Apache-2.0 |

`swift-syntax` is a transitive dependency of `JSON.swift`; the remaining
packages are direct dependencies. Full license texts are available from each
project's repository.
