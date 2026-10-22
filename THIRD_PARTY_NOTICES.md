# Third-Party Notices

VonSwift is an independent Swift + MLX port of the von decision model. This
project is licensed under the Apache License 2.0 (see `LICENSE`). The following
third-party works are incorporated, referenced, or distributed with VonSwift.

## Reference model and tokenizer

VonSwift ports the "System One" decision model from the
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
checkpoint and may be redistributed under that license; VonSwift defaults to a
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
