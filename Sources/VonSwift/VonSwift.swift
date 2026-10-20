/// VonSwift — a Swift implementation of the "System One" decision model
/// architecture used by wfzyx/von (non-autoregressive choice / noul / score).
///
/// The implementation is built up step by step:
///   1. Tokenization (HF tokenizer parity for ModernBERT)
///   2. Model loading (safetensors -> typed weights)
///   3. ModernBERT backbone (layers, attention, MLPs)
///   4. Decision head (logits -> calibrated softmax)
///   5. Public API: decide / judge / rate / systemOne
///   6. Optional HTTP server with /v1/systemone parity

public enum VonSwift {
    public static let version = "0.1.0"
}