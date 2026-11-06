// Swift port of `Qwen3_5Attention` and `Qwen3_5RMSNorm` from Hugging Face
// transformers (src/transformers/models/qwen3_5/modeling_qwen3_5.py),
// translated and modified. Copyright 2025 The Qwen Team and The HuggingFace
// Inc. team. Licensed under the Apache License, Version 2.0; see LICENSE and
// THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

/// Qwen3.5's zero-centred RMSNorm: normalise in float32, scale by `1 + w`,
/// return in the input's dtype.
struct JevNorm {
    /// `1 + w` in float32.
    let scale: MLXArray
    let eps: Float

    init(weight: MLXArray, eps: Float) {
        scale = 1 + weight.asType(.float32)
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: scale, eps: eps).asType(x.dtype)
    }
}

/// Gated full attention: GQA with per-head q/k norms, RoPE on the leading
/// `partial_rotary_factor` of each head, a causal mask, and a sigmoid output
/// gate that `q_proj` emits beside each head's query.
struct JevGatedAttention {
    let query: Linear
    let key: Linear
    let value: Linear
    let output: Linear
    let queryNorm: JevNorm
    let keyNorm: JevNorm
    let heads: Int
    let keyValueHeads: Int
    let headDim: Int
    let rotaryDimensions: Int
    let ropeTheta: Float

    init(weights: JevWeights, prefix: String) {
        let text = weights.configuration.text
        func linear(_ name: String) -> Linear { Linear(weight: weights.array("\(prefix).\(name).weight")) }
        query = linear("q_proj")
        key = linear("k_proj")
        value = linear("v_proj")
        output = linear("o_proj")
        queryNorm = JevNorm(weight: weights.array("\(prefix).q_norm.weight"), eps: text.rmsNormEps)
        keyNorm = JevNorm(weight: weights.array("\(prefix).k_norm.weight"), eps: text.rmsNormEps)
        heads = text.numAttentionHeads
        keyValueHeads = text.numKeyValueHeads
        headDim = text.headDim
        rotaryDimensions = text.rotaryDimensions
        ropeTheta = text.ropeParameters.ropeTheta
    }

    /// `[B, L, D]` in, `[B, L, D]` out. Right padding needs no mask beyond the
    /// causal one: real positions never read later positions.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        // Each head's query and gate sit side by side: [q (head_dim) | gate (head_dim)].
        let projected = query(x).reshaped(batch, length, heads, 2 * headDim)
        let gate = projected[.ellipsis, headDim...].reshaped(batch, length, heads * headDim)
        var queries = queryNorm(projected[.ellipsis, ..<headDim]).transposed(0, 2, 1, 3)
        var keys = keyNorm(key(x).reshaped(batch, length, keyValueHeads, headDim)).transposed(0, 2, 1, 3)
        let values = value(x).reshaped(batch, length, keyValueHeads, headDim).transposed(0, 2, 1, 3)

        // Text-only positions are equal on every mRoPE axis, so mRoPE is plain
        // RoPE over the rotary dimensions; the remaining dimensions pass through.
        queries = MLXFast.RoPE(queries, dimensions: rotaryDimensions, traditional: false,
                               base: ropeTheta, scale: 1, offset: 0)
        keys = MLXFast.RoPE(keys, dimensions: rotaryDimensions, traditional: false,
                            base: ropeTheta, scale: 1, offset: 0)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / Float(headDim).squareRoot(),
            mask: .causal
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDim)
        return output(merged * sigmoid(gate))
    }
}
