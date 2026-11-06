// Swift port of `Qwen3_5GatedDeltaNet` and `Qwen3_5RMSNormGated` from
// Hugging Face transformers (src/transformers/models/qwen3_5/modeling_qwen3_5.py),
// translated and modified. Copyright 2025 The Qwen Team and The HuggingFace
// Inc. team. Licensed under the Apache License, Version 2.0; see LICENSE and
// THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

/// A Gated DeltaNet token mixer: projections, a causal depthwise convolution
/// over queries, keys and values, the gated delta rule, and a gated RMSNorm.
struct JevGatedDeltaNet {
    let inputQKV: Linear
    let inputZ: Linear
    let inputB: Linear
    let inputA: Linear
    let output: Linear
    /// `[K, C]` depthwise taps, oldest first.
    let kernel: MLXArray
    /// float32 `A_log`.
    let logA: MLXArray
    /// float32 `dt_bias` (stored bfloat16; the reference adds it to a float32 `a`).
    let timeStepBias: MLXArray
    /// float32 gated-norm weight.
    let normWeight: MLXArray
    let eps: Float
    let keyHeads: Int
    let valueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int

    init(weights: JevWeights, prefix: String) {
        let text = weights.configuration.text
        func linear(_ name: String) -> Linear { Linear(weight: weights.array("\(prefix).\(name).weight")) }
        inputQKV = linear("in_proj_qkv")
        inputZ = linear("in_proj_z")
        inputB = linear("in_proj_b")
        inputA = linear("in_proj_a")
        output = linear("out_proj")
        // [C, 1, K] -> [K, C] in float32: taps accumulate in float32.
        kernel = weights.array("\(prefix).conv1d.weight").squeezed(axis: 1).transposed().asType(.float32)
        logA = weights.array("\(prefix).A_log").asType(.float32)
        timeStepBias = weights.array("\(prefix).dt_bias").asType(.float32)
        normWeight = weights.array("\(prefix).norm.weight").asType(.float32)
        eps = text.rmsNormEps
        keyHeads = text.linearNumKeyHeads
        valueHeads = text.linearNumValueHeads
        keyHeadDim = text.linearKeyHeadDim
        valueHeadDim = text.linearValueHeadDim
    }

    /// `[B, L, D]` in, `[B, L, D]` out. Every step is causal, so right padding
    /// never reaches a real position.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let dataType = x.dtype
        let mixed = inputQKV(x)
        let z = inputZ(x).reshaped(batch, length, valueHeads, valueHeadDim)
        let b = inputB(x)
        let a = inputA(x)

        // Causal depthwise convolution: out[t] = Σ_k w[k] · x[t + k - (K - 1)].
        let taps = kernel.dim(0)
        let channels = mixed.dim(2)
        let padded = concatenated(
            [MLXArray.zeros([batch, taps - 1, channels], dtype: .float32), mixed.asType(.float32)], axis: 1)
        var convolved = padded[0..., 0..<length] * kernel[0]
        for tap in 1..<taps {
            convolved = convolved + padded[0..., tap..<(tap + length)] * kernel[tap]
        }
        convolved = silu(convolved.asType(dataType))

        let keyWidth = keyHeads * keyHeadDim
        let parts = split(convolved, indices: [keyWidth, 2 * keyWidth], axis: -1)
        var q = parts[0].reshaped(batch, length, keyHeads, keyHeadDim)
        var k = parts[1].reshaped(batch, length, keyHeads, keyHeadDim)
        let v = parts[2].reshaped(batch, length, valueHeads, valueHeadDim)

        let beta = sigmoid(b)
        let decay = -exp(logA) * softplus(a.asType(.float32) + timeStepBias)
        if valueHeads > keyHeads {
            // repeat_interleave: each key head serves consecutive value heads.
            q = repeated(q, count: valueHeads / keyHeads, axis: 2)
            k = repeated(k, count: valueHeads / keyHeads, axis: 2)
        }
        let core = JevDeltaRule.chunked(query: q, key: k, value: v, decay: decay, beta: beta).output
            .asType(dataType)

        // Gated RMSNorm: normalise in float32, back to the compute dtype, scale,
        // then gate with SiLU(z) in float32.
        let core32 = core.asType(.float32)
        let normalised = (core32 * rsqrt((core32 * core32).mean(axis: -1, keepDims: true) + eps)).asType(dataType)
        let gated = (normWeight * normalised) * silu(z.asType(.float32))
        return output(gated.asType(dataType).reshaped(batch, length, valueHeads * valueHeadDim))
    }
}
