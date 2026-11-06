// Swift port of the gated delta rule from Hugging Face transformers
// (src/transformers/models/qwen3_5/modeling_qwen3_5.py,
// `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`),
// translated and modified. Copyright 2025 The Qwen Team and The HuggingFace
// Inc. team. Licensed under the Apache License, Version 2.0; see LICENSE and
// THIRD_PARTY_NOTICES.md.

import Foundation
import MLX

/// The gated delta rule of a Gated DeltaNet layer, in float32.
///
/// Per head, with state `S` (`[dk, dv]`), decay `g ≤ 0` and learning rate `β`:
/// `S ← exp(g)·S + k ⊗ β(v − Sᵀk)` and the output reads `Sᵀq`, with `q` and `k`
/// L2-normalised and `q` scaled by `1/√dk`.
enum JevDeltaRule {
    static let chunkSize = 64

    /// The chunked form transformers uses for prefill: within each chunk of 64
    /// positions the updates collapse into a unit lower-triangular system,
    /// solved here by forward substitution (the reference's export path,
    /// equal to its `solve_triangular`), and the state advances once per chunk.
    ///
    /// - Parameters:
    ///   - query, key: `[B, L, H, dk]`, already repeated to the value heads.
    ///   - value: `[B, L, H, dv]`.
    ///   - decay: `g`, `[B, L, H]`.
    ///   - beta: `[B, L, H]`.
    /// - Returns: the output `[B, L, H, dv]` and the final state `[B, H, dk, dv]`, both float32.
    static func chunked(
        query: MLXArray, key: MLXArray, value: MLXArray, decay: MLXArray, beta: MLXArray
    ) -> (output: MLXArray, state: MLXArray) {
        let (batch, length, heads, keyDim) = (key.dim(0), key.dim(1), key.dim(2), key.dim(3))
        let valueDim = value.dim(3)
        let chunk = chunkSize
        // [B, H, L, ...] in float32.
        var q = l2Normalised(query.transposed(0, 2, 1, 3).asType(.float32))
        var k = l2Normalised(key.transposed(0, 2, 1, 3).asType(.float32))
        var v = value.transposed(0, 2, 1, 3).asType(.float32)
        var b = beta.transposed(0, 2, 1).asType(.float32)
        var g = decay.transposed(0, 2, 1).asType(.float32)
        q = q * (1 / Float(keyDim).squareRoot())

        // Zero-pad to whole chunks: padded positions have β = 0 and k = 0, so
        // they never write to the state, and g = 0 leaves it undecayed.
        let padded = (length + chunk - 1) / chunk * chunk
        if padded != length {
            let extra = padded - length
            q = concatenated([q, MLXArray.zeros([batch, heads, extra, keyDim])], axis: 2)
            k = concatenated([k, MLXArray.zeros([batch, heads, extra, keyDim])], axis: 2)
            v = concatenated([v, MLXArray.zeros([batch, heads, extra, valueDim])], axis: 2)
            b = concatenated([b, MLXArray.zeros([batch, heads, extra])], axis: 2)
            g = concatenated([g, MLXArray.zeros([batch, heads, extra])], axis: 2)
        }
        let chunks = padded / chunk

        let vBeta = (v * b.expandedDimensions(axis: -1)).reshaped(batch, heads, chunks, chunk, valueDim)
        let kBeta = (k * b.expandedDimensions(axis: -1)).reshaped(batch, heads, chunks, chunk, keyDim)
        q = q.reshaped(batch, heads, chunks, chunk, keyDim)
        k = k.reshaped(batch, heads, chunks, chunk, keyDim)
        g = g.reshaped(batch, heads, chunks, chunk)

        // cum[..., t]: log of the decay from the chunk's start to position t.
        let cumulative = cumsum(g, axis: -1)
        let strictlyUpper = triu(MLXArray.ones([chunk, chunk], dtype: .bool), k: 1)
        let pairwise = MLX.where(
            strictlyUpper,
            MLXArray(Float(0)),
            exp(cumulative.expandedDimensions(axis: -1) - cumulative.expandedDimensions(axis: -2))
        )
        let utSystem = matmul(kBeta, k.swappedAxes(-1, -2)) * pairwise
        let intraChunk = matmul(q, k.swappedAxes(-1, -2)) * pairwise
        let decayedKBeta = kBeta * exp(cumulative).expandedDimensions(axis: -1)

        // (I + L)⁻¹ for L the strictly lower part of utSystem, row by row:
        // row i of the inverse's lower part is -L[i,:i] - L[i,:i] @ N[:i,:i].
        var inverse = -tril(utSystem, k: -1)
        for row in 1..<chunk {
            let current = inverse[.ellipsis, row, ..<row]
            let previous = inverse[.ellipsis, ..<row, ..<row]
            inverse[.ellipsis, row, ..<row] = current
                + matmul(current.expandedDimensions(axis: -2), previous).squeezed(axis: -2)
        }
        inverse = inverse + MLXArray.eye(chunk, dtype: .float32)
        let newValues = matmul(inverse, vBeta)
        let keyCumulativeDecay = matmul(inverse, decayedKBeta)

        let decayedQuery = q * exp(cumulative).expandedDimensions(axis: -1)
        let lastCumulative = cumulative[.ellipsis, (chunk - 1)...]
        let decayedKey = k * exp(lastCumulative - cumulative).expandedDimensions(axis: -1)
        let chunkDecay = exp(lastCumulative).expandedDimensions(axis: -1)  // [B, H, N, 1, 1]

        var state = MLXArray.zeros([batch, heads, keyDim, valueDim])
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(chunks)
        for index in 0..<chunks {
            let correction = newValues[0..., 0..., index] - matmul(keyCumulativeDecay[0..., 0..., index], state)
            outputs.append(
                matmul(decayedQuery[0..., 0..., index], state) + matmul(intraChunk[0..., 0..., index], correction)
            )
            state = state * chunkDecay[0..., 0..., index]
                + matmul(decayedKey[0..., 0..., index].swappedAxes(-1, -2), correction)
        }
        let output = stacked(outputs, axis: 2)
            .reshaped(batch, heads, padded, valueDim)[0..., 0..., ..<length]
            .transposed(0, 2, 1, 3)
        return (output, state)
    }

    /// The token-by-token form, kept as an independent check of `chunked`.
    static func recurrent(
        query: MLXArray, key: MLXArray, value: MLXArray, decay: MLXArray, beta: MLXArray
    ) -> (output: MLXArray, state: MLXArray) {
        let (batch, length, heads, keyDim) = (key.dim(0), key.dim(1), key.dim(2), key.dim(3))
        let valueDim = value.dim(3)
        let q = l2Normalised(query.transposed(0, 2, 1, 3).asType(.float32)) * (1 / Float(keyDim).squareRoot())
        let k = l2Normalised(key.transposed(0, 2, 1, 3).asType(.float32))
        let v = value.transposed(0, 2, 1, 3).asType(.float32)
        let b = beta.transposed(0, 2, 1).asType(.float32)
        let g = decay.transposed(0, 2, 1).asType(.float32)

        var state = MLXArray.zeros([batch, heads, keyDim, valueDim])
        var outputs: [MLXArray] = []
        for t in 0..<length {
            let qt = q[0..., 0..., t], kt = k[0..., 0..., t], vt = v[0..., 0..., t]
            state = state * exp(g[0..., 0..., t]).reshaped(batch, heads, 1, 1)
            let memory = (state * kt.expandedDimensions(axis: -1)).sum(axis: -2)
            let delta = (vt - memory) * b[0..., 0..., t].expandedDimensions(axis: -1)
            state = state + kt.expandedDimensions(axis: -1) * delta.expandedDimensions(axis: -2)
            outputs.append((state * qt.expandedDimensions(axis: -1)).sum(axis: -2))
        }
        return (stacked(outputs, axis: 2).transposed(0, 2, 1, 3), state)
    }

    /// `x * rsqrt(Σx² + 1e-6)` over the last axis, as the FLA library normalises.
    static func l2Normalised(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
    }
}
