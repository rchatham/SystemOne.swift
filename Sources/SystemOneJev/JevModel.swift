// Swift port of `Qwen3_5TextModel`, `Qwen3_5DecoderLayer` and `Qwen3_5MLP`
// from Hugging Face transformers (src/transformers/models/qwen3_5/modeling_qwen3_5.py),
// translated and modified. Copyright 2025 The Qwen Team and The HuggingFace
// Inc. team. Licensed under the Apache License, Version 2.0; see LICENSE and
// THIRD_PARTY_NOTICES.md.

import Foundation
import MLX
import MLXNN

/// The JEV System 1 network: the Qwen3.5 text backbone with its LoRA merged,
/// read at each row's last position through the 24-slot decision head.
final class JevModel {
    private enum Mixer {
        case linear(JevGatedDeltaNet)
        case full(JevGatedAttention)
    }

    private struct Layer {
        let inputNorm: JevNorm
        let mixer: Mixer
        let postNorm: JevNorm
        let gate: Linear
        let up: Linear
        let down: Linear
    }

    let dataType: DType
    private let embedding: Embedding
    private let layers: [Layer]
    private let finalNorm: JevNorm
    private let headWeight: MLXArray
    private let headBias: MLXArray

    /// `dataType` overrides the weights' compute dtype, so one set of weights
    /// can back both a float32 and a bfloat16 model.
    init(weights: JevWeights, dataType: DType? = nil) {
        let text = weights.configuration.text
        self.dataType = dataType ?? weights.dataType
        embedding = Embedding(weight: weights.array("model.embed_tokens.weight"))
        finalNorm = JevNorm(weight: weights.array("model.norm.weight"), eps: text.rmsNormEps)
        layers = (0..<text.numHiddenLayers).map { index in
            let prefix = "model.layers.\(index)"
            func linear(_ name: String) -> Linear { Linear(weight: weights.array("\(prefix).\(name).weight")) }
            return Layer(
                inputNorm: JevNorm(weight: weights.array("\(prefix).input_layernorm.weight"), eps: text.rmsNormEps),
                mixer: text.isFullAttention(index)
                    ? .full(JevGatedAttention(weights: weights, prefix: "\(prefix).self_attn"))
                    : .linear(JevGatedDeltaNet(weights: weights, prefix: "\(prefix).linear_attn")),
                postNorm: JevNorm(
                    weight: weights.array("\(prefix).post_attention_layernorm.weight"), eps: text.rmsNormEps),
                gate: linear("mlp.gate_proj"),
                up: linear("mlp.up_proj"),
                down: linear("mlp.down_proj")
            )
        }
        headWeight = weights.headWeight
        headBias = weights.headBias
    }

    /// Runs right-padded rows through the backbone.
    ///
    /// - Parameters:
    ///   - rows: token ids per row, each non-empty.
    ///   - captureLayers: also returns the embedding, every layer's output, and
    ///     the final norm over all positions, for parity probes.
    /// - Returns: the final-norm hidden state at each row's last real position,
    ///   `[B, D]` in the compute dtype, and any captured `[B, L, D]` states.
    func lastHiddenStates(_ rows: [[Int]], captureLayers: Bool = false) -> (last: MLXArray, layers: [MLXArray]) {
        precondition(!rows.isEmpty && rows.allSatisfy { !$0.isEmpty }, "JEV rows need at least one token")
        let length = rows.map(\.count).max()!
        let ids = MLXArray(rows.flatMap { $0.map(Int32.init) + Array(repeating: 0, count: length - $0.count) })
            .reshaped(rows.count, length)
        var hidden = embedding(ids).asType(dataType)
        var captured: [MLXArray] = captureLayers ? [hidden] : []
        for layer in layers {
            let normalised = layer.inputNorm(hidden)
            switch layer.mixer {
            case .linear(let mixer): hidden = hidden + mixer(normalised)
            case .full(let mixer): hidden = hidden + mixer(normalised)
            }
            let feedForward = layer.postNorm(hidden)
            hidden = hidden + layer.down(silu(layer.gate(feedForward)) * layer.up(feedForward))
            if captureLayers { captured.append(hidden) }
        }
        if captureLayers { captured.append(finalNorm(hidden)) }

        // The norm is per position, so normalising only the gathered rows is exact.
        let lastPositions = MLXArray(rows.enumerated().map { Int32($0.offset * length + $0.element.count - 1) })
        let last = hidden.reshaped(rows.count * length, -1).take(lastPositions, axis: 0)
        return (finalNorm(last), captured)
    }

    /// The head's 24 slot logits per row, float32 `[B, slots]`: `W·h + b` on the
    /// float32 last hidden state.
    func slotLogits(_ rows: [[Int]]) -> MLXArray {
        let last = lastHiddenStates(rows).last.asType(.float32)
        return matmul(last, headWeight.T) + headBias
    }
}
