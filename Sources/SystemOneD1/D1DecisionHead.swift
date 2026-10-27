// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation
import MLX
import MLXNN

/// `encoder.py` `DecisionHead`: a question-type embedding, two pre-norm
/// transformer layers over the text positions, and one shared MLP scoring the
/// hidden state at each option marker.
final class D1DecisionHead {
    /// `QTYPES` in `prompt.py`.
    enum QuestionType: Int {
        case choice = 0
        case score = 1
        case noul = 2
    }

    private struct Layer {
        let inProjection: Linear
        let outProjection: Linear
        let norm1Weight: MLXArray, norm1Bias: MLXArray
        let norm2Weight: MLXArray, norm2Bias: MLXArray
        let expand: Linear
        let contract: Linear
    }

    /// PyTorch's `nn.LayerNorm` default.
    private static let normEpsilon: Float = 1e-5
    private static let headDimension = 64

    private let typeEmbedding: MLXArray
    private let layers: [Layer]
    private let scorerNormWeight: MLXArray
    private let scorerNormBias: MLXArray
    private let scorerDense: Linear
    private let scorerOutput: Linear

    init(weights: D1Weights) {
        typeEmbedding = weights.array("head.type_emb.weight")
        layers = (0..<weights.configuration.headLayers).map { index in
            let prefix = "head.head.layers.\(index)"
            func linear(_ name: String) -> Linear {
                Linear(weight: weights.array("\(prefix).\(name).weight"), bias: weights.array("\(prefix).\(name).bias"))
            }
            return Layer(
                inProjection: Linear(
                    weight: weights.array("\(prefix).self_attn.in_proj_weight"),
                    bias: weights.array("\(prefix).self_attn.in_proj_bias")
                ),
                outProjection: linear("self_attn.out_proj"),
                norm1Weight: weights.array("\(prefix).norm1.weight"),
                norm1Bias: weights.array("\(prefix).norm1.bias"),
                norm2Weight: weights.array("\(prefix).norm2.weight"),
                norm2Bias: weights.array("\(prefix).norm2.bias"),
                expand: linear("linear1"),
                contract: linear("linear2")
            )
        }
        scorerNormWeight = weights.array("head.scorer.0.weight")
        scorerNormBias = weights.array("head.scorer.0.bias")
        scorerDense = Linear(weight: weights.array("head.scorer.1.weight"), bias: weights.array("head.scorer.1.bias"))
        scorerOutput = Linear(weight: weights.array("head.scorer.3.weight"), bias: weights.array("head.scorer.3.bias"))
    }

    /// Scores marker positions of right-padded text rows.
    ///
    /// - Parameters:
    ///   - text: `[B, T, D]` trunk output over each row's text positions.
    ///   - lengths: real text positions per row.
    ///   - questionTypes: one type per row.
    ///   - markers: marker positions per row, relative to the row's text.
    /// - Returns: `[B, K]` raw logits; positions past a row's marker count are unspecified.
    func callAsFunction(
        _ text: MLXArray,
        lengths: [Int],
        questionTypes: [QuestionType],
        markers: [[Int]]
    ) -> MLXArray {
        let batch = text.dim(0)
        let length = text.dim(1)
        let width = text.dim(2)
        let heads = width / Self.headDimension

        let types = MLXArray(questionTypes.map { Int32($0.rawValue) })
        var hidden = text + typeEmbedding[types].expandedDimensions(axis: 1)

        var mask: MLXArray?
        if lengths.contains(where: { $0 != length }) {
            let positions = MLXArray(0..<Int32(length)).reshaped(1, length)
            let real = positions .< MLXArray(lengths.map(Int32.init)).reshaped(batch, 1)
            mask = MLX.where(real, MLXArray(Float(0)).asType(text.dtype), MLXArray(-Float.infinity).asType(text.dtype))
                .reshaped(batch, 1, 1, length)
        }

        for layer in layers {
            let normalized = MLXFast.layerNorm(hidden, weight: layer.norm1Weight, bias: layer.norm1Bias,
                                               eps: Self.normEpsilon)
            let qkv = layer.inProjection(normalized)
                .reshaped(batch, length, 3, heads, Self.headDimension)
                .transposed(2, 0, 3, 1, 4)
            let attended = MLXFast.scaledDotProductAttention(
                queries: qkv[0], keys: qkv[1], values: qkv[2],
                scale: 1 / Float(Self.headDimension).squareRoot(),
                mask: mask
            )
            hidden = hidden + layer.outProjection(attended.transposed(0, 2, 1, 3).reshaped(batch, length, width))
            let feedForwardInput = MLXFast.layerNorm(hidden, weight: layer.norm2Weight, bias: layer.norm2Bias,
                                                     eps: Self.normEpsilon)
            hidden = hidden + layer.contract(relu(layer.expand(feedForwardInput)))
        }

        let optionCount = markers.map(\.count).max() ?? 0
        let flatPositions = markers.enumerated().flatMap { row, positions in
            (0..<optionCount).map { Int32(row * length + (positions.indices.contains($0) ? positions[$0] : 0)) }
        }
        let gathered = hidden.reshaped(batch * length, width)[MLXArray(flatPositions)]
        var scored = MLXFast.layerNorm(gathered, weight: scorerNormWeight, bias: scorerNormBias,
                                       eps: Self.normEpsilon)
        scored = scorerOutput(gelu(scorerDense(scored)))
        return scored.reshaped(batch, optionCount)
    }
}
