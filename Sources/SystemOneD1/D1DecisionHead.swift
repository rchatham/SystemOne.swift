// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation
import MLX
import MLXNN

/// `encoder.py` `DecisionHead`: a question-type embedding, two pre-norm
/// transformer layers over the text positions, and one shared MLP scoring the
/// hidden state at each option marker.
///
/// The forward pass is a function of a named parameter set, so training can
/// pass the weights in as arguments to `valueAndGrad` and inference uses the
/// loaded ones.
final class D1DecisionHead {
    /// `QTYPES` in `prompt.py`.
    enum QuestionType: Int {
        case choice = 0
        case score = 1
        case noul = 2
    }

    /// PyTorch's `nn.LayerNorm` default.
    private static let normEpsilon: Float = 1e-5
    private static let headDimension = 64

    /// Every head weight, keyed by its checkpoint name (`head.scorer.1.weight`).
    let parameters: [String: MLXArray]
    private let layerCount: Int

    init(weights: D1Weights) {
        layerCount = weights.configuration.headLayers
        parameters = Dictionary(uniqueKeysWithValues: D1Weights.schema(for: weights.configuration)
            .map(\.name)
            .filter { $0.hasPrefix("head.") }
            .map { ($0, weights.array($0)) })
    }

    /// Scores marker positions of right-padded text rows.
    ///
    /// - Parameters:
    ///   - text: `[B, T, D]` trunk output over each row's text positions.
    ///   - lengths: real text positions per row.
    ///   - questionTypes: one type per row.
    ///   - markers: marker positions per row, relative to the row's text.
    ///   - parameters: head weights to use in place of the loaded ones, by checkpoint name.
    /// - Returns: `[B, K]` raw logits; positions past a row's marker count are unspecified.
    func callAsFunction(
        _ text: MLXArray,
        lengths: [Int],
        questionTypes: [QuestionType],
        markers: [[Int]],
        parameters: [String: MLXArray]? = nil
    ) -> MLXArray {
        let p = parameters ?? self.parameters
        // `nn.Linear`, computed as MLXNN's `Linear` does.
        func linear(_ x: MLXArray, _ name: String) -> MLXArray {
            addMM(p["\(name).bias"]!, x, p["\(name).weight"]!.T)
        }
        func layerNorm(_ x: MLXArray, _ name: String) -> MLXArray {
            MLXFast.layerNorm(x, weight: p["\(name).weight"]!, bias: p["\(name).bias"]!, eps: Self.normEpsilon)
        }

        let batch = text.dim(0)
        let length = text.dim(1)
        let width = text.dim(2)
        let heads = width / Self.headDimension

        let types = MLXArray(questionTypes.map { Int32($0.rawValue) })
        var hidden = text + p["head.type_emb.weight"]![types].expandedDimensions(axis: 1)

        var mask: MLXArray?
        if lengths.contains(where: { $0 != length }) {
            let positions = MLXArray(0..<Int32(length)).reshaped(1, length)
            let real = positions .< MLXArray(lengths.map(Int32.init)).reshaped(batch, 1)
            mask = MLX.where(real, MLXArray(Float(0)).asType(text.dtype), MLXArray(-Float.infinity).asType(text.dtype))
                .reshaped(batch, 1, 1, length)
        }

        for index in 0..<layerCount {
            let layer = "head.head.layers.\(index)"
            let qkv = addMM(p["\(layer).self_attn.in_proj_bias"]!, layerNorm(hidden, "\(layer).norm1"),
                            p["\(layer).self_attn.in_proj_weight"]!.T)
                .reshaped(batch, length, 3, heads, Self.headDimension)
                .transposed(2, 0, 3, 1, 4)
            let attended = MLXFast.scaledDotProductAttention(
                queries: qkv[0], keys: qkv[1], values: qkv[2],
                scale: 1 / Float(Self.headDimension).squareRoot(),
                mask: mask
            )
            hidden = hidden + linear(attended.transposed(0, 2, 1, 3).reshaped(batch, length, width),
                                     "\(layer).self_attn.out_proj")
            let feedForwardInput = layerNorm(hidden, "\(layer).norm2")
            hidden = hidden + linear(relu(linear(feedForwardInput, "\(layer).linear1")), "\(layer).linear2")
        }

        let optionCount = markers.map(\.count).max() ?? 0
        let flatPositions = markers.enumerated().flatMap { row, positions in
            (0..<optionCount).map { Int32(row * length + (positions.indices.contains($0) ? positions[$0] : 0)) }
        }
        let gathered = hidden.reshaped(batch * length, width)[MLXArray(flatPositions)]
        let scored = linear(gelu(linear(layerNorm(gathered, "head.scorer.0"), "head.scorer.1")), "head.scorer.3")
        return scored.reshaped(batch, optionCount)
    }
}
