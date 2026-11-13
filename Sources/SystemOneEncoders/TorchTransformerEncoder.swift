import Foundation
import MLX
import MLXNN
import SystemOne

/// A stack of PyTorch `nn.TransformerEncoderLayer`s (`batch_first`, ReLU,
/// LayerNorm ε = 1e-5, no dropout at inference), as decision heads use them
/// over encoder states. Each layer is self-attention with a fused `in_proj`
/// and a 4×-wide feed-forward; `normFirst` applies the norms before each block.
/// Padded keys are masked; padded queries are computed but never read.
final class TorchTransformerEncoder {
    private struct Layer {
        let inProjection: Linear
        let outProjection: Linear
        let expand: Linear
        let contract: Linear
        let norm1: (weight: MLXArray, bias: MLXArray)
        let norm2: (weight: MLXArray, bias: MLXArray)
    }

    private let layers: [Layer]
    private let heads: Int
    private let normFirst: Bool
    private let dataType: DType

    /// The tensors one layer stores under `prefix` (`head.layers.0.` say).
    static func schema(prefix: String, width: Int) -> [WeightSpecification] {
        [
            WeightSpecification(name: "\(prefix)self_attn.in_proj_weight", shape: [3 * width, width]),
            WeightSpecification(name: "\(prefix)self_attn.in_proj_bias", shape: [3 * width]),
            WeightSpecification(name: "\(prefix)self_attn.out_proj.weight", shape: [width, width]),
            WeightSpecification(name: "\(prefix)self_attn.out_proj.bias", shape: [width]),
            WeightSpecification(name: "\(prefix)linear1.weight", shape: [4 * width, width]),
            WeightSpecification(name: "\(prefix)linear1.bias", shape: [4 * width]),
            WeightSpecification(name: "\(prefix)linear2.weight", shape: [width, 4 * width]),
            WeightSpecification(name: "\(prefix)linear2.bias", shape: [width]),
            WeightSpecification(name: "\(prefix)norm1.weight", shape: [width]),
            WeightSpecification(name: "\(prefix)norm1.bias", shape: [width]),
            WeightSpecification(name: "\(prefix)norm2.weight", shape: [width]),
            WeightSpecification(name: "\(prefix)norm2.bias", shape: [width]),
        ]
    }

    /// Loads `count` layers stored as `\(prefix)\(index).…`.
    init(
        store: any TensorStore, prefix: String, count: Int, width: Int, heads: Int,
        normFirst: Bool = true, dataType: DType
    ) throws {
        func array(_ name: String) throws -> MLXArray { try store.requiredTensor(name).mlxArray(dataType) }
        layers = try (0..<count).map { index in
            let layer = "\(prefix)\(index)."
            func linear(_ name: String) throws -> Linear {
                Linear(weight: try array("\(layer)\(name).weight"), bias: try array("\(layer)\(name).bias"))
            }
            return Layer(
                inProjection: Linear(
                    weight: try array("\(layer)self_attn.in_proj_weight"), bias: try array("\(layer)self_attn.in_proj_bias")
                ),
                outProjection: try linear("self_attn.out_proj"),
                expand: try linear("linear1"),
                contract: try linear("linear2"),
                norm1: (try array("\(layer)norm1.weight"), try array("\(layer)norm1.bias")),
                norm2: (try array("\(layer)norm2.weight"), try array("\(layer)norm2.bias"))
            )
        }
        self.heads = heads
        self.normFirst = normFirst
        self.dataType = dataType
    }

    /// `[rows, width, d]` states through every layer; `lengths` marks real tokens.
    func callAsFunction(_ x: MLXArray, lengths: [Int]) -> MLXArray {
        let width = x.dim(1)
        let blocked: Float = dataType == .float16 ? -65_504 : -Float.greatestFiniteMagnitude
        let keys = lengths.flatMap { length in (0..<width).map { $0 < length ? Float(0) : blocked } }
        let mask = MLXArray(keys, [lengths.count, 1, 1, width]).asType(dataType)
        var hidden = x
        for layer in layers {
            func norm(_ value: MLXArray, _ parameters: (weight: MLXArray, bias: MLXArray)) -> MLXArray {
                MLXFast.layerNorm(value, weight: parameters.weight, bias: parameters.bias, eps: 1e-5)
            }
            if normFirst {
                hidden = hidden + attention(norm(hidden, layer.norm1), layer, mask: mask)
                hidden = hidden + layer.contract(relu(layer.expand(norm(hidden, layer.norm2))))
            } else {
                hidden = norm(hidden + attention(hidden, layer, mask: mask), layer.norm1)
                hidden = norm(hidden + layer.contract(relu(layer.expand(hidden))), layer.norm2)
            }
        }
        return hidden
    }

    /// `nn.MultiheadAttention` self-attention: query, key, and value are the
    /// three row blocks of `in_proj`, split into heads, scaled by `1/√d_head`.
    private func attention(_ x: MLXArray, _ layer: Layer, mask: MLXArray) -> MLXArray {
        let (rows, width, d) = (x.dim(0), x.dim(1), x.dim(2))
        let parts = layer.inProjection(x)
            .reshaped(rows, width, 3, heads, d / heads)
            .split(parts: 3, axis: 2)
            .map { $0.squeezed(axis: 2).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: parts[0], keys: parts[1], values: parts[2],
            scale: 1 / Float(d / heads).squareRoot(), mask: mask
        )
        return layer.outProjection(attended.transposed(0, 2, 1, 3).reshaped(rows, width, d))
    }
}
