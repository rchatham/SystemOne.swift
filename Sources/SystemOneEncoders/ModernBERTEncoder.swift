import Foundation
import MLX
import MLXNN
import SystemOne

/// A ModernBERT-family encoder (ModernBERT, mmBERT, Ettin) built from its
/// `config.json`: pre-norm blocks of fused-QKV attention with RoPE, global
/// attention every few layers and a sliding window on the rest, and a gated
/// GELU MLP, as transformers' `ModernBertModel` computes them.
///
/// Weights load from any `TensorStore` under `prefix` with the Hugging Face
/// names (`embeddings.tok_embeddings`, `layers.N.attn.Wqkv`, ...), dense or
/// MLX-quantized. `EncoderAttention` adds what the decision models built on
/// these encoders need beyond padding: explicit position ids and per-row
/// visibility, with the sliding window measured on the position ids.
public final class ModernBERTEncoder {
    public let configuration: ModernBERTConfiguration
    /// The dtype weights and activations use.
    public let dataType: DType

    private struct Norm {
        let weight: MLXArray
        let bias: MLXArray?
        let eps: Float

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
        }
    }

    private struct Layer {
        let attentionNorm: Norm?
        let queryKeyValue: Linear
        let attentionOutput: Linear
        let mlpNorm: Norm
        let mlpInput: Linear
        let mlpOutput: Linear
    }

    private let embedding: Embedding
    private let embeddingNorm: Norm
    private let layers: [Layer]
    private let finalNorm: Norm

    /// Every tensor the encoder reads under `prefix`, as float32.
    public static func schema(for configuration: ModernBERTConfiguration, prefix: String = "") -> [WeightSpecification] {
        var schema: [WeightSpecification] = []
        let d = configuration.hiddenSize
        func add(_ name: String, _ shape: [Int]) {
            schema.append(WeightSpecification(name: prefix + name, shape: shape))
        }
        func norm(_ name: String) {
            add("\(name).weight", [d])
            if configuration.normBias { add("\(name).bias", [d]) }
        }
        func linear(_ name: String, _ output: Int, _ input: Int, bias: Bool) {
            add("\(name).weight", [output, input])
            if bias { add("\(name).bias", [output]) }
        }
        add("embeddings.tok_embeddings.weight", [configuration.vocabSize, d])
        norm("embeddings.norm")
        for index in 0..<configuration.numHiddenLayers {
            let layer = "layers.\(index)"
            if index > 0 { norm("\(layer).attn_norm") }
            linear("\(layer).attn.Wqkv", 3 * d, d, bias: configuration.attentionBias)
            linear("\(layer).attn.Wo", d, d, bias: configuration.attentionBias)
            norm("\(layer).mlp_norm")
            linear("\(layer).mlp.Wi", 2 * configuration.intermediateSize, d, bias: configuration.mlpBias)
            linear("\(layer).mlp.Wo", d, configuration.intermediateSize, bias: configuration.mlpBias)
        }
        norm("final_norm")
        return schema
    }

    /// Loads the encoder's tensors from `store` under `prefix`, checking each
    /// one's shape (quantized weights are checked by their packed shapes).
    /// Other tensors in the store, such as a decision head's, are left alone.
    public init(
        store: any TensorStore, configuration: ModernBERTConfiguration, prefix: String = "",
        dataType: DType = .float32
    ) throws {
        self.configuration = configuration
        self.dataType = dataType
        for specification in Self.schema(for: configuration, prefix: prefix) {
            let module = String(specification.name.dropLast(".weight".count))
            if specification.name.hasSuffix(".weight"), specification.shape.count == 2, store.isQuantized(module) {
                _ = try store.quantizationParameters(of: module, inputWidth: specification.shape[1])
                continue
            }
            let descriptor = try store.requiredTensor(specification.name).descriptor
            guard descriptor.shape == specification.shape else {
                throw WeightsError.shapeMismatch(name: specification.name, expected: specification.shape, actual: descriptor.shape)
            }
            guard descriptor.dataType.isFloatingPoint else {
                throw WeightsError.dataTypeMismatch(
                    name: specification.name, expected: specification.dataType, actual: descriptor.dataType
                )
            }
        }

        let d = configuration.hiddenSize
        func norm(_ name: String) throws -> Norm {
            Norm(
                weight: try store.requiredTensor(prefix + "\(name).weight").mlxArray(dataType),
                bias: try configuration.normBias ? store.requiredTensor(prefix + "\(name).bias").mlxArray(dataType) : nil,
                eps: configuration.normEps
            )
        }
        func linear(_ name: String, input: Int, bias: Bool) throws -> Linear {
            try store.linear(prefix + name, inputWidth: input, bias: bias, dataType: dataType)
        }
        embedding = try store.embedding(prefix + "embeddings.tok_embeddings", width: d, dataType: dataType)
        embeddingNorm = try norm("embeddings.norm")
        layers = try (0..<configuration.numHiddenLayers).map { index in
            let layer = "layers.\(index)"
            return Layer(
                attentionNorm: try index == 0 ? nil : norm("\(layer).attn_norm"),
                queryKeyValue: try linear("\(layer).attn.Wqkv", input: d, bias: configuration.attentionBias),
                attentionOutput: try linear("\(layer).attn.Wo", input: d, bias: configuration.attentionBias),
                mlpNorm: try norm("\(layer).mlp_norm"),
                mlpInput: try linear("\(layer).mlp.Wi", input: d, bias: configuration.mlpBias),
                mlpOutput: try linear("\(layer).mlp.Wo", input: configuration.intermediateSize, bias: configuration.mlpBias)
            )
        }
        finalNorm = try norm("final_norm")
    }

    /// Final hidden states `[rows, width, hidden]` for right-padded token rows.
    /// Positions past a row's length are padding; their states are unspecified.
    public func callAsFunction(_ ids: [[Int]], attention: EncoderAttention? = nil) throws -> MLXArray {
        guard let width = ids.first?.count, width > 0 else { throw EncoderError.emptyBatch }
        for (row, tokens) in ids.enumerated() {
            guard tokens.count == width else { throw EncoderError.raggedBatch(row: row) }
            for (column, token) in tokens.enumerated() where !(0..<configuration.vocabSize).contains(token) {
                throw EncoderError.invalidTokenID(row: row, column: column, value: token)
            }
        }
        let attention = attention ?? EncoderAttention(lengths: Array(repeating: width, count: ids.count))
        guard attention.lengths.count == ids.count else {
            throw EncoderError.invalidLength(row: min(attention.lengths.count, ids.count), length: 0)
        }
        try attention.validate(width: width)
        let positions = attention.positionIDs(width: width)
        let highest = Int(positions.max() ?? 0)
        guard highest < configuration.maxPositionEmbeddings else {
            throw EncoderError.sequenceTooLong(maximum: configuration.maxPositionEmbeddings, actual: highest + 1)
        }

        let batch = ids.count
        let heads = configuration.numAttentionHeads
        let headDimension = configuration.headDimension
        let tokens = MLXArray(ids.flatMap { $0.map(Int32.init) }, [batch, width])
        let fullMask = additiveMask(attention, width: width, radius: nil)
        let slidingMask = additiveMask(attention, width: width, radius: configuration.slidingRadius)
        let positionIDs = attention.hasDefaultPositions ? nil : MLXArray(positions, [batch, width])
        var rotations: [Float: (cos: MLXArray, sin: MLXArray)] = [:]

        var hidden = embeddingNorm(embedding(tokens))
        for (index, layer) in layers.enumerated() {
            let input = layer.attentionNorm.map { $0(hidden) } ?? hidden
            let parts = layer.queryKeyValue(input)
                .reshaped(batch, width, 3, heads, headDimension)
                .split(parts: 3, axis: 2)
                .map { $0.squeezed(axis: 2).transposed(0, 2, 1, 3) }
            var (queries, keys) = (parts[0], parts[1])
            let theta = configuration.ropeTheta(layer: index)
            if let positionIDs {
                if rotations[theta] == nil {
                    rotations[theta] = Self.rotation(positions: positionIDs, theta: theta, dimension: headDimension, dataType: dataType)
                }
                let rotation = rotations[theta]!
                queries = Self.rotate(queries, rotation)
                keys = Self.rotate(keys, rotation)
            } else {
                queries = MLXFast.RoPE(queries, dimensions: headDimension, traditional: false, base: theta, scale: 1, offset: 0)
                keys = MLXFast.RoPE(keys, dimensions: headDimension, traditional: false, base: theta, scale: 1, offset: 0)
            }
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: parts[2],
                scale: 1 / Float(headDimension).squareRoot(),
                mask: configuration.isFullAttention(index) ? fullMask : slidingMask
            )
            hidden = hidden + layer.attentionOutput(attended.transposed(0, 2, 1, 3).reshaped(batch, width, -1))

            let (gated, gate) = layer.mlpInput(layer.mlpNorm(hidden)).split(axis: -1)
            hidden = hidden + layer.mlpOutput(activation(gated) * gate)
        }
        return finalNorm(hidden)
    }

    private func activation(_ x: MLXArray) -> MLXArray {
        switch configuration.hiddenActivation {
        case "gelu_pytorch_tanh", "gelu_new", "gelu_fast":
            return geluApproximate(x)
        default:
            return gelu(x)
        }
    }

    /// The additive mask for full (`radius: nil`) or sliding layers: key padding
    /// alone when the layout allows it, else a per-query `[rows, 1, width, width]` mask.
    private func additiveMask(_ attention: EncoderAttention, width: Int, radius: Int?) -> MLXArray {
        let blocked: Float
        switch dataType {
        case .float16: blocked = -65_504
        case .bfloat16: blocked = -3.0e38
        default: blocked = -Float.greatestFiniteMagnitude
        }
        if radius == nil, !attention.needsQueryMask {
            let values = attention.keyValidity(width: width).map { $0 ? Float(0) : blocked }
            return MLXArray(values, [attention.lengths.count, 1, 1, width]).asType(dataType)
        }
        let values = attention.visibility(width: width, radius: radius).map { $0 ? Float(0) : blocked }
        return MLXArray(values, [attention.lengths.count, 1, width, width]).asType(dataType)
    }

    /// RoPE's cosines and sines `[rows, 1, width, dimension]` for explicit
    /// positions, computed in float32 as transformers does
    /// (`1 / theta^(2i/d)`, frequencies repeated over both halves).
    static func rotation(
        positions: MLXArray, theta: Float, dimension: Int, dataType: DType
    ) -> (cos: MLXArray, sin: MLXArray) {
        let inverse = (0..<(dimension / 2)).map { 1 / powf(theta, Float(2 * $0) / Float(dimension)) }
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1) * MLXArray(inverse)
        let angles = concatenated([frequencies, frequencies], axis: -1).expandedDimensions(axis: 1)
        return (MLX.cos(angles).asType(dataType), MLX.sin(angles).asType(dataType))
    }

    /// Non-interleaved RoPE: `x · cos + rotate_half(x) · sin`.
    static func rotate(_ x: MLXArray, _ rotation: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let halves = x.split(parts: 2, axis: -1)
        let rotated = concatenated([-halves[1], halves[0]], axis: -1)
        return x * rotation.cos + rotated * rotation.sin
    }
}
