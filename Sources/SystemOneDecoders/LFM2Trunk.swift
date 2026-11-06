import Foundation
import MLX
import MLXNN
import SystemOne

/// An LFM2 / LFM2.5 text trunk's settings, read from `config.json` (nested
/// under `text_config` in LFM2-VL checkpoints such as d1-3B).
public struct LFM2Configuration: Equatable, Sendable {
    public enum LayerType: String, Equatable, Sendable {
        /// A gated causal short convolution (`conv_L_cache` taps).
        case convolution = "conv"
        /// Grouped-query attention with query and key RMSNorm and RoPE.
        case fullAttention = "full_attention"
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    /// The feed-forward width after `block_auto_adjust_ff_dim`.
    public let intermediateSize: Int
    public let layerTypes: [LayerType]
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let normEpsilon: Float
    public let ropeTheta: Float
    public let convolutionKernel: Int
    public let convolutionBias: Bool
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int

    public init(
        vocabSize: Int, hiddenSize: Int, intermediateSize: Int, layerTypes: [LayerType],
        attentionHeads: Int, keyValueHeads: Int, headDimension: Int, normEpsilon: Float = 1e-5,
        ropeTheta: Float = 1_000_000, convolutionKernel: Int = 3, convolutionBias: Bool = false,
        tieWordEmbeddings: Bool = true, maxPositionEmbeddings: Int = 128_000
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerTypes = layerTypes
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headDimension = headDimension
        self.normEpsilon = normEpsilon
        self.ropeTheta = ropeTheta
        self.convolutionKernel = convolutionKernel
        self.convolutionBias = convolutionBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    public init(modelFolder: URL) throws {
        let url = modelFolder.appendingPathComponent("config.json")
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw LFM2ConfigurationError.invalid("config.json is not a JSON object")
        }
        try self.init(json: json)
    }

    public init(json root: [String: Any]) throws {
        let text = root["text_config"] as? [String: Any] ?? root
        func int(_ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = text[key] as? Int { return value }
            if let fallback { return fallback }
            throw LFM2ConfigurationError.missing(key)
        }
        if let type = text["model_type"] as? String, type != "lfm2" {
            throw LFM2ConfigurationError.unsupported("model_type \(type)")
        }
        let layerCount = try int("num_hidden_layers")
        let layerTypes: [LayerType]
        if let names = text["layer_types"] as? [String] {
            layerTypes = try names.map {
                guard let type = LayerType(rawValue: $0) else { throw LFM2ConfigurationError.unsupported("layer type \($0)") }
                return type
            }
            guard layerTypes.count == layerCount else {
                throw LFM2ConfigurationError.invalid("layer_types has \(layerTypes.count) entries for \(layerCount) layers")
            }
        } else if let attention = text["full_attn_idxs"] as? [Int] {
            layerTypes = (0..<layerCount).map { attention.contains($0) ? .fullAttention : .convolution }
        } else {
            throw LFM2ConfigurationError.missing("layer_types")
        }

        // Lfm2MLP: with block_auto_adjust_ff_dim, 2/3 of the width, scaled, rounded up.
        var intermediate = try int("intermediate_size")
        if text["block_auto_adjust_ff_dim"] as? Bool == true {
            intermediate = 2 * intermediate / 3
            if let multiplier = (text["block_ffn_dim_multiplier"] as? NSNumber)?.doubleValue {
                intermediate = Int(multiplier * Double(intermediate))
                let multiple = try int("block_multiple_of", 256)
                intermediate = multiple * ((intermediate + multiple - 1) / multiple)
            }
        }
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        if let type = rope["rope_type"] as? String, type != "default" {
            throw LFM2ConfigurationError.unsupported("rope_type \(type)")
        }
        let theta = (rope["rope_theta"] as? NSNumber ?? text["rope_theta"] as? NSNumber)?.floatValue ?? 1_000_000
        let hidden = try int("hidden_size")
        let heads = try int("num_attention_heads")
        self.init(
            vocabSize: try int("vocab_size"),
            hiddenSize: hidden,
            intermediateSize: intermediate,
            layerTypes: layerTypes,
            attentionHeads: heads,
            keyValueHeads: try int("num_key_value_heads", heads),
            headDimension: try int("head_dim", hidden / heads),
            normEpsilon: ((text["norm_eps"] ?? text["block_norm_eps"]) as? NSNumber)?.floatValue ?? 1e-5,
            ropeTheta: theta,
            convolutionKernel: try int("conv_L_cache", 3),
            convolutionBias: text["conv_bias"] as? Bool ?? false,
            tieWordEmbeddings: text["tie_word_embeddings"] as? Bool ?? root["tie_word_embeddings"] as? Bool ?? true,
            maxPositionEmbeddings: try int("max_position_embeddings", 128_000)
        )
        guard heads % keyValueHeads == 0 else {
            throw LFM2ConfigurationError.invalid("\(heads) attention heads do not group into \(keyValueHeads) key-value heads")
        }
    }
}

public enum LFM2ConfigurationError: Error, Equatable, Sendable {
    case missing(String)
    case invalid(String)
    case unsupported(String)
}

extension LFM2ConfigurationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missing(let key): return "The LFM2 config has no \(key)."
        case .invalid(let reason): return "The LFM2 config is invalid: \(reason)."
        case .unsupported(let feature): return "The LFM2 trunk does not support \(feature)."
        }
    }
}

/// The causal LFM2 / LFM2.5 text trunk (gated short convolutions and GQA
/// layers), returning final-norm hidden states. Apache-2.0, following
/// transformers' `Lfm2Model`.
///
/// It takes the same right-padded rows, ``DecoderAttention`` plans (on its
/// attention layers; the convolutions are causal), and ``DecoderCache``
/// prefixes as ``Qwen35Trunk``.
public final class LFM2Trunk {
    public let configuration: LFM2Configuration
    public let dataType: DType

    private struct Attention {
        let query: Linear
        let key: Linear
        let value: Linear
        let output: Linear
        let queryNorm: MLXArray
        let keyNorm: MLXArray
    }

    private struct Convolution {
        let input: Linear
        /// `[kernel, channels]`.
        let kernel: MLXArray
        let bias: MLXArray?
        let output: Linear
    }

    private struct Layer {
        let attention: Attention?
        let convolution: Convolution?
        let operatorNorm: MLXArray
        let feedForwardNorm: MLXArray
        let gate: Linear
        let up: Linear
        let down: Linear
    }

    private let embedding: Embedding
    private let readout: Embedding
    private let layers: [Layer]
    private let finalNorm: MLXArray

    public convenience init(modelFolder: URL, dataType: DType = .float32) throws {
        try self.init(
            store: SafetensorsCheckpoint(modelFolder: modelFolder),
            configuration: LFM2Configuration(modelFolder: modelFolder),
            dataType: dataType
        )
    }

    /// Loads the trunk from `store` under any published prefix
    /// (`model.language_model.`, mlx-vlm's `language_model.model.`, `model.`).
    /// Convolution kernels may be `[channels, 1, kernel]` or MLX's
    /// `[channels, kernel, 1]`. Vision tensors are ignored.
    public init(store: any TensorStore, configuration: LFM2Configuration, dataType: DType = .float32) throws {
        self.configuration = configuration
        self.dataType = dataType
        let c = configuration
        let d = c.hiddenSize
        guard let prefix = ["model.language_model.", "language_model.model.", "language_model.", "model.", ""]
            .first(where: { store.descriptors["\($0)embed_tokens.weight"] != nil })
        else { throw WeightsError.tensorSetMismatch(missing: ["embed_tokens.weight"], unexpected: []) }

        func check(_ name: String, _ shape: [Int]) throws {
            let module = String(name.dropLast(".weight".count))
            if name.hasSuffix(".weight"), shape.count == 2, store.isQuantized(module) {
                _ = try store.quantizationParameters(of: module, inputWidth: shape[1])
                return
            }
            let actual = try store.requiredTensor(name).descriptor.shape
            guard actual == shape else { throw WeightsError.shapeMismatch(name: name, expected: shape, actual: actual) }
        }
        func linear(_ name: String, _ output: Int, _ input: Int, bias: Bool = false) throws -> Linear {
            try check("\(name).weight", [output, input])
            if bias { try check("\(name).bias", [output]) }
            return try store.linear(name, inputWidth: input, bias: bias, dataType: dataType)
        }
        func norm(_ name: String, _ width: Int) throws -> MLXArray {
            try check(name, [width])
            return try store.requiredTensor(name).mlxArray(dataType)
        }

        try check("\(prefix)embed_tokens.weight", [c.vocabSize, d])
        let input = try store.embedding("\(prefix)embed_tokens", width: d, dataType: dataType)
        embedding = input
        if c.tieWordEmbeddings {
            readout = input
        } else {
            guard let head = ["lm_head", "language_model.lm_head"].first(where: { store.descriptors["\($0).weight"] != nil }) else {
                throw WeightsError.tensorSetMismatch(missing: ["lm_head.weight"], unexpected: [])
            }
            try check("\(head).weight", [c.vocabSize, d])
            readout = try store.embedding(head, width: d, dataType: dataType)
        }

        layers = try c.layerTypes.enumerated().map { index, type -> Layer in
            let p = "\(prefix)layers.\(index)."
            var attention: Attention?
            var convolution: Convolution?
            switch type {
            case .fullAttention:
                let keyWidth = c.keyValueHeads * c.headDimension
                attention = Attention(
                    query: try linear("\(p)self_attn.q_proj", c.attentionHeads * c.headDimension, d),
                    key: try linear("\(p)self_attn.k_proj", keyWidth, d),
                    value: try linear("\(p)self_attn.v_proj", keyWidth, d),
                    output: try linear("\(p)self_attn.out_proj", d, c.attentionHeads * c.headDimension),
                    queryNorm: try norm("\(p)self_attn.q_layernorm.weight", c.headDimension),
                    keyNorm: try norm("\(p)self_attn.k_layernorm.weight", c.headDimension)
                )
            case .convolution:
                let name = "\(p)conv.conv.weight"
                let shape = try store.requiredTensor(name).descriptor.shape
                guard shape == [d, 1, c.convolutionKernel] || shape == [d, c.convolutionKernel, 1] else {
                    throw WeightsError.shapeMismatch(name: name, expected: [d, 1, c.convolutionKernel], actual: shape)
                }
                convolution = Convolution(
                    input: try linear("\(p)conv.in_proj", 3 * d, d, bias: c.convolutionBias),
                    kernel: try store.requiredTensor(name).mlxArray(dataType).reshaped(d, c.convolutionKernel).transposed(),
                    bias: try c.convolutionBias ? norm("\(p)conv.conv.bias", d) : nil,
                    output: try linear("\(p)conv.out_proj", d, d, bias: c.convolutionBias)
                )
            }
            return Layer(
                attention: attention,
                convolution: convolution,
                operatorNorm: try norm("\(p)operator_norm.weight", d),
                feedForwardNorm: try norm("\(p)ffn_norm.weight", d),
                gate: try linear("\(p)feed_forward.w1", c.intermediateSize, d),
                up: try linear("\(p)feed_forward.w3", c.intermediateSize, d),
                down: try linear("\(p)feed_forward.w2", d, c.intermediateSize)
            )
        }
        finalNorm = try norm("\(prefix)embedding_norm.weight", d)
    }

    /// Final-norm hidden states `[rows, width, hidden]` for right-padded rows,
    /// after `prefix` when given (read, not changed). `features` stand in for
    /// placeholder tokens' embeddings (image features at `<image>`).
    public func callAsFunction(
        _ ids: [[Int]], attention: DecoderAttention? = nil, prefix: DecoderCache? = nil, features: DecoderFeatures? = nil
    ) throws -> MLXArray {
        let width = try DecoderBatch.validate(ids, vocabularySize: configuration.vocabSize)
        let attention = attention ?? DecoderAttention(lengths: Array(repeating: width, count: ids.count))
        guard attention.lengths.count == ids.count else {
            throw DecoderError.invalidLength(row: min(attention.lengths.count, ids.count), length: 0)
        }
        try attention.validate(width: width)
        return try forward(ids, width: width, attention: attention, read: prefix, write: nil, features: features)
    }

    /// Extends `cache` with one causal row and returns its hidden states;
    /// `features` (one row) fill the row's placeholder tokens.
    @discardableResult
    public func prefill(_ ids: [Int], cache: DecoderCache, features: DecoderFeatures? = nil) throws -> MLXArray {
        let width = try DecoderBatch.validate([ids], vocabularySize: configuration.vocabSize)
        return try forward(
            [ids], width: width, attention: DecoderAttention(lengths: [width]), read: cache, write: cache, features: features
        )
    }

    /// Output-embedding rows `[ids.count, hidden]`, dequantized.
    public func readoutRows(_ ids: [Int]) throws -> MLXArray {
        for id in ids where !(0..<configuration.vocabSize).contains(id) {
            throw DecoderError.invalidTokenID(row: 0, column: 0, value: id)
        }
        return readout(MLXArray(ids.map(Int32.init)))
    }

    private func forward(
        _ ids: [[Int]], width: Int, attention: DecoderAttention, read: DecoderCache?, write: DecoderCache?,
        features: DecoderFeatures?
    ) throws -> MLXArray {
        let c = configuration
        let rows = ids.count
        let prefixLength = read?.length ?? 0
        let positions = attention.positionIDs(width: width, prefix: prefixLength)
        let highest = Int(positions.max() ?? 0)
        guard highest < c.maxPositionEmbeddings else {
            throw DecoderError.sequenceTooLong(maximum: c.maxPositionEmbeddings, actual: highest + 1)
        }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = attention.isPlainCausal(width: width)
            ? .causal
            : .array(attention.mask(width: width, prefix: prefixLength))
        let rotation = attention.hasDefaultPositions(prefix: prefixLength)
            ? nil
            : Qwen35Trunk.rotation(positions: MLXArray(positions, [rows, width]), theta: c.ropeTheta, dimension: c.headDimension)

        var hidden = try DecoderBatch.embed(ids, width: width, embedding: embedding, features: features)
        for (index, layer) in layers.enumerated() {
            let input = MLXFast.rmsNorm(hidden, weight: layer.operatorNorm, eps: c.normEpsilon)
            if let block = layer.attention {
                var queries = MLXFast.rmsNorm(block.query(input).reshaped(rows, width, c.attentionHeads, -1), weight: block.queryNorm, eps: c.normEpsilon)
                    .transposed(0, 2, 1, 3)
                var keys = MLXFast.rmsNorm(block.key(input).reshaped(rows, width, c.keyValueHeads, -1), weight: block.keyNorm, eps: c.normEpsilon)
                    .transposed(0, 2, 1, 3)
                var values = block.value(input).reshaped(rows, width, c.keyValueHeads, -1).transposed(0, 2, 1, 3)
                if let rotation {
                    queries = Qwen35Trunk.rotate(queries, rotation, dimension: c.headDimension)
                    keys = Qwen35Trunk.rotate(keys, rotation, dimension: c.headDimension)
                } else {
                    queries = MLXFast.RoPE(queries, dimensions: c.headDimension, traditional: false, base: c.ropeTheta, scale: 1, offset: prefixLength)
                    keys = MLXFast.RoPE(keys, dimensions: c.headDimension, traditional: false, base: c.ropeTheta, scale: 1, offset: prefixLength)
                }
                if let cachedKeys = read?.keys[index], let cachedValues = read?.values[index] {
                    keys = concatenated([DecoderBatch.rows(cachedKeys, rows), keys], axis: 2)
                    values = concatenated([DecoderBatch.rows(cachedValues, rows), values], axis: 2)
                }
                write?.keys[index] = keys
                write?.values[index] = values
                let attended = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values,
                    scale: 1 / Float(c.headDimension).squareRoot(), mask: mask
                ).transposed(0, 2, 1, 3).reshaped(rows, width, -1)
                hidden = hidden + block.output(attended)
            } else if let block = layer.convolution {
                // B, C, x = in_proj; y = C · causal_conv(B · x); out_proj.
                let parts = block.input(input).split(parts: 3, axis: -1)
                let gated = parts[0] * parts[2]
                let kernel = c.convolutionKernel
                let window = read?.convolution[index].map { DecoderBatch.rows($0, rows) }
                    ?? MLXArray.zeros([rows, kernel - 1, c.hiddenSize], dtype: gated.dtype)
                let padded = concatenated([window, gated], axis: 1)
                write?.convolution[index] = padded[0..., width ..< (width + kernel - 1), 0...]
                var convolved = padded[0..., 0 ..< width, 0...] * block.kernel[0]
                for tap in 1 ..< kernel {
                    convolved = convolved + padded[0..., tap ..< (tap + width), 0...] * block.kernel[tap]
                }
                if let bias = block.bias { convolved = convolved + bias }
                hidden = hidden + block.output(parts[1] * convolved)
            }
            let ffnInput = MLXFast.rmsNorm(hidden, weight: layer.feedForwardNorm, eps: c.normEpsilon)
            hidden = hidden + layer.down(silu(layer.gate(ffnInput)) * layer.up(ffnInput))
        }
        write?.length = prefixLength + width
        return MLXFast.rmsNorm(hidden, weight: finalNorm, eps: c.normEpsilon)
    }
}

/// Batch checks and cache broadcasting the decoder trunks share.
enum DecoderBatch {
    /// The common width of `ids`, after checking every row and token.
    static func validate(_ ids: [[Int]], vocabularySize: Int) throws -> Int {
        guard let width = ids.first?.count, width > 0 else { throw DecoderError.emptyBatch }
        for (row, tokens) in ids.enumerated() {
            guard tokens.count == width else { throw DecoderError.raggedBatch(row: row) }
            for (column, token) in tokens.enumerated() where !(0..<vocabularySize).contains(token) {
                throw DecoderError.invalidTokenID(row: row, column: column, value: token)
            }
        }
        return width
    }

    /// A cached `[1, …]` array repeated over `count` rows.
    static func rows(_ array: MLXArray, _ count: Int) -> MLXArray {
        array.dim(0) == count ? array : MLX.broadcast(array, to: [count] + array.shape.dropFirst())
    }
}
