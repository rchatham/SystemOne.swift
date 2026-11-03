import Foundation
import MLX
import MLXNN
import SystemOne

/// An LFM3 text trunk's settings (`text_config` of LFM3-VL checkpoints such as
/// d1-2B).
public struct LFM3Configuration: Equatable, Sendable {
    public enum LayerType: String, Equatable, Sendable {
        /// A gated causal short convolution (`conv_L_cache` taps).
        case convolution = "conv"
        /// Gated grouped-query attention over the whole sequence.
        case fullAttention = "full_attention"
        /// Gated grouped-query attention over the last `sliding_window` tokens.
        case slidingAttention = "sliding_attention"
    }

    /// RoPE for one attention layer type.
    public struct Rotary: Equatable, Sendable {
        public let theta: Float
        /// Frequencies rotated per head half; the rest of each half is left as is.
        public let angles: Int
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerTypes: [LayerType]
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let normEpsilon: Float
    public let convolutionKernel: Int
    public let convolutionBias: Bool
    public let slidingWindow: Int
    public let sandwichNorms: Bool
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    public let fullRotary: Rotary
    public let slidingRotary: Rotary

    public init(
        vocabSize: Int, hiddenSize: Int, intermediateSize: Int, layerTypes: [LayerType],
        attentionHeads: Int, keyValueHeads: Int, headDimension: Int, normEpsilon: Float = 1e-5,
        convolutionKernel: Int = 7, convolutionBias: Bool = false, slidingWindow: Int = 256,
        sandwichNorms: Bool = true, tieWordEmbeddings: Bool = true, maxPositionEmbeddings: Int = 262_144,
        fullRotary: Rotary, slidingRotary: Rotary
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerTypes = layerTypes
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headDimension = headDimension
        self.normEpsilon = normEpsilon
        self.convolutionKernel = convolutionKernel
        self.convolutionBias = convolutionBias
        self.slidingWindow = slidingWindow
        self.sandwichNorms = sandwichNorms
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.fullRotary = fullRotary
        self.slidingRotary = slidingRotary
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
        if let type = text["model_type"] as? String, type != "lfm3" {
            throw LFM2ConfigurationError.unsupported("model_type \(type)")
        }
        let layerCount = try int("num_hidden_layers")
        guard let names = text["layer_types"] as? [String] else { throw LFM2ConfigurationError.missing("layer_types") }
        let layerTypes = try names.map {
            guard let type = LayerType(rawValue: $0) else { throw LFM2ConfigurationError.unsupported("layer type \($0)") }
            return type
        }
        guard layerTypes.count == layerCount else {
            throw LFM2ConfigurationError.invalid("layer_types has \(layerTypes.count) entries for \(layerCount) layers")
        }
        let hidden = try int("hidden_size")
        let heads = try int("num_attention_heads")
        let headDimension = try int("head_dim", hidden / heads)

        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        func rotary(_ type: String, theta: Float, partial: Double) throws -> Rotary {
            let parameters = rope[type] as? [String: Any] ?? [:]
            if let kind = parameters["rope_type"] as? String, kind != "default" {
                throw LFM2ConfigurationError.unsupported("rope_type \(kind)")
            }
            let factor = (parameters["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? partial
            let angles = Int(Double(headDimension) * factor) / 2
            guard angles > 0, angles <= headDimension / 2 else {
                throw LFM2ConfigurationError.invalid("partial_rotary_factor \(factor)")
            }
            return Rotary(theta: (parameters["rope_theta"] as? NSNumber)?.floatValue ?? theta, angles: angles)
        }
        // Without rope_parameters the reference takes full θ 1e6 with half the
        // pairs rotated and sliding θ 1e4; a given entry defaults to θ 1e6, all pairs.
        let defaults = rope.isEmpty
        self.init(
            vocabSize: try int("vocab_size"),
            hiddenSize: hidden,
            intermediateSize: try int("intermediate_size"),
            layerTypes: layerTypes,
            attentionHeads: heads,
            keyValueHeads: try int("num_key_value_heads", heads),
            headDimension: headDimension,
            normEpsilon: (text["norm_eps"] as? NSNumber)?.floatValue ?? 1e-5,
            convolutionKernel: try int("conv_L_cache", 7),
            convolutionBias: text["conv_bias"] as? Bool ?? false,
            slidingWindow: try int("sliding_window", 256),
            sandwichNorms: text["use_sandwich_norm"] as? Bool ?? true,
            tieWordEmbeddings: text["tie_word_embeddings"] as? Bool ?? root["tie_word_embeddings"] as? Bool ?? true,
            maxPositionEmbeddings: try int("max_position_embeddings", 262_144),
            fullRotary: try rotary("full_attention", theta: 1_000_000, partial: defaults ? 0.5 : 1),
            slidingRotary: try rotary("sliding_attention", theta: defaults ? 10_000 : 1_000_000, partial: 1)
        )
        guard heads % keyValueHeads == 0 else {
            throw LFM2ConfigurationError.invalid("\(heads) attention heads do not group into \(keyValueHeads) key-value heads")
        }
        guard slidingWindow > 0 else { throw LFM2ConfigurationError.invalid("sliding_window \(slidingWindow)") }
    }
}

/// The LFM3 text trunk (d1-2B's backbone), following mlx-vlm's `lfm3` port of
/// the LFM3 reference (Apache-2.0):
///
/// - gated short convolutions with `conv_L_cache` (7) taps, causal;
/// - full-attention layers: GQA with q/k RMSNorm, a per-head sigmoid output
///   gate (`gate_proj`), and partial RoPE that rotates the first `angles`
///   coordinate pairs of each head half (frequencies over the whole head);
/// - sliding-attention layers: the same over the last `sliding_window` tokens,
///   causal, with full RoPE;
/// - sandwich norms around both blocks, and `embedding_norm` at the end.
///
/// A ``DecoderAttention`` plan governs the full-attention layers only; sliding
/// layers and convolutions stay causal, as d1-2B's set attention runs them.
public final class LFM3Trunk {
    public let configuration: LFM3Configuration
    public let dataType: DType

    private struct Attention {
        let query: Linear
        let key: Linear
        let value: Linear
        let output: Linear
        let gate: Linear
        let queryNorm: MLXArray
        let keyNorm: MLXArray
        let sliding: Bool
    }

    private struct Convolution {
        let input: Linear
        let kernel: MLXArray
        let bias: MLXArray?
        let output: Linear
    }

    private struct Layer {
        let attention: Attention?
        let convolution: Convolution?
        let operatorNorm: MLXArray
        let postOperatorNorm: MLXArray?
        let feedForwardNorm: MLXArray
        let postFeedForwardNorm: MLXArray?
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
            configuration: LFM3Configuration(modelFolder: modelFolder),
            dataType: dataType
        )
    }

    /// Loads the trunk under `model.language_model.` (transformers), mlx-vlm's
    /// `language_model.model.`, or `model.`; convolution kernels in either
    /// layout. Vision and audio tensors are ignored.
    public init(store: any TensorStore, configuration: LFM3Configuration, dataType: DType = .float32) throws {
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
            case .fullAttention, .slidingAttention:
                let keyWidth = c.keyValueHeads * c.headDimension
                attention = Attention(
                    query: try linear("\(p)self_attn.q_proj", c.attentionHeads * c.headDimension, d),
                    key: try linear("\(p)self_attn.k_proj", keyWidth, d),
                    value: try linear("\(p)self_attn.v_proj", keyWidth, d),
                    output: try linear("\(p)self_attn.o_proj", d, c.attentionHeads * c.headDimension),
                    gate: try linear("\(p)self_attn.gate_proj", c.attentionHeads, d),
                    queryNorm: try norm("\(p)self_attn.q_norm.weight", c.headDimension),
                    keyNorm: try norm("\(p)self_attn.k_norm.weight", c.headDimension),
                    sliding: type == .slidingAttention
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
                postOperatorNorm: try c.sandwichNorms ? norm("\(p)post_operator_norm.weight", d) : nil,
                feedForwardNorm: try norm("\(p)ffn_norm.weight", d),
                postFeedForwardNorm: try c.sandwichNorms ? norm("\(p)post_ffn_norm.weight", d) : nil,
                gate: try linear("\(p)feed_forward.w1", c.intermediateSize, d),
                up: try linear("\(p)feed_forward.w3", c.intermediateSize, d),
                down: try linear("\(p)feed_forward.w2", d, c.intermediateSize)
            )
        }
        finalNorm = try norm("\(prefix)embedding_norm.weight", d)
    }

    /// Final-norm hidden states `[rows, width, hidden]` for right-padded rows,
    /// after `prefix` when given (read, not changed).
    public func callAsFunction(_ ids: [[Int]], attention: DecoderAttention? = nil, prefix: DecoderCache? = nil) throws -> MLXArray {
        let width = try DecoderBatch.validate(ids, vocabularySize: configuration.vocabSize)
        let attention = attention ?? DecoderAttention(lengths: Array(repeating: width, count: ids.count))
        guard attention.lengths.count == ids.count else {
            throw DecoderError.invalidLength(row: min(attention.lengths.count, ids.count), length: 0)
        }
        try attention.validate(width: width)
        return try forward(ids, width: width, attention: attention, read: prefix, write: nil)
    }

    /// Extends `cache` with one causal row and returns its hidden states.
    @discardableResult
    public func prefill(_ ids: [Int], cache: DecoderCache) throws -> MLXArray {
        let width = try DecoderBatch.validate([ids], vocabularySize: configuration.vocabSize)
        return try forward([ids], width: width, attention: DecoderAttention(lengths: [width]), read: cache, write: cache)
    }

    /// Output-embedding rows `[ids.count, hidden]`, dequantized.
    public func readoutRows(_ ids: [Int]) throws -> MLXArray {
        for id in ids where !(0..<configuration.vocabSize).contains(id) {
            throw DecoderError.invalidTokenID(row: 0, column: 0, value: id)
        }
        return readout(MLXArray(ids.map(Int32.init)))
    }

    /// LFM3's rotation: pair `i` with `i + d/2` and turn it by
    /// `position · θ^(−2i/d)` for `i < angles`; the other pairs stay put.
    static func rotation(positions: MLXArray, rotary: LFM3Configuration.Rotary, dimension: Int) -> (cos: MLXArray, sin: MLXArray) {
        let inverse = (0..<(dimension / 2)).map { $0 < rotary.angles ? powf(rotary.theta, -Float(2 * $0) / Float(dimension)) : 0 }
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1) * MLXArray(inverse)
        let angles = concatenated([frequencies, frequencies], axis: -1).expandedDimensions(axis: 1)
        return (MLX.cos(angles), MLX.sin(angles))
    }

    /// `[rows, 1, width, prefix + width]`: causal over each row's real tokens
    /// and the prefix, within the last `window` tokens; a padding query sees
    /// only itself.
    static func slidingMask(lengths: [Int], width: Int, prefix: Int, window: Int) -> MLXArray {
        MLXArray(slidingMaskValues(lengths: lengths, width: width, prefix: prefix, window: window), [lengths.count, 1, width, prefix + width])
    }

    /// ``slidingMask(lengths:width:prefix:window:)``'s values, row-major.
    static func slidingMaskValues(lengths: [Int], width: Int, prefix: Int, window: Int) -> [Bool] {
        let keys = prefix + width
        var values = [Bool](repeating: false, count: lengths.count * width * keys)
        for (row, length) in lengths.enumerated() {
            for query in 0..<width {
                let base = (row * width + query) * keys
                guard query < length else {
                    values[base + prefix + query] = true
                    continue
                }
                let at = prefix + query
                for key in max(0, at - window + 1)...at { values[base + key] = true }
            }
        }
        return values
    }

    private func forward(
        _ ids: [[Int]], width: Int, attention: DecoderAttention, read: DecoderCache?, write: DecoderCache?
    ) throws -> MLXArray {
        let c = configuration
        let rows = ids.count
        let prefixLength = read?.length ?? 0
        let positionIDs = attention.positionIDs(width: width, prefix: prefixLength)
        let highest = Int(positionIDs.max() ?? 0)
        guard highest < c.maxPositionEmbeddings else {
            throw DecoderError.sequenceTooLong(maximum: c.maxPositionEmbeddings, actual: highest + 1)
        }
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode = attention.isPlainCausal(width: width)
            ? .causal
            : .array(attention.mask(width: width, prefix: prefixLength))
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode =
            attention.lengths.allSatisfy({ $0 == width }) && prefixLength + width <= c.slidingWindow
                ? .causal
                : .array(Self.slidingMask(lengths: attention.lengths, width: width, prefix: prefixLength, window: c.slidingWindow))
        let positions = MLXArray(positionIDs, [rows, width])
        let fullRotation = Self.rotation(positions: positions, rotary: c.fullRotary, dimension: c.headDimension)
        let slidingRotation = Self.rotation(positions: positions, rotary: c.slidingRotary, dimension: c.headDimension)

        var hidden = embedding(MLXArray(ids.flatMap { $0.map(Int32.init) }, [rows, width]))
        for (index, layer) in layers.enumerated() {
            let input = MLXFast.rmsNorm(hidden, weight: layer.operatorNorm, eps: c.normEpsilon)
            var mixed: MLXArray
            if let block = layer.attention {
                let rotation = block.sliding ? slidingRotation : fullRotation
                func rotate(_ x: MLXArray) -> MLXArray {
                    let halves = x.split(parts: 2, axis: -1)
                    let turned = concatenated([-halves[1], halves[0]], axis: -1)
                    return x * rotation.cos.asType(x.dtype) + turned * rotation.sin.asType(x.dtype)
                }
                let queries = rotate(MLXFast.rmsNorm(block.query(input).reshaped(rows, width, c.attentionHeads, -1), weight: block.queryNorm, eps: c.normEpsilon)
                    .transposed(0, 2, 1, 3))
                var keys = rotate(MLXFast.rmsNorm(block.key(input).reshaped(rows, width, c.keyValueHeads, -1), weight: block.keyNorm, eps: c.normEpsilon)
                    .transposed(0, 2, 1, 3))
                var values = block.value(input).reshaped(rows, width, c.keyValueHeads, -1).transposed(0, 2, 1, 3)
                if let cachedKeys = read?.keys[index], let cachedValues = read?.values[index] {
                    keys = concatenated([DecoderBatch.rows(cachedKeys, rows), keys], axis: 2)
                    values = concatenated([DecoderBatch.rows(cachedValues, rows), values], axis: 2)
                }
                write?.keys[index] = keys
                write?.values[index] = values
                let attended = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values,
                    scale: 1 / Float(c.headDimension).squareRoot(), mask: block.sliding ? slidingMask : fullMask
                ).transposed(0, 2, 1, 3)
                let gated = attended * MLX.sigmoid(block.gate(input)).expandedDimensions(axis: -1)
                mixed = block.output(gated.reshaped(rows, width, -1))
            } else if let block = layer.convolution {
                let parts = block.input(input).split(parts: 3, axis: -1)
                let gatedInput = parts[0] * parts[2]
                let kernel = c.convolutionKernel
                let window = read?.convolution[index].map { DecoderBatch.rows($0, rows) }
                    ?? MLXArray.zeros([rows, kernel - 1, c.hiddenSize], dtype: gatedInput.dtype)
                let padded = concatenated([window, gatedInput], axis: 1)
                write?.convolution[index] = padded[0..., width ..< (width + kernel - 1), 0...]
                var convolved = padded[0..., 0 ..< width, 0...] * block.kernel[0]
                for tap in 1 ..< kernel {
                    convolved = convolved + padded[0..., tap ..< (tap + width), 0...] * block.kernel[tap]
                }
                if let bias = block.bias { convolved = convolved + bias }
                mixed = block.output(parts[1] * convolved)
            } else {
                mixed = MLXArray.zeros(like: hidden)
            }
            if let norm = layer.postOperatorNorm { mixed = MLXFast.rmsNorm(mixed, weight: norm, eps: c.normEpsilon) }
            hidden = hidden + mixed
            let ffnInput = MLXFast.rmsNorm(hidden, weight: layer.feedForwardNorm, eps: c.normEpsilon)
            var ffn = layer.down(silu(layer.gate(ffnInput)) * layer.up(ffnInput))
            if let norm = layer.postFeedForwardNorm { ffn = MLXFast.rmsNorm(ffn, weight: norm, eps: c.normEpsilon) }
            hidden = hidden + ffn
        }
        write?.length = prefixLength + width
        return MLXFast.rmsNorm(hidden, weight: finalNorm, eps: c.normEpsilon)
    }
}
