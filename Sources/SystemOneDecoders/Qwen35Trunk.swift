import Foundation
import MLX
import MLXNN
import SystemOne

/// How a checkpoint stores the trunk.
public enum Qwen35WeightLayout: Equatable, Sendable {
    /// transformers: zero-centred RMSNorm weights (`x · (1 + w)`) and
    /// `conv1d.weight` as `[channels, 1, kernel]`.
    case transformers
    /// mlx-lm exports: norm weights already include the `+ 1`, `conv1d.weight`
    /// as `[channels, kernel, 1]`, possibly quantized.
    case mlx
}

/// The Qwen3.5 text trunk (Gated DeltaNet and gated attention layers, 3:1 in
/// the released models) returning final-norm hidden states, as decision
/// readouts need them. Apache-2.0, following transformers' `Qwen3_5TextModel`.
///
/// Hidden states come from right-padded rows under a ``DecoderAttention`` plan,
/// optionally after a cached prefix. The language-model head is never applied
/// in full: ``readoutRows(_:)`` gathers the output-embedding rows a readout
/// needs (tied to the input embeddings when the config says so).
public final class Qwen35Trunk {
    public let configuration: Qwen35Configuration
    public let layout: Qwen35WeightLayout
    public let dataType: DType

    private struct Attention {
        let query: Linear
        let key: Linear
        let value: Linear
        let output: Linear
        let queryNorm: MLXArray
        let keyNorm: MLXArray
    }

    private struct DeltaNet {
        let queryKeyValue: Linear
        let gate: Linear
        let beta: Linear
        let alpha: Linear
        /// `[kernel, channels]`.
        let convolution: MLXArray
        let decayRate: MLXArray
        let timeBias: MLXArray
        let norm: MLXArray
        let output: Linear
    }

    private struct Layer {
        let attention: Attention?
        let deltaNet: DeltaNet?
        let inputNorm: MLXArray
        let mlpNorm: MLXArray
        let gateProjection: Linear
        let upProjection: Linear
        let downProjection: Linear
    }

    private let embedding: Embedding
    private let readout: Embedding?
    private let layers: [Layer]
    private let finalNorm: MLXArray

    /// Opens `config.json` and the safetensors in `modelFolder` (one file or a
    /// sharded index).
    public convenience init(modelFolder: URL, dataType: DType = .float32) throws {
        try self.init(
            store: SafetensorsCheckpoint(modelFolder: modelFolder),
            configuration: Qwen35Configuration(modelFolder: modelFolder),
            dataType: dataType
        )
    }

    /// Loads the trunk from `store`, finding its tensors under any of the
    /// published prefixes (`model.language_model.`, `language_model.`,
    /// `model.`, mlx-lm's `language_model.model.`). Vision and multi-token
    /// prediction tensors are ignored.
    ///
    /// - Parameters:
    ///   - adapter: a LoRA merged into the linear layers it targets as they load.
    ///   - loadsReadout: `false` skips an untied language-model head, for models
    ///     read through their own head; ``readoutRows(_:)`` then throws.
    public init(
        store: any TensorStore, configuration: Qwen35Configuration, dataType: DType = .float32,
        adapter: LoRAAdapter? = nil, loadsReadout: Bool = true
    ) throws {
        self.configuration = configuration
        self.dataType = dataType
        let prefix = try Self.prefix(in: store)
        let layout = try Self.layout(of: store, prefix: prefix, configuration: configuration)
        self.layout = layout
        let c = configuration
        let d = c.hiddenSize

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
            if let adapter {
                return try adapter.linear(
                    name, base: store, inputWidth: input, outputWidth: output, bias: bias, dataType: dataType)
            }
            return try store.linear(name, inputWidth: input, bias: bias, dataType: dataType)
        }
        /// A norm weight in float32, with transformers' `+ 1` folded in.
        func norm(_ name: String, _ width: Int, centred: Bool = true) throws -> MLXArray {
            try check(name, [width])
            let weight = try store.requiredTensor(name).mlxArray(.float32)
            return centred && layout == .transformers ? weight + 1 : weight
        }
        func vector(_ name: String, _ width: Int) throws -> MLXArray {
            try check(name, [width])
            return try store.requiredTensor(name).mlxArray(.float32)
        }

        try check("\(prefix)embed_tokens.weight", [c.vocabSize, d])
        let input = try store.embedding("\(prefix)embed_tokens", width: d, dataType: dataType)
        embedding = input
        if c.tieWordEmbeddings {
            readout = input
        } else if !loadsReadout {
            readout = nil
        } else {
            let head = try Self.headPrefix(in: store, trunkPrefix: prefix)
            try check("\(head).weight", [c.vocabSize, d])
            readout = try store.embedding(head, width: d, dataType: dataType)
        }

        layers = try c.layerTypes.enumerated().map { index, type -> Layer in
            let p = "\(prefix)layers.\(index)."
            var attention: Attention?
            var deltaNet: DeltaNet?
            switch type {
            case .fullAttention:
                let queryWidth = c.attentionHeads * c.headDimension * (c.attentionOutputGate ? 2 : 1)
                let keyWidth = c.keyValueHeads * c.headDimension
                attention = Attention(
                    query: try linear("\(p)self_attn.q_proj", queryWidth, d, bias: c.attentionBias),
                    key: try linear("\(p)self_attn.k_proj", keyWidth, d, bias: c.attentionBias),
                    value: try linear("\(p)self_attn.v_proj", keyWidth, d, bias: c.attentionBias),
                    output: try linear("\(p)self_attn.o_proj", d, c.attentionHeads * c.headDimension, bias: c.attentionBias),
                    queryNorm: try norm("\(p)self_attn.q_norm.weight", c.headDimension),
                    keyNorm: try norm("\(p)self_attn.k_norm.weight", c.headDimension)
                )
            case .linearAttention:
                let channels = c.convolutionDimension
                let convolution = "\(p)linear_attn.conv1d.weight"
                let kernelShape = layout == .transformers ? [channels, 1, c.convolutionKernel] : [channels, c.convolutionKernel, 1]
                try check(convolution, kernelShape)
                deltaNet = DeltaNet(
                    queryKeyValue: try linear("\(p)linear_attn.in_proj_qkv", channels, d),
                    gate: try linear("\(p)linear_attn.in_proj_z", c.linearValueDimension, d),
                    beta: try linear("\(p)linear_attn.in_proj_b", c.linearValueHeads, d),
                    alpha: try linear("\(p)linear_attn.in_proj_a", c.linearValueHeads, d),
                    convolution: try store.requiredTensor(convolution).mlxArray(dataType)
                        .reshaped(channels, c.convolutionKernel).transposed(),
                    decayRate: try vector("\(p)linear_attn.A_log", c.linearValueHeads),
                    timeBias: try vector("\(p)linear_attn.dt_bias", c.linearValueHeads),
                    norm: try norm("\(p)linear_attn.norm.weight", c.linearValueHeadDimension, centred: false),
                    output: try linear("\(p)linear_attn.out_proj", d, c.linearValueDimension)
                )
            }
            return Layer(
                attention: attention,
                deltaNet: deltaNet,
                inputNorm: try norm("\(p)input_layernorm.weight", d),
                mlpNorm: try norm("\(p)post_attention_layernorm.weight", d),
                gateProjection: try linear("\(p)mlp.gate_proj", c.intermediateSize, d),
                upProjection: try linear("\(p)mlp.up_proj", c.intermediateSize, d),
                downProjection: try linear("\(p)mlp.down_proj", d, c.intermediateSize)
            )
        }
        finalNorm = try norm("\(prefix)norm.weight", d)
        try adapter?.requireAllUsed()
    }

    // MARK: - Public forward passes

    /// Final-norm hidden states `[rows, width, hidden]` for right-padded token
    /// rows, after `prefix` when given. States past a row's length are
    /// unspecified. `prefix` is read, not changed; a one-row prefix serves
    /// every row.
    public func callAsFunction(
        _ ids: [[Int]], attention: DecoderAttention? = nil, prefix: DecoderCache? = nil
    ) throws -> MLXArray {
        let width = try validate(ids)
        let attention = attention ?? DecoderAttention(lengths: Array(repeating: width, count: ids.count))
        guard attention.lengths.count == ids.count else {
            throw DecoderError.invalidLength(row: min(attention.lengths.count, ids.count), length: 0)
        }
        try attention.validate(width: width)
        return try forward(ids, width: width, attention: attention, read: prefix, write: nil)
    }

    /// Extends `cache` with one row of tokens under a causal plan and returns
    /// their hidden states `[1, length, hidden]`.
    @discardableResult
    public func prefill(_ ids: [Int], cache: DecoderCache) throws -> MLXArray {
        let width = try validate([ids])
        let attention = DecoderAttention(lengths: [width])
        return try forward([ids], width: width, attention: attention, read: cache, write: cache)
    }

    /// Output-embedding rows `[ids.count, hidden]` (the language-model head's,
    /// or the input embeddings when tied), dequantized. A readout's logits are
    /// `hidden · rowsᵀ`.
    public func readoutRows(_ ids: [Int]) throws -> MLXArray {
        guard let readout else { throw DecoderError.noReadout }
        for id in ids where !(0..<configuration.vocabSize).contains(id) {
            throw DecoderError.invalidTokenID(row: 0, column: 0, value: id)
        }
        return readout(MLXArray(ids.map(Int32.init)))
    }

    // MARK: - Forward

    private func validate(_ ids: [[Int]]) throws -> Int {
        guard let width = ids.first?.count, width > 0 else { throw DecoderError.emptyBatch }
        for (row, tokens) in ids.enumerated() {
            guard tokens.count == width else { throw DecoderError.raggedBatch(row: row) }
            for (column, token) in tokens.enumerated() where !(0..<configuration.vocabSize).contains(token) {
                throw DecoderError.invalidTokenID(row: row, column: column, value: token)
            }
        }
        return width
    }

    private func forward(
        _ ids: [[Int]], width: Int, attention: DecoderAttention, read: DecoderCache?, write: DecoderCache?
    ) throws -> MLXArray {
        let rows = ids.count
        let prefixLength = read?.length ?? 0
        let positions = attention.positionIDs(width: width, prefix: prefixLength)
        let highest = Int(positions.max() ?? 0)
        guard highest < configuration.maxPositionEmbeddings else {
            throw DecoderError.sequenceTooLong(maximum: configuration.maxPositionEmbeddings, actual: highest + 1)
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = attention.isPlainCausal(width: width)
            ? .causal
            : .array(attention.mask(width: width, prefix: prefixLength))
        let rotation = attention.hasDefaultPositions(prefix: prefixLength)
            ? nil
            : Self.rotation(
                positions: MLXArray(positions, [rows, width]), theta: configuration.ropeTheta,
                dimension: configuration.rotaryDimension
            )

        var hidden = embedding(MLXArray(ids.flatMap { $0.map(Int32.init) }, [rows, width]))
        for (index, layer) in layers.enumerated() {
            let input = normalize(hidden, layer.inputNorm)
            if let block = layer.attention {
                hidden = hidden + attend(
                    input, block, layer: index, mask: mask, rotation: rotation,
                    offset: prefixLength, read: read, write: write
                )
            } else if let block = layer.deltaNet {
                hidden = hidden + mix(input, block, layer: index, read: read, write: write)
            }
            let mlpInput = normalize(hidden, layer.mlpNorm)
            hidden = hidden + layer.downProjection(silu(layer.gateProjection(mlpInput)) * layer.upProjection(mlpInput))
        }
        write?.length = prefixLength + width
        return normalize(hidden, finalNorm)
    }

    /// RMSNorm in float32 with a float32 weight, as transformers computes it.
    private func normalize(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: configuration.rmsNormEpsilon).asType(dataType)
    }

    /// A cached `[1, …]` array repeated over the batch.
    private func batched(_ array: MLXArray, rows: Int) -> MLXArray {
        array.dim(0) == rows ? array : MLX.broadcast(array, to: [rows] + array.shape.dropFirst())
    }

    private func attend(
        _ x: MLXArray, _ block: Attention, layer: Int, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        rotation: (cos: MLXArray, sin: MLXArray)?, offset: Int, read: DecoderCache?, write: DecoderCache?
    ) -> MLXArray {
        let c = configuration
        let (rows, width) = (x.dim(0), x.dim(1))
        var queries = block.query(x).reshaped(rows, width, c.attentionHeads, -1)
        var gate: MLXArray?
        if c.attentionOutputGate {
            let parts = queries.split(parts: 2, axis: -1)
            queries = parts[0]
            gate = parts[1].reshaped(rows, width, -1)
        }
        queries = headNorm(queries, block.queryNorm).transposed(0, 2, 1, 3)
        var keys = headNorm(block.key(x).reshaped(rows, width, c.keyValueHeads, -1), block.keyNorm).transposed(0, 2, 1, 3)
        var values = block.value(x).reshaped(rows, width, c.keyValueHeads, -1).transposed(0, 2, 1, 3)

        if let rotation {
            queries = Self.rotate(queries, rotation, dimension: c.rotaryDimension)
            keys = Self.rotate(keys, rotation, dimension: c.rotaryDimension)
        } else {
            queries = MLXFast.RoPE(queries, dimensions: c.rotaryDimension, traditional: false, base: c.ropeTheta, scale: 1, offset: offset)
            keys = MLXFast.RoPE(keys, dimensions: c.rotaryDimension, traditional: false, base: c.ropeTheta, scale: 1, offset: offset)
        }
        // Read the prefix before recording: in a prefill they are one cache.
        if let cachedKeys = read?.keys[layer], let cachedValues = read?.values[layer] {
            keys = concatenated([batched(cachedKeys, rows: rows), keys], axis: 2)
            values = concatenated([batched(cachedValues, rows: rows), values], axis: 2)
        }
        write?.keys[layer] = keys
        write?.values[layer] = values
        var attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / Float(c.headDimension).squareRoot(), mask: mask
        ).transposed(0, 2, 1, 3).reshaped(rows, width, -1)
        if let gate { attended = attended * MLX.sigmoid(gate) }
        return block.output(attended)
    }

    /// Per-head RMSNorm over the last axis.
    private func headNorm(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        normalize(x, weight)
    }

    private func mix(_ x: MLXArray, _ block: DeltaNet, layer: Int, read: DecoderCache?, write: DecoderCache?) -> MLXArray {
        let c = configuration
        let (rows, width) = (x.dim(0), x.dim(1))
        let kernel = c.convolutionKernel

        // Causal depthwise convolution over [window; new tokens], then SiLU.
        let projected = block.queryKeyValue(x)
        let window = read?.convolution[layer].map { batched($0, rows: rows) }
            ?? MLXArray.zeros([rows, kernel - 1, c.convolutionDimension], dtype: projected.dtype)
        let padded = concatenated([window, projected], axis: 1)
        write?.convolution[layer] = padded[0..., width ..< (width + kernel - 1), 0...]
        var convolved = padded[0..., 0 ..< width, 0...] * block.convolution[0]
        for tap in 1 ..< kernel {
            convolved = convolved + padded[0..., tap ..< (tap + width), 0...] * block.convolution[tap]
        }
        let mixed = silu(convolved)

        let parts = mixed.split(indices: [c.linearKeyDimension, 2 * c.linearKeyDimension], axis: -1)
        var queries = parts[0].reshaped(rows, width, c.linearKeyHeads, c.linearKeyHeadDimension).asType(.float32)
        var keys = parts[1].reshaped(rows, width, c.linearKeyHeads, c.linearKeyHeadDimension).asType(.float32)
        let values = parts[2].reshaped(rows, width, c.linearValueHeads, c.linearValueHeadDimension)
        queries = GatedDeltaRule.l2Normalized(queries) * (1 / Float(c.linearKeyHeadDimension).squareRoot())
        keys = GatedDeltaRule.l2Normalized(keys)
        let repeats = c.linearValueHeads / c.linearKeyHeads
        if repeats > 1 {
            queries = repeated(queries, count: repeats, axis: 2)
            keys = repeated(keys, count: repeats, axis: 2)
        }

        let beta = MLX.sigmoid(block.beta(x)).asType(.float32)
        let decay = -MLX.exp(block.decayRate) * softplus(block.alpha(x).asType(.float32) + block.timeBias)
        let (output, state) = GatedDeltaRule.run(
            query: queries, key: keys, value: values, decay: decay, beta: beta,
            state: read?.recurrent[layer].map { batched($0, rows: rows) }
        )
        write?.recurrent[layer] = state

        // Gated RMSNorm on the output in the model type: normalize in float32,
        // weight in the model type, SiLU gate in float32.
        let gate = block.gate(x).reshaped(rows, width, c.linearValueHeads, c.linearValueHeadDimension)
        let unit = MLXArray.ones([c.linearValueHeadDimension])
        let normalized = (MLXFast.rmsNorm(output.asType(dataType).asType(.float32), weight: unit, eps: c.rmsNormEpsilon)
            .asType(dataType) * block.norm.asType(dataType)).asType(.float32)
        let gated = (normalized * silu(gate.asType(.float32))).asType(dataType)
        return block.output(gated.reshaped(rows, width, -1))
    }

    // MARK: - Weights and RoPE

    /// The prefix the trunk's tensors share in `store`.
    static func prefix(in store: any TensorStore) throws -> String {
        for prefix in ["model.language_model.", "language_model.model.", "language_model.", "model.", ""] {
            if store.descriptors["\(prefix)embed_tokens.weight"] != nil { return prefix }
        }
        throw WeightsError.tensorSetMismatch(missing: ["embed_tokens.weight"], unexpected: [])
    }

    /// The untied language-model head's prefix.
    static func headPrefix(in store: any TensorStore, trunkPrefix: String) throws -> String {
        let candidates = trunkPrefix == "language_model.model." ? ["language_model.lm_head", "lm_head"] : ["lm_head", "language_model.lm_head"]
        for head in candidates where store.descriptors["\(head).weight"] != nil { return head }
        throw WeightsError.tensorSetMismatch(missing: ["lm_head.weight"], unexpected: [])
    }

    /// transformers or mlx-lm storage, told apart by the convolution's shape.
    static func layout(of store: any TensorStore, prefix: String, configuration: Qwen35Configuration) throws -> Qwen35WeightLayout {
        guard let index = configuration.layerTypes.firstIndex(of: .linearAttention) else {
            // No recurrent layer to tell by: mlx-lm keeps the transformers names
            // under `language_model.model.`.
            return prefix == "language_model.model." ? .mlx : .transformers
        }
        let name = "\(prefix)layers.\(index).linear_attn.conv1d.weight"
        let shape = try store.requiredTensor(name).descriptor.shape
        return shape.count == 3 && shape[1] == 1 ? .transformers : .mlx
    }

    /// Cosines and sines `[rows, 1, width, dimension]` for explicit positions,
    /// in float32 as transformers computes them.
    static func rotation(positions: MLXArray, theta: Float, dimension: Int) -> (cos: MLXArray, sin: MLXArray) {
        let inverse = (0..<(dimension / 2)).map { 1 / powf(theta, Float(2 * $0) / Float(dimension)) }
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1) * MLXArray(inverse)
        let angles = concatenated([frequencies, frequencies], axis: -1).expandedDimensions(axis: 1)
        return (MLX.cos(angles), MLX.sin(angles))
    }

    /// Non-interleaved RoPE on the first `dimension` features of each head.
    static func rotate(_ x: MLXArray, _ rotation: (cos: MLXArray, sin: MLXArray), dimension: Int) -> MLXArray {
        let rotary = x[.ellipsis, 0 ..< dimension].asType(.float32)
        let halves = rotary.split(parts: 2, axis: -1)
        let turned = rotary * rotation.cos + concatenated([-halves[1], halves[0]], axis: -1) * rotation.sin
        guard dimension < x.dim(-1) else { return turned.asType(x.dtype) }
        return concatenated([turned.asType(x.dtype), x[.ellipsis, dimension...]], axis: -1)
    }
}
