import Foundation
import MLX
import MLXNN

/// The shared LFM2.5-Encoder trunk (`encoder.py` `Trunk`): LFM2 blocks made
/// bidirectional, with a centred 3-tap short convolution and no causal mask.
///
/// A media prefix enters as embeddings in front of the text. Prefix positions
/// attend only to the prefix, and the convolution never lets the last prefix
/// position read the first text position, so the prefix is a function of the
/// media alone: a request records its prefix's states once (`prefixState(of:)`)
/// and each question then runs only its text positions. Text, image, and audio
/// rows run the same weights.
final class D1Trunk {
    private enum Operator {
        case convolution(inProjection: Linear, kernel: MLXArray, outProjection: Linear)
        case attention(
            query: Linear, key: Linear, value: Linear, output: Linear,
            queryNorm: MLXArray, keyNorm: MLXArray
        )
    }

    private struct Layer {
        let operatorNorm: MLXArray
        let op: Operator
        let feedForwardNorm: MLXArray
        let gate: Linear
        let up: Linear
        let down: Linear
    }

    /// Additive mask value for blocked keys, as `encoder.py` uses in float32.
    static let blocked: Float = -1e9

    let embedding: Embedding
    private let layers: [Layer]
    private let finalNorm: MLXArray
    private let configuration: D1Configuration.Text

    init(weights: D1Weights) {
        let configuration = weights.configuration.text
        self.configuration = configuration
        embedding = Embedding(weight: weights.array("encoder.embed_tokens.weight"))
        finalNorm = weights.array("encoder.embedding_norm.weight")
        layers = (0..<configuration.numHiddenLayers).map { index in
            let prefix = "encoder.layers.\(index)"
            func linear(_ name: String) -> Linear { Linear(weight: weights.array("\(prefix).\(name).weight")) }
            let op: Operator
            if configuration.isAttentionLayer(index) {
                op = .attention(
                    query: linear("self_attn.q_proj"),
                    key: linear("self_attn.k_proj"),
                    value: linear("self_attn.v_proj"),
                    output: linear("self_attn.out_proj"),
                    queryNorm: weights.array("\(prefix).self_attn.q_layernorm.weight"),
                    keyNorm: weights.array("\(prefix).self_attn.k_layernorm.weight")
                )
            } else {
                // [D, 1, 3] depthwise taps, kept as [3, D] for per-tap broadcasting.
                let kernel = weights.array("\(prefix).conv.conv.weight").squeezed(axis: 1).transposed()
                op = .convolution(
                    inProjection: linear("conv.in_proj"),
                    kernel: kernel,
                    outProjection: linear("conv.out_proj")
                )
            }
            return Layer(
                operatorNorm: weights.array("\(prefix).operator_norm.weight"),
                op: op,
                feedForwardNorm: weights.array("\(prefix).ffn_norm.weight"),
                gate: linear("feed_forward.w1"),
                up: linear("feed_forward.w3"),
                down: linear("feed_forward.w2")
            )
        }
    }

    /// The per-layer states of a media prefix that text positions read. They
    /// depend on the media alone, so one request computes them once and every
    /// question reuses them.
    struct PrefixState {
        enum Layer {
            /// `[1, 1, C]` `b * u` of the last prefix position: the left tap of
            /// the first text position.
            case convolution(lastProduct: MLXArray)
            /// `[1, Hkv, P, head]` keys after RMSNorm and RoPE, and values.
            case attention(keys: MLXArray, values: MLXArray)
        }

        let length: Int
        let layers: [Layer]

        var arrays: [MLXArray] {
            layers.flatMap { layer -> [MLXArray] in
                switch layer {
                case .convolution(let lastProduct): return [lastProduct]
                case .attention(let keys, let values): return [keys, values]
                }
            }
        }
    }

    /// Runs right-padded rows of embeddings.
    ///
    /// - Parameters:
    ///   - hidden: `[B, L, D]` embeddings, media prefix first, zero padding last.
    ///   - lengths: real positions per row (prefix plus text).
    ///   - prefixLengths: media prefix positions per row (0 for text-only rows).
    ///   - captureLayers: when true, also returns every layer boundary for parity probes.
    func callAsFunction(
        _ hidden: MLXArray,
        lengths: [Int],
        prefixLengths: [Int],
        captureLayers: Bool = false
    ) -> (output: MLXArray, layers: [MLXArray]) {
        let run = forward(hidden, lengths: lengths, prefixLengths: prefixLengths, after: nil,
                          captureLayers: captureLayers)
        return (run.output, run.layers)
    }

    /// Runs right-padded text rows that all follow the same media prefix,
    /// reading the prefix through its cached states instead of recomputing it.
    ///
    /// - Parameters:
    ///   - hidden: `[B, T, D]` text embeddings, zero padding last.
    ///   - lengths: real text positions per row.
    ///   - prefix: the states `prefixState(of:)` recorded for the shared prefix.
    func callAsFunction(_ hidden: MLXArray, lengths: [Int], after prefix: PrefixState) -> MLXArray {
        forward(hidden, lengths: lengths, prefixLengths: lengths.map { _ in 0 }, after: prefix).output
    }

    /// Records the states text positions read from a `[P, D]` media prefix.
    ///
    /// The prefix runs alone: it attends only to itself, and its last position's
    /// right tap is the zero padding `keepRight` would otherwise impose.
    func prefixState(of embeddings: MLXArray) -> PrefixState {
        let length = embeddings.dim(0)
        let run = forward(embeddings.expandedDimensions(axis: 0), lengths: [length], prefixLengths: [0],
                          after: nil, recording: true)
        return PrefixState(length: length, layers: run.recorded)
    }

    private func forward(
        _ hidden: MLXArray,
        lengths: [Int],
        prefixLengths: [Int],
        after prefix: PrefixState?,
        recording: Bool = false,
        captureLayers: Bool = false
    ) -> (output: MLXArray, layers: [MLXArray], recorded: [PrefixState.Layer]) {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let eps = configuration.normEps
        let positions = MLXArray(0..<Int32(length))
        let lengthColumn = MLXArray(lengths.map(Int32.init)).reshaped(batch, 1)
        let prefixColumn = MLXArray(prefixLengths.map(Int32.init)).reshaped(batch, 1)
        // Text after a cached prefix sits at positions P..<P+T.
        let offset = prefix?.length ?? 0

        let real = positions.reshaped(1, length) .< lengthColumn
        let padding = real.asType(.float32).expandedDimensions(axis: -1)
        // The last prefix position never reads the first text position.
        let keepRight = (positions.reshaped(1, length) .!= (prefixColumn - 1))
            .asType(.float32).expandedDimensions(axis: -1)

        var mask: MLXArray?
        if prefix != nil {
            // Text reads every prefix key and its own row's real text keys.
            if lengths.contains(where: { $0 != length }) {
                let keys = MLXArray(0..<Int32(offset + length)).reshaped(1, offset + length)
                let paddedKey = keys .>= (lengthColumn + Int32(offset))
                mask = MLX.where(paddedKey, MLXArray(Self.blocked), MLXArray(Float(0)))
                    .reshaped(batch, 1, 1, offset + length)
            }
        } else if lengths.contains(where: { $0 != length }) || prefixLengths.contains(where: { $0 > 0 }) {
            let mediaQuery = (positions.reshaped(1, length) .< prefixColumn).expandedDimensions(axis: 2)
            let textKey = (positions.reshaped(1, length) .>= prefixColumn).expandedDimensions(axis: 1)
            let paddedKey = logicalNot(real).expandedDimensions(axis: 1)
            let isBlocked = logicalOr(paddedKey, logicalAnd(mediaQuery, textKey))
            mask = MLX.where(isBlocked, MLXArray(Self.blocked), MLXArray(Float(0)))
                .expandedDimensions(axis: 1)
        }

        var hidden = hidden
        var captured: [MLXArray] = captureLayers ? [hidden] : []
        var recorded: [PrefixState.Layer] = []
        for (index, layer) in layers.enumerated() {
            let normalized = MLXFast.rmsNorm(hidden, weight: layer.operatorNorm, eps: eps)
            switch layer.op {
            case .convolution(let inProjection, let kernel, let outProjection):
                let projected = inProjection(normalized * padding)
                let parts = projected.split(parts: 3, axis: -1)
                let (b, c, u) = (parts[0], parts[1], parts[2])
                let product = b * u
                if recording {
                    // Projected on its own so the cache holds one position, not a
                    // view pinning the whole `[1, P, C]` product.
                    let last = inProjection(normalized[0..., (length - 1)...]).split(parts: 3, axis: -1)
                    recorded.append(.convolution(lastProduct: last[0] * last[2]))
                }
                let zero = MLXArray.zeros([batch, 1, product.dim(2)], dtype: product.dtype)
                var before = zero
                if case .convolution(let lastProduct)? = prefix?.layers[index] {
                    before = MLX.broadcast(lastProduct, to: zero.shape)
                }
                let bx = concatenated([before, product, zero], axis: 1)
                let left = bx[0..., 0..<length]
                let centre = bx[0..., 1..<(length + 1)]
                let right = bx[0..., 2..<(length + 2)] * keepRight
                let y = left * kernel[0] + centre * kernel[1] + right * kernel[2]
                hidden = hidden + outProjection(c * y)

            case .attention(let query, let key, let value, let output, let queryNorm, let keyNorm):
                let headDimension = configuration.headDimension
                func heads(_ x: MLXArray, count: Int) -> MLXArray {
                    x.reshaped(batch, length, count, headDimension)
                }
                var q = MLXFast.rmsNorm(heads(query(normalized), count: configuration.numAttentionHeads),
                                        weight: queryNorm, eps: eps).transposed(0, 2, 1, 3)
                var k = MLXFast.rmsNorm(heads(key(normalized), count: configuration.numKeyValueHeads),
                                        weight: keyNorm, eps: eps).transposed(0, 2, 1, 3)
                var v = heads(value(normalized), count: configuration.numKeyValueHeads).transposed(0, 2, 1, 3)
                q = MLXFast.RoPE(q, dimensions: headDimension, traditional: false,
                                 base: configuration.ropeTheta, scale: 1, offset: offset)
                k = MLXFast.RoPE(k, dimensions: headDimension, traditional: false,
                                 base: configuration.ropeTheta, scale: 1, offset: offset)
                if recording { recorded.append(.attention(keys: k, values: v)) }
                if case .attention(let keys, let values)? = prefix?.layers[index] {
                    func shared(_ x: MLXArray) -> MLXArray {
                        MLX.broadcast(x, to: [batch, x.dim(1), x.dim(2), x.dim(3)])
                    }
                    k = concatenated([shared(keys), k], axis: 2)
                    v = concatenated([shared(values), v], axis: 2)
                }
                let attended = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v,
                    scale: 1 / Float(headDimension).squareRoot(),
                    mask: mask
                )
                hidden = hidden + output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
            }
            let ffnInput = MLXFast.rmsNorm(hidden, weight: layer.feedForwardNorm, eps: eps)
            hidden = hidden + layer.down(silu(layer.gate(ffnInput)) * layer.up(ffnInput))
            if captureLayers { captured.append(hidden) }
        }
        let output = MLXFast.rmsNorm(hidden, weight: finalNorm, eps: eps)
        if captureLayers { captured.append(output) }
        return (output, captured, recorded)
    }
}
