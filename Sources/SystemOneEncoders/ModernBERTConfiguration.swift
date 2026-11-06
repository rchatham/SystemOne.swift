import Foundation

/// `config.json` of a ModernBERT-family encoder: ModernBERT, mmBERT, and the
/// Ettin encoders share this architecture and differ only in these values.
public struct ModernBERTConfiguration: Equatable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    /// `full_attention` or `sliding_attention` for every layer.
    public let layerTypes: [String]
    /// The whole local window; a sliding layer sees keys within half of it.
    public let localAttention: Int
    public let globalRoPETheta: Float
    public let localRoPETheta: Float
    public let maxPositionEmbeddings: Int
    public let normEps: Float
    public let normBias: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let hiddenActivation: String
    public let padTokenID: Int?

    public init(
        vocabSize: Int, hiddenSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        intermediateSize: Int, layerTypes: [String], localAttention: Int,
        globalRoPETheta: Float, localRoPETheta: Float, maxPositionEmbeddings: Int,
        normEps: Float = 1e-5, normBias: Bool = false, attentionBias: Bool = false, mlpBias: Bool = false,
        hiddenActivation: String = "gelu", padTokenID: Int? = nil
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.layerTypes = layerTypes
        self.localAttention = localAttention
        self.globalRoPETheta = globalRoPETheta
        self.localRoPETheta = localRoPETheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normEps = normEps
        self.normBias = normBias
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.hiddenActivation = hiddenActivation
        self.padTokenID = padTokenID
    }

    /// Reads `config.json` from a model folder.
    public init(modelFolder: URL) throws {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
        self = try JSONDecoder().decode(Self.self, from: data)
    }

    public var headDimension: Int { hiddenSize / numAttentionHeads }

    /// Keys a sliding layer's query may see on each side, inclusive.
    public var slidingRadius: Int { localAttention / 2 }

    public func isFullAttention(_ layer: Int) -> Bool {
        layerTypes[layer] == "full_attention"
    }

    public func ropeTheta(layer: Int) -> Float {
        isFullAttention(layer) ? globalRoPETheta : localRoPETheta
    }
}

extension ModernBERTConfiguration: Decodable {
    private enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case layerTypes = "layer_types"
        case globalAttentionEveryNLayers = "global_attn_every_n_layers"
        case localAttention = "local_attention"
        case globalRoPETheta = "global_rope_theta"
        case localRoPETheta = "local_rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case normBias = "norm_bias"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case hiddenActivation = "hidden_activation"
        case padTokenID = "pad_token_id"
    }

    private struct RoPE: Decodable {
        let ropeTheta: Float
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    /// Accepts both the transformers 5 form (`layer_types`, `rope_parameters`
    /// per layer type) and the original one (`global_attn_every_n_layers`,
    /// `global_rope_theta`, `local_rope_theta`), with ModernBERT's defaults.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let layers = try container.decode(Int.self, forKey: .numHiddenLayers)
        let layerTypes: [String]
        if let types = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            let every = try container.decodeIfPresent(Int.self, forKey: .globalAttentionEveryNLayers) ?? 3
            layerTypes = (0..<layers).map { $0 % every == 0 ? "full_attention" : "sliding_attention" }
        }
        guard layerTypes.count == layers,
              layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "layer_types must name full_attention or sliding_attention for each of \(layers) layers"
            )
        }
        var global = try container.decodeIfPresent(Float.self, forKey: .globalRoPETheta) ?? 160_000
        var local = try container.decodeIfPresent(Float.self, forKey: .localRoPETheta) ?? 10_000
        if let parameters = try container.decodeIfPresent([String: RoPE].self, forKey: .ropeParameters) {
            if let full = parameters["full_attention"] { global = full.ropeTheta }
            if let sliding = parameters["sliding_attention"] { local = sliding.ropeTheta }
        }
        self.init(
            vocabSize: try container.decode(Int.self, forKey: .vocabSize),
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            numHiddenLayers: layers,
            numAttentionHeads: try container.decode(Int.self, forKey: .numAttentionHeads),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            layerTypes: layerTypes,
            localAttention: try container.decodeIfPresent(Int.self, forKey: .localAttention) ?? 128,
            globalRoPETheta: global,
            localRoPETheta: local,
            maxPositionEmbeddings: try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8_192,
            normEps: try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5,
            normBias: try container.decodeIfPresent(Bool.self, forKey: .normBias) ?? false,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            hiddenActivation: try container.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu",
            padTokenID: try container.decodeIfPresent(Int.self, forKey: .padTokenID)
        )
    }
}
