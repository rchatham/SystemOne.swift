import Foundation

/// The token mixer a Qwen3.5 layer uses.
public enum Qwen35LayerType: String, Equatable, Sendable {
    /// Gated DeltaNet: a causal convolution and a gated delta-rule recurrence.
    case linearAttention = "linear_attention"
    /// Gated full attention: grouped-query attention with partial RoPE, query
    /// and key RMSNorm, and a sigmoid output gate.
    case fullAttention = "full_attention"
}

/// A Qwen3.5 text trunk's settings, read from `config.json`.
///
/// Both forms are accepted: the vision-language config with a nested
/// `text_config` (`Qwen3_5ForConditionalGeneration`, `Qwen3_5Model`) and a
/// text-only config (`Qwen3_5ForCausalLM`, mlx-lm exports). Layer types come
/// from `layer_types` when present (pruned models such as d3-edge mix them
/// freely), otherwise every `full_attention_interval`-th layer is full attention.
public struct Qwen35Configuration: Equatable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerTypes: [Qwen35LayerType]
    public let rmsNormEpsilon: Float
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int

    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let attentionBias: Bool
    /// Whether `q_proj` also produces a per-head sigmoid output gate.
    public let attentionOutputGate: Bool
    public let ropeTheta: Float
    /// The leading features of each head RoPE rotates (`head_dim × partial_rotary_factor`).
    public let rotaryDimension: Int

    public let linearKeyHeads: Int
    public let linearValueHeads: Int
    public let linearKeyHeadDimension: Int
    public let linearValueHeadDimension: Int
    public let convolutionKernel: Int

    public var layerCount: Int { layerTypes.count }
    public var linearKeyDimension: Int { linearKeyHeads * linearKeyHeadDimension }
    public var linearValueDimension: Int { linearValueHeads * linearValueHeadDimension }
    /// Channels of the causal convolution: queries, keys, and values.
    public var convolutionDimension: Int { 2 * linearKeyDimension + linearValueDimension }

    public init(
        vocabSize: Int, hiddenSize: Int, intermediateSize: Int, layerTypes: [Qwen35LayerType],
        rmsNormEpsilon: Float = 1e-6, tieWordEmbeddings: Bool, maxPositionEmbeddings: Int = 262_144,
        attentionHeads: Int, keyValueHeads: Int, headDimension: Int, attentionBias: Bool = false,
        attentionOutputGate: Bool = true, ropeTheta: Float, rotaryDimension: Int,
        linearKeyHeads: Int, linearValueHeads: Int, linearKeyHeadDimension: Int,
        linearValueHeadDimension: Int, convolutionKernel: Int = 4
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerTypes = layerTypes
        self.rmsNormEpsilon = rmsNormEpsilon
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headDimension = headDimension
        self.attentionBias = attentionBias
        self.attentionOutputGate = attentionOutputGate
        self.ropeTheta = ropeTheta
        self.rotaryDimension = rotaryDimension
        self.linearKeyHeads = linearKeyHeads
        self.linearValueHeads = linearValueHeads
        self.linearKeyHeadDimension = linearKeyHeadDimension
        self.linearValueHeadDimension = linearValueHeadDimension
        self.convolutionKernel = convolutionKernel
    }

    public init(modelFolder: URL) throws {
        let url = modelFolder.appendingPathComponent("config.json")
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw Qwen35ConfigurationError.invalid("config.json is not a JSON object")
        }
        try self.init(json: json)
    }

    public init(json root: [String: Any]) throws {
        let text = root["text_config"] as? [String: Any] ?? root
        func int(_ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = text[key] as? Int { return value }
            if let fallback { return fallback }
            throw Qwen35ConfigurationError.missing(key)
        }
        func bool(_ key: String, _ fallback: Bool) -> Bool { text[key] as? Bool ?? root[key] as? Bool ?? fallback }

        if let type = text["model_type"] as? String,
           !["qwen3_5_text", "qwen3_5", "qwen3_next"].contains(type) {
            throw Qwen35ConfigurationError.unsupported("model_type \(type)")
        }
        if let experts = text["num_experts"] as? Int, experts > 0 {
            throw Qwen35ConfigurationError.unsupported("mixture-of-experts layers")
        }
        if let activation = text["hidden_act"] as? String, activation != "silu" {
            throw Qwen35ConfigurationError.unsupported("hidden_act \(activation)")
        }

        let layerCount = try int("num_hidden_layers")
        let layerTypes: [Qwen35LayerType]
        if let names = text["layer_types"] as? [String] {
            layerTypes = try names.map {
                guard let type = Qwen35LayerType(rawValue: $0) else { throw Qwen35ConfigurationError.unsupported("layer type \($0)") }
                return type
            }
            guard layerTypes.count == layerCount else {
                throw Qwen35ConfigurationError.invalid("layer_types has \(layerTypes.count) entries for \(layerCount) layers")
            }
        } else {
            let interval = try int("full_attention_interval", 4)
            layerTypes = (0..<layerCount).map { ($0 + 1) % interval == 0 ? .fullAttention : .linearAttention }
        }

        let rope = text["rope_parameters"] as? [String: Any] ?? text["rope_scaling"] as? [String: Any] ?? [:]
        if let type = (rope["rope_type"] ?? rope["type"]) as? String, type != "default" {
            throw Qwen35ConfigurationError.unsupported("rope_type \(type)")
        }
        func double(_ key: String, _ fallback: Double) -> Double {
            (rope[key] as? NSNumber)?.doubleValue ?? (text[key] as? NSNumber)?.doubleValue ?? fallback
        }

        let hiddenSize = try int("hidden_size")
        let heads = try int("num_attention_heads")
        let headDimension = try int("head_dim", hiddenSize / heads)
        self.init(
            vocabSize: try int("vocab_size"),
            hiddenSize: hiddenSize,
            intermediateSize: try int("intermediate_size"),
            layerTypes: layerTypes,
            rmsNormEpsilon: Float((text["rms_norm_eps"] as? NSNumber)?.doubleValue ?? 1e-6),
            tieWordEmbeddings: bool("tie_word_embeddings", false),
            maxPositionEmbeddings: try int("max_position_embeddings", 262_144),
            attentionHeads: heads,
            keyValueHeads: try int("num_key_value_heads", heads),
            headDimension: headDimension,
            attentionBias: bool("attention_bias", false),
            attentionOutputGate: bool("attn_output_gate", true),
            ropeTheta: Float(double("rope_theta", 10_000_000)),
            rotaryDimension: Int(Double(headDimension) * double("partial_rotary_factor", 0.25)),
            linearKeyHeads: try int("linear_num_key_heads"),
            linearValueHeads: try int("linear_num_value_heads"),
            linearKeyHeadDimension: try int("linear_key_head_dim"),
            linearValueHeadDimension: try int("linear_value_head_dim"),
            convolutionKernel: try int("linear_conv_kernel_dim", 4)
        )
        try validate()
    }

    func validate() throws {
        guard attentionHeads % keyValueHeads == 0 else {
            throw Qwen35ConfigurationError.invalid("\(attentionHeads) attention heads do not group into \(keyValueHeads) key-value heads")
        }
        guard linearValueHeads % linearKeyHeads == 0 else {
            throw Qwen35ConfigurationError.invalid("\(linearValueHeads) value heads do not group into \(linearKeyHeads) key heads")
        }
        guard rotaryDimension > 0, rotaryDimension % 2 == 0, rotaryDimension <= headDimension else {
            throw Qwen35ConfigurationError.invalid("rotary dimension \(rotaryDimension) for head dimension \(headDimension)")
        }
        guard convolutionKernel >= 2 else { throw Qwen35ConfigurationError.invalid("convolution kernel \(convolutionKernel)") }
    }
}

public enum Qwen35ConfigurationError: Error, Equatable, Sendable {
    case missing(String)
    case invalid(String)
    case unsupported(String)
}

extension Qwen35ConfigurationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missing(let key): return "The Qwen3.5 config has no \(key)."
        case .invalid(let reason): return "The Qwen3.5 config is invalid: \(reason)."
        case .unsupported(let feature): return "The Qwen3.5 trunk does not support \(feature)."
        }
    }
}
