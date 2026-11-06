import Foundation

/// Errors raised while reading a local JEV model folder.
public enum JevConfigurationError: Error, Equatable, Sendable {
    case missingFile(String)
    case invalid(file: String, reason: String)
}

extension JevConfigurationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "The local JEV model folder is missing \(name)."
        case .invalid(let file, let reason):
            return "\(file) is not a supported JEV configuration: \(reason)."
        }
    }
}

/// The three configuration files of a JEV checkpoint: the Qwen3.5 text
/// backbone (`config.json`), the decision head and adapter layout
/// (`judge_config.json`), and the per-kind temperatures (`calibration.json`).
/// Shapes, slot ranges and temperatures are read from here, not hard-coded.
public struct JevConfiguration: Equatable, Sendable {
    /// `Qwen3_5TextConfig`: the hybrid Gated DeltaNet / gated attention backbone.
    public struct Text: Decodable, Equatable, Sendable {
        public struct Rope: Decodable, Equatable, Sendable {
            public let ropeTheta: Float
            public let partialRotaryFactor: Float

            enum CodingKeys: String, CodingKey {
                case ropeTheta = "rope_theta"
                case partialRotaryFactor = "partial_rotary_factor"
            }
        }

        public let modelType: String
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let layerTypes: [String]
        public let linearConvKernelDim: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let linearNumKeyHeads: Int
        public let linearNumValueHeads: Int
        public let attnOutputGate: Bool
        public let tieWordEmbeddings: Bool
        public let ropeParameters: Rope

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case layerTypes = "layer_types"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case attnOutputGate = "attn_output_gate"
            case tieWordEmbeddings = "tie_word_embeddings"
            case ropeParameters = "rope_parameters"
        }

        /// Leading head dimensions RoPE rotates; the rest pass through.
        public var rotaryDimensions: Int { Int(Float(headDim) * ropeParameters.partialRotaryFactor) }
        /// Width of the Gated DeltaNet queries and of its keys.
        public var linearKeyDimension: Int { linearKeyHeadDim * linearNumKeyHeads }
        /// Width of the Gated DeltaNet values.
        public var linearValueDimension: Int { linearValueHeadDim * linearNumValueHeads }
        /// Channels of the depthwise convolution: queries, keys, values.
        public var convolutionChannels: Int { 2 * linearKeyDimension + linearValueDimension }

        public func isFullAttention(_ layer: Int) -> Bool { layerTypes[layer] == "full_attention" }
    }

    /// `judge_config.json`: the 24-slot head's layout and where the adapter lives.
    public struct Judge: Decodable, Equatable, Sendable {
        public struct Slots: Decodable, Equatable, Sendable {
            public let numSlots: Int
            public let ranges: [String: [Int]]
            public let templateVersion: String

            enum CodingKeys: String, CodingKey {
                case numSlots = "num_slots"
                case ranges
                case templateVersion = "template_version"
            }
        }

        public let hiddenSize: Int
        public let slots: Slots
        public let adapterSubfolder: String

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case slots
            case adapterSubfolder = "adapter_subfolder"
        }
    }

    /// peft's `adapter_config.json`: the System 1 LoRA, merged at load.
    public struct Adapter: Decodable, Equatable, Sendable {
        public let peftType: String
        public let r: Int
        public let loraAlpha: Double
        public let targetModules: [String]
        public let bias: String
        public let useDora: Bool
        public let useRslora: Bool
        public let fanInFanOut: Bool
        public let rankPattern: [String: Int]
        public let alphaPattern: [String: Double]

        enum CodingKeys: String, CodingKey {
            case peftType = "peft_type"
            case r
            case loraAlpha = "lora_alpha"
            case targetModules = "target_modules"
            case bias
            case useDora = "use_dora"
            case useRslora = "use_rslora"
            case fanInFanOut = "fan_in_fan_out"
            case rankPattern = "rank_pattern"
            case alphaPattern = "alpha_pattern"
        }

        /// peft's `lora_alpha / r` scaling.
        public var scale: Float { Float(loraAlpha / Double(r)) }

        /// Decoder projections a JEV adapter may target, by module name.
        static let supportedTargets: Set<String> = [
            "in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    }

    public let text: Text
    public let judge: Judge
    public let adapter: Adapter
    /// `calibration.json` `per_kind`: the temperature each kind's logits are divided by.
    public let temperatures: [JevKind: Float]

    public init(modelFolder: URL) throws {
        func data(_ name: String) throws -> Data {
            let url = modelFolder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw JevConfigurationError.missingFile(name)
            }
            return try Data(contentsOf: url)
        }
        func decode<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
            do {
                return try JSONDecoder().decode(type, from: data(name))
            } catch let error as JevConfigurationError {
                throw error
            } catch {
                throw JevConfigurationError.invalid(file: name, reason: "\(error)")
            }
        }
        struct Calibration: Decodable {
            let perKind: [String: Double]
            enum CodingKeys: String, CodingKey { case perKind = "per_kind" }
        }

        let text = try decode(Text.self, "config.json")
        let judge = try decode(Judge.self, "judge_config.json")
        guard !judge.adapterSubfolder.isEmpty, !judge.adapterSubfolder.contains(".."),
              !judge.adapterSubfolder.hasPrefix("/")
        else {
            throw JevConfigurationError.invalid(
                file: "judge_config.json", reason: "adapter_subfolder must be a folder inside the model folder")
        }
        let adapter = try decode(Adapter.self, "\(judge.adapterSubfolder)/adapter_config.json")
        let calibration = try decode(Calibration.self, "calibration.json")
        var temperatures: [JevKind: Float] = [:]
        for kind in JevKind.allCases {
            guard let temperature = calibration.perKind[kind.rawValue], temperature > 0 else {
                throw JevConfigurationError.invalid(
                    file: "calibration.json", reason: "per_kind needs a positive \(kind.rawValue) temperature")
            }
            temperatures[kind] = Float(temperature)
        }
        try self.init(text: text, judge: judge, adapter: adapter, temperatures: temperatures)
    }

    init(text: Text, judge: Judge, adapter: Adapter, temperatures: [JevKind: Float]) throws {
        func require(_ condition: Bool, _ file: String, _ reason: String) throws {
            guard condition else { throw JevConfigurationError.invalid(file: file, reason: reason) }
        }
        try require(text.modelType == "qwen3_5_text", "config.json", "model_type must be qwen3_5_text")
        try require(text.layerTypes.count == text.numHiddenLayers, "config.json",
                    "layer_types must name every layer")
        try require(Set(text.layerTypes).isSubset(of: ["linear_attention", "full_attention"]), "config.json",
                    "layer_types must be linear_attention or full_attention")
        try require(text.attnOutputGate, "config.json", "only gated attention (attn_output_gate) is supported")
        try require(text.numAttentionHeads % text.numKeyValueHeads == 0, "config.json",
                    "attention heads must be a multiple of key/value heads")
        try require(text.linearNumValueHeads % text.linearNumKeyHeads == 0, "config.json",
                    "linear value heads must be a multiple of linear key heads")
        try require(text.rotaryDimensions % 2 == 0 && text.rotaryDimensions <= text.headDim, "config.json",
                    "partial_rotary_factor must give an even rotary width within head_dim")
        try require(judge.hiddenSize == text.hiddenSize, "judge_config.json", "hidden_size must match config.json")
        try require(judge.slots.templateVersion == "bare-v1", "judge_config.json",
                    "only the bare-v1 template is supported")
        for kind in JevKind.allCases {
            guard let range = judge.slots.ranges[kind.rawValue], range.count == 2,
                  0 <= range[0], range[0] < range[1], range[1] <= judge.slots.numSlots
            else {
                throw JevConfigurationError.invalid(
                    file: "judge_config.json", reason: "slots.ranges.\(kind.rawValue) must lie within num_slots")
            }
            try require(range[1] - range[0] >= kind.minimumSlots, "judge_config.json",
                        "slots.ranges.\(kind.rawValue) needs at least \(kind.minimumSlots) slots")
        }
        let adapterFile = "\(judge.adapterSubfolder)/adapter_config.json"
        try require(adapter.peftType == "LORA" && adapter.r > 0, adapterFile, "only plain LoRA adapters are supported")
        try require(!adapter.useDora && !adapter.useRslora && !adapter.fanInFanOut && adapter.bias == "none",
                    adapterFile, "DoRA, rsLoRA, fan_in_fan_out and LoRA biases are not supported")
        try require(adapter.rankPattern.isEmpty && adapter.alphaPattern.isEmpty, adapterFile,
                    "per-module rank or alpha patterns are not supported")
        let unknown = Set(adapter.targetModules).subtracting(Adapter.supportedTargets)
        try require(unknown.isEmpty, adapterFile, "unsupported target modules \(unknown.sorted())")
        self.text = text
        self.judge = judge
        self.adapter = adapter
        self.temperatures = temperatures
    }

    /// The head slots one kind reads, in order.
    public func slots(for kind: JevKind) -> Range<Int> {
        let range = judge.slots.ranges[kind.rawValue]!
        return range[0]..<range[1]
    }
}

/// The three question kinds of the `bare-v1` template.
public enum JevKind: String, CaseIterable, Sendable {
    case noul, choice, score

    /// Slots a kind needs: `false`/`true`, the levels `0`–`5`, two options.
    var minimumSlots: Int {
        switch self {
        case .noul: return 2
        case .score: return 6
        case .choice: return 2
        }
    }
}
