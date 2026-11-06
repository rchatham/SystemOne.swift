import Foundation
import MLX
import MLXNN
import SystemOne

/// How a marker-head checkpoint renders questions: Laya's `rl_common.py` or
/// Julia-1's `julia/data.py` and `julia/typed.py`. Both pack
/// `[CLS] "<type> question: <instructions>" [SEP] ([MASK] " option")… [SEP] state [SEP]`.
public enum MarkerPromptStyle: String, Sendable {
    /// Options `key: description` (`level i: …` for scores; `false: …`/`true: …` with defaults).
    case laya
    /// Options are the descriptions themselves; noul is `false`/`true` or both criteria.
    case julia
}

/// A marker-head model folder's settings.
public struct MarkerModelConfiguration: Equatable, Sendable {
    public let style: MarkerPromptStyle
    public let maxLength: Int
    public let headLength: Int
    public let headLayers: Int
    /// Per question type: choice, score, noul.
    public let temperatures: [Double]
    /// By `type:size` bucket (`choice:2`, `choice:3-5`, ...), overriding `temperatures`.
    public let temperaturesByOptions: [String: Double]

    public init(
        style: MarkerPromptStyle, maxLength: Int, headLength: Int, headLayers: Int,
        temperatures: [Double] = [1, 1, 1], temperaturesByOptions: [String: Double] = [:]
    ) {
        self.style = style
        self.maxLength = maxLength
        self.headLength = headLength
        self.headLayers = headLayers
        self.temperatures = temperatures
        self.temperaturesByOptions = temperaturesByOptions
    }

    /// Laya folders have `rl_agent_config.json`; Julia folders `julia_config.json`
    /// and `inference-policy.json`.
    public init(modelFolder: URL) throws {
        func object(_ name: String) throws -> [String: Any]? {
            let url = modelFolder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw EncoderTokenizerError.invalidJSON(url.path)
            }
            return object
        }
        if let laya = try object("rl_agent_config.json") {
            self.init(
                style: .laya,
                maxLength: laya["max_len"] as? Int ?? 512,
                headLength: laya["head_max_len"] as? Int ?? 192,
                headLayers: laya["head_layers"] as? Int ?? 2,
                temperatures: (laya["temperature"] as? [Double]) ?? [1, 1, 1],
                temperaturesByOptions: (laya["temperature_by_options"] as? [String: Double]) ?? [:]
            )
        } else if let julia = try object("julia_config.json") {
            let policy = try object("inference-policy.json") ?? [:]
            self.init(
                style: .julia,
                maxLength: policy["max_length"] as? Int ?? 8_192,
                headLength: policy["head_length"] as? Int ?? 256,
                headLayers: julia["head_layers"] as? Int ?? 2
            )
        } else {
            throw EncoderTokenizerError.missingFile(modelFolder.appendingPathComponent("rl_agent_config.json").path)
        }
    }

    /// `temp_bucket`: the question type and its option count's size class.
    public static func bucket(type: Int, options: Int) -> String {
        let size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+"
        return "\(["choice", "score", "noul"][type]):\(size)"
    }

    public func temperature(type: Int, options: Int) -> Double {
        temperaturesByOptions[Self.bucket(type: type, options: options)] ?? temperatures[type]
    }
}

/// Prompt rendering for marker-head models.
public enum MarkerPrompt {
    static let defaultFalse = "no, the statement does not hold"
    static let defaultTrue = "yes, the statement holds"

    /// Option texts in answer order (noul: `[false, true]`).
    public static func options(for question: Question, style: MarkerPromptStyle) throws -> [String] {
        switch (style, question) {
        case (.laya, .choice(let choice)):
            return choice.criteria.map { option in
                guard let description = option.description, !description.isEmpty else { return option.key }
                return "\(option.key): \(description)"
            }
        case (.laya, .score(let score)):
            return score.criteria.enumerated().map { "level \($0.offset): \(EncoderText.pythonString($0.element))" }
        case (.laya, .noul(let noul)):
            let falseText = noul.criteria?.falseDescription.flatMap { $0.isEmpty ? nil : $0 } ?? defaultFalse
            let trueText = noul.criteria?.trueDescription.flatMap { $0.isEmpty ? nil : $0 } ?? defaultTrue
            return ["false: " + falseText, "true: " + trueText]
        case (.julia, .choice(let choice)):
            return try choice.criteria.map { option in
                guard let description = option.description, !description.isEmpty else {
                    throw MarkerPromptError.needsDescriptions
                }
                return description
            }
        case (.julia, .score(let score)):
            return try score.criteria.map { level in
                guard case .text(let text) = level, !text.isEmpty else { throw MarkerPromptError.needsDescriptions }
                return text
            }
        case (.julia, .noul(let noul)):
            guard let criteria = noul.criteria else { return ["false", "true"] }
            guard let falseText = criteria.falseDescription, let trueText = criteria.trueDescription,
                  !falseText.isEmpty, !trueText.isEmpty
            else { throw MarkerPromptError.needsBothNoulCriteria }
            return [falseText, trueText]
        }
    }

    /// Laya's `build_sequence` or Julia's `sequence`: options capped at 48
    /// tokens and shrunk evenly to fit the head budget, the question head cut to
    /// what is left, the state cut to the remaining context.
    public static func encode(
        state: SystemOneState, question: Question, options: [String],
        configuration: MarkerModelConfiguration, tokenizer: EncoderTokenizer, tokens: EncoderSpecialTokens
    ) throws -> EncoderRow {
        let mask = tokens.maskText
        func clean(_ text: String) -> String { text.replacingOccurrences(of: mask, with: " ") }
        let type = EncoderText.typeName(question)
        let head = tokenizer.encode("\(type) question: \(clean(question.instructions))")
        var optionIDs = options.map { [tokens.mask] + Array(tokenizer.encode(" " + clean($0)).prefix(48)) }
        var budget = configuration.headLength - optionIDs.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let perOption = max(4, (configuration.headLength - 16) / max(1, optionIDs.count))
            optionIDs = optionIDs.map { Array($0.prefix(perOption)) }
            budget = configuration.headLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        var ids = [tokens.cls] + Array(head.prefix(max(8, budget))) + [tokens.sep]
        var markers: [Int] = []
        for option in optionIDs {
            markers.append(ids.count)
            ids += option
        }
        ids.append(tokens.sep)
        let stateIDs = tokenizer.encode(clean(EncoderText.pythonJSON(state)))
        switch configuration.style {
        case .laya:
            let room = max(0, configuration.maxLength - ids.count - 1)
            ids = Array((ids + stateIDs.prefix(room) + [tokens.sep]).prefix(configuration.maxLength))
            markers = markers.filter { $0 < configuration.maxLength }
            guard markers.count == options.count else { throw MarkerPromptError.optionsDoNotFit(headLength: configuration.headLength) }
        case .julia:
            let room = configuration.maxLength - ids.count - 1
            guard room >= 1 else { throw MarkerPromptError.optionsDoNotFit(headLength: configuration.headLength) }
            ids += stateIDs.prefix(room) + [tokens.sep]
        }
        return EncoderRow(ids: ids, markers: markers, questionType: EncoderText.typeID(question))
    }
}

public enum MarkerPromptError: Error, Equatable, Sendable {
    case needsDescriptions
    case needsBothNoulCriteria
    case optionsDoNotFit(headLength: Int)
}

extension MarkerPromptError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .needsDescriptions: return "This model reads each option's description; every option needs one."
        case .needsBothNoulCriteria: return "Noul criteria must describe both false and true."
        case .optionsDoNotFit(let headLength): return "The options do not fit the \(headLength)-token question head."
        }
    }
}

/// The marker-head network Laya and Julia-1 share: an encoder, a question-type
/// embedding added to every state, PyTorch transformer layers, and a scorer
/// (LayerNorm → Linear → GELU → Linear) at each `[MASK]`.
final class MarkerHeadModel {
    let encoder: ModernBERTEncoder
    private let typeEmbedding: MLXArray
    private let head: TorchTransformerEncoder?
    private let scorerNorm: (weight: MLXArray, bias: MLXArray)
    private let scorerDense: Linear
    private let scorerOutput: Linear
    let pad: Int

    /// Every tensor the network reads; the checkpoint may hold more (the act
    /// head and the temperature buffer are not used for answers).
    static func schema(encoder: ModernBERTConfiguration, headLayers: Int) -> [WeightSpecification] {
        let d = encoder.hiddenSize
        return ModernBERTEncoder.schema(for: encoder, prefix: "encoder.")
            + (0..<headLayers).flatMap { TorchTransformerEncoder.schema(prefix: "head.layers.\($0).", width: d) }
            + [
                WeightSpecification(name: "type_emb.weight", shape: [3, d]),
                WeightSpecification(name: "scorer.0.weight", shape: [d]),
                WeightSpecification(name: "scorer.0.bias", shape: [d]),
                WeightSpecification(name: "scorer.1.weight", shape: [d, d]),
                WeightSpecification(name: "scorer.1.bias", shape: [d]),
                WeightSpecification(name: "scorer.3.weight", shape: [1, d]),
                WeightSpecification(name: "scorer.3.bias", shape: [1]),
            ]
    }

    init(
        store: any TensorStore, encoder configuration: ModernBERTConfiguration, headLayers: Int, pad: Int,
        dataType: DType = .float32
    ) throws {
        // The encoder and the scorer's linears read MLX-quantized weights; the head layers and type embedding load dense.
        for specification in Self.schema(encoder: configuration, headLayers: headLayers)
        where !(specification.name.hasPrefix("encoder.") || specification.name.hasPrefix("scorer."))
            || !store.isQuantized(String(specification.name.dropLast(".weight".count)))
        {
            let descriptor = try store.requiredTensor(specification.name).descriptor
            guard descriptor.shape == specification.shape else {
                throw WeightsError.shapeMismatch(name: specification.name, expected: specification.shape, actual: descriptor.shape)
            }
        }
        let d = configuration.hiddenSize
        encoder = try ModernBERTEncoder(store: store, configuration: configuration, prefix: "encoder.", dataType: dataType)
        typeEmbedding = try store.requiredTensor("type_emb.weight").mlxArray(dataType)
        head = headLayers > 0
            ? try TorchTransformerEncoder(
                store: store, prefix: "head.layers.", count: headLayers, width: d, heads: max(1, d / 64), dataType: dataType
            )
            : nil
        scorerNorm = (
            try store.requiredTensor("scorer.0.weight").mlxArray(dataType),
            try store.requiredTensor("scorer.0.bias").mlxArray(dataType)
        )
        scorerDense = try store.linear("scorer.1", inputWidth: d, bias: true, dataType: dataType)
        scorerOutput = try store.linear("scorer.3", inputWidth: d, bias: true, dataType: dataType)
        self.pad = pad
    }

    func logits(_ rows: [EncoderRow]) throws -> [[Float]] {
        let lengths = rows.map(\.ids.count)
        var hidden = try encoder(paddedIDs(rows.map(\.ids), pad: pad), attention: EncoderAttention(lengths: lengths))
        hidden = hidden + typeEmbedding[MLXArray(rows.map { Int32($0.questionType) })].expandedDimensions(axis: 1)
        if let head { hidden = head(hidden, lengths: lengths) }
        let markers = MLXFast.layerNorm(
            gatherMarkers(hidden, rows: rows), weight: scorerNorm.weight, bias: scorerNorm.bias, eps: 1e-5
        )
        return splitLogits(scorerOutput(gelu(scorerDense(markers))).squeezed(axis: -1), rows: rows)
    }
}

/// Laya (`convaiinnovations/laya`, English and multilingual) and Julia-1
/// (`SupersonicLabs/Julia-1`) as a System One backend.
///
/// Laya divides each question's logits by its fitted temperature and reports
/// `1 - normalized entropy` as confidence; Julia applies no temperature and
/// reports the top probability. A noul is `P(true)`.
public final class MarkerHeadBackend: DecisionBackend {
    public let modelID: String
    let configuration: MarkerModelConfiguration
    let tokenizer: EncoderTokenizer
    let tokens: EncoderSpecialTokens
    private let model: MarkerHeadModel?

    /// Loads a Laya or Julia folder: its config files, `encoder/config.json`,
    /// `tokenizer/`, and `model.safetensors`. For Laya multilingual, pass the
    /// repository's `multilingual/` folder.
    public convenience init(modelFolder: URL, modelID: String? = nil, dataType: DType = .float32) throws {
        let configuration = try MarkerModelConfiguration(modelFolder: modelFolder)
        let encoder = try ModernBERTConfiguration(modelFolder: modelFolder.appendingPathComponent("encoder"))
        let tokenizer = try HubEncoderTokenizer(folder: modelFolder.appendingPathComponent("tokenizer"))
        let tokens = try EncoderSpecialTokens(tokenizer: tokenizer, names: tokenizer.specialTokens)
        let store = try SafetensorsFile(contentsOf: modelFolder.appendingPathComponent("model.safetensors"))
        self.init(
            modelID: modelID ?? (configuration.style == .laya ? "laya" : "julia-1"),
            configuration: configuration, tokenizer: tokenizer, tokens: tokens,
            model: try MarkerHeadModel(
                store: store, encoder: encoder, headLayers: configuration.headLayers, pad: tokens.pad, dataType: dataType
            )
        )
    }

    init(
        modelID: String, configuration: MarkerModelConfiguration, tokenizer: EncoderTokenizer,
        tokens: EncoderSpecialTokens, model: MarkerHeadModel?
    ) {
        self.modelID = modelID
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.model = model
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<EncoderRow> {
        guard media == nil else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        var rows: [EncoderRow] = []
        var decisions: [PlannedDecision] = []
        for entry in request.questions {
            let question = entry.question
            let options: [String]
            let row: EncoderRow
            do {
                options = try MarkerPrompt.options(for: question, style: configuration.style)
                if configuration.style == .julia, !(2...20).contains(options.count) {
                    throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: "Julia reads 2 to 20 options")
                }
                row = try MarkerPrompt.encode(
                    state: request.state, question: question, options: options,
                    configuration: configuration, tokenizer: tokenizer, tokens: tokens
                )
            } catch let error as MarkerPromptError {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: error.localizedDescription)
            }
            let temperature = configuration.temperature(type: row.questionType, options: options.count)
            let style = configuration.style
            decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                let p = EncoderAnswers.softmax(logits[0], temperature: temperature)
                let confidence = style == .laya ? EncoderAnswers.entropyConfidence(p) : (p.max() ?? 0)
                switch question {
                case .noul:
                    return .noul(NoulAnswer(noul: p[1]))
                case .choice(let choice):
                    return EncoderAnswers.choice(keys: choice.criteria.map(\.key), probabilities: p, confidence: confidence)
                case .score(let score):
                    return EncoderAnswers.score(levels: score.criteria.map(EncoderText.pythonString), probabilities: p, confidence: confidence)
                }
            })
            rows.append(row)
        }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: rows.reduce(0) { $0 + $1.ids.count }))
    }

    public func logits(for rows: [EncoderRow]) async throws -> BatchLogitsOutput {
        guard let model else { preconditionFailure("MarkerHeadBackend has no model") }
        return BatchLogitsOutput(rows: try model.logits(rows))
    }
}

extension SystemOneEngine {
    /// Laya or Julia-1 from a local folder.
    public static func markerHead(modelFolder: URL, dataType: DType = .float32) throws -> SystemOneEngine {
        SystemOneEngine(backend: try MarkerHeadBackend(modelFolder: modelFolder, dataType: dataType))
    }
}
