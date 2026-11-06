import Foundation
import MLX
import MLXNN
import SystemOne

/// How DecisionTune 1.0 (`decision-tune/decisiontune-1.0`, `engine.py`) packs a
/// question: `[CLS] instructions [SEP] ([MASK] option)… [SEP] state [SEP]`,
/// one sequence per question, nothing truncated.
public enum DecisionTunePrompt {
    /// Special token texts replaced by a space in caller text.
    public static let reservedTokens = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]

    public static func clean(_ text: String) -> String {
        reservedTokens.reduce(text) { $0.replacingOccurrences(of: $1, with: " ") }
    }

    /// A text state as is; fields as `key: value | key: value`; no state as empty text.
    public static func stateText(_ state: SystemOneState) -> String {
        switch state {
        case .media: return ""
        case .text(let text): return text
        case .fields(let fields): return fields.map { "\($0.key): \($0.value)" }.joined(separator: " | ")
        }
    }

    /// The description, or the key with underscores as spaces when the
    /// description is missing or empty. A whitespace-only description is kept.
    public static func optionText(key: String, description: String?) -> String {
        guard let description, !description.isEmpty else { return key.replacingOccurrences(of: "_", with: " ") }
        return description
    }

    /// Option texts in answer order. A noul reads `["yes", "no"]` (its criteria
    /// are not part of the reference). Score is not part of the reference either:
    /// its levels are read like choice options, in order.
    public static func options(for question: Question) -> [String] {
        switch question {
        case .choice(let choice): return choice.criteria.map { optionText(key: $0.key, description: $0.description) }
        case .noul: return ["yes", "no"]
        case .score(let score): return score.criteria.map(EncoderText.pythonString)
        }
    }

    public static func encode(
        state: SystemOneState, instructions: String, options: [String],
        tokenizer: EncoderTokenizer, tokens: EncoderSpecialTokens
    ) -> EncoderRow {
        func ids(_ text: String) -> [Int] { tokenizer.encode(clean(text)) }
        var sequence = [tokens.cls] + ids(instructions) + [tokens.sep]
        var markers: [Int] = []
        for option in options {
            markers.append(sequence.count)
            sequence += [tokens.mask] + ids(option)
        }
        sequence += [tokens.sep] + ids(stateText(state)) + [tokens.sep]
        return EncoderRow(ids: sequence, markers: markers)
    }
}

/// DecisionTune's network: a ModernBERT-large encoder and one `Linear(d, 1)`
/// over the state at every `[MASK]`.
final class DecisionTuneModel {
    let encoder: ModernBERTEncoder
    let head: Linear
    let pad: Int

    init(encoder: ModernBERTEncoder, head: any TensorStore, pad: Int) throws {
        let d = encoder.configuration.hiddenSize
        try head.validate(schema: [
            WeightSpecification(name: "head.weight", shape: [1, d]),
            WeightSpecification(name: "head.bias", shape: [1]),
        ])
        self.encoder = encoder
        self.head = try head.linear("head", inputWidth: d, bias: true, dataType: encoder.dataType)
        self.pad = pad
    }

    func logits(_ rows: [EncoderRow]) throws -> [[Float]] {
        let hidden = try encoder(paddedIDs(rows.map(\.ids), pad: pad), attention: EncoderAttention(lengths: rows.map(\.ids.count)))
        return splitLogits(head(gatherMarkers(hidden, rows: rows)).squeezed(axis: -1), rows: rows)
    }
}

/// DecisionTune 1.0 as a System One backend. Probabilities are the plain
/// softmax over the marker scores (no calibration); a choice's confidence is
/// its top probability, and a noul is `P(yes)`.
public final class DecisionTuneBackend: DecisionBackend {
    public let modelID: String
    let tokenizer: EncoderTokenizer
    let tokens: EncoderSpecialTokens
    let contextLimit: Int
    private let model: DecisionTuneModel?

    /// Loads a local DecisionTune folder: `config.json`, `tokenizer.json`,
    /// `model.safetensors` (the encoder), and `head.safetensors`.
    public convenience init(modelFolder: URL, modelID: String = "decisiontune-1.0", dataType: DType = .float32) throws {
        let configuration = try ModernBERTConfiguration(modelFolder: modelFolder)
        let tokenizer = try HubEncoderTokenizer(folder: modelFolder)
        let tokens = try EncoderSpecialTokens(tokenizer: tokenizer, names: tokenizer.specialTokens)
        let store = try SafetensorsFile(contentsOf: modelFolder.appendingPathComponent("model.safetensors"))
        let model = try DecisionTuneModel(
            encoder: ModernBERTEncoder(store: store, configuration: configuration, dataType: dataType),
            head: SafetensorsFile(contentsOf: modelFolder.appendingPathComponent("head.safetensors")),
            pad: tokens.pad
        )
        self.init(
            modelID: modelID, tokenizer: tokenizer, tokens: tokens,
            contextLimit: configuration.maxPositionEmbeddings, model: model
        )
    }

    init(
        modelID: String, tokenizer: EncoderTokenizer, tokens: EncoderSpecialTokens, contextLimit: Int,
        model: DecisionTuneModel?
    ) {
        self.modelID = modelID
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.contextLimit = contextLimit
        self.model = model
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<EncoderRow> {
        guard media == nil else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        var rows: [EncoderRow] = []
        var decisions: [PlannedDecision] = []
        for entry in request.questions {
            let question = entry.question
            let options = DecisionTunePrompt.options(for: question)
            if case .choice = question, options.count < 2 {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: "choice needs at least 2 options")
            }
            let row = DecisionTunePrompt.encode(
                state: request.state, instructions: question.instructions, options: options,
                tokenizer: tokenizer, tokens: tokens
            )
            guard row.ids.count <= contextLimit else {
                throw SystemOneEngineError.invalidQuestion(
                    id: entry.id, reason: "sequence of \(row.ids.count) tokens exceeds the \(contextLimit)-token context"
                )
            }
            decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                let p = EncoderAnswers.softmax(logits[0])
                switch question {
                case .noul:
                    return .noul(NoulAnswer(noul: p[0]))
                case .choice(let choice):
                    return EncoderAnswers.choice(keys: choice.criteria.map(\.key), probabilities: p, confidence: p.max() ?? 0)
                case .score:
                    return EncoderAnswers.score(levels: options, probabilities: p, confidence: p.max() ?? 0)
                }
            })
            rows.append(row)
        }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: rows.reduce(0) { $0 + $1.ids.count }))
    }

    public func logits(for rows: [EncoderRow]) async throws -> BatchLogitsOutput {
        guard let model else { preconditionFailure("DecisionTuneBackend has no model") }
        return BatchLogitsOutput(rows: try model.logits(rows))
    }
}

extension BatchLogitsOutput {
    /// Row-major logits and offsets from per-row arrays.
    init(rows: [[Float]]) {
        var offsets = [0]
        for row in rows { offsets.append(offsets.last! + row.count) }
        self.init(logits: rows.flatMap { $0 }, rowOffsets: offsets)
    }
}

extension SystemOneEngine {
    /// DecisionTune 1.0 from a local folder.
    public static func decisionTune(modelFolder: URL, dataType: DType = .float32) throws -> SystemOneEngine {
        SystemOneEngine(backend: try DecisionTuneBackend(modelFolder: modelFolder, dataType: dataType))
    }
}
