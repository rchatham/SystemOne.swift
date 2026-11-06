import Foundation
import MLX
import SystemOne

/// One `bare-v1` prompt, read at its last token.
public struct JevRow: PlannedRow {
    public let ids: [Int]
    public let kind: JevKind
    /// Active slots: 2 for a noul, 6 for a score, the option count for a choice.
    public let optionCount: Int

    public var sequenceLength: Int { ids.count }
    public var logitCount: Int { optionCount }
}

/// A JEV checkpoint (autotrust/JEV-9B) as a System One backend.
///
/// Every question becomes one `bare-v1` prompt. The backbone's last-position
/// hidden state goes through the 24-slot head; the kind's active slots,
/// divided by its `calibration.json` temperature, give the softmax answer.
/// A noul's options are `false`/`true` and a score's `0`–`5`, so their
/// criteria travel in the question text (`JevPrompt.foldedQuestion`).
public final class JevBackend: DecisionBackend {
    public let modelID: String
    /// Longest prompt in tokens; longer states are cut to their head and tail.
    public let maxLength: Int

    let configuration: JevConfiguration
    let coder: any JevTextCoding
    private let model: JevModel?
    private let logitsProvider: (([JevRow]) async throws -> [[Float]])?

    /// Loads a local JEV model folder: `config.json`, `judge_config.json`,
    /// `calibration.json`, the tokenizer, the sharded backbone, the adapter and
    /// `head.safetensors`. Pass `artifacts: nil` to skip the byte-for-byte pins.
    public convenience init(
        modelFolder: URL,
        modelID: String = "JEV-9B",
        artifacts: JevArtifacts? = .jev9B,
        precision: JevPrecision = .bfloat16,
        maxLength: Int = 4_096
    ) async throws {
        let configuration = try JevConfiguration(modelFolder: modelFolder)
        let tokenizer = try await JevTokenizer(modelFolder: modelFolder)
        let weights = try JevWeights(
            modelFolder: modelFolder, configuration: configuration, artifacts: artifacts, precision: precision)
        self.init(
            modelID: modelID, configuration: configuration, coder: tokenizer, maxLength: maxLength,
            model: JevModel(weights: weights), logitsProvider: nil
        )
    }

    init(
        modelID: String,
        configuration: JevConfiguration,
        coder: any JevTextCoding,
        maxLength: Int,
        model: JevModel?,
        logitsProvider: (([JevRow]) async throws -> [[Float]])?
    ) {
        self.modelID = modelID
        self.configuration = configuration
        self.coder = coder
        self.maxLength = maxLength
        self.model = model
        self.logitsProvider = logitsProvider
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<JevRow> {
        guard media == nil else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        let state = JevPrompt.serialize(request.state)
        var rows: [JevRow] = []
        var decisions: [PlannedDecision] = []
        for entry in request.questions {
            let question = entry.question
            if case .choice(let choice) = question, choice.criteria.count == 1 {
                let key = choice.criteria[0].key
                decisions.append(PlannedDecision(id: entry.id, rowIndices: []) { _ in
                    .choice(DecisionMath.choiceAnswer(keys: [key], probabilities: [1]))
                })
                continue
            }
            try validate(entry)
            let (kind, options) = JevPrompt.options(for: question)
            let ids = try encode(
                id: entry.id, kind: kind, state: state, question: JevPrompt.foldedQuestion(question), options: options)
            let temperature = configuration.temperatures[kind]!
            decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                Self.answer(question, probabilities: DecisionMath.softmax(logits[0], temperature: temperature))
            })
            rows.append(JevRow(ids: ids, kind: kind, optionCount: options.count))
        }
        let read = rows.reduce(0) { $0 + $1.sequenceLength }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: read, outputTokens: 0))
    }

    public func logits(for rows: [JevRow]) async throws -> BatchLogitsOutput {
        let perRow: [[Float]]
        if let logitsProvider {
            perRow = try await logitsProvider(rows)
        } else {
            guard let model else { preconditionFailure("JevBackend has no logits provider") }
            let logits = model.slotLogits(rows.map(\.ids))
            eval(logits)  // MLX eval: materialise the lazy logits.
            let width = logits.dim(1)
            let flat = logits.asArray(Float.self)
            perRow = rows.enumerated().map { index, row in
                let slots = configuration.slots(for: row.kind)
                let start = index * width + slots.lowerBound
                return Array(flat[start..<(start + row.optionCount)])
            }
        }
        var offsets = [0]
        for row in perRow { offsets.append(offsets.last! + row.count) }
        return BatchLogitsOutput(logits: perRow.flatMap { $0 }, rowOffsets: offsets)
    }
}

extension JevBackend {
    /// JEV's fixed answer sets: a choice reads 2 to 16 slots, a score exactly six.
    func validate(_ entry: QuestionEntry) throws {
        switch entry.question {
        case .choice(let choice):
            let available = min(JevPrompt.maximumChoiceOptions, configuration.slots(for: .choice).count)
            guard (2...available).contains(choice.criteria.count) else {
                throw SystemOneEngineError.invalidQuestion(
                    id: entry.id, reason: "a JEV choice takes 1 to \(available) options")
            }
        case .score(let score):
            guard score.criteria.count == JevPrompt.scoreLevels else {
                throw SystemOneEngineError.invalidQuestion(
                    id: entry.id, reason: "a JEV score has exactly \(JevPrompt.scoreLevels) levels, 0 to 5")
            }
        case .noul:
            break
        }
    }

    /// Token ids of one prompt. A prompt over `maxLength` keeps the head 60 %
    /// and tail 40 % of its state's tokens, as JEV's own server truncates.
    func encode(id: String, kind: JevKind, state: String, question: String, options: [String]) throws -> [Int] {
        let ids = coder.encode(JevPrompt.render(kind: kind, state: state, question: question, options: options))
        guard ids.count > maxLength else { return ids }
        let empty = coder.encode(JevPrompt.render(kind: kind, state: "", question: question, options: options))
        guard empty.count <= maxLength else {
            throw JevPromptError.promptTooLong(id: id, tokens: empty.count, maximum: maxLength)
        }
        let stateIDs = coder.encode(state)
        var budget = min(maxLength - empty.count, stateIDs.count - 1)
        while budget > 0 {
            let head = budget * 3 / 5
            let tail = budget - head
            let kept = coder.decode(Array(stateIDs[..<head])) + coder.decode(Array(stateIDs[(stateIDs.count - tail)...]))
            let ids = coder.encode(JevPrompt.render(kind: kind, state: kept, question: question, options: options))
            if ids.count <= maxLength { return ids }
            // Re-tokenising across the cut can merge differently: shrink by the overshoot.
            budget -= ids.count - maxLength
        }
        return empty
    }

    static func answer(_ question: Question, probabilities: [Float]) -> Answer {
        switch question {
        case .noul:
            return .noul(DecisionMath.noulAnswer(probabilities: probabilities))
        case .choice(let choice):
            return .choice(DecisionMath.choiceAnswer(keys: choice.criteria.map(\.key), probabilities: probabilities))
        case .score(let score):
            return .score(DecisionMath.scoreAnswer(
                legend: score.criteria.map(JevPrompt.criterion), probabilities: probabilities))
        }
    }
}

/// JEV's prompt shape for request validation, without loading weights.
public struct JevModelProfile: ModelProfile {
    public let modelID: String
    public let maximumForwardUTF8Bytes: Int

    /// `maxLength` is in tokens. The byte cap allows eight bytes per token,
    /// beyond which a state is surely truncated anyway.
    public init(modelID: String = "JEV-9B", maxLength: Int = 4_096) {
        self.modelID = modelID
        maximumForwardUTF8Bytes = maxLength * 8
    }

    /// `bare-v1` has no option markers: the answer is read at the final position.
    public var markerText: String? { nil }

    public func renderState(_ state: SystemOneState) -> String {
        JevPrompt.serialize(state)
    }

    public func sketch(_ question: Question, renderedState: String) -> PromptSketch {
        let (kind, options) = JevPrompt.options(for: question)
        let prompt = JevPrompt.render(
            kind: kind, state: renderedState, question: JevPrompt.foldedQuestion(question), options: options)
        return PromptSketch(instructions: question.instructions, options: options, forwards: [prompt])
    }
}

extension SystemOneEngine {
    /// Loads a JEV checkpoint from a local model folder, verifying the pinned
    /// `autotrust/JEV-9B` files unless other artifacts (or `nil`) are given.
    public static func jev(
        modelFolder: URL,
        modelID: String = "JEV-9B",
        artifacts: JevArtifacts? = .jev9B,
        precision: JevPrecision = .bfloat16,
        maxLength: Int = 4_096
    ) async throws -> SystemOneEngine {
        SystemOneEngine(backend: try await JevBackend(
            modelFolder: modelFolder, modelID: modelID, artifacts: artifacts, precision: precision,
            maxLength: maxLength
        ))
    }
}
