import Foundation
import MLX
import SystemOne

/// A d1-omni checkpoint as a System One backend.
///
/// Every question becomes one row: `<bos> state <q> instructions` followed by
/// one `<opt> <mask> option </opt>` block per option. Images or audio enter as a
/// prefix encoded once per request. Answers are the softmax over each marker's
/// logit, with per-type temperatures from `config.json` for text-only rows.
public final class D1Backend: DecisionBackend {
    public let modelID: String

    let configuration: D1Configuration
    let tokenizer: D1Tokenizer
    private let model: D1Model?
    private let logitsProvider: (([D1Row]) async throws -> [[Float]])?

    /// Loads a local d1 model folder (`config.json`, tokenizer files, and
    /// `model.safetensors`). Pass `artifact: nil` to skip the byte-for-byte pin,
    /// for example when loading a fine-tuned checkpoint.
    public convenience init(
        modelFolder: URL,
        modelID: String = "d1-omni-600M",
        artifact: PinnedArtifact? = D1Weights.d1Omni600MArtifact
    ) async throws {
        let configuration = try D1Configuration(modelFolder: modelFolder)
        let tokenizer = try await D1Tokenizer(modelFolder: modelFolder)
        let url = modelFolder.appendingPathComponent("model.safetensors")
        let weights = try artifact.map {
            try D1Weights(contentsOf: url, pinnedTo: $0, configuration: configuration)
        } ?? D1Weights(contentsOf: url, configuration: configuration)
        self.init(
            modelID: modelID,
            configuration: configuration,
            tokenizer: tokenizer,
            model: D1Model(weights: weights),
            logitsProvider: nil
        )
    }

    init(
        modelID: String,
        configuration: D1Configuration,
        tokenizer: D1Tokenizer,
        model: D1Model?,
        logitsProvider: (([D1Row]) async throws -> [[Float]])?
    ) {
        self.modelID = modelID
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.model = model
        self.logitsProvider = logitsProvider
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<D1Row> {
        let mode: D1PromptMode
        let modeLength: Int
        var state = D1Prompt.serialize(request.state)
        switch media {
        case nil:
            (mode, modeLength) = (.text, configuration.maxLength)
        case .images(let images):
            guard !images.isEmpty else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
            (mode, modeLength) = (.image, configuration.imageTextLength)
        case .audio(let clip):
            guard clip.sampleRate == D1Audio.sampleRate, !clip.samples.isEmpty else {
                throw SystemOneEngineError.unsupportedMedia(model: modelID)
            }
            (mode, modeLength) = (.audio, configuration.audioTextLength)
            // Audio questions were trained with `{}` when the clip is the whole state.
            if case .text("") = request.state { state = "{}" }
        }

        let prefix = media.map { D1MediaPrefix(media: $0, length: D1Model.prefixLength(of: $0)) }
        let prefixLength = prefix?.length ?? 0
        let maxLength = min(modeLength, configuration.maxLength - prefixLength)
        guard maxLength >= 64 else {
            throw D1PromptError.mediaTooLong(prefixLength: prefixLength, maximum: configuration.maxLength)
        }

        var rows: [D1Row] = []
        var decisions: [PlannedDecision] = []
        for entry in request.questions {
            try validate(entry)
            let (instructions, type) = Self.describe(entry.question)
            let encoded = try D1Prompt.encode(
                state: state, question: entry.question, instructions: instructions, id: entry.id,
                maxLength: maxLength, mode: mode, tokenizer: tokenizer
            )
            let temperature = mode == .text ? temperature(for: entry.question) : 1
            let question = entry.question
            decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                Self.answer(question, probabilities: Self.softmax(logits[0], temperature: temperature))
            })
            rows.append(D1Row(ids: encoded.ids, markers: encoded.markers, questionType: type, prefix: prefix))
        }
        let read = rows.reduce(0) { $0 + $1.sequenceLength }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: read, outputTokens: 0))
    }

    /// The shared padded-attention bound over each row's own text: a request's
    /// rows share one media prefix whose trunk states run once, so padding
    /// costs only text positions.
    public func microbatches(for rows: [D1Row], strategy: BatchingStrategy) -> [[Int]] {
        SystemOneEngine.microbatchIndices(
            sequenceLengths: rows.map(\.computedLength),
            maxAmplification: strategy.maxAmplification
        )
    }

    public func logits(for rows: [D1Row]) async throws -> BatchLogitsOutput {
        let perRow: [[Float]]
        if let logitsProvider {
            perRow = try await logitsProvider(rows)
        } else {
            guard let model else { preconditionFailure("D1Backend has no logits provider") }
            let logits = try model.logits(for: rows)
            eval(logits)
            let width = logits.dim(1)
            let flat = logits.asArray(Float.self)
            perRow = rows.indices.map { Array(flat[($0 * width)..<($0 * width + rows[$0].logitCount)]) }
        }
        var offsets = [0]
        for row in perRow { offsets.append(offsets.last! + row.count) }
        return BatchLogitsOutput(logits: perRow.flatMap { $0 }, rowOffsets: offsets)
    }
}

extension D1Backend {
    static func describe(_ question: Question) -> (instructions: String, type: D1DecisionHead.QuestionType) {
        switch question {
        case .choice(let choice): return (choice.instructions, .choice)
        case .score(let score): return (score.instructions, .score)
        case .noul(let noul): return (noul.instructions, .noul)
        }
    }

    /// `Question.__post_init__`: a choice needs two options; a score 2 to 10 levels.
    func validate(_ entry: QuestionEntry) throws {
        switch entry.question {
        case .choice(let choice) where choice.criteria.count < 2:
            throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: "a choice needs at least two options")
        case .score(let score) where !(2...10).contains(score.criteria.count):
            throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: "a score needs 2 to 10 levels")
        default:
            break
        }
    }

    /// Per-bucket temperature, falling back to the type's, then to 1.
    func temperature(for question: Question) -> Double {
        let (type, count): (String, Int)
        switch question {
        case .choice(let choice): (type, count) = ("choice", choice.criteria.count)
        case .score(let score): (type, count) = ("score", score.criteria.count)
        case .noul: (type, count) = ("noul", 2)
        }
        return configuration.temperatures[D1Prompt.temperatureKey(type: type, optionCount: count)]
            ?? configuration.temperatures[type] ?? 1
    }

    /// float32 `softmax(z / T)`, as `_forward` computes it.
    static func softmax(_ logits: [Float], temperature: Double) -> [Float] {
        let scaled = logits.map { $0 / Float(temperature) }
        let maximum = scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    /// `prompt.answer`. A Noul's distribution arrives as `[false, true]`.
    static func answer(_ question: Question, probabilities: [Float]) -> Answer {
        let p = probabilities.map(Double.init)
        switch question {
        case .noul:
            return .noul(NoulAnswer(noul: p[1]))
        case .choice(let choice):
            let best = p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) }!
            return .choice(ChoiceAnswer(
                choice: choice.criteria[best].key,
                probabilities: zip(choice.criteria, p).map { ProbabilityEntry(key: $0.key, probability: $1) },
                confidence: p[best]
            ))
        case .score(let score):
            let best = p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) }!
            let keys = p.indices.map(String.init)
            return .score(ScoreAnswer(
                score: p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element },
                confidence: p[best],
                legend: zip(keys, score.criteria).map { LegendEntry(key: $0, description: D1Prompt.criterion($1)) },
                probabilities: zip(keys, p).map { ProbabilityEntry(key: $0, probability: $1) }
            ))
        }
    }
}

extension SystemOneEngine {
    /// Loads d1 from a local model folder, verifying the pinned
    /// `LiquidAI/d1-omni-600M` weights unless another artifact is given.
    public static func d1(
        modelFolder: URL,
        modelID: String = "d1-omni-600M",
        artifact: PinnedArtifact? = D1Weights.d1Omni600MArtifact
    ) async throws -> SystemOneEngine {
        SystemOneEngine(backend: try await D1Backend(modelFolder: modelFolder, modelID: modelID, artifact: artifact))
    }
}
