import Foundation

/// Controls how `systemOne` fans out one forward pass per planned row or
/// groups compatible rows into padded microbatches.
public enum BatchingStrategy: Sendable, Equatable {
    /// Evaluate each planned row in its own forward pass (no batching).
    case sequential

    /// Use a conservative microbatch policy that keeps amplification low
    /// (max 1.5×). The absolute 1,048,576 padded-work bound is preserved.
    case automatic

    /// Use the current bounded greedy microbatch policy with the absolute
    /// 1,048,576 padded-work bound and an amplification ceiling of 2.0×.
    case batched

    /// Maximum amplification factor for the strategy.
    public var maxAmplification: Double? {
        switch self {
        case .sequential: return nil
        case .automatic: return 1.5
        case .batched: return 2.0
        }
    }
}

/// Validation and orchestration errors raised by the public Von API.
public enum SystemOneEngineError: Error, Equatable, Sendable {
    case emptyChoiceOptions
    case emptyScoreLevels
    case duplicateStateFieldKey(String)
    case duplicateChoiceKey(String)
    case duplicateQuestionID(String)
    case unexpectedMaskCount(expected: Int, actual: Int)
    case unexpectedLogitCount(expected: Int, actual: Int)
    /// A batch provider returned row boundaries that do not match the planned rows.
    case unexpectedBatchOffsets(expected: [Int], actual: [Int])
    /// The model cannot read this kind of media.
    case unsupportedMedia(model: String)
    /// A question violates a model-specific constraint.
    case invalidQuestion(id: String, reason: String)
}

extension SystemOneEngineError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyChoiceOptions:
            return "Choice requires at least one option."
        case .emptyScoreLevels:
            return "Score requires at least one level."
        case .duplicateStateFieldKey(let key):
            return "State field key '\(key)' appears more than once."
        case .duplicateChoiceKey(let key):
            return "Choice option key '\(key)' appears more than once."
        case .duplicateQuestionID(let id):
            return "Question id '\(id)' appears more than once."
        case .unexpectedMaskCount(let expected, let actual):
            return "Packed input contains \(actual) mask tokens; expected \(expected), one per option."
        case .unexpectedLogitCount(let expected, let actual):
            return "The logits provider returned \(actual) logits; expected \(expected)."
        case .unexpectedBatchOffsets(let expected, let actual):
            return "The batch logits provider returned offsets \(actual); expected \(expected)."
        case .unsupportedMedia(let model):
            return "Model \(model) does not accept this media."
        case .invalidQuestion(let id, let reason):
            return "Question '\(id)' is invalid: \(reason)"
        }
    }
}

/// Local, actor-isolated System One inference engine.
///
/// The engine owns what every model shares: input validation, planning every
/// row before inference, bounded microbatch fan-out, cancellation, and answer
/// assembly. A `DecisionBackend` supplies prompt packing, the network, and
/// calibration. It performs no model lookup, network access, or HTTP serving.
public actor SystemOneEngine {
    public static let defaultChoiceInstructions = "Which option best describes the state?"
    public static let defaultScoreInstructions = "Rate where the state falls on this scale:"

    /// The identifier the backend reports in every response.
    public nonisolated let modelID: String

    private let backend: any DecisionBackend

    /// Wraps a loaded model backend.
    public init(backend: some DecisionBackend) {
        self.backend = backend
        modelID = backend.modelID
    }

    /// Evaluates ordered questions with bounded, stable microbatch fan-out.
    public func systemOne(
        _ request: SystemOneRequest,
        media: SystemOneMedia? = nil,
        batchingStrategy: BatchingStrategy = .automatic
    ) async throws -> SystemOneResponse {
        try validate(request.state)
        try validate(request.questions)
        return try await run(backend, request: request, media: media, strategy: batchingStrategy)
    }

    /// Convenience form of `systemOne` retaining explicit question order.
    public func systemOne(
        state: SystemOneState,
        questions: [QuestionEntry],
        media: SystemOneMedia? = nil,
        model: String? = nil,
        batchingStrategy: BatchingStrategy = .automatic
    ) async throws -> SystemOneResponse {
        try await systemOne(
            SystemOneRequest(model: model ?? modelID, state: state, questions: questions),
            media: media,
            batchingStrategy: batchingStrategy
        )
    }

    /// Chooses one option from an ordered list.
    public func decide(
        state: SystemOneState,
        choices: [ChoiceOption],
        instructions: String = SystemOneEngine.defaultChoiceInstructions,
        media: SystemOneMedia? = nil,
        model: String? = nil
    ) async throws -> ChoiceAnswer {
        let answer = try await single(
            .choice(Choice(instructions: instructions, criteria: choices)),
            id: "decision", state: state, media: media, model: model
        )
        guard case .choice(let choice) = answer else {
            preconditionFailure("SystemOneEngine produced a mismatched decision answer")
        }
        return choice
    }

    /// Evaluates a yes/no condition and returns the probability that it is true.
    public func judge(
        state: SystemOneState,
        instructions: String,
        criteria: NoulCriteria? = nil,
        media: SystemOneMedia? = nil,
        model: String? = nil
    ) async throws -> Double {
        let answer = try await single(
            .noul(Noul(instructions: instructions, criteria: criteria)),
            id: "judgment", state: state, media: media, model: model
        )
        guard case .noul(let noul) = answer else {
            preconditionFailure("SystemOneEngine produced a mismatched judgment answer")
        }
        return noul.noul
    }

    /// Rates state over ordered ordinal levels.
    public func rate(
        state: SystemOneState,
        criteria: [ScoreLevel],
        instructions: String = SystemOneEngine.defaultScoreInstructions,
        media: SystemOneMedia? = nil,
        model: String? = nil
    ) async throws -> ScoreAnswer {
        let answer = try await single(
            .score(Score(instructions: instructions, criteria: criteria)),
            id: "rating", state: state, media: media, model: model
        )
        guard case .score(let score) = answer else {
            preconditionFailure("SystemOneEngine produced a mismatched rating answer")
        }
        return score
    }
}

extension SystemOneEngine {
    /// Stable length-sorted row indices grouped by bounded padded attention work.
    ///
    /// A row of length `L` costs `L × (sharedPrefixLength + L)`: its queries
    /// attend to its own keys plus a prefix every row shares and that is
    /// already computed (`L²` when there is none).
    /// When `maxAmplification` is `nil` each row is its own singleton batch.
    /// Otherwise the absolute boundary `max(candidateMaximum, 1_048_576)` and the
    /// relative `paddedWork <= maxAmplification × usefulWork` gate are applied.
    public static func microbatchIndices(
        sequenceLengths: [Int],
        sharedPrefixLength: Int = 0,
        maxAmplification: Double?
    ) -> [[Int]] {
        let sortedIndices = sequenceLengths.indices.sorted { lhs, rhs in
            let left = sequenceLengths[lhs]
            let right = sequenceLengths[rhs]
            return left == right ? lhs < rhs : left > right
        }
        guard !sortedIndices.isEmpty else { return [] }

        func work(_ length: Int) -> Int? {
            guard length >= 0, sharedPrefixLength >= 0 else { return nil }
            let (keys, keysOverflow) = sharedPrefixLength.addingReportingOverflow(length)
            let (result, overflow) = length.multipliedReportingOverflow(by: keys)
            return keysOverflow || overflow ? nil : result
        }

        var batches: [[Int]] = []
        var current: [Int] = []
        var maximumWork = 0
        var sumWork = 0
        var sumWorkIsUnbounded = false

        for index in sortedIndices {
            guard let rowWork = work(sequenceLengths[index]) else {
                if !current.isEmpty {
                    batches.append(current)
                    current = []
                    maximumWork = 0
                    sumWork = 0
                    sumWorkIsUnbounded = false
                }
                batches.append([index])
                continue
            }

            guard !current.isEmpty else {
                current = [index]
                maximumWork = rowWork
                sumWork = rowWork
                sumWorkIsUnbounded = false
                continue
            }

            guard let amplification = maxAmplification else {
                // nil means sequential: each row is a singleton batch
                batches.append(current)
                current = [index]
                maximumWork = rowWork
                sumWork = rowWork
                sumWorkIsUnbounded = false
                continue
            }

            let candidateMaximum = max(maximumWork, rowWork)
            let (candidateCount, countOverflow) = current.count.addingReportingOverflow(1)
            let (paddedWork, paddedOverflow) = candidateCount.multipliedReportingOverflow(
                by: candidateMaximum
            )
            let (candidateSum, sumOverflow) = sumWork.addingReportingOverflow(rowWork)
            let candidateSumIsUnbounded = sumWorkIsUnbounded || sumOverflow
            let (amplifiedWork, usefulOverflow) = candidateSum.multipliedReportingOverflow(
                by: Int(amplification * 256)
            )
            let usefulWorkIsUnbounded = candidateSumIsUnbounded || usefulOverflow
            let absoluteLimit = max(candidateMaximum, 1_048_576)
            let fits = !countOverflow
                && !paddedOverflow
                && paddedWork <= absoluteLimit
                && (usefulWorkIsUnbounded || paddedWork * 256 <= amplifiedWork)

            if fits {
                current.append(index)
                maximumWork = candidateMaximum
                sumWork = candidateSumIsUnbounded ? 0 : candidateSum
                sumWorkIsUnbounded = candidateSumIsUnbounded
            } else {
                batches.append(current)
                current = [index]
                maximumWork = rowWork
                sumWork = rowWork
                sumWorkIsUnbounded = false
            }
        }
        if !current.isEmpty {
            batches.append(current)
        }
        return batches
    }
}

private extension SystemOneEngine {
    func single(
        _ question: Question,
        id: String,
        state: SystemOneState,
        media: SystemOneMedia?,
        model: String?
    ) async throws -> Answer {
        let response = try await systemOne(
            state: state,
            questions: [QuestionEntry(id: id, question: question)],
            media: media,
            model: model
        )
        guard let answer = response[id] else {
            preconditionFailure("SystemOneEngine dropped answer \(id)")
        }
        return answer
    }

    /// Opens the existential so planning and execution share one concrete `Row`.
    ///
    /// Decisions resolve in plan order as soon as all of their rows have run,
    /// so a postprocessing failure stops inference before later microbatches.
    func run<Backend: DecisionBackend>(
        _ backend: Backend,
        request: SystemOneRequest,
        media: SystemOneMedia?,
        strategy: BatchingStrategy
    ) async throws -> SystemOneResponse {
        let plan = try backend.plan(request, media: media)
        var rowLogits = [[Float]?](repeating: nil, count: plan.rows.count)
        var answers: [AnswerEntry] = []
        answers.reserveCapacity(plan.decisions.count)

        func resolveReadyDecisions() throws {
            while answers.count < plan.decisions.count {
                let decision = plan.decisions[answers.count]
                let logits = decision.rowIndices.compactMap { rowLogits[$0] }
                guard logits.count == decision.rowIndices.count else { return }
                answers.append(AnswerEntry(id: decision.id, answer: try decision.resolve(logits)))
            }
        }

        try resolveReadyDecisions()
        for indices in backend.microbatches(for: plan.rows, strategy: strategy) {
            let chunk = indices.map { plan.rows[$0] }
            try Task.checkCancellation()
            let output = try await backend.logits(for: chunk)
            for (index, logits) in zip(indices, try separate(output, logitCounts: chunk.map(\.logitCount))) {
                rowLogits[index] = logits
            }
            try resolveReadyDecisions()
        }
        guard answers.count == plan.decisions.count else {
            preconditionFailure("SystemOneEngine failed to populate a planned inference row")
        }
        return SystemOneResponse(model: backend.modelID, answers: answers, usage: plan.usage)
    }

    func separate(_ output: BatchLogitsOutput, logitCounts: [Int]) throws -> [[Float]] {
        var expectedOffsets = [0]
        expectedOffsets.reserveCapacity(logitCounts.count + 1)
        for count in logitCounts {
            let (offset, overflow) = expectedOffsets.last!.addingReportingOverflow(count)
            guard !overflow else {
                throw SystemOneEngineError.unexpectedLogitCount(expected: Int.max, actual: output.logits.count)
            }
            expectedOffsets.append(offset)
        }
        guard output.rowOffsets == expectedOffsets else {
            throw SystemOneEngineError.unexpectedBatchOffsets(
                expected: expectedOffsets,
                actual: output.rowOffsets
            )
        }
        let expectedCount = expectedOffsets.last!
        guard output.logits.count == expectedCount else {
            throw SystemOneEngineError.unexpectedLogitCount(
                expected: expectedCount,
                actual: output.logits.count
            )
        }
        return logitCounts.indices.map { index in
            Array(output.logits[expectedOffsets[index]..<expectedOffsets[index + 1]])
        }
    }

    func validate(_ state: SystemOneState) throws {
        guard case .fields(let fields) = state else { return }
        var keys = Set<String>()
        for field in fields {
            guard keys.insert(field.key).inserted else {
                throw SystemOneEngineError.duplicateStateFieldKey(field.key)
            }
        }
    }

    func validate(_ questions: [QuestionEntry]) throws {
        var questionIDs = Set<String>()
        for entry in questions {
            guard questionIDs.insert(entry.id).inserted else {
                throw SystemOneEngineError.duplicateQuestionID(entry.id)
            }
            switch entry.question {
            case .choice(let question):
                guard !question.criteria.isEmpty else { throw SystemOneEngineError.emptyChoiceOptions }
                var keys = Set<String>()
                for option in question.criteria {
                    guard keys.insert(option.key).inserted else {
                        throw SystemOneEngineError.duplicateChoiceKey(option.key)
                    }
                }
            case .score(let question):
                guard !question.criteria.isEmpty else { throw SystemOneEngineError.emptyScoreLevels }
            case .noul:
                break
            }
        }
    }
}
