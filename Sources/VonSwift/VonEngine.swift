import Foundation
import MLX

/// Controls how `systemOne` fans out one forward pass per planned row or
/// groups compatible rows into padded microbatches.
public enum VonBatchingStrategy: Sendable, Equatable {
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
public enum VonEngineError: Error, Equatable, Sendable {
    case emptyChoiceOptions
    case emptyScoreLevels
    case duplicateStateFieldKey(String)
    case duplicateChoiceKey(String)
    case duplicateQuestionID(String)
    case unexpectedMaskCount(expected: Int, actual: Int)
    case unexpectedLogitCount(expected: Int, actual: Int)
    /// A batch provider returned row boundaries that do not match the planned rows.
    case unexpectedBatchOffsets(expected: [Int], actual: [Int])
}

extension VonEngineError: LocalizedError {
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
        }
    }
}

struct VonBatchLogitsOutput: Equatable, Sendable {
    let logits: [Float]
    let rowOffsets: [Int]

    init(logits: [Float], rowOffsets: [Int]) {
        self.logits = logits
        self.rowOffsets = rowOffsets
    }
}

struct VonPlannedInferenceRow: Equatable, Sendable {
    let input: VonPackedInput
    let expectedLogitCount: Int
}

enum VonPlannedAnswer: Equatable, Sendable {
    case choice(keys: [String])
    case noul
    case score(descriptions: [String])
}

struct VonPlannedQuestion: Equatable, Sendable {
    let id: String
    let answer: VonPlannedAnswer
    let primaryRowIndex: Int
    let nullRowIndex: Int?
    let stateTokenCount: Int
}

/// Local, actor-isolated System One inference engine.
///
/// Production initialization accepts only an explicit tokenizer folder and the
/// byte-for-byte verified production safetensors artifact. It performs no model
/// lookup, network access, HTTP serving, or unbounded/concurrent batching.
public actor VonEngine {
    public static let modelID = "von-1.1.0"
    public static let defaultModelAlias = "von-latest"
    public static let defaultChoiceInstructions = "Which option best describes the state?"
    public static let defaultScoreInstructions = "Rate where the state falls on this scale:"

    private let tokenizer: VonTokenizer
    private let model: VonOptionMarkerModel?
    private let injectedLogitsProvider: (@Sendable (VonPackedInput) async throws -> [Float])?
    private let injectedBatchLogitsProvider: (@Sendable ([VonPackedInput]) async throws -> VonBatchLogitsOutput)?

    /// Loads the local tokenizer and verifies the exact pinned production weights.
    public init(tokenizerFolder: URL, verifiedWeightsURL: URL) async throws {
        tokenizer = try await VonTokenizer(modelFolder: tokenizerFolder)
        let weights = try VonWeights.verifiedProduction(contentsOf: verifiedWeightsURL)
        model = VonOptionMarkerModel(weights: weights)
        injectedLogitsProvider = nil
        injectedBatchLogitsProvider = nil
    }

    /// Offline test seam. The provider receives one packed input per sequential forward.
    init(
        tokenizer: VonTokenizer,
        logitsProvider: @escaping @Sendable (VonPackedInput) async throws -> [Float]
    ) {
        self.tokenizer = tokenizer
        model = nil
        injectedLogitsProvider = logitsProvider
        injectedBatchLogitsProvider = nil
    }

    /// Offline batched test seam. Microbatches are issued sequentially in sorted order.
    init(
        tokenizer: VonTokenizer,
        batchLogitsProvider: @escaping @Sendable ([VonPackedInput]) async throws -> VonBatchLogitsOutput
    ) {
        self.tokenizer = tokenizer
        model = nil
        injectedLogitsProvider = nil
        injectedBatchLogitsProvider = batchLogitsProvider
    }

    /// Evaluates ordered questions with bounded, stable microbatch fan-out.
    public func systemOne(
        _ request: SystemOneRequest,
        batchingStrategy: VonBatchingStrategy = .automatic
    ) async throws -> SystemOneResponse {
        try validate(request.state)
        try validate(request.questions)

        let stateText = VonPythonSemantics.formatState(request.state)
        let stateTokenCount = tokenizer.stateTokenCount(stateText)
        let plan = try plan(
            questions: request.questions,
            stateText: stateText,
            stateTokenCount: stateTokenCount
        )
        let answers: [AnswerEntry]
        if injectedLogitsProvider != nil {
            answers = try await executeLegacy(plan)
        } else {
            let rowLogits = try await execute(plan.rows, strategy: batchingStrategy)
            answers = try plan.questions.map { question in
                AnswerEntry(
                    id: question.id,
                    answer: try answer(for: question, rowLogits: rowLogits)
                )
            }
        }

        let stateCharacters = VonPythonSemantics.characterCount(stateText)
        let usage = Usage(
            inputTokens: max(1, stateCharacters / 4) + max(1, plan.instructionCharacters / 4),
            outputTokens: answers.count
        )
        return SystemOneResponse(model: Self.modelID, answers: answers, usage: usage)
    }

    /// Convenience form of `systemOne` retaining explicit question order.
    public func systemOne(
        state: VonState,
        questions: [QuestionEntry],
        model: String = VonEngine.defaultModelAlias,
        batchingStrategy: VonBatchingStrategy = .automatic
    ) async throws -> SystemOneResponse {
        try await systemOne(
            SystemOneRequest(model: model, state: state, questions: questions),
            batchingStrategy: batchingStrategy
        )
    }

    /// Chooses one option from an ordered list.
    public func decide(
        state: VonState,
        choices: [ChoiceOption],
        instructions: String = VonEngine.defaultChoiceInstructions,
        model: String = VonEngine.defaultModelAlias
    ) async throws -> ChoiceAnswer {
        let response = try await systemOne(
            state: state,
            questions: [
                QuestionEntry(
                    id: "decision",
                    question: .choice(Choice(instructions: instructions, criteria: choices))
                ),
            ],
            model: model
        )
        guard case .choice(let answer)? = response["decision"] else {
            preconditionFailure("VonEngine produced a mismatched decision answer")
        }
        return answer
    }

    /// Evaluates a yes/no condition and returns the probability that it is true.
    public func judge(
        state: VonState,
        instructions: String,
        criteria: NoulCriteria? = nil,
        model: String = VonEngine.defaultModelAlias
    ) async throws -> Double {
        let response = try await systemOne(
            state: state,
            questions: [
                QuestionEntry(
                    id: "judgment",
                    question: .noul(Noul(instructions: instructions, criteria: criteria))
                ),
            ],
            model: model
        )
        guard case .noul(let answer)? = response["judgment"] else {
            preconditionFailure("VonEngine produced a mismatched judgment answer")
        }
        return answer.noul
    }

    /// Rates state over ordered ordinal levels.
    public func rate(
        state: VonState,
        criteria: [ScoreLevel],
        instructions: String = VonEngine.defaultScoreInstructions,
        model: String = VonEngine.defaultModelAlias
    ) async throws -> ScoreAnswer {
        let response = try await systemOne(
            state: state,
            questions: [
                QuestionEntry(
                    id: "rating",
                    question: .score(Score(instructions: instructions, criteria: criteria))
                ),
            ],
            model: model
        )
        guard case .score(let answer)? = response["rating"] else {
            preconditionFailure("VonEngine produced a mismatched rating answer")
        }
        return answer
    }
}

extension VonEngine {
    /// Stable length-sorted row indices grouped by bounded padded attention work.
    ///
    /// When `maxAmplification` is `nil` each row is its own singleton batch.
    /// Otherwise the absolute boundary `max(candidateMaximum, 1_048_576)` and the
    /// relative `paddedWork <= maxAmplification × usefulWork` gate are applied.
    public static func microbatchIndices(
        sequenceLengths: [Int],
        maxAmplification: Double?
    ) -> [[Int]] {
        let sortedIndices = sequenceLengths.indices.sorted { lhs, rhs in
            let left = sequenceLengths[lhs]
            let right = sequenceLengths[rhs]
            return left == right ? lhs < rhs : left > right
        }
        guard !sortedIndices.isEmpty else { return [] }

        func squared(_ length: Int) -> Int? {
            guard length >= 0 else { return nil }
            let (result, overflow) = length.multipliedReportingOverflow(by: length)
            return overflow ? nil : result
        }

        var batches: [[Int]] = []
        var current: [Int] = []
        var maximumSquared = 0
        var sumSquared = 0
        var sumSquaredIsUnbounded = false

        for index in sortedIndices {
            guard let rowSquared = squared(sequenceLengths[index]) else {
                if !current.isEmpty {
                    batches.append(current)
                    current = []
                    maximumSquared = 0
                    sumSquared = 0
                    sumSquaredIsUnbounded = false
                }
                batches.append([index])
                continue
            }

            guard !current.isEmpty else {
                current = [index]
                maximumSquared = rowSquared
                sumSquared = rowSquared
                sumSquaredIsUnbounded = false
                continue
            }

            guard let amplification = maxAmplification else {
                // nil means sequential: each row is a singleton batch
                batches.append(current)
                current = [index]
                maximumSquared = rowSquared
                sumSquared = rowSquared
                sumSquaredIsUnbounded = false
                continue
            }

            let candidateMaximum = max(maximumSquared, rowSquared)
            let (candidateCount, countOverflow) = current.count.addingReportingOverflow(1)
            let (paddedWork, paddedOverflow) = candidateCount.multipliedReportingOverflow(
                by: candidateMaximum
            )
            let (candidateSum, sumOverflow) = sumSquared.addingReportingOverflow(rowSquared)
            let candidateSumIsUnbounded = sumSquaredIsUnbounded || sumOverflow
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
                maximumSquared = candidateMaximum
                sumSquared = candidateSumIsUnbounded ? 0 : candidateSum
                sumSquaredIsUnbounded = candidateSumIsUnbounded
            } else {
                batches.append(current)
                current = [index]
                maximumSquared = rowSquared
                sumSquared = rowSquared
                sumSquaredIsUnbounded = false
            }
        }
        if !current.isEmpty {
            batches.append(current)
        }
        return batches
    }
}

private extension VonEngine {
    typealias InferencePlan = (
        rows: [VonPlannedInferenceRow],
        questions: [VonPlannedQuestion],
        instructionCharacters: Int
    )

    func plan(
        questions: [QuestionEntry],
        stateText: String,
        stateTokenCount: Int
    ) throws -> InferencePlan {
        var rows: [VonPlannedInferenceRow] = []
        var plannedQuestions: [VonPlannedQuestion] = []
        var instructionCharacters = 0
        rows.reserveCapacity(questions.count)
        plannedQuestions.reserveCapacity(questions.count)

        for entry in questions {
            let options: [String]
            let instructions: String
            let needsNullRow: Bool
            let plannedAnswer: VonPlannedAnswer
            switch entry.question {
            case .choice(let question):
                options = VonPythonSemantics.renderChoiceOptions(question.criteria)
                instructions = question.instructions
                needsNullRow = false
                plannedAnswer = .choice(keys: question.criteria.map(\.key))
            case .noul(let question):
                let rendered = VonPythonSemantics.renderNoulCriteria(question.criteria)
                options = rendered.descriptions
                instructions = question.instructions
                needsNullRow = !rendered.hasExplicitCriteria
                plannedAnswer = .noul
            case .score(let question):
                let rendered = VonPythonSemantics.renderScoreLevels(question.criteria)
                options = rendered
                instructions = question.instructions
                needsNullRow = false
                plannedAnswer = .score(descriptions: rendered)
            }

            let primaryRowIndex = rows.count
            rows.append(try plannedRow(state: stateText, question: instructions, options: options))
            let nullRowIndex: Int?
            if needsNullRow {
                nullRowIndex = rows.count
                rows.append(try plannedRow(state: "", question: instructions, options: options))
            } else {
                nullRowIndex = nil
            }
            plannedQuestions.append(VonPlannedQuestion(
                id: entry.id,
                answer: plannedAnswer,
                primaryRowIndex: primaryRowIndex,
                nullRowIndex: nullRowIndex,
                stateTokenCount: stateTokenCount
            ))
            instructionCharacters += VonPythonSemantics.characterCount(instructions)
        }
        return (rows, plannedQuestions, instructionCharacters)
    }

    func plannedRow(
        state: String,
        question: String,
        options: [String]
    ) throws -> VonPlannedInferenceRow {
        let packed = tokenizer.packedInput(state: state, question: question, options: options)
        guard packed.maskPositions.count == options.count else {
            throw VonEngineError.unexpectedMaskCount(
                expected: options.count,
                actual: packed.maskPositions.count
            )
        }
        return VonPlannedInferenceRow(input: packed, expectedLogitCount: options.count)
    }

    func executeLegacy(_ plan: InferencePlan) async throws -> [AnswerEntry] {
        guard let injectedLogitsProvider else {
            preconditionFailure("VonEngine has no legacy logits provider")
        }
        var rowLogits = [[Float]](repeating: [], count: plan.rows.count)
        var answers: [AnswerEntry] = []
        answers.reserveCapacity(plan.questions.count)

        for question in plan.questions {
            let rowIndices = [question.primaryRowIndex]
                + (question.nullRowIndex.map { [$0] } ?? [])
            for rowIndex in rowIndices {
                let row = plan.rows[rowIndex]
                try Task.checkCancellation()
                let logits = try await injectedLogitsProvider(row.input)
                guard logits.count == row.expectedLogitCount else {
                    throw VonEngineError.unexpectedLogitCount(
                        expected: row.expectedLogitCount,
                        actual: logits.count
                    )
                }
                rowLogits[rowIndex] = logits
            }
            answers.append(AnswerEntry(
                id: question.id,
                answer: try answer(for: question, rowLogits: rowLogits)
            ))
        }
        return answers
    }

    func execute(
        _ rows: [VonPlannedInferenceRow],
        strategy: VonBatchingStrategy
    ) async throws -> [[Float]] {
        guard !rows.isEmpty else { return [] }
        let batches = Self.microbatchIndices(
            sequenceLengths: rows.map { $0.input.inputIDs.count },
            maxAmplification: strategy.maxAmplification
        )
        var scattered = [[Float]?](repeating: nil, count: rows.count)
        for indices in batches {
            let chunk = indices.map { rows[$0] }
            let inputs = chunk.map(\.input)
            let output: VonBatchLogitsOutput
            if let injectedBatchLogitsProvider {
                try Task.checkCancellation()
                output = try await injectedBatchLogitsProvider(inputs)
            } else {
                guard let model else { preconditionFailure("VonEngine has no logits provider") }
                try Task.checkCancellation()
                let modelOutput = try model.forward(inputs)
                eval(modelOutput.logits)
                output = VonBatchLogitsOutput(
                    logits: modelOutput.logits.asArray(Float.self),
                    rowOffsets: modelOutput.rowOffsets
                )
            }
            let batchLogits = try separate(output, for: chunk)
            for (index, logits) in zip(indices, batchLogits) {
                scattered[index] = logits
            }
        }
        return scattered.map { logits in
            guard let logits else {
                preconditionFailure("VonEngine failed to populate a planned inference row")
            }
            return logits
        }
    }

    func separate(
        _ output: VonBatchLogitsOutput,
        for rows: [VonPlannedInferenceRow]
    ) throws -> [[Float]] {
        var expectedOffsets = [0]
        expectedOffsets.reserveCapacity(rows.count + 1)
        for row in rows {
            let (offset, overflow) = expectedOffsets.last!.addingReportingOverflow(row.expectedLogitCount)
            guard !overflow else {
                throw VonEngineError.unexpectedLogitCount(expected: Int.max, actual: output.logits.count)
            }
            expectedOffsets.append(offset)
        }
        guard output.rowOffsets == expectedOffsets else {
            throw VonEngineError.unexpectedBatchOffsets(
                expected: expectedOffsets,
                actual: output.rowOffsets
            )
        }
        let expectedCount = expectedOffsets.last!
        guard output.logits.count == expectedCount else {
            throw VonEngineError.unexpectedLogitCount(
                expected: expectedCount,
                actual: output.logits.count
            )
        }
        return rows.indices.map { index in
            Array(output.logits[expectedOffsets[index]..<expectedOffsets[index + 1]])
        }
    }

    func answer(
        for planned: VonPlannedQuestion,
        rowLogits: [[Float]]
    ) throws -> Answer {
        let logits = rowLogits[planned.primaryRowIndex]
        switch planned.answer {
        case .choice(let keys):
            let result = try VonDecisionMath.choice(
                logits: logits,
                stateTokenCount: planned.stateTokenCount
            )
            return .choice(ChoiceAnswer(
                choice: keys[result.selectedIndex],
                probabilities: zip(keys, result.roundedProbabilities).map {
                    ProbabilityEntry(key: $0.0, probability: $0.1)
                },
                confidence: result.confidence
            ))
        case .noul:
            let nullLogits = planned.nullRowIndex.map { rowLogits[$0] }
            let result = try VonDecisionMath.noul(
                logits: logits,
                nullLogits: nullLogits,
                stateTokenCount: planned.stateTokenCount
            )
            return .noul(NoulAnswer(noul: result.probabilityTrue))
        case .score(let descriptions):
            let result = try VonDecisionMath.score(
                logits: logits,
                stateTokenCount: planned.stateTokenCount
            )
            let keys = descriptions.indices.map(String.init)
            return .score(ScoreAnswer(
                score: result.score,
                confidence: result.confidence,
                legend: zip(keys, descriptions).map {
                    LegendEntry(key: $0.0, description: $0.1)
                },
                probabilities: zip(keys, result.roundedProbabilities).map {
                    ProbabilityEntry(key: $0.0, probability: $0.1)
                }
            ))
        }
    }

    func validate(_ state: VonState) throws {
        guard case .fields(let fields) = state else { return }
        var keys = Set<String>()
        for field in fields {
            guard keys.insert(field.key).inserted else {
                throw VonEngineError.duplicateStateFieldKey(field.key)
            }
        }
    }

    func validate(_ questions: [QuestionEntry]) throws {
        var questionIDs = Set<String>()
        for entry in questions {
            guard questionIDs.insert(entry.id).inserted else {
                throw VonEngineError.duplicateQuestionID(entry.id)
            }
            switch entry.question {
            case .choice(let question):
                guard !question.criteria.isEmpty else { throw VonEngineError.emptyChoiceOptions }
                var keys = Set<String>()
                for option in question.criteria {
                    guard keys.insert(option.key).inserted else {
                        throw VonEngineError.duplicateChoiceKey(option.key)
                    }
                }
            case .score(let question):
                guard !question.criteria.isEmpty else { throw VonEngineError.emptyScoreLevels }
            case .noul:
                break
            }
        }
    }
}
