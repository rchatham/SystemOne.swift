import Foundation
import MLX

/// Validation and orchestration errors raised by the public Von API.
public enum VonEngineError: Error, Equatable, Sendable {
    case emptyChoiceOptions
    case emptyScoreLevels
    case duplicateStateFieldKey(String)
    case duplicateChoiceKey(String)
    case duplicateQuestionID(String)
    case unexpectedMaskCount(expected: Int, actual: Int)
    case unexpectedLogitCount(expected: Int, actual: Int)
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
        }
    }
}

/// Local, actor-isolated System One inference engine.
///
/// Production initialization accepts only an explicit tokenizer folder and the
/// byte-for-byte verified production safetensors artifact. It performs no model
/// lookup, network access, HTTP serving, or batching.
public actor VonEngine {
    public static let modelID = "von-1.1.0"
    public static let defaultModelAlias = "von-latest"
    public static let defaultChoiceInstructions = "Which option best describes the state?"
    public static let defaultScoreInstructions = "Rate where the state falls on this scale:"

    private let tokenizer: VonTokenizer
    private let model: VonOptionMarkerModel?
    private let injectedLogitsProvider: (@Sendable (VonPackedInput) async throws -> [Float])?

    /// Loads the local tokenizer and verifies the exact pinned production weights.
    public init(tokenizerFolder: URL, verifiedWeightsURL: URL) async throws {
        tokenizer = try await VonTokenizer(modelFolder: tokenizerFolder)
        let weights = try VonWeights.verifiedProduction(contentsOf: verifiedWeightsURL)
        model = VonOptionMarkerModel(weights: weights)
        injectedLogitsProvider = nil
    }

    /// Offline test seam. The provider receives one packed input per sequential forward.
    init(
        tokenizer: VonTokenizer,
        logitsProvider: @escaping @Sendable (VonPackedInput) async throws -> [Float]
    ) {
        self.tokenizer = tokenizer
        model = nil
        injectedLogitsProvider = logitsProvider
    }

    /// Evaluates ordered questions sequentially and returns Python-shaped typed answers.
    public func systemOne(_ request: SystemOneRequest) async throws -> SystemOneResponse {
        try validate(request.state)
        try validate(request.questions)

        let stateText = VonPythonSemantics.formatState(request.state)
        var answers: [AnswerEntry] = []
        answers.reserveCapacity(request.questions.count)
        var instructionCharacters = 0

        for entry in request.questions {
            let answer: Answer
            switch entry.question {
            case .choice(let question):
                answer = .choice(try await evaluateChoice(question, stateText: stateText))
                instructionCharacters += VonPythonSemantics.characterCount(question.instructions)
            case .noul(let question):
                answer = .noul(try await evaluateNoul(question, stateText: stateText))
                instructionCharacters += VonPythonSemantics.characterCount(question.instructions)
            case .score(let question):
                answer = .score(try await evaluateScore(question, stateText: stateText))
                instructionCharacters += VonPythonSemantics.characterCount(question.instructions)
            }
            answers.append(AnswerEntry(id: entry.id, answer: answer))
        }

        let stateCharacters = VonPythonSemantics.characterCount(stateText)
        let usage = Usage(
            inputTokens: max(1, stateCharacters / 4) + max(1, instructionCharacters / 4),
            outputTokens: answers.count
        )
        return SystemOneResponse(model: Self.modelID, answers: answers, usage: usage)
    }

    /// Convenience form of `systemOne` retaining explicit question order.
    public func systemOne(
        state: VonState,
        questions: [QuestionEntry],
        model: String = VonEngine.defaultModelAlias
    ) async throws -> SystemOneResponse {
        try await systemOne(SystemOneRequest(model: model, state: state, questions: questions))
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

private extension VonEngine {
    func evaluateChoice(_ question: Choice, stateText: String) async throws -> ChoiceAnswer {
        let descriptions = VonPythonSemantics.renderChoiceOptions(question.criteria)
        let logits = try await forward(
            state: stateText,
            question: question.instructions,
            options: descriptions
        )
        let result = try VonDecisionMath.choice(
            logits: logits,
            stateTokenCount: tokenizer.stateTokenCount(stateText)
        )
        return ChoiceAnswer(
            choice: question.criteria[result.selectedIndex].key,
            probabilities: zip(question.criteria, result.roundedProbabilities).map {
                ProbabilityEntry(key: $0.0.key, probability: $0.1)
            },
            confidence: result.confidence
        )
    }

    func evaluateNoul(_ question: Noul, stateText: String) async throws -> NoulAnswer {
        let rendered = VonPythonSemantics.renderNoulCriteria(question.criteria)
        let logits = try await forward(
            state: stateText,
            question: question.instructions,
            options: rendered.descriptions
        )
        let nullLogits: [Float]?
        if rendered.hasExplicitCriteria {
            nullLogits = nil
        } else {
            nullLogits = try await forward(
                state: "",
                question: question.instructions,
                options: rendered.descriptions
            )
        }
        let result = try VonDecisionMath.noul(
            logits: logits,
            nullLogits: nullLogits,
            stateTokenCount: tokenizer.stateTokenCount(stateText)
        )
        return NoulAnswer(noul: result.probabilityTrue)
    }

    func evaluateScore(_ question: Score, stateText: String) async throws -> ScoreAnswer {
        let descriptions = VonPythonSemantics.renderScoreLevels(question.criteria)
        let logits = try await forward(
            state: stateText,
            question: question.instructions,
            options: descriptions
        )
        let result = try VonDecisionMath.score(
            logits: logits,
            stateTokenCount: tokenizer.stateTokenCount(stateText)
        )
        let keys = descriptions.indices.map(String.init)
        return ScoreAnswer(
            score: result.score,
            confidence: result.confidence,
            legend: zip(keys, descriptions).map { LegendEntry(key: $0.0, description: $0.1) },
            probabilities: zip(keys, result.roundedProbabilities).map {
                ProbabilityEntry(key: $0.0, probability: $0.1)
            }
        )
    }

    func forward(state: String, question: String, options: [String]) async throws -> [Float] {
        let packed = tokenizer.packedInput(state: state, question: question, options: options)
        guard packed.maskPositions.count == options.count else {
            throw VonEngineError.unexpectedMaskCount(
                expected: options.count,
                actual: packed.maskPositions.count
            )
        }
        let logits: [Float]
        if let injectedLogitsProvider {
            logits = try await injectedLogitsProvider(packed)
        } else {
            guard let model else { preconditionFailure("VonEngine has no logits provider") }
            let output = try model.forward(
                inputIDs: packed.inputIDs,
                attentionMask: packed.attentionMask,
                maskPositions: packed.maskPositions
            )
            eval(output)
            logits = output.asArray(Float.self)
        }
        guard logits.count == options.count else {
            throw VonEngineError.unexpectedLogitCount(expected: options.count, actual: logits.count)
        }
        return logits
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
