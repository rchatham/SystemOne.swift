import Foundation

/// One ordered key/value state field. Fields are rendered in their supplied order.
public struct StateField: Codable, Equatable, Sendable {
    public let key: String
    public let value: String

    public init(key: String, value: String) {
        self.key = key
        self.value = value
    }
}

/// State accepted by Von. Key/value state deliberately uses an ordered array.
public enum SystemOneState: Equatable, Sendable {
    case text(String)
    case fields([StateField])
}

/// One ordered Choice option.
public struct ChoiceOption: Equatable, Sendable {
    public let key: String
    public let description: String?

    public init(key: String, description: String? = nil) {
        self.key = key
        self.description = description
    }
}

/// Python-compatible Noul criteria. The wire keys are `true` and `false`.
public struct NoulCriteria: Codable, Equatable, Sendable {
    public let trueDescription: String?
    public let falseDescription: String?

    public init(true trueDescription: String? = nil, false falseDescription: String? = nil) {
        self.trueDescription = trueDescription
        self.falseDescription = falseDescription
    }

    enum CodingKeys: String, CodingKey {
        case trueDescription = "true"
        case falseDescription = "false"
    }
}

/// One ordered Score level, matching Python's string or `{what, examples}` forms.
public enum ScoreLevel: Codable, Equatable, Sendable {
    case text(String)
    case described(what: String, examples: [String])

    public init(what: String = "", examples: [String] = []) {
        self = .described(what: what, examples: examples)
    }

    private enum CodingKeys: String, CodingKey {
        case what
        case examples
    }

    public init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if let text = try? single.decode(String.self) {
            self = .text(text)
            return
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self = .described(
            what: try container.decodeIfPresent(String.self, forKey: .what) ?? "",
            examples: try container.decodeIfPresent([String].self, forKey: .examples) ?? []
        )
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let text):
            var container = encoder.singleValueContainer()
            try container.encode(text)
        case .described(let what, let examples):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(what, forKey: .what)
            try container.encode(examples, forKey: .examples)
        }
    }
}

public struct Choice: Equatable, Sendable {
    public let instructions: String
    public let criteria: [ChoiceOption]

    public init(instructions: String, criteria: [ChoiceOption]) {
        self.instructions = instructions
        self.criteria = criteria
    }
}

public struct Noul: Codable, Equatable, Sendable {
    public let instructions: String
    public let criteria: NoulCriteria?

    public init(instructions: String, criteria: NoulCriteria? = nil) {
        self.instructions = instructions
        self.criteria = criteria
    }
}

public struct Score: Codable, Equatable, Sendable {
    public let instructions: String
    public let criteria: [ScoreLevel]

    public init(instructions: String, criteria: [ScoreLevel]) {
        self.instructions = instructions
        self.criteria = criteria
    }
}

/// A typed Python-compatible question union.
public enum Question: Equatable, Sendable {
    case choice(Choice)
    case noul(Noul)
    case score(Score)
}

/// One ordered question entry.
public struct QuestionEntry: Equatable, Sendable {
    public let id: String
    public let question: Question

    public init(id: String, question: Question) {
        self.id = id
        self.question = question
    }
}

/// A System One request. `questions` remains ordered for sequential evaluation.
public struct SystemOneRequest: Codable, Equatable, Sendable {
    public let model: String
    public let state: SystemOneState
    public let questions: [QuestionEntry]

    public init(
        model: String = "von-latest",
        state: SystemOneState,
        questions: [QuestionEntry]
    ) {
        self.model = model
        self.state = state
        self.questions = questions
    }
}

public struct ProbabilityEntry: Codable, Equatable, Sendable {
    public let key: String
    public let probability: Double

    public init(key: String, probability: Double) {
        self.key = key
        self.probability = probability
    }
}

public struct LegendEntry: Codable, Equatable, Sendable {
    public let key: String
    public let description: String

    public init(key: String, description: String) {
        self.key = key
        self.description = description
    }
}

public struct ChoiceAnswer: Equatable, Sendable {
    public let choice: String
    public let probabilities: [ProbabilityEntry]
    public let confidence: Double

    public init(choice: String, probabilities: [ProbabilityEntry], confidence: Double) {
        self.choice = choice
        self.probabilities = probabilities
        self.confidence = confidence
    }
}

public struct NoulAnswer: Codable, Equatable, Sendable {
    public let noul: Double

    public init(noul: Double) {
        self.noul = noul
    }
}

public struct ScoreAnswer: Equatable, Sendable {
    public let score: Double
    public let confidence: Double
    public let legend: [LegendEntry]
    public let probabilities: [ProbabilityEntry]

    public init(
        score: Double,
        confidence: Double,
        legend: [LegendEntry],
        probabilities: [ProbabilityEntry]
    ) {
        self.score = score
        self.confidence = confidence
        self.legend = legend
        self.probabilities = probabilities
    }
}

public enum Answer: Equatable, Sendable {
    case choice(ChoiceAnswer)
    case noul(NoulAnswer)
    case score(ScoreAnswer)
}

public struct AnswerEntry: Equatable, Sendable {
    public let id: String
    public let answer: Answer

    public init(id: String, answer: Answer) {
        self.id = id
        self.answer = answer
    }
}

public struct Usage: Codable, Equatable, Sendable {
    public let inputTokens: Int
    public let outputTokens: Int

    public init(inputTokens: Int = 0, outputTokens: Int = 0) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

public struct SystemOneResponse: Codable, Equatable, Sendable {
    public let model: String
    public let answers: [AnswerEntry]
    public let usage: Usage

    public init(model: String, answers: [AnswerEntry], usage: Usage) {
        self.model = model
        self.answers = answers
        self.usage = usage
    }

    public subscript(id: String) -> Answer? {
        answers.first { $0.id == id }?.answer
    }
}

private struct DynamicCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

private enum TaggedCodingKeys: String, CodingKey {
    case type
    case instructions
    case criteria
    case criteriaOrder = "criteria_order"
    case choice
    case noul
    case score
    case probabilities
    case probabilityOrder = "probability_order"
    case confidence
    case legend
    case legendOrder = "legend_order"
}

private enum EnvelopeCodingKeys: String, CodingKey {
    case model
    case state
    case stateOrder = "state_order"
    case questions
    case questionOrder = "question_order"
    case answers
    case answerOrder = "answer_order"
    case usage
}

private enum SystemOneCodingError {
    static func invalid(_ value: String, path: [CodingKey]) -> DecodingError {
        .dataCorrupted(.init(codingPath: path, debugDescription: "Unknown Von type '\(value)'."))
    }
}

private func validateOrder(_ order: [String], keys: [String], path: [CodingKey]) throws {
    guard Set(order).count == order.count, Set(order) == Set(keys) else {
        throw DecodingError.dataCorrupted(.init(
            codingPath: path,
            debugDescription: "Order metadata must contain every keyed entry exactly once."
        ))
    }
}

private func validateUniqueKeys(
    _ keys: [String],
    field: String,
    path: [CodingKey]
) throws {
    var seen = Set<String>()
    for key in keys where !seen.insert(key).inserted {
        throw EncodingError.invalidValue(
            keys,
            .init(
                codingPath: path,
                debugDescription: "\(field) contains duplicate key '\(key)'."
            )
        )
    }
}

extension SystemOneState: Codable {
    public init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if let text = try? single.decode(String.self) {
            self = .text(text)
        } else {
            self = .fields(try single.decode([StateField].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .fields(let fields): try container.encode(fields)
        }
    }
}

extension Choice: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TaggedCodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "choice" else { throw SystemOneCodingError.invalid(type, path: decoder.codingPath) }
        instructions = try container.decode(String.self, forKey: .instructions)
        let criteriaContainer = try container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .criteria
        )
        let order = try container.decode([String].self, forKey: .criteriaOrder)
        try validateOrder(order, keys: criteriaContainer.allKeys.map(\.stringValue), path: decoder.codingPath)
        criteria = try order.map { name in
            let key = DynamicCodingKey(stringValue: name)!
            return ChoiceOption(
                key: name,
                description: try criteriaContainer.decodeIfPresent(String.self, forKey: key)
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        let criteriaOrder = criteria.map(\.key)
        try validateUniqueKeys(
            criteriaOrder,
            field: "Choice.criteria",
            path: encoder.codingPath
        )
        var container = encoder.container(keyedBy: TaggedCodingKeys.self)
        try container.encode("choice", forKey: .type)
        try container.encode(instructions, forKey: .instructions)
        try container.encode(criteriaOrder, forKey: .criteriaOrder)
        var criteriaContainer = container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .criteria
        )
        for option in criteria {
            let key = DynamicCodingKey(stringValue: option.key)!
            try criteriaContainer.encode(option.description, forKey: key)
        }
    }
}

extension Noul {
    enum CodingKeys: String, CodingKey { case type, instructions, criteria }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "noul" else { throw SystemOneCodingError.invalid(type, path: decoder.codingPath) }
        instructions = try container.decode(String.self, forKey: .instructions)
        criteria = try container.decodeIfPresent(NoulCriteria.self, forKey: .criteria)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("noul", forKey: .type)
        try container.encode(instructions, forKey: .instructions)
        try container.encodeIfPresent(criteria, forKey: .criteria)
    }
}

extension Score {
    enum CodingKeys: String, CodingKey { case type, instructions, criteria }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "score" else { throw SystemOneCodingError.invalid(type, path: decoder.codingPath) }
        instructions = try container.decode(String.self, forKey: .instructions)
        criteria = try container.decode([ScoreLevel].self, forKey: .criteria)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("score", forKey: .type)
        try container.encode(instructions, forKey: .instructions)
        try container.encode(criteria, forKey: .criteria)
    }
}

extension Question: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TaggedCodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "choice": self = .choice(try Choice(from: decoder))
        case "noul": self = .noul(try Noul(from: decoder))
        case "score": self = .score(try Score(from: decoder))
        default: throw SystemOneCodingError.invalid(type, path: decoder.codingPath)
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .choice(let value): try value.encode(to: encoder)
        case .noul(let value): try value.encode(to: encoder)
        case .score(let value): try value.encode(to: encoder)
        }
    }
}

extension QuestionEntry: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let pair = try container.decode([String: Question].self)
        guard pair.count == 1, let first = pair.first else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected one question entry.")
        }
        self.init(id: first.key, question: first.value)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicCodingKey.self)
        try container.encode(question, forKey: DynamicCodingKey(stringValue: id)!)
    }
}

extension SystemOneRequest {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: EnvelopeCodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model) ?? "von-latest"
        if let text = try? container.decode(String.self, forKey: .state) {
            state = .text(text)
        } else {
            let stateContainer = try container.nestedContainer(
                keyedBy: DynamicCodingKey.self,
                forKey: .state
            )
            let stateOrder = try container.decode([String].self, forKey: .stateOrder)
            try validateOrder(
                stateOrder,
                keys: stateContainer.allKeys.map(\.stringValue),
                path: decoder.codingPath
            )
            state = .fields(try stateOrder.map { name in
                let key = DynamicCodingKey(stringValue: name)!
                return StateField(key: name, value: try stateContainer.decode(String.self, forKey: key))
            })
        }
        let questionsContainer = try container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .questions
        )
        let questionOrder = try container.decode([String].self, forKey: .questionOrder)
        try validateOrder(
            questionOrder,
            keys: questionsContainer.allKeys.map(\.stringValue),
            path: decoder.codingPath
        )
        questions = try questionOrder.map { id in
            let key = DynamicCodingKey(stringValue: id)!
            return QuestionEntry(id: id, question: try questionsContainer.decode(Question.self, forKey: key))
        }
    }

    public func encode(to encoder: Encoder) throws {
        if case .fields(let fields) = state {
            try validateUniqueKeys(
                fields.map(\.key),
                field: "SystemOneRequest.state",
                path: encoder.codingPath
            )
        }
        let questionOrder = questions.map(\.id)
        try validateUniqueKeys(
            questionOrder,
            field: "SystemOneRequest.questions",
            path: encoder.codingPath
        )
        var container = encoder.container(keyedBy: EnvelopeCodingKeys.self)
        try container.encode(model, forKey: .model)
        switch state {
        case .text(let text):
            try container.encode(text, forKey: .state)
        case .fields(let fields):
            try container.encode(fields.map(\.key), forKey: .stateOrder)
            var stateContainer = container.nestedContainer(
                keyedBy: DynamicCodingKey.self,
                forKey: .state
            )
            for field in fields {
                try stateContainer.encode(field.value, forKey: DynamicCodingKey(stringValue: field.key)!)
            }
        }
        try container.encode(questionOrder, forKey: .questionOrder)
        var questionsContainer = container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .questions
        )
        for entry in questions {
            try questionsContainer.encode(entry.question, forKey: DynamicCodingKey(stringValue: entry.id)!)
        }
    }
}

extension ChoiceAnswer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TaggedCodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "choice" else { throw SystemOneCodingError.invalid(type, path: decoder.codingPath) }
        choice = try container.decode(String.self, forKey: .choice)
        confidence = try container.decode(Double.self, forKey: .confidence)
        let probabilitiesContainer = try container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .probabilities
        )
        let probabilityOrder = try container.decode([String].self, forKey: .probabilityOrder)
        try validateOrder(
            probabilityOrder,
            keys: probabilitiesContainer.allKeys.map(\.stringValue),
            path: decoder.codingPath
        )
        probabilities = try probabilityOrder.map { name in
            let key = DynamicCodingKey(stringValue: name)!
            return ProbabilityEntry(
                key: name,
                probability: try probabilitiesContainer.decode(Double.self, forKey: key)
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        let probabilityOrder = probabilities.map(\.key)
        try validateUniqueKeys(
            probabilityOrder,
            field: "ChoiceAnswer.probabilities",
            path: encoder.codingPath
        )
        var container = encoder.container(keyedBy: TaggedCodingKeys.self)
        try container.encode("choice", forKey: .type)
        try container.encode(choice, forKey: .choice)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(probabilityOrder, forKey: .probabilityOrder)
        var probabilitiesContainer = container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .probabilities
        )
        for probability in probabilities {
            try probabilitiesContainer.encode(
                probability.probability,
                forKey: DynamicCodingKey(stringValue: probability.key)!
            )
        }
    }
}

extension NoulAnswer {
    enum CodingKeys: String, CodingKey { case type, noul }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "noul" else { throw SystemOneCodingError.invalid(type, path: decoder.codingPath) }
        noul = try container.decode(Double.self, forKey: .noul)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("noul", forKey: .type)
        try container.encode(noul, forKey: .noul)
    }
}

extension ScoreAnswer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TaggedCodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "score" else { throw SystemOneCodingError.invalid(type, path: decoder.codingPath) }
        score = try container.decode(Double.self, forKey: .score)
        confidence = try container.decode(Double.self, forKey: .confidence)
        let legendContainer = try container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .legend)
        let legendOrder = try container.decode([String].self, forKey: .legendOrder)
        try validateOrder(
            legendOrder,
            keys: legendContainer.allKeys.map(\.stringValue),
            path: decoder.codingPath
        )
        legend = try legendOrder.map { name in
            let key = DynamicCodingKey(stringValue: name)!
            return LegendEntry(
                key: name,
                description: try legendContainer.decode(String.self, forKey: key)
            )
        }
        let probabilitiesContainer = try container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .probabilities
        )
        let probabilityOrder = try container.decode([String].self, forKey: .probabilityOrder)
        try validateOrder(
            probabilityOrder,
            keys: probabilitiesContainer.allKeys.map(\.stringValue),
            path: decoder.codingPath
        )
        probabilities = try probabilityOrder.map { name in
            let key = DynamicCodingKey(stringValue: name)!
            return ProbabilityEntry(
                key: name,
                probability: try probabilitiesContainer.decode(Double.self, forKey: key)
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        let legendOrder = legend.map(\.key)
        let probabilityOrder = probabilities.map(\.key)
        try validateUniqueKeys(
            legendOrder,
            field: "ScoreAnswer.legend",
            path: encoder.codingPath
        )
        try validateUniqueKeys(
            probabilityOrder,
            field: "ScoreAnswer.probabilities",
            path: encoder.codingPath
        )
        var container = encoder.container(keyedBy: TaggedCodingKeys.self)
        try container.encode("score", forKey: .type)
        try container.encode(score, forKey: .score)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(legendOrder, forKey: .legendOrder)
        try container.encode(probabilityOrder, forKey: .probabilityOrder)
        var legendContainer = container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .legend)
        for entry in legend {
            try legendContainer.encode(entry.description, forKey: DynamicCodingKey(stringValue: entry.key)!)
        }
        var probabilitiesContainer = container.nestedContainer(
            keyedBy: DynamicCodingKey.self,
            forKey: .probabilities
        )
        for probability in probabilities {
            try probabilitiesContainer.encode(
                probability.probability,
                forKey: DynamicCodingKey(stringValue: probability.key)!
            )
        }
    }
}

extension Answer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TaggedCodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "choice": self = .choice(try ChoiceAnswer(from: decoder))
        case "noul": self = .noul(try NoulAnswer(from: decoder))
        case "score": self = .score(try ScoreAnswer(from: decoder))
        default: throw SystemOneCodingError.invalid(type, path: decoder.codingPath)
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .choice(let value): try value.encode(to: encoder)
        case .noul(let value): try value.encode(to: encoder)
        case .score(let value): try value.encode(to: encoder)
        }
    }
}

extension AnswerEntry: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let pair = try container.decode([String: Answer].self)
        guard pair.count == 1, let first = pair.first else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected one answer entry.")
        }
        self.init(id: first.key, answer: first.value)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicCodingKey.self)
        try container.encode(answer, forKey: DynamicCodingKey(stringValue: id)!)
    }
}

extension SystemOneResponse {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: EnvelopeCodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        usage = try container.decode(Usage.self, forKey: .usage)
        let answersContainer = try container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .answers)
        let answerOrder = try container.decode([String].self, forKey: .answerOrder)
        try validateOrder(
            answerOrder,
            keys: answersContainer.allKeys.map(\.stringValue),
            path: decoder.codingPath
        )
        answers = try answerOrder.map { id in
            let key = DynamicCodingKey(stringValue: id)!
            return AnswerEntry(id: id, answer: try answersContainer.decode(Answer.self, forKey: key))
        }
    }

    public func encode(to encoder: Encoder) throws {
        let answerOrder = answers.map(\.id)
        try validateUniqueKeys(
            answerOrder,
            field: "SystemOneResponse.answers",
            path: encoder.codingPath
        )
        var container = encoder.container(keyedBy: EnvelopeCodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encode(usage, forKey: .usage)
        try container.encode(answerOrder, forKey: .answerOrder)
        var answersContainer = container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .answers)
        for entry in answers {
            try answersContainer.encode(entry.answer, forKey: DynamicCodingKey(stringValue: entry.id)!)
        }
    }
}
