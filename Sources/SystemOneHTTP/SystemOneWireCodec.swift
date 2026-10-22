import Foundation
import JSON
import SystemOne

/// Failures while adapting metadata-free Python Von HTTP JSON to Swift types.
public enum SystemOneWireCodecError: Error, Equatable, Sendable {
    case invalidRequest(path: String, message: String)
    case unsupportedState(path: String, message: String)
    case safetyLimitExceeded(name: String, limit: Int)
    case invalidResponse(path: String, message: String)
}

/// Conservative tokenizer-free limits on work admitted to the inference engine.
public struct WireSafetyLimits: Equatable, Sendable {
    public let maximumQuestions: Int
    public let maximumPlannedForwards: Int
    public let maximumOptionsPerQuestion: Int
    public let maximumPlannedMarkers: Int
    public let maximumScoreExamplesPerLevel: Int
    public let maximumStateUTF8Bytes: Int
    public let maximumInstructionUTF8BytesPerQuestion: Int
    public let maximumTotalInstructionUTF8Bytes: Int
    public let maximumRenderedOptionsUTF8BytesPerQuestion: Int
    public let maximumTotalRenderedOptionsUTF8Bytes: Int
    public let maximumPackedUTF8BytesPerForward: Int
    public let maximumTotalPackedUTF8Bytes: Int

    public init(
        maximumQuestions: Int = 16,
        maximumPlannedForwards: Int = 24,
        maximumOptionsPerQuestion: Int = 64,
        maximumPlannedMarkers: Int = 256,
        maximumScoreExamplesPerLevel: Int = 256,
        maximumStateUTF8Bytes: Int = 4_096,
        maximumInstructionUTF8BytesPerQuestion: Int = 2_048,
        maximumTotalInstructionUTF8Bytes: Int = 8_192,
        maximumRenderedOptionsUTF8BytesPerQuestion: Int = 4_096,
        maximumTotalRenderedOptionsUTF8Bytes: Int = 16_384,
        maximumPackedUTF8BytesPerForward: Int = 8_190,
        maximumTotalPackedUTF8Bytes: Int = 32_768
    ) {
        self.maximumQuestions = maximumQuestions
        self.maximumPlannedForwards = maximumPlannedForwards
        self.maximumOptionsPerQuestion = maximumOptionsPerQuestion
        self.maximumPlannedMarkers = maximumPlannedMarkers
        self.maximumScoreExamplesPerLevel = maximumScoreExamplesPerLevel
        self.maximumStateUTF8Bytes = maximumStateUTF8Bytes
        self.maximumInstructionUTF8BytesPerQuestion = maximumInstructionUTF8BytesPerQuestion
        self.maximumTotalInstructionUTF8Bytes = maximumTotalInstructionUTF8Bytes
        self.maximumRenderedOptionsUTF8BytesPerQuestion = maximumRenderedOptionsUTF8BytesPerQuestion
        self.maximumTotalRenderedOptionsUTF8Bytes = maximumTotalRenderedOptionsUTF8Bytes
        self.maximumPackedUTF8BytesPerForward = maximumPackedUTF8BytesPerForward
        self.maximumTotalPackedUTF8Bytes = maximumTotalPackedUTF8Bytes
    }

    public static let httpDefault = WireSafetyLimits()
}

private struct RenderedQuestionPlan {
    let instructions: String
    let options: [String]
    let includesNullNoulForward: Bool

    func forwardStates(_ state: String) -> [String] {
        guard !options.isEmpty else { return [] }
        return includesNullNoulForward ? [state, ""] : [state]
    }
}

/// The metadata-free JSON adapter used at the Python-compatible HTTP boundary.
/// Parsing limits and duplicate-key rejection are applied before typed decoding.
public struct SystemOneWireCodec: Sendable {
    public let limits: OrderedJSONLimits
    public let safetyLimits: WireSafetyLimits

    public init(
        limits: OrderedJSONLimits = .httpDefault,
        safetyLimits: WireSafetyLimits = .httpDefault
    ) {
        self.limits = limits
        self.safetyLimits = safetyLimits
    }

    public func decodeRequest(_ data: Data) throws -> SystemOneRequest {
        let root = try OrderedJSONParser(limits: limits).parse(data)
        let envelope = try object(root, path: "$")
        let model: String
        if let value = member("model", in: envelope) {
            model = try string(value, path: "$.model")
        } else {
            model = "von-latest"
        }
        guard let stateValue = member("state", in: envelope) else {
            throw invalid("$.state", "field is required")
        }
        guard let questionsValue = member("questions", in: envelope) else {
            throw invalid("$.questions", "field is required")
        }
        let questionsObject = try object(questionsValue, path: "$.questions")
        let questions = try questionsObject.map { entry in
            QuestionEntry(
                id: entry.key,
                question: try decodeQuestion(
                    entry.value, path: "$.questions.\(pathComponent(entry.key))")
            )
        }
        let request = SystemOneRequest(
            model: model, state: try decodeState(stateValue), questions: questions)
        try validateSemanticBudget(request)
        return request
    }

    public func encodeResponse(_ response: SystemOneResponse) throws -> Data {
        try requireUnique(response.answers.map(\.id), path: "$.answers")
        let answers = try response.answers.map { entry in
            OrderedJSONMember(
                key: entry.id,
                value: try encodeAnswer(entry.answer, path: "$.answers.\(pathComponent(entry.id))"))
        }
        let root: OrderedJSON = .object([
            .init(key: "model", value: .string(response.model)),
            .init(key: "answers", value: .object(answers)),
            .init(
                key: "usage",
                value: .object([
                    .init(
                        key: "input_tokens",
                        value: .number(OrderedJSONNumber(response.usage.inputTokens))),
                    .init(
                        key: "output_tokens",
                        value: .number(OrderedJSONNumber(response.usage.outputTokens))),
                ])),
        ])
        return try OrderedJSONSerializer(limits: limits).serialize(root)
    }

    private func decodeState(_ value: OrderedJSON) throws -> SystemOneState {
        switch value {
        case .string(let text):
            return .text(text)
        case .object(let members):
            return .fields(
                try members.map { member in
                    StateField(
                        key: member.key,
                        value: try pythonString(
                            member.value, nested: false,
                            path: "$.state.\(pathComponent(member.key))")
                    )
                })
        default:
            return .text(try pythonString(value, nested: false, path: "$.state"))
        }
    }

    private func decodeQuestion(_ value: OrderedJSON, path: String) throws -> Question {
        let question = try object(value, path: path)
        let type: String
        if let typeValue = member("type", in: question) {
            type = try string(typeValue, path: "\(path).type")
        } else {
            type = "choice"
        }
        guard let instructionsValue = member("instructions", in: question) else {
            throw invalid("\(path).instructions", "field is required")
        }
        let instructions = try string(instructionsValue, path: "\(path).instructions")
        switch type {
        case "choice":
            guard let criteriaValue = member("criteria", in: question) else {
                throw invalid("\(path).criteria", "field is required for choice")
            }
            let criteriaObject = try object(criteriaValue, path: "\(path).criteria")
            guard !criteriaObject.isEmpty else {
                throw invalid("\(path).criteria", "choice criteria must not be empty")
            }
            let criteria = try criteriaObject.map { item -> ChoiceOption in
                switch item.value {
                case .null: return ChoiceOption(key: item.key)
                case .string(let description):
                    return ChoiceOption(key: item.key, description: description)
                default:
                    throw invalid(
                        "\(path).criteria.\(pathComponent(item.key))",
                        "choice descriptions must be strings or null")
                }
            }
            return .choice(Choice(instructions: instructions, criteria: criteria))
        case "noul":
            guard let criteriaValue = member("criteria", in: question) else {
                return .noul(Noul(instructions: instructions))
            }
            if case .null = criteriaValue {
                return .noul(Noul(instructions: instructions))
            }
            let criteriaObject = try object(criteriaValue, path: "\(path).criteria")
            var positive: String?
            var negative: String?
            for item in criteriaObject {
                let description = try string(
                    item.value, path: "\(path).criteria.\(pathComponent(item.key))")
                if item.key == "true" { positive = description }
                if item.key == "false" { negative = description }
            }
            return .noul(
                Noul(
                    instructions: instructions,
                    criteria: NoulCriteria(true: positive, false: negative)
                ))
        case "score":
            guard let criteriaValue = member("criteria", in: question) else {
                throw invalid("\(path).criteria", "field is required for score")
            }
            guard case .array(let values) = criteriaValue else {
                throw invalid("\(path).criteria", "score criteria must be an array")
            }
            guard !values.isEmpty else {
                throw invalid("\(path).criteria", "score criteria must not be empty")
            }
            let levels = try values.enumerated().map { index, value in
                try decodeScoreLevel(value, path: "\(path).criteria[\(index)]")
            }
            return .score(Score(instructions: instructions, criteria: levels))
        default:
            throw invalid("\(path).type", "unknown question type '\(type)'")
        }
    }

    private func decodeScoreLevel(_ value: OrderedJSON, path: String) throws -> ScoreLevel {
        if case .string(let text) = value { return .text(text) }
        let level = try object(value, path: path)
        let what: String
        if let value = member("what", in: level) {
            what = try pythonString(value, nested: false, path: "\(path).what")
        } else {
            what = ""
        }
        var examples: [String] = []
        if let value = member("examples", in: level), pythonTruthy(value) {
            switch value {
            case .array(let values):
                try requireScoreExampleCount(values.count)
                examples = try values.enumerated().map { index, value in
                    try string(value, path: "\(path).examples[\(index)]")
                }
            case .string(let text):
                // Python `", ".join(text)` iterates Unicode code points.
                let scalarCount = text.unicodeScalars.count
                try requireScoreExampleCount(scalarCount)
                examples = text.unicodeScalars.map(String.init)
            case .object(let members):
                // Iterating a Python dict yields its keys; values are irrelevant to join().
                try requireScoreExampleCount(members.count)
                examples = members.map(\.key)
            default:
                throw invalid(
                    "\(path).examples",
                    "truthy examples must be a string, array of strings, or object")
            }
        }
        return ScoreLevel(what: what, examples: examples)
    }

    private func requireScoreExampleCount(_ count: Int) throws {
        guard count <= safetyLimits.maximumScoreExamplesPerLevel else {
            throw SystemOneWireCodecError.safetyLimitExceeded(
                name: "score examples per level",
                limit: safetyLimits.maximumScoreExamplesPerLevel
            )
        }
    }

    private func validateSemanticBudget(_ request: SystemOneRequest) throws {
        try requireAtMost(request.questions.count, safetyLimits.maximumQuestions, name: "questions")
        let state = renderState(request.state)
        try requireAtMost(
            state.utf8.count, safetyLimits.maximumStateUTF8Bytes, name: "rendered state UTF-8 bytes"
        )

        var plannedForwards = 0
        var plannedMarkers = 0
        var totalInstructionBytes = 0
        var totalOptionBytes = 0
        var totalPackedBytes = 0

        for entry in request.questions {
            let plan = renderPlan(entry.question)
            let instructionBytes = plan.instructions.utf8.count
            try requireAtMost(
                instructionBytes,
                safetyLimits.maximumInstructionUTF8BytesPerQuestion,
                name: "instruction UTF-8 bytes per question"
            )
            totalInstructionBytes = try checkedAdd(
                totalInstructionBytes, instructionBytes, name: "total instruction UTF-8 bytes")
            try requireAtMost(
                totalInstructionBytes,
                safetyLimits.maximumTotalInstructionUTF8Bytes,
                name: "total instruction UTF-8 bytes"
            )
            try requireAtMost(
                plan.options.count, safetyLimits.maximumOptionsPerQuestion,
                name: "options per question")

            let optionBytes = try plan.options.reduce(0) { partial, option in
                try checkedAdd(
                    partial, option.utf8.count, name: "rendered option UTF-8 bytes per question")
            }
            try requireAtMost(
                optionBytes,
                safetyLimits.maximumRenderedOptionsUTF8BytesPerQuestion,
                name: "rendered option UTF-8 bytes per question"
            )
            totalOptionBytes = try checkedAdd(
                totalOptionBytes, optionBytes, name: "total rendered option UTF-8 bytes")
            try requireAtMost(
                totalOptionBytes,
                safetyLimits.maximumTotalRenderedOptionsUTF8Bytes,
                name: "total rendered option UTF-8 bytes"
            )

            for forwardState in plan.forwardStates(state) {
                plannedForwards = try checkedAdd(plannedForwards, 1, name: "planned forwards")
                plannedMarkers = try checkedAdd(
                    plannedMarkers, plan.options.count, name: "planned markers")
                let packed = VonTokenizer.packSequence(
                    state: forwardState,
                    question: plan.instructions,
                    options: plan.options
                ).precomposedStringWithCanonicalMapping
                guard occurrences(of: "[MASK]", in: packed) == plan.options.count else {
                    throw invalid(
                        "$.questions.\(pathComponent(entry.id))",
                        "literal [MASK] injection is not supported")
                }
                let packedBytes = packed.utf8.count
                // The pinned byte-level tokenizer emits at most one token per NFC UTF-8 byte,
                // plus its two template tokens. Keep this hard model cap even if policy is relaxed.
                let packedLimit = min(
                    safetyLimits.maximumPackedUTF8BytesPerForward,
                    VonModernBERT.maximumSequenceLength - 2
                )
                try requireAtMost(
                    packedBytes,
                    packedLimit,
                    name: "packed UTF-8 bytes per forward"
                )
                totalPackedBytes = try checkedAdd(
                    totalPackedBytes, packedBytes, name: "total packed UTF-8 bytes")
            }
        }

        try requireAtMost(
            plannedForwards, safetyLimits.maximumPlannedForwards, name: "planned forwards")
        try requireAtMost(
            plannedMarkers, safetyLimits.maximumPlannedMarkers, name: "planned markers")
        try requireAtMost(
            totalPackedBytes,
            safetyLimits.maximumTotalPackedUTF8Bytes,
            name: "total packed UTF-8 bytes"
        )
        try validateWorstCaseResponseBudget(request)
    }

    /// Request-controlled keys and labels can be repeated in the response and JSON
    /// control characters expand to six-byte escapes. Serialize a conservative
    /// successful response before inference so an admitted request cannot turn
    /// into a post-inference 500 solely because its response exceeds the body cap.
    private func validateWorstCaseResponseBudget(_ request: SystemOneRequest) throws {
        let maximumFiniteNumber = Double.greatestFiniteMagnitude
        let answers = request.questions.map { entry -> AnswerEntry in
            let answer: Answer
            switch entry.question {
            case .choice(let choice):
                let selected =
                    choice.criteria.max {
                        escapedJSONStringByteCount($0.key) < escapedJSONStringByteCount($1.key)
                    }?.key ?? ""
                answer = .choice(
                    ChoiceAnswer(
                        choice: selected,
                        probabilities: choice.criteria.map {
                            ProbabilityEntry(key: $0.key, probability: maximumFiniteNumber)
                        },
                        confidence: maximumFiniteNumber
                    ))
            case .noul:
                answer = .noul(NoulAnswer(noul: maximumFiniteNumber))
            case .score(let score):
                let descriptions = renderPlan(.score(score)).options
                let keys = descriptions.indices.map(String.init)
                answer = .score(
                    ScoreAnswer(
                        score: maximumFiniteNumber,
                        confidence: maximumFiniteNumber,
                        legend: zip(keys, descriptions).map {
                            LegendEntry(key: $0.0, description: $0.1)
                        },
                        probabilities: keys.map {
                            ProbabilityEntry(key: $0, probability: maximumFiniteNumber)
                        }
                    ))
            }
            return AnswerEntry(id: entry.id, answer: answer)
        }
        let response = SystemOneResponse(
            model: SystemOneEngine.modelID,
            answers: answers,
            usage: Usage(inputTokens: Int.max, outputTokens: Int.max)
        )
        do {
            _ = try encodeResponse(response)
        } catch OrderedJSONError.bodyTooLarge {
            throw SystemOneWireCodecError.safetyLimitExceeded(
                name: "worst-case serialized response bytes",
                limit: limits.maximumBodyBytes
            )
        }
    }

    private func escapedJSONStringByteCount(_ value: String) -> Int {
        value.unicodeScalars.reduce(2) { count, scalar in
            let added: Int
            switch scalar.value {
            case 0x22, 0x5C, 0x08, 0x0C, 0x0A, 0x0D, 0x09:
                added = 2
            case 0x00...0x1F:
                added = 6
            default:
                added = String(scalar).utf8.count
            }
            let (sum, overflow) = count.addingReportingOverflow(added)
            return overflow ? Int.max : sum
        }
    }

    private func renderState(_ state: SystemOneState) -> String {
        switch state {
        case .text(let text): return text
        case .fields(let fields):
            return fields.map { "\($0.key): \($0.value)" }.joined(separator: "\n")
        }
    }

    private func renderPlan(_ question: Question) -> RenderedQuestionPlan {
        switch question {
        case .choice(let choice):
            let options = choice.criteria.map { option in
                guard let description = option.description, !description.isEmpty else {
                    return pythonStrip(option.key)
                }
                return pythonStrip(description)
            }
            return RenderedQuestionPlan(
                instructions: choice.instructions, options: options, includesNullNoulForward: false)
        case .noul(let noul):
            let positive = noul.criteria?.trueDescription
            let negative = noul.criteria?.falseDescription
            let explicit = !(positive ?? "").isEmpty || !(negative ?? "").isEmpty
            return RenderedQuestionPlan(
                instructions: noul.instructions,
                options: [
                    positive.flatMap { $0.isEmpty ? nil : $0 } ?? "Yes, condition holds true.",
                    negative.flatMap { $0.isEmpty ? nil : $0 } ?? "No, condition is false.",
                ],
                includesNullNoulForward: !explicit
            )
        case .score(let score):
            let options = score.criteria.map { level in
                switch level {
                case .text(let text): return pythonStrip(text)
                case .described(let what, let examples):
                    let suffix =
                        examples.isEmpty ? "" : " Examples: \(examples.joined(separator: ", "))"
                    return pythonStrip("\(what)\(suffix)")
                }
            }
            return RenderedQuestionPlan(
                instructions: score.instructions, options: options, includesNullNoulForward: false)
        }
    }

    private func checkedAdd(_ lhs: Int, _ rhs: Int, name: String) throws -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw SystemOneWireCodecError.safetyLimitExceeded(name: name, limit: Int.max)
        }
        return sum
    }

    private func requireAtMost(_ value: Int, _ limit: Int, name: String) throws {
        guard value <= limit else {
            throw SystemOneWireCodecError.safetyLimitExceeded(name: name, limit: limit)
        }
    }

    private func occurrences(of needle: String, in text: String) -> Int {
        var count = 0
        var searchRange = text.startIndex..<text.endIndex
        while let range = text.range(of: needle, range: searchRange) {
            count += 1
            searchRange = range.upperBound..<text.endIndex
        }
        return count
    }

    private func pythonStrip(_ text: String) -> String {
        let scalars = text.unicodeScalars
        var lower = scalars.startIndex
        var upper = scalars.endIndex
        while lower < upper, isPythonWhitespace(scalars[lower]) { scalars.formIndex(after: &lower) }
        while lower < upper {
            let previous = scalars.index(before: upper)
            guard isPythonWhitespace(scalars[previous]) else { break }
            upper = previous
        }
        return String(scalars[lower..<upper])
    }

    private func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x0009...0x000D, 0x001C...0x0020, 0x0085, 0x00A0, 0x1680,
            0x2000...0x200A, 0x2028...0x2029, 0x202F, 0x205F, 0x3000:
            return true
        default:
            return false
        }
    }

    private func encodeAnswer(_ answer: Answer, path: String) throws -> OrderedJSON {
        switch answer {
        case .choice(let value):
            try requireFinite(value.confidence, path: "\(path).confidence")
            try requireUnique(value.probabilities.map(\.key), path: "\(path).probabilities")
            return .object([
                .init(key: "type", value: .string("choice")),
                .init(key: "choice", value: .string(value.choice)),
                .init(
                    key: "probabilities",
                    value: .object(
                        try value.probabilities.map { entry in
                            try requireFinite(
                                entry.probability,
                                path: "\(path).probabilities.\(pathComponent(entry.key))")
                            return .init(
                                key: entry.key,
                                value: .number(try OrderedJSONNumber(entry.probability)))
                        })),
                .init(key: "confidence", value: .number(try OrderedJSONNumber(value.confidence))),
            ])
        case .noul(let value):
            try requireFinite(value.noul, path: "\(path).noul")
            return .object([
                .init(key: "type", value: .string("noul")),
                .init(key: "noul", value: .number(try OrderedJSONNumber(value.noul))),
            ])
        case .score(let value):
            try requireFinite(value.score, path: "\(path).score")
            try requireFinite(value.confidence, path: "\(path).confidence")
            try requireUnique(value.legend.map(\.key), path: "\(path).legend")
            try requireUnique(value.probabilities.map(\.key), path: "\(path).probabilities")
            return .object([
                .init(key: "type", value: .string("score")),
                .init(key: "score", value: .number(try OrderedJSONNumber(value.score))),
                .init(key: "confidence", value: .number(try OrderedJSONNumber(value.confidence))),
                .init(
                    key: "legend",
                    value: .object(
                        value.legend.map {
                            .init(key: $0.key, value: .string($0.description))
                        })),
                .init(
                    key: "probabilities",
                    value: .object(
                        try value.probabilities.map { entry in
                            try requireFinite(
                                entry.probability,
                                path: "\(path).probabilities.\(pathComponent(entry.key))")
                            return .init(
                                key: entry.key,
                                value: .number(try OrderedJSONNumber(entry.probability)))
                        })),
            ])
        }
    }

    private func pythonString(_ value: OrderedJSON, nested: Bool, path: String) throws -> String {
        switch value {
        case .null: return "None"
        case .bool(let value): return value ? "True" : "False"
        case .number(let number): return try pythonNumber(number, path: path)
        case .string(let value): return nested ? pythonRepresentation(value) : value
        case .array(let values):
            return "["
                + (try values.enumerated().map {
                    try pythonString($0.element, nested: true, path: "\(path)[\($0.offset)]")
                }).joined(separator: ", ") + "]"
        case .object(let members):
            return "{"
                + (try members.map { member in
                    "\(pythonRepresentation(member.key)): \(try pythonString(member.value, nested: true, path: "\(path).\(pathComponent(member.key))"))"
                }).joined(separator: ", ") + "}"
        }
    }

    private func pythonNumber(_ number: OrderedJSONNumber, path: String) throws -> String {
        let raw = number.rawValue
        if !raw.contains(".") && !raw.contains("e") && !raw.contains("E") {
            if raw == "-0" { return "0" }
            return raw
        }
        guard let value = Double(raw), value.isFinite else {
            throw SystemOneWireCodecError.unsupportedState(
                path: path, message: "number is not a finite CPython float")
        }
        return String(value)
    }

    private func pythonTruthy(_ value: OrderedJSON) -> Bool {
        switch value {
        case .null: return false
        case .bool(let value): return value
        case .number(let number):
            guard let value = Double(number.rawValue) else { return true }
            return value != 0
        case .string(let value): return !value.isEmpty
        case .array(let values): return !values.isEmpty
        case .object(let members): return !members.isEmpty
        }
    }

    private func pythonRepresentation(_ value: String) -> String {
        let containsSingleQuote = value.unicodeScalars.contains { $0.value == 0x27 }
        let containsDoubleQuote = value.unicodeScalars.contains { $0.value == 0x22 }
        let quote: Unicode.Scalar = containsSingleQuote && !containsDoubleQuote ? "\"" : "'"
        var result = String(quote)
        for scalar in value.unicodeScalars {
            let code = scalar.value
            if scalar == quote || code == 0x5C {
                result += "\\"
                result.unicodeScalars.append(scalar)
            } else {
                switch code {
                case 0x09: result += "\\t"
                case 0x0A: result += "\\n"
                case 0x0D: result += "\\r"
                default:
                    if isPythonPrintable(scalar) {
                        result.unicodeScalars.append(scalar)
                    } else if code <= 0xFF {
                        result += String(format: "\\x%02x", code)
                    } else if code <= 0xFFFF {
                        result += String(format: "\\u%04x", code)
                    } else {
                        result += String(format: "\\U%08x", code)
                    }
                }
            }
        }
        result.unicodeScalars.append(quote)
        return result
    }

    private func isPythonPrintable(_ scalar: Unicode.Scalar) -> Bool {
        if scalar.value == 0x20 { return true }
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
            .nonspacingMark, .spacingMark, .enclosingMark,
            .decimalNumber, .letterNumber, .otherNumber,
            .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
            .initialPunctuation, .finalPunctuation, .otherPunctuation,
            .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol:
            return true
        default:
            return false
        }
    }

    private func object(_ value: OrderedJSON, path: String) throws -> [OrderedJSONMember] {
        guard case .object(let members) = value else { throw invalid(path, "expected object") }
        return members
    }

    private func string(_ value: OrderedJSON, path: String) throws -> String {
        guard case .string(let string) = value else { throw invalid(path, "expected string") }
        return string
    }

    private func member(_ key: String, in members: [OrderedJSONMember]) -> OrderedJSON? {
        members.first { $0.key == key }?.value
    }

    private func requireUnique(_ keys: [String], path: String) throws {
        var seen = Set<String>()
        for key in keys where !seen.insert(key).inserted {
            throw SystemOneWireCodecError.invalidResponse(path: path, message: "duplicate key '\(key)'")
        }
    }

    private func requireFinite(_ value: Double, path: String) throws {
        guard value.isFinite else {
            throw SystemOneWireCodecError.invalidResponse(path: path, message: "non-finite number")
        }
    }

    private func invalid(_ path: String, _ message: String) -> SystemOneWireCodecError {
        .invalidRequest(path: path, message: message)
    }

    private func pathComponent(_ value: String) -> String {
        value.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
            ? value
            : "['\(value.replacingOccurrences(of: "'", with: "\\'"))']"
    }
}
