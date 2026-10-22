import Foundation
import SystemOne

/// What accompanies the text of a d1 request. It selects the option wording,
/// the text budget, and whether text temperatures apply, as `modeling_d1.py` does.
public enum D1PromptMode: Equatable, Sendable {
    case text
    case image
    case audio
}

/// One question encoded over one state: token ids and each option's marker position.
public struct D1EncodedQuestion: Equatable, Sendable {
    public let ids: [Int]
    public let markers: [Int]
}

/// Errors raised while encoding a question that cannot fit its context.
public enum D1PromptError: Error, Equatable, Sendable {
    case optionsDoNotFit(questionID: String)
    case mediaTooLong(prefixLength: Int, maximum: Int)
}

extension D1PromptError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .optionsDoNotFit(let id):
            return "The options of question '\(id)' do not fit in the context."
        case .mediaTooLong(let prefixLength, let maximum):
            return "The media take \(prefixLength) of the \(maximum) positions; send fewer images."
        }
    }
}

/// `prompt.py`: questions in, token sequences out.
///
///     <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> ... <decide>
public enum D1Prompt {
    /// Option tokens each option may use before the shared budget is split.
    static let tokensPerOption = 24

    /// `<|name|>` becomes `<¦name¦>` so caller text can never emit a delimiter.
    public static func escape(_ text: String) -> String {
        guard text.contains("<|") else { return text }
        var out = ""
        var rest = Substring(text)
        while let open = rest.range(of: "<|") {
            out += rest[..<open.lowerBound]
            let afterOpen = rest[open.upperBound...]
            let name = afterOpen.prefix { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_") }
            if !name.isEmpty, afterOpen.dropFirst(name.count).hasPrefix("|>") {
                out += "<¦\(name)¦>"
                rest = afterOpen.dropFirst(name.count + 2)
            } else {
                out += "<|"
                rest = afterOpen
            }
        }
        return out + rest
    }

    /// A string passes through; ordered fields become a JSON object, matching
    /// `json.dumps(state, ensure_ascii=False)` for string values.
    public static func serialize(_ state: SystemOneState) -> String {
        switch state {
        case .text(let text):
            return text
        case .fields(let fields):
            let members = fields.map {
                "\(PythonSemantics.jsonString($0.key)): \(PythonSemantics.jsonString($0.value))"
            }
            return "{" + members.joined(separator: ", ") + "}"
        }
    }

    /// `_criterion`: strings pass through; a `{what, examples}` level is JSON.
    static func criterion(_ level: ScoreLevel) -> String {
        switch level {
        case .text(let text):
            return text
        case .described(let what, let examples):
            var members = ["\"what\": \(PythonSemantics.jsonString(what))"]
            if !examples.isEmpty {
                members.append("\"examples\": [\(examples.map(PythonSemantics.jsonString).joined(separator: ", "))]")
            }
            return "{" + members.joined(separator: ", ") + "}"
        }
    }

    /// Option texts in the model's order; a Noul is read as `[false, true]`.
    public static func renderOptions(_ question: Question, mode: D1PromptMode) -> [String] {
        switch question {
        case .choice(let choice):
            return choice.criteria.enumerated().map { index, option in
                let description = option.description.flatMap { $0.isEmpty ? nil : $0 }
                if mode == .audio {
                    return String(format: "option_%03d: ", index) + (description ?? option.key)
                }
                return description.map { "\(option.key): \($0)" } ?? option.key
            }
        case .score(let score):
            return score.criteria.enumerated().map { "level \($0.offset): \(criterion($0.element))" }
        case .noul(let noul):
            if mode == .audio {
                return ["false: no", "true: yes"]
            }
            // Python reads `q.criteria or noul_default`: an absent or empty
            // criteria dict falls back to the image default of plain yes/no.
            let explicit = noul.criteria.map { $0.trueDescription != nil || $0.falseDescription != nil } ?? false
            let falseSide: String?
            let trueSide: String?
            if explicit {
                falseSide = noul.criteria?.falseDescription
                trueSide = noul.criteria?.trueDescription
            } else if mode == .image {
                falseSide = "no"
                trueSide = "yes"
            } else {
                falseSide = nil
                trueSide = nil
            }
            return [
                "false: " + (falseSide.flatMap { $0.isEmpty ? nil : $0 } ?? "no, the statement does not hold"),
                "true: " + (trueSide.flatMap { $0.isEmpty ? nil : $0 } ?? "yes, the statement holds"),
            ]
        }
    }

    /// Token ids of one question over one state and each option's marker position.
    ///
    /// The option block gets `max(96, min(24k + 32, maxLength / 2))` tokens,
    /// shared evenly; the state is truncated on the right to the room left.
    public static func encode(
        state: String,
        question: Question,
        instructions: String,
        id: String,
        maxLength: Int,
        mode: D1PromptMode,
        tokenizer: D1Tokenizer
    ) throws -> D1EncodedQuestion {
        let special = tokenizer.special
        let encode = { tokenizer.encode(escape($0)) }
        let options = renderOptions(question, mode: mode)
        let budget = max(96, min(options.count * tokensPerOption + 32, maxLength / 2))
        let perOption = max(2, (budget - 3 * options.count) / options.count)
        var block = Array(([special.question] + encode(instructions)).prefix(max(16, budget)))
        var markers: [Int] = []
        for text in options {
            markers.append(block.count + 1)
            block += [special.option, special.mask] + encode(" " + text).prefix(perOption) + [special.optionEnd]
        }
        block.append(special.decide)

        let room = max(0, maxLength - block.count - 2)
        let stateIDs = [special.state] + encode(state).prefix(room)
        let ids = Array(([special.bos] + stateIDs + block).prefix(maxLength))
        markers = markers.map { $0 + 1 + stateIDs.count }
        guard let last = markers.last, last < maxLength else {
            throw D1PromptError.optionsDoNotFit(questionID: id)
        }
        return D1EncodedQuestion(ids: ids, markers: markers)
    }

    /// `temperature_key`: the calibration bucket for a type and option count.
    static func temperatureKey(type: String, optionCount: Int) -> String {
        let bucket: String
        switch optionCount {
        case ...2: bucket = "2"
        case ...5: bucket = "3-5"
        case ...10: bucket = "6-10"
        default: bucket = "11+"
        }
        return "\(type):\(bucket)"
    }
}
