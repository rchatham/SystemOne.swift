import Foundation
import SystemOne

/// Errors raised while turning a request into `bare-v1` prompts.
public enum JevPromptError: Error, Equatable, Sendable {
    /// Even an empty state leaves no room under `maxLength` tokens.
    case promptTooLong(id: String, tokens: Int, maximum: Int)
}

extension JevPromptError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .promptTooLong(let id, let tokens, let maximum):
            return "Question \(id) needs \(tokens) tokens without its state; the JEV limit is \(maximum)."
        }
    }
}

/// The `bare-v1` template JEV models were trained on:
///
///     [kind] choice
///     [state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
///     [question] Supplier response for this scenario.
///     [options]
///     A) issue_warning
///     B) renegotiate
///     [decision]:
///
/// A noul's options are always `false`, `true`; a score's are `0` … `5`. The
/// decision is read at the final `:`.
public enum JevPrompt {
    /// Option labels of the 16 choice slots.
    static let choiceLabels = "ABCDEFGHIJKLMNOP".map(String.init)
    /// The maximum options a choice can have: one head slot per label.
    public static let maximumChoiceOptions = 16
    /// The fixed levels of a score.
    public static let scoreLevels = 6

    /// The full prompt for one question.
    public static func render(kind: JevKind, state: String, question: String, options: [String]) -> String {
        let lines = kind == .choice
            ? options.enumerated().map { "\(choiceLabels[$0.offset])) \($0.element)" }
            : options
        return "[kind] \(kind.rawValue)\n[state] \(state)\n[question] \(question)\n[options]\n"
            + lines.joined(separator: "\n") + "\n[decision]:"
    }

    /// A request's state as prompt text: text as is, fields as a JSON object
    /// with Python's `json.dumps(..., ensure_ascii=False)` separators.
    public static func serialize(_ state: SystemOneState) -> String {
        switch state {
        case .text(let text):
            return text
        case .fields(let fields):
            let members = fields.map { "\(PythonSemantics.jsonString($0.key)): \(PythonSemantics.jsonString($0.value))" }
            return "{" + members.joined(separator: ", ") + "}"
        }
    }

    /// A question's kind and the option lines its prompt lists. A choice
    /// option reads `key: description`, or just `key`.
    public static func options(for question: Question) -> (kind: JevKind, options: [String]) {
        switch question {
        case .noul:
            return (.noul, ["false", "true"])
        case .score:
            return (.score, (0..<scoreLevels).map(String.init))
        case .choice(let choice):
            return (.choice, choice.criteria.map { option in
                let description = option.description.flatMap { $0.isEmpty ? nil : $0 }
                return description.map { "\(option.key): \($0)" } ?? option.key
            })
        }
    }

    /// A score level as legend text: the text itself, or `what` with its examples.
    public static func criterion(_ level: ScoreLevel) -> String {
        switch level {
        case .text(let text):
            return text
        case .described(let what, let examples):
            return examples.isEmpty ? what : "\(what) (e.g. \(examples.joined(separator: "; ")))"
        }
    }

    /// The `[question]` line. JEV's options are fixed for nouls and scores, so
    /// what a noul's `true`/`false` or a score's levels mean must travel in the
    /// question text itself. A choice keeps its instructions as they are.
    ///
    /// The result must stay on one line: `\n[options]` follows it.
    public static func foldedQuestion(_ question: Question) -> String {
        switch question {
        case .choice(let choice):
            return choice.instructions
        case .noul(let noul):
            // TODO(human): fold `noul.criteria?.trueDescription` and
            // `falseDescription` into the instructions.
            return noul.instructions
        case .score(let score):
            // TODO(human): fold the six levels (see `criterion(_:)`) into the
            // instructions, so the model knows what 0 and 5 mean.
            return score.instructions
        }
    }
}
