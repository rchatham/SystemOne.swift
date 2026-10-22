import Foundation
import SystemOne

/// How the pinned von backend renders states and question criteria into text.
public enum VonPrompt {
    public static let defaultTrueDescription = "Yes, condition holds true."
    public static let defaultFalseDescription = "No, condition is false."

    /// Text states pass through; ordered fields render as `key: value` lines.
    public static func formatState(_ state: SystemOneState) -> String {
        switch state {
        case .text(let text):
            return text
        case .fields(let fields):
            return fields.map { "\($0.key): \($0.value)" }.joined(separator: "\n")
        }
    }

    /// Uses the stripped key only when a description is `nil` or exactly empty.
    public static func renderChoiceOptions(_ options: [ChoiceOption]) -> [String] {
        options.map { option in
            guard let description = option.description, !description.isEmpty else {
                return PythonSemantics.pythonStrip(option.key)
            }
            return PythonSemantics.pythonStrip(description)
        }
    }

    /// Either non-empty side (including whitespace) is explicit and disables
    /// null-state debiasing; each missing or empty side gets its own default.
    public static func renderNoulCriteria(_ criteria: NoulCriteria?) -> (
        descriptions: [String],
        hasExplicitCriteria: Bool
    ) {
        let positive = criteria?.trueDescription
        let negative = criteria?.falseDescription
        let hasExplicitCriteria = !(positive ?? "").isEmpty || !(negative ?? "").isEmpty
        return (
            [
                positive.flatMap { $0.isEmpty ? nil : $0 } ?? defaultTrueDescription,
                negative.flatMap { $0.isEmpty ? nil : $0 } ?? defaultFalseDescription,
            ],
            hasExplicitCriteria
        )
    }

    /// Renders `{what, examples}` levels with the exact `" Examples: "` and `", "` separators.
    public static func renderScoreLevels(_ levels: [ScoreLevel]) -> [String] {
        levels.map { level in
            switch level {
            case .text(let text):
                return PythonSemantics.pythonStrip(text)
            case .described(let what, let examples):
                let suffix = examples.isEmpty ? "" : " Examples: \(examples.joined(separator: ", "))"
                return PythonSemantics.pythonStrip("\(what)\(suffix)")
            }
        }
    }
}
