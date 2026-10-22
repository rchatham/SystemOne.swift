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

/// von's prompt shape and Python-server metadata, without loading weights.
public struct VonModelProfile: ModelProfile {
    public init() {}

    public var modelID: String { VonBackend.modelID }
    public var defaultModelAlias: String { VonBackend.defaultModelAlias }
    public var markerText: String? { "[MASK]" }

    /// The byte-level tokenizer emits at most one token per NFC UTF-8 byte,
    /// plus its two template tokens.
    public var maximumForwardUTF8Bytes: Int { VonModernBERT.maximumSequenceLength - 2 }

    public func renderState(_ state: SystemOneState) -> String {
        VonPrompt.formatState(state)
    }

    public func sketch(_ question: Question, renderedState: String) -> PromptSketch {
        let options: [String]
        var states = [renderedState]
        switch question {
        case .choice(let choice):
            options = VonPrompt.renderChoiceOptions(choice.criteria)
        case .noul(let noul):
            let rendered = VonPrompt.renderNoulCriteria(noul.criteria)
            options = rendered.descriptions
            // Default Noul adds a context-free null pass for debiasing.
            if !rendered.hasExplicitCriteria { states.append("") }
        case .score(let score):
            options = VonPrompt.renderScoreLevels(score.criteria)
        }
        let forwards = options.isEmpty ? [] : states.map {
            VonTokenizer.packSequence(state: $0, question: question.instructions, options: options)
        }
        return PromptSketch(instructions: question.instructions, options: options, forwards: forwards)
    }

    public var healthJSON: String {
        #"{"status":"ok","service":"von-decision-server","version":"1.1.0","engine":"von-1.1","homage":"John von Neumann & Ludwig von Mises"}"#
    }

    public var modelsJSON: String {
        #"{"models":[{"name":"von-latest","description":"Current Von System One decision model","release_date":"2026-09-21"},{"name":"von-1.1.0","description":"Von 1.1 stable release","release_date":"2026-09-21"},{"name":"jev-latest","description":"TypeSafe Jev compatibility alias","release_date":"2026-09-21"}],"object":"list","data":[{"id":"von-latest","object":"model","owned_by":"von"},{"id":"von-1.1.0","object":"model","owned_by":"von"},{"id":"jev-latest","object":"model","owned_by":"von"}]}"#
    }
}
