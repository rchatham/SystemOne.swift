import Foundation

enum VonPythonSemantics {
    static let defaultTrueDescription = "Yes, condition holds true."
    static let defaultFalseDescription = "No, condition is false."

    static func formatState(_ state: VonState) -> String {
        switch state {
        case .text(let text):
            return text
        case .fields(let fields):
            return fields.map { "\($0.key): \($0.value)" }.joined(separator: "\n")
        }
    }

    static func renderChoiceOptions(_ options: [ChoiceOption]) -> [String] {
        options.map { option in
            guard let description = option.description, !description.isEmpty else {
                return pythonStrip(option.key)
            }
            return pythonStrip(description)
        }
    }

    static func renderNoulCriteria(_ criteria: NoulCriteria?) -> (
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

    static func renderScoreLevels(_ levels: [ScoreLevel]) -> [String] {
        levels.map { level in
            switch level {
            case .text(let text):
                return pythonStrip(text)
            case .described(let what, let examples):
                let suffix = examples.isEmpty ? "" : " Examples: \(examples.joined(separator: ", "))"
                return pythonStrip("\(what)\(suffix)")
            }
        }
    }

    /// CPython `len(str)` counts Unicode scalar values, not Swift grapheme clusters.
    static func characterCount(_ text: String) -> Int {
        text.unicodeScalars.count
    }

    /// Matches CPython 3.12 `str.strip()` whitespace behavior.
    static func pythonStrip(_ text: String) -> String {
        let scalars = text.unicodeScalars
        var lowerBound = scalars.startIndex
        var upperBound = scalars.endIndex

        while lowerBound < upperBound, isPythonWhitespace(scalars[lowerBound]) {
            scalars.formIndex(after: &lowerBound)
        }
        while lowerBound < upperBound {
            let previous = scalars.index(before: upperBound)
            guard isPythonWhitespace(scalars[previous]) else { break }
            upperBound = previous
        }
        return String(scalars[lowerBound..<upperBound])
    }

    private static func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x0009...0x000D,
             0x001C...0x0020,
             0x0085,
             0x00A0,
             0x1680,
             0x2000...0x200A,
             0x2028...0x2029,
             0x202F,
             0x205F,
             0x3000:
            return true
        default:
            return false
        }
    }
}
