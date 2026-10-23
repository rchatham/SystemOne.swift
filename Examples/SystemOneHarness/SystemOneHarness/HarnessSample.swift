import Foundation
import SystemOne

/// Built-in question sets, one per input kind. Each mixes a Choice, a Noul,
/// and a Score so every answer type is exercised.
enum HarnessSample: String, CaseIterable, Hashable {
    case text
    case photo
    case voice

    var label: String {
        switch self {
        case .text: "Text triage"
        case .photo: "Photo"
        case .voice: "Voice note"
        }
    }

    /// The default state text; photo and voice requests often need none.
    var state: String {
        switch self {
        case .text:
            """
            Checkout error rate rose from 0.2% to 14% at 09:12 UTC, immediately after \
            deploy 4812 rolled out to the Tokyo region. Card authorizations are timing \
            out at the payment gateway; other regions are unaffected. On-call has paged \
            the payments team and is preparing a rollback.
            """
        case .photo: ""
        case .voice: "Voice note from a user."
        }
    }

    var questions: [QuestionEntry] {
        switch self {
        case .text:
            [
                choice("root_cause", "What most likely caused the incident?", [
                    ("infrastructure", "Server, network, or region failure"),
                    ("deploy", "A recent code or config change"),
                    ("third_party", "An external provider outage"),
                ]),
                noul("customer_impact", "Are customers currently unable to pay?"),
                score("severity", "How severe is the incident?", [
                    "Cosmetic, no customer impact",
                    "Minor degradation for a few customers",
                    "Major feature degraded in one region",
                    "Core revenue path down",
                ]),
            ]
        case .photo:
            [
                choice("scene", "Where was this photo taken?", [
                    ("indoors", "Inside a building"),
                    ("outdoors", "Outside"),
                    ("vehicle", "Inside a vehicle"),
                ]),
                noul("person", "Is there a person in the photo?"),
                score("light", "How bright is the photo?", ["Dark", "Dim", "Normal", "Bright"]),
            ]
        case .voice:
            [
                choice("intent", "What does the speaker want?", [
                    ("information", "To find something out"),
                    ("action", "To get something done"),
                    ("chat", "Just to talk"),
                ]),
                noul("question", "Is the speaker asking a question?"),
                score("urgency", "How urgent does the speaker sound?", ["Not urgent", "Somewhat", "Very urgent"]),
            ]
        }
    }

    static func describe(_ entry: AnswerEntry) -> String {
        switch entry.answer {
        case .choice(let answer):
            "\(entry.id): \(answer.choice) (\(percent(answer.confidence)))"
        case .noul(let answer):
            "\(entry.id): P(yes) = \(percent(answer.noul))"
        case .score(let answer):
            "\(entry.id): \(answer.score.formatted(.number.precision(.fractionLength(2)))) (\(percent(answer.confidence)))"
        }
    }

    private func choice(_ id: String, _ instructions: String, _ options: [(String, String)]) -> QuestionEntry {
        QuestionEntry(id: id, question: .choice(Choice(
            instructions: instructions,
            criteria: options.map { ChoiceOption(key: $0.0, description: $0.1) }
        )))
    }

    private func noul(_ id: String, _ instructions: String) -> QuestionEntry {
        QuestionEntry(id: id, question: .noul(Noul(instructions: instructions)))
    }

    private func score(_ id: String, _ instructions: String, _ levels: [String]) -> QuestionEntry {
        QuestionEntry(id: id, question: .score(Score(instructions: instructions, criteria: levels.map { .text($0) })))
    }

    private static func percent(_ value: Double) -> String {
        value.formatted(.percent.precision(.fractionLength(1)))
    }
}
