import Foundation
import SystemOne
import SystemOneVon

/// The harness's built-in incident triage request: one question of each kind,
/// including a default Noul that triggers the extra null-state pass.
enum HarnessSample {
    static let state = """
        Checkout error rate rose from 0.2% to 14% at 09:12 UTC, immediately after \
        deploy 4812 rolled out to the Tokyo region. Card authorizations are timing \
        out at the payment gateway; other regions are unaffected. On-call has paged \
        the payments team and is preparing a rollback.
        """

    static let questions: [QuestionEntry] = [
        QuestionEntry(
            id: "root_cause",
            question: .choice(Choice(
                instructions: SystemOneEngine.defaultChoiceInstructions,
                criteria: [
                    ChoiceOption(key: "infrastructure", description: "Server, network, or region failure"),
                    ChoiceOption(key: "deploy", description: "A recent code or config change"),
                    ChoiceOption(key: "third_party", description: "An external provider outage"),
                    ChoiceOption(key: "billing", description: "Payment or invoice issue"),
                ]
            ))
        ),
        QuestionEntry(
            id: "customer_impact",
            question: .noul(Noul(instructions: "Are customers currently unable to pay?"))
        ),
        QuestionEntry(
            id: "rollback_required",
            question: .noul(Noul(
                instructions: "Should the latest deploy be rolled back?",
                criteria: NoulCriteria(
                    true: "The deploy is the likely cause and rolling back is safe.",
                    false: "The deploy is unrelated or rollback would cause harm."
                )
            ))
        ),
        QuestionEntry(
            id: "severity",
            question: .score(Score(
                instructions: SystemOneEngine.defaultScoreInstructions,
                criteria: [
                    .text("SEV4: cosmetic, no customer impact"),
                    .text("SEV3: minor degradation for a few customers"),
                    .text("SEV2: major feature degraded in one region"),
                    .text("SEV1: core revenue path down"),
                ]
            ))
        ),
    ]

    static func describe(_ entry: AnswerEntry) -> String {
        switch entry.answer {
        case .choice(let answer):
            return "\(entry.id): \(answer.choice) (confidence \(percent(answer.confidence)))"
        case .noul(let answer):
            return "\(entry.id): P(true) = \(percent(answer.noul))"
        case .score(let answer):
            return "\(entry.id): \(answer.score.formatted()) (confidence \(percent(answer.confidence)))"
        }
    }

    private static func percent(_ value: Double) -> String {
        value.formatted(.percent.precision(.fractionLength(1)))
    }
}
