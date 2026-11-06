import Foundation
import SystemOne
import XCTest
@testable import SystemOneJev

final class JevPromptTests: XCTestCase {
    func testRendersTheModelCardTemplate() {
        // The exact string from the autotrust/JEV-9B model card.
        XCTAssertEqual(
            JevPrompt.render(
                kind: .choice,
                state: "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
                question: "Supplier response for this scenario.",
                options: ["issue_warning", "renegotiate", "dual_source", "maintain"]
            ),
            """
            [kind] choice
            [state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
            [question] Supplier response for this scenario.
            [options]
            A) issue_warning
            B) renegotiate
            C) dual_source
            D) maintain
            [decision]:
            """
        )
        XCTAssertEqual(
            JevPrompt.render(kind: .noul, state: "S", question: "Q?", options: ["false", "true"]),
            "[kind] noul\n[state] S\n[question] Q?\n[options]\nfalse\ntrue\n[decision]:"
        )
        let sixteen = JevPrompt.render(kind: .choice, state: "", question: "q", options: (0..<16).map { "o\($0)" })
        XCTAssertTrue(sixteen.hasSuffix("O) o14\nP) o15\n[decision]:"))
    }

    func testSerializesFieldsAsPythonJSON() {
        XCTAssertEqual(JevPrompt.serialize(.text("as is\n")), "as is\n")
        XCTAssertEqual(
            JevPrompt.serialize(.fields([StateField(key: "status", value: "degraded"),
                                         StateField(key: "quote", value: "say \"hi\" — 日本")])),
            #"{"status": "degraded", "quote": "say \"hi\" — 日本"}"#
        )
    }

    func testOptionsForEachKind() {
        XCTAssertEqual(JevPrompt.options(for: .noul(Noul(instructions: "q", criteria: nil))).options, ["false", "true"])
        XCTAssertEqual(
            JevPrompt.options(for: .score(Score(instructions: "q", criteria: (0..<6).map { .text("l\($0)") }))).options,
            ["0", "1", "2", "3", "4", "5"]
        )
        let choice = Question.choice(Choice(instructions: "q", criteria: [
            ChoiceOption(key: "billing", description: "Charges and refunds"),
            ChoiceOption(key: "fraud"),
            ChoiceOption(key: "other", description: ""),
        ]))
        XCTAssertEqual(JevPrompt.options(for: choice).kind, .choice)
        XCTAssertEqual(JevPrompt.options(for: choice).options, ["billing: Charges and refunds", "fraud", "other"])
    }

    func testCriterionText() {
        XCTAssertEqual(JevPrompt.criterion(.text("Blocking now")), "Blocking now")
        XCTAssertEqual(JevPrompt.criterion(.described(what: "Low", examples: [])), "Low")
        XCTAssertEqual(JevPrompt.criterion(.described(what: "High", examples: ["outage", "data loss"])),
                       "High (e.g. outage; data loss)")
    }

    func testFoldedQuestionsStayOnOneLine() {
        let questions: [Question] = [
            .choice(Choice(instructions: "Which team?", criteria: [ChoiceOption(key: "a"), ChoiceOption(key: "b")])),
            .noul(Noul(instructions: "Is it urgent?", criteria: NoulCriteria(true: "needs action today", false: "can wait"))),
            .score(Score(instructions: "How severe?", criteria: [
                .text("none"), .text("cosmetic"), .text("minor"), .text("major"),
                .described(what: "critical", examples: ["outage"]), .text("catastrophic"),
            ])),
        ]
        for question in questions {
            let folded = JevPrompt.foldedQuestion(question)
            XCTAssertFalse(folded.contains("\n"), folded)
            XCTAssertTrue(folded.hasPrefix(question.instructions), folded)
        }
        XCTAssertEqual(JevPrompt.foldedQuestion(questions[0]), "Which team?")
    }
}
