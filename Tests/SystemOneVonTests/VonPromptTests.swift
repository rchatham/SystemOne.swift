import XCTest
@testable import SystemOne
@testable import SystemOneVon

final class VonPromptTests: XCTestCase {
    func testOrderedStateAndPythonCharacterCounting() {
        XCTAssertEqual(
            VonPrompt.formatState(.fields([
                StateField(key: "z", value: "first"),
                StateField(key: "a", value: "second"),
            ])),
            "z: first\na: second"
        )
        XCTAssertEqual(PythonSemantics.characterCount("é"), 1)
        XCTAssertEqual(PythonSemantics.characterCount("e\u{0301}"), 2)
        XCTAssertEqual(PythonSemantics.characterCount("👨‍👩‍👧‍👦"), 7)
    }

    func testChoiceFallbackMatchesPythonTruthinessBeforeStrip() {
        let rendered = VonPrompt.renderChoiceOptions([
            ChoiceOption(key: "  alpha  ", description: nil),
            ChoiceOption(key: "  beta  ", description: ""),
            ChoiceOption(key: "gamma", description: " \t\n "),
            ChoiceOption(key: "delta", description: "  custom  "),
        ])
        XCTAssertEqual(rendered, ["alpha", "beta", "", "custom"])
    }

    func testNoulPartialCriteriaAndWhitespaceTruthiness() {
        let defaults = VonPrompt.renderNoulCriteria(nil)
        XCTAssertFalse(defaults.hasExplicitCriteria)
        XCTAssertEqual(defaults.descriptions, [
            VonPrompt.defaultTrueDescription,
            VonPrompt.defaultFalseDescription,
        ])

        let empty = VonPrompt.renderNoulCriteria(NoulCriteria(true: "", false: ""))
        XCTAssertFalse(empty.hasExplicitCriteria)
        XCTAssertEqual(empty.descriptions, defaults.descriptions)

        let partialTrue = VonPrompt.renderNoulCriteria(NoulCriteria(true: "custom"))
        XCTAssertTrue(partialTrue.hasExplicitCriteria)
        XCTAssertEqual(partialTrue.descriptions, ["custom", VonPrompt.defaultFalseDescription])

        let partialFalse = VonPrompt.renderNoulCriteria(NoulCriteria(false: "custom"))
        XCTAssertTrue(partialFalse.hasExplicitCriteria)
        XCTAssertEqual(partialFalse.descriptions, [VonPrompt.defaultTrueDescription, "custom"])

        let whitespace = VonPrompt.renderNoulCriteria(NoulCriteria(true: "   "))
        XCTAssertTrue(whitespace.hasExplicitCriteria)
        XCTAssertEqual(whitespace.descriptions, ["   ", VonPrompt.defaultFalseDescription])
    }

    func testScoreLevelRenderingMatchesPython() {
        XCTAssertEqual(
            VonPrompt.renderScoreLevels([
                .text("  Low  "),
                ScoreLevel(what: "Medium"),
                ScoreLevel(what: "High", examples: ["outage", "data loss"]),
                ScoreLevel(what: "", examples: ["example"]),
            ]),
            ["Low", "Medium", "High Examples: outage, data loss", "Examples: example"]
        )
    }
}
