import XCTest
@testable import SystemOne

final class PythonSemanticsTests: XCTestCase {
    func testOrderedStateAndPythonCharacterCounting() {
        XCTAssertEqual(
            PythonSemantics.formatState(.fields([
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
        let rendered = PythonSemantics.renderChoiceOptions([
            ChoiceOption(key: "  alpha  ", description: nil),
            ChoiceOption(key: "  beta  ", description: ""),
            ChoiceOption(key: "gamma", description: " \t\n "),
            ChoiceOption(key: "delta", description: "  custom  "),
        ])
        XCTAssertEqual(rendered, ["alpha", "beta", "", "custom"])
    }

    func testNoulPartialCriteriaAndWhitespaceTruthiness() {
        let defaults = PythonSemantics.renderNoulCriteria(nil)
        XCTAssertFalse(defaults.hasExplicitCriteria)
        XCTAssertEqual(defaults.descriptions, [
            PythonSemantics.defaultTrueDescription,
            PythonSemantics.defaultFalseDescription,
        ])

        let empty = PythonSemantics.renderNoulCriteria(NoulCriteria(true: "", false: ""))
        XCTAssertFalse(empty.hasExplicitCriteria)
        XCTAssertEqual(empty.descriptions, defaults.descriptions)

        let partialTrue = PythonSemantics.renderNoulCriteria(NoulCriteria(true: "custom"))
        XCTAssertTrue(partialTrue.hasExplicitCriteria)
        XCTAssertEqual(partialTrue.descriptions, ["custom", PythonSemantics.defaultFalseDescription])

        let partialFalse = PythonSemantics.renderNoulCriteria(NoulCriteria(false: "custom"))
        XCTAssertTrue(partialFalse.hasExplicitCriteria)
        XCTAssertEqual(partialFalse.descriptions, [PythonSemantics.defaultTrueDescription, "custom"])

        let whitespace = PythonSemantics.renderNoulCriteria(NoulCriteria(true: "   "))
        XCTAssertTrue(whitespace.hasExplicitCriteria)
        XCTAssertEqual(whitespace.descriptions, ["   ", PythonSemantics.defaultFalseDescription])
    }

    func testScoreLevelRenderingMatchesPython() {
        XCTAssertEqual(
            PythonSemantics.renderScoreLevels([
                .text("  Low  "),
                ScoreLevel(what: "Medium"),
                ScoreLevel(what: "High", examples: ["outage", "data loss"]),
                ScoreLevel(what: "", examples: ["example"]),
            ]),
            ["Low", "Medium", "High Examples: outage, data loss", "Examples: example"]
        )
    }
}
