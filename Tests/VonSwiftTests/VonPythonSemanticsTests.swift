import XCTest
@testable import VonSwift

final class VonPythonSemanticsTests: XCTestCase {
    func testOrderedStateAndPythonCharacterCounting() {
        XCTAssertEqual(
            VonPythonSemantics.formatState(.fields([
                VonStateField(key: "z", value: "first"),
                VonStateField(key: "a", value: "second"),
            ])),
            "z: first\na: second"
        )
        XCTAssertEqual(VonPythonSemantics.characterCount("é"), 1)
        XCTAssertEqual(VonPythonSemantics.characterCount("e\u{0301}"), 2)
        XCTAssertEqual(VonPythonSemantics.characterCount("👨‍👩‍👧‍👦"), 7)
    }

    func testChoiceFallbackMatchesPythonTruthinessBeforeStrip() {
        let rendered = VonPythonSemantics.renderChoiceOptions([
            ChoiceOption(key: "  alpha  ", description: nil),
            ChoiceOption(key: "  beta  ", description: ""),
            ChoiceOption(key: "gamma", description: " \t\n "),
            ChoiceOption(key: "delta", description: "  custom  "),
        ])
        XCTAssertEqual(rendered, ["alpha", "beta", "", "custom"])
    }

    func testNoulPartialCriteriaAndWhitespaceTruthiness() {
        let defaults = VonPythonSemantics.renderNoulCriteria(nil)
        XCTAssertFalse(defaults.hasExplicitCriteria)
        XCTAssertEqual(defaults.descriptions, [
            VonPythonSemantics.defaultTrueDescription,
            VonPythonSemantics.defaultFalseDescription,
        ])

        let empty = VonPythonSemantics.renderNoulCriteria(NoulCriteria(true: "", false: ""))
        XCTAssertFalse(empty.hasExplicitCriteria)
        XCTAssertEqual(empty.descriptions, defaults.descriptions)

        let partialTrue = VonPythonSemantics.renderNoulCriteria(NoulCriteria(true: "custom"))
        XCTAssertTrue(partialTrue.hasExplicitCriteria)
        XCTAssertEqual(partialTrue.descriptions, ["custom", VonPythonSemantics.defaultFalseDescription])

        let partialFalse = VonPythonSemantics.renderNoulCriteria(NoulCriteria(false: "custom"))
        XCTAssertTrue(partialFalse.hasExplicitCriteria)
        XCTAssertEqual(partialFalse.descriptions, [VonPythonSemantics.defaultTrueDescription, "custom"])

        let whitespace = VonPythonSemantics.renderNoulCriteria(NoulCriteria(true: "   "))
        XCTAssertTrue(whitespace.hasExplicitCriteria)
        XCTAssertEqual(whitespace.descriptions, ["   ", VonPythonSemantics.defaultFalseDescription])
    }

    func testScoreLevelRenderingMatchesPython() {
        XCTAssertEqual(
            VonPythonSemantics.renderScoreLevels([
                .text("  Low  "),
                ScoreLevel(what: "Medium"),
                ScoreLevel(what: "High", examples: ["outage", "data loss"]),
                ScoreLevel(what: "", examples: ["example"]),
            ]),
            ["Low", "Medium", "High Examples: outage, data loss", "Examples: example"]
        )
    }
}
