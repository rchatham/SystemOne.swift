import XCTest
@testable import SystemOne

final class DecisionMathTests: XCTestCase {
    func testSoftmaxDividesByTemperatureAndIsShiftInvariant() {
        let p = DecisionMath.softmax([1, 2, 3], temperature: 2)
        let e = [exp(Float(0.5)), exp(Float(1)), exp(Float(1.5))]
        let total = e.reduce(0, +)
        for (actual, expected) in zip(p, e.map { $0 / total }) {
            XCTAssertEqual(actual, expected, accuracy: 1e-7)
        }
        XCTAssertEqual(DecisionMath.softmax([1001, 1002, 1003], temperature: 2), p)
        XCTAssertEqual(DecisionMath.softmax([0]), [1])
    }

    func testArgmaxPrefersTheLowestIndexOnTies() {
        XCTAssertEqual(DecisionMath.argmax([0.2, 0.4, 0.4]), 1)
        XCTAssertEqual(DecisionMath.argmax([0.5, 0.5]), 0)
        XCTAssertEqual(DecisionMath.argmax([0.1, 0.2, 0.7]), 2)
    }

    func testAnswersFromDistributions() {
        XCTAssertEqual(DecisionMath.noulAnswer(probabilities: [0.25, 0.75]), NoulAnswer(noul: 0.75))

        let choice = DecisionMath.choiceAnswer(keys: ["a", "b", "c"], probabilities: [0.25, 0.5, 0.25])
        XCTAssertEqual(choice.choice, "b")
        XCTAssertEqual(choice.confidence, 0.5)
        XCTAssertEqual(choice.probabilities.map(\.key), ["a", "b", "c"])

        let score = DecisionMath.scoreAnswer(legend: ["low", "mid", "high"], probabilities: [0.25, 0.25, 0.5])
        XCTAssertEqual(score.score, 1.25, accuracy: 1e-12)
        XCTAssertEqual(score.confidence, 0.5)
        XCTAssertEqual(score.legend, [
            LegendEntry(key: "0", description: "low"),
            LegendEntry(key: "1", description: "mid"),
            LegendEntry(key: "2", description: "high"),
        ])
        XCTAssertEqual(score.probabilities.map(\.key), ["0", "1", "2"])
    }
}
