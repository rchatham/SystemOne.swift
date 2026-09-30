import Foundation
import XCTest
@testable import VonSwift

final class VonDecisionMathTests: XCTestCase {
    func testGoldenCalibrationAndDecisionsForAllFourCasesAndFiveForwards() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.cases.count, 4)
        XCTAssertEqual(fixture.forwards.count, 5)
        XCTAssertEqual(fixture.forwards.map(\.rawLogits.count), [4, 2, 2, 2, 4])

        for testCase in fixture.cases {
            let primary = fixture.forwards[testCase.forwardIndices[0]].rawLogits.map(Float.init)
            let actualCalibration: VonCalibrationResult

            switch testCase.primitive {
            case "choice":
                let result = try VonDecisionMath.choice(
                    logits: primary,
                    stateTokenCount: testCase.calibration.stateTokenCount
                )
                actualCalibration = result.calibration
                XCTAssertEqual(result.selectedIndex, testCase.postprocess.selectedOptionIndex)
                XCTAssertEqual(result.roundedProbabilities, answerProbabilities(testCase))
                XCTAssertEqual(result.confidence, answerNumber(testCase, key: "confidence"))

            case "noul":
                let nullLogits = testCase.forwardIndices.count == 2
                    ? fixture.forwards[testCase.forwardIndices[1]].rawLogits.map(Float.init)
                    : nil
                let result = try VonDecisionMath.noul(
                    logits: primary,
                    nullLogits: nullLogits,
                    stateTokenCount: testCase.calibration.stateTokenCount
                )
                actualCalibration = result.calibration
                XCTAssertEqual(result.debiasApplied, testCase.forwardIndices.count == 2)
                XCTAssertEqual(result.probabilityTrue, answerNumber(testCase, key: "noul"))

            case "score":
                let result = try VonDecisionMath.score(
                    logits: primary,
                    stateTokenCount: testCase.calibration.stateTokenCount
                )
                actualCalibration = result.calibration
                XCTAssertEqual(result.roundedProbabilities, answerProbabilities(testCase))
                XCTAssertEqual(result.score, answerNumber(testCase, key: "score"))
                XCTAssertEqual(result.confidence, answerNumber(testCase, key: "confidence"))

            default:
                XCTFail("Unexpected primitive \(testCase.primitive)")
                continue
            }

            assertCalibration(actualCalibration, equals: testCase.calibration, caseID: testCase.id)
        }
    }

    func testSoftmaxTemperatureClampsOverridesAndFirstMaximumTie() throws {
        XCTAssertEqual(try VonDecisionMath.softmax([1_000, 1_000]), [0.5, 0.5])
        let saturated = try VonDecisionMath.softmax([1_000, -1_000])
        XCTAssertEqual(saturated[0], 1)
        XCTAssertEqual(saturated[1], 0)

        let lower = try VonDecisionMath.calibrate(logits: [0, 0], stateTokenCount: 1)
        XCTAssertEqual(lower.effectiveTemperature, 0.3)
        let upper = try VonDecisionMath.calibrate(logits: [0], stateTokenCount: Int.max)
        XCTAssertEqual(upper.effectiveTemperature, 12.0)
        XCTAssertEqual(upper.entropyNumerator, 0)
        XCTAssertEqual(upper.entropyNormalizer, 1)
        XCTAssertEqual(upper.normalizedEntropy, 0)
        XCTAssertEqual(upper.probabilities, [1])

        let overridden = try VonDecisionMath.calibrate(
            logits: [1, 0],
            stateTokenCount: 1,
            temperatureOverride: -2
        )
        XCTAssertEqual(overridden.effectiveTemperature, -2)
        XCTAssertEqual(overridden.scaledLogits, [10_000, 0])
        XCTAssertEqual(overridden.probabilities, [1, 0])

        let tie = try VonDecisionMath.choice(logits: [3, 3, 2], stateTokenCount: 1)
        XCTAssertEqual(tie.selectedIndex, 0, "argmax must keep the first maximum")

        // Pinned Torch 2.14 CPU boundary: softmax multiplies by one float32
        // reciprocal instead of dividing each lane independently. The latter
        // produces 0.0845500007 and rounds to the wrong fourth decimal.
        let boundary = try VonDecisionMath.choice(
            logits: [3.884742, -17.598158],
            stateTokenCount: 100
        )
        XCTAssertEqual(boundary.calibration.probabilities[1], Float(0.08454999327659607))
        XCTAssertEqual(boundary.roundedProbabilities[1], 0.0845)
    }

    func testDefaultNoulUsesFloat32AndRoundingUsesNearestEven() throws {
        let result = try VonDecisionMath.noul(
            logits: [8.395277976989746, -3.166865587234497],
            nullLogits: [-7.495823860168457, -0.5461345911026001],
            stateTokenCount: 15
        )
        XCTAssertEqual(result.calibration.inputLogits[0], Float(13.26006031036377))
        XCTAssertEqual(result.calibration.inputLogits[1], Float(-3.166865587234497))

        XCTAssertEqual(VonDecisionMath.rounded(1.25, places: 1), 1.2)
        XCTAssertEqual(VonDecisionMath.rounded(1.75, places: 1), 1.8)
        XCTAssertEqual(VonDecisionMath.rounded(2.5, places: 0), 2)
        XCTAssertEqual(VonDecisionMath.rounded(3.5, places: 0), 4)
    }

    func testDecisionMathRejectsInvalidInputs() {
        XCTAssertThrowsError(try VonDecisionMath.softmax([])) {
            XCTAssertEqual($0 as? VonDecisionMathError, .emptyLogits)
        }
        XCTAssertThrowsError(try VonDecisionMath.softmax([0, .infinity])) {
            XCTAssertEqual(
                $0 as? VonDecisionMathError,
                .nonFiniteLogit(index: 1, value: .infinity)
            )
        }
        XCTAssertThrowsError(
            try VonDecisionMath.calibrate(logits: [0], stateTokenCount: 0)
        ) {
            XCTAssertEqual($0 as? VonDecisionMathError, .invalidStateTokenCount(0))
        }
        XCTAssertThrowsError(
            try VonDecisionMath.calibrate(
                logits: [0],
                stateTokenCount: 1,
                temperatureOverride: .nan
            )
        ) { error in
            guard case .invalidTemperature(let value) = error as? VonDecisionMathError else {
                return XCTFail("Unexpected error \(error)")
            }
            XCTAssertTrue(value.isNaN)
        }
        XCTAssertThrowsError(
            try VonDecisionMath.noul(logits: [0], stateTokenCount: 1)
        ) {
            XCTAssertEqual($0 as? VonDecisionMathError, .noulRequiresTwoLogits(actual: 1))
        }
        XCTAssertThrowsError(
            try VonDecisionMath.noul(
                logits: [0, 1],
                nullLogits: [0, 1, 2],
                stateTokenCount: 1
            )
        ) {
            XCTAssertEqual($0 as? VonDecisionMathError, .noulRequiresTwoLogits(actual: 3))
        }
    }
}

private extension VonDecisionMathTests {
    struct GoldenFixture: Decodable {
        let cases: [GoldenCase]
        let forwards: [GoldenForward]
    }

    struct GoldenCase: Decodable {
        let id: String
        let primitive: String
        let orderedOptions: [GoldenOption]
        let forwardIndices: [Int]
        let calibration: GoldenCalibration
        let postprocess: GoldenPostprocess
        let answerJSON: String

        enum CodingKeys: String, CodingKey {
            case id
            case primitive
            case orderedOptions = "ordered_options"
            case forwardIndices = "forward_indices"
            case calibration
            case postprocess
            case answerJSON = "answer_model_dump_json"
        }
    }

    struct GoldenOption: Decodable {
        let key: String
    }

    struct GoldenForward: Decodable {
        let rawLogits: [Double]

        enum CodingKeys: String, CodingKey {
            case rawLogits = "raw_logits"
        }
    }

    struct GoldenCalibration: Decodable {
        let inputLogits: [Double]
        let unscaledProbabilities: [Double]
        let entropyNumerator: Double
        let entropyNormalizer: Double
        let normalizedEntropy: Double
        let stateTokenCount: Int
        let rawTemperature: Double
        let effectiveTemperature: Double
        let scaledLogits: [Double]
        let probabilities: [Double]

        enum CodingKeys: String, CodingKey {
            case inputLogits = "input_logits"
            case unscaledProbabilities = "unscaled_probabilities"
            case entropyNumerator = "entropy_numerator"
            case entropyNormalizer = "entropy_normalizer"
            case normalizedEntropy = "normalized_entropy"
            case stateTokenCount = "state_token_count"
            case rawTemperature = "raw_temperature"
            case effectiveTemperature = "effective_temperature"
            case scaledLogits = "scaled_logits"
            case probabilities
        }
    }

    struct GoldenPostprocess: Decodable {
        let selectedOptionIndex: Int?

        enum CodingKeys: String, CodingKey {
            case selectedOptionIndex = "selected_option_index"
        }
    }

    func loadFixture() throws -> GoldenFixture {
        try JSONDecoder().decode(
            GoldenFixture.self,
            from: Data(contentsOf: fixtureURL("von_golden_v1.json"))
        )
    }

    func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("testdata/\(name)", isDirectory: false)
    }

    func answer(_ testCase: GoldenCase) -> [String: Any] {
        let data = testCase.answerJSON.data(using: .utf8)!
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    func answerNumber(_ testCase: GoldenCase, key: String) -> Double {
        answer(testCase)[key] as! Double
    }

    func answerProbabilities(_ testCase: GoldenCase) -> [Double] {
        let probabilities = answer(testCase)["probabilities"] as! [String: Double]
        return testCase.orderedOptions.map { probabilities[$0.key]! }
    }

    func assertCalibration(
        _ actual: VonCalibrationResult,
        equals expected: GoldenCalibration,
        caseID: String
    ) {
        assertFloatArray(actual.inputLogits, equals: expected.inputLogits, accuracy: 0, "\(caseID) input")
        assertFloatArray(
            actual.unscaledProbabilities,
            equals: expected.unscaledProbabilities,
            accuracy: 2e-7,
            "\(caseID) unscaled"
        )
        XCTAssertEqual(
            actual.entropyNumerator,
            Float(expected.entropyNumerator),
            accuracy: 2e-7,
            "\(caseID) entropy numerator"
        )
        XCTAssertEqual(actual.entropyNormalizer, expected.entropyNormalizer, accuracy: 1e-15)
        XCTAssertEqual(actual.normalizedEntropy, expected.normalizedEntropy, accuracy: 3e-7)
        XCTAssertEqual(actual.stateTokenCount, expected.stateTokenCount)
        XCTAssertEqual(actual.rawTemperature, expected.rawTemperature, accuracy: 2e-6)
        XCTAssertEqual(actual.effectiveTemperature, expected.effectiveTemperature, accuracy: 2e-6)
        assertFloatArray(
            actual.scaledLogits,
            equals: expected.scaledLogits,
            accuracy: 2e-6,
            "\(caseID) scaled"
        )
        assertFloatArray(
            actual.probabilities,
            equals: expected.probabilities,
            accuracy: 2e-7,
            "\(caseID) probabilities"
        )
    }

    func assertFloatArray(
        _ actual: [Float],
        equals expected: [Double],
        accuracy: Float,
        _ message: String
    ) {
        XCTAssertEqual(actual.count, expected.count, message)
        for index in actual.indices {
            XCTAssertEqual(actual[index], Float(expected[index]), accuracy: accuracy, "\(message)[\(index)]")
        }
    }
}
