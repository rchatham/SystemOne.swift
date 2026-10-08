import Foundation
import MLX
import XCTest
@testable import VonSwift

final class VonOptionMarkerModelTests: XCTestCase {
    func testMarkerValidationRejectsMissingOutOfRangeAndWrongTokens() throws {
        XCTAssertThrowsError(
            try VonOptionMarkerModel.validateMaskPositions([], inputIDs: [50_281, 50_282])
        ) { XCTAssertEqual($0 as? VonOptionMarkerModelError, .emptyMaskPositions) }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.validateMaskPositions([2], inputIDs: [50_281, 50_282])
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .invalidMaskPosition(index: 0, value: 2, sequenceLength: 2)
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.validateMaskPositions([-1], inputIDs: [50_281, 50_282])
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .invalidMaskPosition(index: 0, value: -1, sequenceLength: 2)
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.validateMaskPositions([1], inputIDs: [50_281, 50_282])
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .markerTokenMismatch(position: 1, tokenID: 50_282)
            )
        }
        XCTAssertNoThrow(
            try VonOptionMarkerModel.validateMaskPositions(
                [1, 1],
                inputIDs: [50_281, VonTokenizer.maskTokenID, 50_282]
            )
        )
    }

    func testBatchPreparationRightPadsAndBuildsFlatMarkerLayout() throws {
        let mask = VonTokenizer.maskTokenID
        let prepared = try VonOptionMarkerModel.prepareBatch(
            inputIDs: [[10, mask, 11], [mask, mask]],
            attentionMask: [[1, 1, 0], [1, 1]],
            maskPositions: [[1], [0, 1]]
        )

        XCTAssertEqual(prepared.maximumSequenceLength, 3)
        XCTAssertEqual(
            prepared.inputIDs,
            [[10, mask, 11], [mask, mask, VonTokenizer.paddingTokenID]]
        )
        XCTAssertEqual(prepared.attentionMask, [[1, 1, 0], [1, 1, 0]])
        XCTAssertEqual(prepared.flattenedMaskPositions, [1, 3, 4])
        XCTAssertEqual(prepared.rowOffsets, [0, 1, 3])

        let defaultAttention = try VonOptionMarkerModel.prepareBatch(
            inputIDs: [[10, mask, 11], [mask, mask]],
            attentionMask: nil,
            maskPositions: [[1], [0, 1]]
        )
        XCTAssertEqual(defaultAttention.attentionMask, [[1, 1, 1], [1, 1, 0]])
    }

    func testBatchPreparationReportsTheOriginalRowForInvalidInputAndMarkers() throws {
        let mask = VonTokenizer.maskTokenID
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [],
                attentionMask: nil,
                maskPositions: []
            )
        ) { XCTAssertEqual($0 as? VonOptionMarkerModelError, .emptyBatch) }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], [mask]],
                attentionMask: [[1]],
                maskPositions: [[0], [0]]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .attentionBatchMismatch(expected: 2, actual: 1)
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], [mask]],
                attentionMask: [[1], [1, 0]],
                maskPositions: [[0], [0]]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .invalidInputRow(
                    batchIndex: 1,
                    error: .attentionSequenceMismatch(batchIndex: 1, expected: 1, actual: 2)
                )
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], []],
                attentionMask: nil,
                maskPositions: [[0], [0]]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .invalidInputRow(batchIndex: 1, error: .emptySequence)
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], [mask, VonModernBERT.vocabularySize]],
                attentionMask: nil,
                maskPositions: [[0], [0]]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .invalidInputRow(
                    batchIndex: 1,
                    error: .invalidTokenID(
                        batchIndex: 1,
                        sequenceIndex: 1,
                        value: VonModernBERT.vocabularySize
                    )
                )
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], [1, 2]],
                attentionMask: nil,
                maskPositions: [[0], [1]]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .markerTokenMismatchInBatch(batchIndex: 1, position: 1, tokenID: 2)
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], [mask]],
                attentionMask: nil,
                maskPositions: [[0], [1]]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .invalidMaskPositionInBatch(
                    batchIndex: 1,
                    index: 0,
                    value: 1,
                    sequenceLength: 1
                )
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask], [mask]],
                attentionMask: nil,
                maskPositions: [[0], []]
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .emptyMaskPositionsInBatch(batchIndex: 1)
            )
        }
        XCTAssertThrowsError(
            try VonOptionMarkerModel.prepareBatch(
                inputIDs: [[mask]],
                attentionMask: nil,
                maskPositions: []
            )
        ) {
            XCTAssertEqual(
                $0 as? VonOptionMarkerModelError,
                .maskPositionsBatchMismatch(expected: 1, actual: 0)
            )
        }
    }

    func testProductionBatchParityWhenConfigured() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["VON_OPTION_MARKER_BATCH_PARITY"] == "1" else {
            throw XCTSkip("Set VON_OPTION_MARKER_BATCH_PARITY=1 to run batch parity.")
        }
        guard let weightsPath = environment["VON_PRODUCTION_WEIGHTS_PATH"], !weightsPath.isEmpty else {
            throw XCTSkip("Set VON_PRODUCTION_WEIGHTS_PATH to the verified production artifact.")
        }

        let deviceName = environment["VON_OPTION_MARKER_BATCH_DEVICE"] ?? "cpu"
        let device: Device
        switch deviceName {
        case "cpu":
            device = .cpu
        case "gpu":
            device = .gpu
        default:
            return XCTFail(
                "VON_OPTION_MARKER_BATCH_DEVICE must be 'cpu' or 'gpu', got \(deviceName)"
            )
        }

        let source = try JSONDecoder().decode(
            GoldenFixture.self,
            from: Data(contentsOf: fixtureURL("von_golden_v1.json"))
        )
        let batchFixture = try JSONDecoder().decode(
            BatchGoldenFixture.self,
            from: Data(contentsOf: fixtureURL("von_batch_golden_v1.json"))
        )
        XCTAssertEqual(source.forwards.count, 5)
        XCTAssertEqual(batchFixture.batchCapture.rows.count, 5)

        try Device.withDefaultDevice(device) {
            let weights = try VonWeights.verifiedProduction(
                contentsOf: URL(fileURLWithPath: weightsPath)
            )
            let model = VonOptionMarkerModel(weights: weights)
            let packedInputs = source.forwards.map { forward in
                VonPackedInput(
                    text: "",
                    inputIDs: forward.inputIDs,
                    attentionMask: forward.attentionMask,
                    maskPositions: forward.maskPositions
                )
            }
            let batch = try model.forward(packedInputs)
            eval(batch.logits)
            let flatBatchLogits = batch.logits.asArray(Float.self)

            var expectedOffsets = [0]
            for (index, row) in batchFixture.batchCapture.rows.enumerated() {
                expectedOffsets.append(expectedOffsets[index] + row.batchedRawLogits.count)
            }
            XCTAssertEqual(batch.rowOffsets, expectedOffsets)

            XCTAssertTrue(flatBatchLogits.allSatisfy(\.isFinite), "batched logits must be finite")
            var maximumBatchOracleError: Float = 0
            var maximumBatchSequentialError: Float = 0
            for index in source.forwards.indices {
                let sourceForward = source.forwards[index]
                let oracle = batchFixture.batchCapture.rows[index]
                let actualBatch = Array(
                    flatBatchLogits[batch.rowOffsets[index]..<batch.rowOffsets[index + 1]]
                )
                let batchError = maxAbsoluteError(
                    actualBatch,
                    oracle.batchedRawLogits.map(Float.init)
                )
                maximumBatchOracleError = max(maximumBatchOracleError, batchError)
                XCTAssertLessThan(batchError, 1e-3, "batched oracle row \(index)")

                let sequential = try model.forward(
                    inputIDs: sourceForward.inputIDs,
                    attentionMask: sourceForward.attentionMask,
                    maskPositions: sourceForward.maskPositions
                )
                eval(sequential)
                let sequentialLogits = sequential.asArray(Float.self)
                XCTAssertTrue(
                    sequentialLogits.allSatisfy(\.isFinite),
                    "sequential logits row \(index) must be finite"
                )
                let sequentialOracleError = maxAbsoluteError(
                    sequentialLogits,
                    oracle.sequentialRawLogits.map(Float.init)
                )
                XCTAssertLessThan(
                    sequentialOracleError,
                    1e-3,
                    "sequential Swift oracle row \(index)"
                )
                let batchSequentialError = maxAbsoluteError(actualBatch, sequentialLogits)
                maximumBatchSequentialError = max(
                    maximumBatchSequentialError,
                    batchSequentialError
                )
                XCTAssertLessThan(
                    batchSequentialError,
                    1e-3,
                    "batched versus sequential Swift row \(index)"
                )
                print(
                    "OptionMarker \(deviceName) batch row \(index) "
                        + "oracle_max_abs=\(batchError) "
                        + "sequential_max_abs=\(batchSequentialError)"
                )
            }
            print(
                "OptionMarker \(deviceName) BATCH_ORACLE_MAX_ABS="
                    + "\(maximumBatchOracleError)"
            )
            print(
                "OptionMarker \(deviceName) BATCH_SEQUENTIAL_MAX_ABS="
                    + "\(maximumBatchSequentialError)"
            )
        }
    }

    func testProductionScorerAndComposedParityWhenConfigured() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["VON_OPTION_MARKER_PARITY"] == "1" else {
            throw XCTSkip("Set VON_OPTION_MARKER_PARITY=1 to run scorer and decision parity.")
        }
        guard let weightsPath = environment["VON_PRODUCTION_WEIGHTS_PATH"], !weightsPath.isEmpty else {
            throw XCTSkip("Set VON_PRODUCTION_WEIGHTS_PATH to the verified production artifact.")
        }

        let deviceName = environment["VON_OPTION_MARKER_DEVICE"] ?? "cpu"
        let device: Device
        switch deviceName {
        case "cpu":
            device = .cpu
        case "gpu":
            device = .gpu
        default:
            return XCTFail("VON_OPTION_MARKER_DEVICE must be 'cpu' or 'gpu', got \(deviceName)")
        }

        let fixture = try JSONDecoder().decode(
            GoldenFixture.self,
            from: Data(contentsOf: fixtureURL("von_golden_v1.json"))
        )
        XCTAssertEqual(fixture.forwards.count, 5)
        XCTAssertEqual(fixture.cases.count, 4)

        try Device.withDefaultDevice(device) {
            let weights = try VonWeights.verifiedProduction(
                contentsOf: URL(fileURLWithPath: weightsPath)
            )
            var isolatedMaximumError: Float = 0
            do {
                let scorer = VonOptionMarkerScorer(weights: weights.scorer)
                for forward in fixture.forwards {
                    let states = MLXArray(
                        forward.finalMaskStates.flatMap { $0 }.map(Float.init),
                        [forward.finalMaskStates.count, VonOptionMarkerScorer.inputSize]
                    )
                    let actual = scorer.forward(states)
                    eval(actual)
                    let error = maxAbsoluteError(
                        actual.asArray(Float.self),
                        forward.rawLogits.map(Float.init)
                    )
                    isolatedMaximumError = max(isolatedMaximumError, error)
                    print(
                        "OptionMarker \(deviceName) isolated scorer forward \(forward.index) "
                            + "max_abs=\(error)"
                    )
                    XCTAssertLessThan(error, 1e-3, "isolated scorer forward \(forward.index)")
                }
            }

            let model = VonOptionMarkerModel(weights: weights)
            var composedMaximumError: Float = 0
            var composedLogits: [[Float]] = []
            composedLogits.reserveCapacity(fixture.forwards.count)
            for forward in fixture.forwards {
                let output = try model.forward(
                    inputIDs: forward.inputIDs,
                    attentionMask: forward.attentionMask,
                    maskPositions: forward.maskPositions
                )
                eval(output)
                let actual = output.asArray(Float.self)
                composedLogits.append(actual)
                let error = maxAbsoluteError(actual, forward.rawLogits.map(Float.init))
                composedMaximumError = max(composedMaximumError, error)
                print(
                    "OptionMarker \(deviceName) composed raw logits forward \(forward.index) "
                        + "max_abs=\(error)"
                )
                XCTAssertLessThan(error, 1e-3, "composed raw logits forward \(forward.index)")
            }

            var decisionProbabilityMaximumError: Float = 0
            for testCase in fixture.cases {
                let primary = composedLogits[testCase.forwardIndices[0]]
                let calibration: VonCalibrationResult
                switch testCase.primitive {
                case "choice":
                    let result = try VonDecisionMath.choice(
                        logits: primary,
                        stateTokenCount: testCase.calibration.stateTokenCount
                    )
                    calibration = result.calibration
                    XCTAssertEqual(
                        result.selectedIndex,
                        try selectedOptionIndex(from: testCase),
                        "\(testCase.id) selected option"
                    )
                    try assertRoundedDecision(result, testCase: testCase)

                case "noul":
                    let null = testCase.forwardIndices.count == 2
                        ? composedLogits[testCase.forwardIndices[1]]
                        : nil
                    let result = try VonDecisionMath.noul(
                        logits: primary,
                        nullLogits: null,
                        stateTokenCount: testCase.calibration.stateTokenCount
                    )
                    calibration = result.calibration
                    try assertRoundedDecision(result, testCase: testCase)

                case "score":
                    let result = try VonDecisionMath.score(
                        logits: primary,
                        stateTokenCount: testCase.calibration.stateTokenCount
                    )
                    calibration = result.calibration
                    try assertRoundedDecision(result, testCase: testCase)

                default:
                    XCTFail("Unexpected primitive \(testCase.primitive)")
                    continue
                }

                let probabilityError = maxAbsoluteError(
                    calibration.probabilities,
                    testCase.calibration.probabilities.map(Float.init)
                )
                decisionProbabilityMaximumError = max(
                    decisionProbabilityMaximumError,
                    probabilityError
                )
                print(
                    "OptionMarker \(deviceName) decision probabilities \(testCase.id) "
                        + "max_abs=\(probabilityError)"
                )
                XCTAssertLessThan(
                    probabilityError,
                    1e-3,
                    "\(deviceName) decision probabilities \(testCase.id)"
                )
            }

            print("OptionMarker \(deviceName) ISOLATED_SCORER_MAX_ABS=\(isolatedMaximumError)")
            print("OptionMarker \(deviceName) COMPOSED_LOGITS_MAX_ABS=\(composedMaximumError)")
            print(
                "OptionMarker \(deviceName) DECISION_PROBABILITIES_MAX_ABS="
                    + "\(decisionProbabilityMaximumError)"
            )
            XCTAssertLessThan(isolatedMaximumError, 1e-3)
            XCTAssertLessThan(composedMaximumError, 1e-3)
            XCTAssertLessThan(decisionProbabilityMaximumError, 1e-3)
        }
    }
}

private extension VonOptionMarkerModelTests {
    struct GoldenFixture: Decodable {
        let cases: [GoldenCase]
        let forwards: [GoldenForward]
    }

    struct BatchGoldenFixture: Decodable {
        let batchCapture: BatchCapture

        enum CodingKeys: String, CodingKey {
            case batchCapture = "batch_capture"
        }
    }

    struct BatchCapture: Decodable {
        let rows: [BatchRow]
    }

    struct BatchRow: Decodable {
        let batchedRawLogits: [Double]
        let sequentialRawLogits: [Double]

        enum CodingKeys: String, CodingKey {
            case batchedRawLogits = "batched_raw_logits"
            case sequentialRawLogits = "sequential_raw_logits"
        }
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

    struct GoldenCalibration: Decodable {
        let stateTokenCount: Int
        let probabilities: [Double]

        enum CodingKeys: String, CodingKey {
            case stateTokenCount = "state_token_count"
            case probabilities
        }
    }

    struct GoldenPostprocess: Decodable {
        let confidenceUnrounded: Double?
        let scoreUnrounded: Double?
        let probabilityTrueUnrounded: Double?

        enum CodingKeys: String, CodingKey {
            case confidenceUnrounded = "confidence_unrounded"
            case scoreUnrounded = "score_unrounded"
            case probabilityTrueUnrounded = "probability_true_unrounded"
        }
    }

    struct GoldenForward: Decodable {
        let index: Int
        let inputIDs: [Int]
        let attentionMask: [Int]
        let maskPositions: [Int]
        let rawLogits: [Double]
        let finalMaskStates: [[Double]]

        enum CodingKeys: String, CodingKey {
            case index
            case inputIDs = "input_ids"
            case attentionMask = "attention_mask"
            case maskPositions = "mask_positions"
            case rawLogits = "raw_logits"
            case finalMaskStates = "final_mask_states"
        }
    }

    func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("testdata/\(name)", isDirectory: false)
    }

    func maxAbsoluteError(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count else {
            XCTFail("Parity arrays differ in length: \(actual.count) vs \(expected.count)")
            return .infinity
        }
        guard actual.allSatisfy(\.isFinite), expected.allSatisfy(\.isFinite) else {
            XCTFail("Parity arrays contain non-finite values")
            return .infinity
        }
        return zip(actual, expected).reduce(0) { maximum, pair in
            max(maximum, abs(pair.0 - pair.1))
        }
    }

    func answer(_ testCase: GoldenCase) throws -> [String: Any] {
        let data = try XCTUnwrap(testCase.answerJSON.data(using: .utf8))
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    func selectedOptionIndex(from testCase: GoldenCase) throws -> Int {
        let choice = try XCTUnwrap(try answer(testCase)["choice"] as? String)
        return try XCTUnwrap(testCase.orderedOptions.firstIndex { $0.key == choice })
    }

    func answerProbabilities(_ testCase: GoldenCase) throws -> [Double] {
        let probabilities = try XCTUnwrap(
            try answer(testCase)["probabilities"] as? [String: Double]
        )
        return try testCase.orderedOptions.map { option in
            try XCTUnwrap(probabilities[option.key])
        }
    }

    func answerNumber(_ testCase: GoldenCase, key: String) throws -> Double {
        try XCTUnwrap(try answer(testCase)[key] as? Double)
    }

    func assertRoundedDecision(
        _ result: VonChoiceMathResult,
        testCase: GoldenCase
    ) throws {
        let expectedRounded = try answerProbabilities(testCase)
        for index in result.calibration.probabilities.indices {
            assertStableRounding(
                actual: Double(result.calibration.probabilities[index]),
                expected: testCase.calibration.probabilities[index],
                places: 4,
                roundedExpected: expectedRounded[index],
                message: "\(testCase.id) probability \(index)"
            )
        }
        let expectedConfidence = try XCTUnwrap(testCase.postprocess.confidenceUnrounded)
        let actualProbabilities = result.calibration.probabilities.map(Double.init).sorted(by: >)
        let actualConfidence = actualProbabilities[0] - actualProbabilities[1]
        assertStableRounding(
            actual: actualConfidence,
            expected: expectedConfidence,
            places: 3,
            roundedExpected: try answerNumber(testCase, key: "confidence"),
            message: "\(testCase.id) confidence"
        )
        XCTAssertEqual(result.roundedProbabilities, expectedRounded)
        XCTAssertEqual(result.confidence, try answerNumber(testCase, key: "confidence"))
    }

    func assertRoundedDecision(
        _ result: VonNoulMathResult,
        testCase: GoldenCase
    ) throws {
        let expected = try XCTUnwrap(testCase.postprocess.probabilityTrueUnrounded)
        assertStableRounding(
            actual: Double(result.calibration.probabilities[0]),
            expected: expected,
            places: 4,
            roundedExpected: try answerNumber(testCase, key: "noul"),
            message: "\(testCase.id) noul"
        )
        XCTAssertEqual(result.probabilityTrue, try answerNumber(testCase, key: "noul"))
    }

    func assertRoundedDecision(
        _ result: VonScoreMathResult,
        testCase: GoldenCase
    ) throws {
        let expectedRounded = try answerProbabilities(testCase)
        for index in result.calibration.probabilities.indices {
            assertStableRounding(
                actual: Double(result.calibration.probabilities[index]),
                expected: testCase.calibration.probabilities[index],
                places: 4,
                roundedExpected: expectedRounded[index],
                message: "\(testCase.id) probability \(index)"
            )
        }
        let actualProbabilities = result.calibration.probabilities.map(Double.init)
        let sorted = actualProbabilities.sorted(by: >)
        assertStableRounding(
            actual: sorted[0] - sorted[1],
            expected: try XCTUnwrap(testCase.postprocess.confidenceUnrounded),
            places: 3,
            roundedExpected: try answerNumber(testCase, key: "confidence"),
            message: "\(testCase.id) confidence"
        )
        let actualScore = actualProbabilities.enumerated().reduce(0.0) {
            $0 + Double($1.offset) * $1.element
        }
        assertStableRounding(
            actual: actualScore,
            expected: try XCTUnwrap(testCase.postprocess.scoreUnrounded),
            places: 2,
            roundedExpected: try answerNumber(testCase, key: "score"),
            message: "\(testCase.id) score"
        )
        XCTAssertEqual(result.roundedProbabilities, expectedRounded)
        XCTAssertEqual(result.confidence, try answerNumber(testCase, key: "confidence"))
        XCTAssertEqual(result.score, try answerNumber(testCase, key: "score"))
    }

    /// Guard exact answer checks against silently crossing a rounding boundary.
    func assertStableRounding(
        actual: Double,
        expected: Double,
        places: Int,
        roundedExpected: Double,
        message: String
    ) {
        let factor = pow(10.0, Double(places))
        let scaled = expected * factor
        let fraction = scaled - floor(scaled)
        let boundaryDistance = abs(fraction - 0.5) / factor
        let observedDrift = abs(actual - expected)
        XCTAssertGreaterThan(
            boundaryDistance,
            observedDrift,
            "\(message): parity drift reaches a nearest-even rounding boundary"
        )
        if boundaryDistance > observedDrift {
            XCTAssertEqual(
                VonDecisionMath.rounded(actual, places: places),
                roundedExpected,
                message
            )
        }
    }
}
