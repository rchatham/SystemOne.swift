import Foundation
import MLX
import XCTest
@testable import VonSwift

final class VonModernBERTTests: XCTestCase {
    func testFullAttentionScheduleAndQKVShapes() {
        XCTAssertEqual(
            (0..<VonModernBERT.layerCount).filter(VonModernBERT.isFullAttention),
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        )
        XCTAssertEqual(
            VonModernBERT.queryKeyValueShape(batchSize: 2, sequenceLength: 7),
            [2, 7, 3, 16, 64]
        )
        XCTAssertEqual(
            VonModernBERT.attentionProjectionShape(batchSize: 2, sequenceLength: 7),
            [2, 16, 7, 64]
        )
    }

    func testAdditiveMasksPreserveFullKeyOnlyMaskingAndInclusiveSlidingRadius() {
        let blocked = -Float.greatestFiniteMagnitude
        let full = VonModernBERT.additiveAttentionMaskValues(
            [[1, 0, 1], [0, 1, 1]],
            type: .full
        )
        XCTAssertEqual(full.shape, [2, 1, 1, 3])
        XCTAssertEqual(
            full.values,
            [0, blocked, 0, blocked, 0, 0],
            "full attention remains key-only even for padded queries"
        )

        var keyMask = [Int](repeating: 1, count: 130)
        keyMask[64] = 0
        let sliding = VonModernBERT.additiveAttentionMaskValues(
            [keyMask],
            type: .sliding
        )
        XCTAssertEqual(sliding.shape, [1, 1, 130, 130])

        func value(query: Int, key: Int) -> Float {
            sliding.values[query * 130 + key]
        }

        XCTAssertEqual(value(query: 0, key: 0), 0)
        XCTAssertEqual(value(query: 0, key: 63), 0)
        XCTAssertEqual(value(query: 0, key: 64), blocked, "masked keys remain blocked")
        XCTAssertEqual(value(query: 0, key: 65), blocked, "radius 65 is excluded")
        XCTAssertEqual(value(query: 63, key: 127), 0, "radius 64 is inclusive")
        XCTAssertEqual(value(query: 63, key: 128), blocked, "radius 65 is excluded")
        XCTAssertEqual(value(query: 65, key: 0), blocked)
        XCTAssertEqual(value(query: 65, key: 1), 0, "left radius 64 is inclusive")
        XCTAssertEqual(value(query: 65, key: 64), blocked, "padding is independent of query")
        XCTAssertEqual(value(query: 65, key: 129), 0, "right radius 64 is inclusive")

        for key in keyMask.indices {
            XCTAssertEqual(
                value(query: 64, key: key),
                key == 64 ? 0 : blocked,
                "a padded query must see itself only"
            )
        }
        XCTAssertTrue(sliding.values.allSatisfy(\.isFinite))
        for query in keyMask.indices {
            let rowStart = query * keyMask.count
            let row = sliding.values[rowStart..<(rowStart + keyMask.count)]
            XCTAssertTrue(row.contains(0), "sliding query row \(query) must not be all-blocked")
        }
    }

    func testInputValidationRejectsMalformedInputs() throws {
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [], attentionMask: nil)
        ) { XCTAssertEqual($0 as? VonModernBERTError, .emptyBatch) }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[]], attentionMask: nil)
        ) { XCTAssertEqual($0 as? VonModernBERTError, .emptySequence) }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[1, 2], [3]], attentionMask: nil)
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .raggedInput(batchIndex: 1, expected: 2, actual: 1)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(
                inputIDs: [[Int](repeating: 1, count: 8_193)],
                attentionMask: nil
            )
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .sequenceTooLong(maximum: 8_192, actual: 8_193)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[-1]], attentionMask: nil)
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .invalidTokenID(batchIndex: 0, sequenceIndex: 0, value: -1)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[50_368]], attentionMask: nil)
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .invalidTokenID(batchIndex: 0, sequenceIndex: 0, value: 50_368)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[1]], attentionMask: [[1], [1]])
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .attentionBatchMismatch(expected: 1, actual: 2)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[1, 2]], attentionMask: [[1]])
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .attentionSequenceMismatch(batchIndex: 0, expected: 2, actual: 1)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[1]], attentionMask: [[2]])
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .invalidAttentionMaskValue(batchIndex: 0, sequenceIndex: 0, value: 2)
            )
        }
        XCTAssertThrowsError(
            try VonModernBERT.validateInputs(inputIDs: [[1, 2]], attentionMask: [[0, 0]])
        ) {
            XCTAssertEqual(
                $0 as? VonModernBERTError,
                .noUnmaskedTokens(batchIndex: 0)
            )
        }

        XCTAssertEqual(
            try VonModernBERT.validateInputs(inputIDs: [[1, 2], [3, 4]], attentionMask: nil),
            [[1, 1], [1, 1]]
        )
    }

    func testProductionBackboneParityWhenConfigured() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["VON_MODERNBERT_PARITY"] == "1" else {
            throw XCTSkip("Set VON_MODERNBERT_PARITY=1 to run ModernBERT numeric parity.")
        }
        guard let weightsPath = environment["VON_PRODUCTION_WEIGHTS_PATH"], !weightsPath.isEmpty else {
            throw XCTSkip("Set VON_PRODUCTION_WEIGHTS_PATH to the verified production artifact.")
        }

        let deviceName = environment["VON_MODERNBERT_DEVICE"] ?? "cpu"
        let device: Device
        switch deviceName {
        case "cpu":
            device = .cpu
        case "gpu":
            device = .gpu
        default:
            return XCTFail("VON_MODERNBERT_DEVICE must be 'cpu' or 'gpu', got \(deviceName)")
        }

        let fixture = try JSONDecoder().decode(
            GoldenFixture.self,
            from: Data(contentsOf: fixtureURL("von_golden_v1.json"))
        )
        XCTAssertEqual(fixture.metadata.backboneEntries.count, 30)
        XCTAssertEqual(fixture.metadata.probeDimensions.count, 16)
        XCTAssertEqual(fixture.forwards.count, 5)

        try Device.withDefaultDevice(device) {
            let weights = try VonWeights.verifiedProduction(
                contentsOf: URL(fileURLWithPath: weightsPath)
            )
            let model = VonModernBERT(weights: weights)
            var globalMaximumError: Float = 0

            for forward in fixture.forwards {
                let output = try model.forward(
                    inputIDs: forward.inputIDs,
                    attentionMask: forward.attentionMask,
                    captureHiddenStates: true
                )
                XCTAssertEqual(output.hiddenStates.count, 30, "forward \(forward.index)")
                eval(output.hiddenStates)

                let maskIndices = MLXArray(forward.maskPositions)
                let probeIndices = MLXArray(fixture.metadata.probeDimensions)
                for (boundaryIndex, expectedBoundary) in forward.backbone.enumerated() {
                    XCTAssertEqual(
                        expectedBoundary.name,
                        fixture.metadata.backboneEntries[boundaryIndex]
                    )
                    let actual = output.hiddenStates[boundaryIndex]
                        .take(maskIndices, axis: 1)
                        .squeezed(axis: 0)
                        .take(probeIndices, axis: 1)
                        .asArray(Float.self)
                    let expected = expectedBoundary.maskStateProbes.flatMap { $0 }.map(Float.init)
                    let maximumError = maxAbsoluteError(actual, expected)
                    globalMaximumError = max(globalMaximumError, maximumError)
                    print(
                        "ModernBERT \(deviceName) forward \(forward.index) "
                            + "\(expectedBoundary.name) max_abs=\(maximumError)"
                    )
                    XCTAssertLessThan(
                        maximumError,
                        1e-3,
                        "\(deviceName) forward \(forward.index) \(expectedBoundary.name)"
                    )
                }

                let actualFinal = output.lastHiddenState
                    .take(maskIndices, axis: 1)
                    .squeezed(axis: 0)
                    .asArray(Float.self)
                let expectedFinal = forward.finalMaskStates.flatMap { $0 }.map(Float.init)
                let finalMaximumError = maxAbsoluteError(actualFinal, expectedFinal)
                globalMaximumError = max(globalMaximumError, finalMaximumError)
                print(
                    "ModernBERT \(deviceName) forward \(forward.index) "
                        + "full_final_vector max_abs=\(finalMaximumError)"
                )
                XCTAssertLessThan(
                    finalMaximumError,
                    1e-3,
                    "\(deviceName) forward \(forward.index) full final mask vectors"
                )
            }

            print("ModernBERT \(deviceName) GLOBAL_MAX_ABS=\(globalMaximumError)")
            XCTAssertLessThan(globalMaximumError, 1e-3)
        }
    }
}

private extension VonModernBERTTests {
    struct GoldenFixture: Decodable {
        let forwards: [GoldenForward]
        let metadata: Metadata
    }

    struct Metadata: Decodable {
        let backboneEntries: [String]
        let probeDimensions: [Int]

        enum CodingKeys: String, CodingKey {
            case backboneEntries = "backbone_entries"
            case probeDimensions = "probe_dimensions"
        }
    }

    struct GoldenForward: Decodable {
        let index: Int
        let inputIDs: [Int]
        let attentionMask: [Int]
        let maskPositions: [Int]
        let backbone: [BackboneBoundary]
        let finalMaskStates: [[Double]]

        enum CodingKeys: String, CodingKey {
            case index
            case inputIDs = "input_ids"
            case attentionMask = "attention_mask"
            case maskPositions = "mask_positions"
            case backbone
            case finalMaskStates = "final_mask_states"
        }
    }

    struct BackboneBoundary: Decodable {
        let name: String
        let maskStateProbes: [[Double]]

        enum CodingKeys: String, CodingKey {
            case name
            case maskStateProbes = "mask_state_probes"
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
        XCTAssertEqual(actual.count, expected.count)
        return zip(actual, expected).reduce(0) { maximum, pair in
            max(maximum, abs(pair.0 - pair.1))
        }
    }
}
