import CryptoKit
import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneD1

final class D1VisionTests: XCTestCase {
    // MARK: Offline: fixture only

    func testLayoutsMatchPython() throws {
        for expected in try D1Fixture.load().vision.layouts {
            let layout = D1Vision.layout(width: expected.width, height: expected.height)
            let label = "\(expected.width)x\(expected.height)"
            XCTAssertEqual([layout.grid.columns, layout.grid.rows], expected.grid, label)
            XCTAssertEqual([layout.thumbnail.height, layout.thumbnail.width], expected.thumbnail, label)
            XCTAssertEqual(layout.tiled, expected.tiled, label)
        }
    }

    func testCropsMatchTorchvisionBitForBit() throws {
        for expected in try D1Fixture.load().vision.images {
            let image = syntheticImage(width: expected.width, height: expected.height, seed: expected.seed)
            let crops = D1Vision.crops(of: image)
            XCTAssertEqual(crops.count, expected.crops.count, expected.name)
            for (crop, reference) in zip(crops, expected.crops) {
                XCTAssertEqual([crop.height, crop.width], [reference.height, reference.width], expected.name)
                XCTAssertEqual(crop.pixels.reduce(0) { $0 + Int($1) }, reference.sum, expected.name)
                let digest = SHA256.hash(data: Data(crop.pixels)).map { String(format: "%02x", $0) }.joined()
                XCTAssertEqual(digest, reference.sha256, "\(expected.name) \(crop.height)x\(crop.width)")
            }
            XCTAssertEqual(D1Vision.prefixLength(width: image.width, height: image.height), expected.prefixLength)
        }
    }

    // MARK: Weights: needs D1_MODEL_FOLDER and D1_MODEL_PARITY=1

    func testTowerAndPrefixMatchPython() throws {
        let weights = try D1TextTests.loadWeights()
        try Device.withDefaultDevice(D1TextTests.device) {
            let vision = D1Vision(weights: weights)
            for expected in try D1Fixture.load().vision.images {
                let image = syntheticImage(width: expected.width, height: expected.height, seed: expected.seed)
                let dimensions = MLXArray(expected.probeDimensions.map(Int32.init))
                for (crop, reference) in zip(D1Vision.crops(of: image), expected.crops) {
                    let rows = MLXArray(reference.probeRows.map(Int32.init))
                    let (patchRows, patchColumns) = (crop.height / D1Vision.patch, crop.width / D1Vision.patch)
                    let positions = vision.positions(rows: patchRows, columns: patchColumns)[0]
                    assertClose(positions[rows].take(dimensions, axis: 1), reference.positionProbe, 1e-4,
                                "\(expected.name) positions")
                    let tower = vision.towerOutput(crop: crop)[0]
                    assertClose(tower[rows].take(dimensions, axis: 1), reference.towerProbe, 2e-3,
                                "\(expected.name) tower")
                }
                let prefix = vision(image)
                XCTAssertEqual(prefix.dim(0), expected.prefixLength)
                let rows = MLXArray(expected.prefixRows.map(Int32.init))
                let probe = MLXArray(D1TextTests.probeDimensions.map(Int32.init))
                assertClose(prefix[rows].take(probe, axis: 1), expected.prefixProbe, 2e-3, "\(expected.name) prefix")
            }
        }
    }

    func testImageRequestsMatchPython() async throws {
        let weights = try D1TextTests.loadWeights()
        let folder = try D1Fixture.modelFolder()
        let backend = D1Backend(
            modelID: "d1-omni-600M",
            configuration: weights.configuration,
            tokenizer: try await D1Tokenizer(modelFolder: folder),
            model: D1Model(weights: weights),
            logitsProvider: nil
        )
        let engine = SystemOneEngine(backend: backend)
        for testCase in try D1Fixture.load().vision.requests {
            let images = testCase.images.map { syntheticImage(width: $0.width, height: $0.height, seed: $0.seed) }
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry),
                    media: .images(images)
                )
            }
            XCTAssertEqual(response.usage.inputTokens, Int(testCase.result["usage"]!["input_tokens"]!.number!))
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                assert(answer.answer, matches: expected, tolerance: 2e-3, "\(testCase.id).\(answer.id)")
            }
        }
    }
}

/// Max absolute error of an MLX probe against fixture rows, relative to the probe's scale.
func assertClose(
    _ actual: MLXArray,
    _ expected: [[Float]],
    _ tolerance: Float,
    _ message: String,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let reference = MLXArray(expected.flatMap { $0 }, actual.shape)
    let error = abs(actual - reference).max().item(Float.self)
    let scale = max(1, abs(reference).max().item(Float.self))
    print("\(message): max error \(error) (scale \(scale))")
    XCTAssertLessThanOrEqual(error / scale, tolerance, message, file: file, line: line)
}
