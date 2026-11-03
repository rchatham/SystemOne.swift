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
            model: try D1Model(weights: weights),
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

    func testImageRowsReusePrefixStates() throws {
        let weights = try D1TextTests.loadWeights()
        let model = try D1Model(weights: weights)
        try Device.withDefaultDevice(D1TextTests.device) {
            for testCase in try D1Fixture.load().vision.requests {
                let images = testCase.images.map { syntheticImage(width: $0.width, height: $0.height, seed: $0.seed) }
                try checkPrefixReuse(model, media: .images(images), rows: testCase.rows, "\(testCase.id)")
            }
        }
    }
}

/// Runs one request's rows with the prefix's cached trunk states and again with
/// the whole prefix per row, and checks both against the reference's logits.
func checkPrefixReuse(
    _ model: D1Model,
    media: SystemOneMedia,
    rows fixtureRows: [D1Fixture.Row],
    _ label: String,
    file: StaticString = #filePath,
    line: UInt = #line
) throws {
    let prefix = D1MediaPrefix(media: media, length: D1Model.prefixLength(of: media))
    let rows = fixtureRows.map {
        D1Row(ids: $0.ids, markers: $0.markers, questionType: .init(rawValue: $0.qtype)!, prefix: prefix)
    }
    func timed(_ body: () throws -> [MLXArray]) rethrows -> ([MLXArray], Double) {
        let start = Date()
        let result = try body()
        eval(result)
        return (result, Date().timeIntervalSince(start))
    }
    let embeddings = try prefix.embeddings(using: model)
    // Warm both paths' kernels; the second call also caches the prefix states.
    _ = try timed { [try model.logits(for: rows, reusingPrefixStates: false), try model.logits(for: rows)] }
    let (full, fullSeconds) = try timed { [try model.logits(for: rows, reusingPrefixStates: false)] }
    let (_, stateSeconds) = timed { model.trunk.prefixState(of: embeddings).arrays }
    let (reused, rowSeconds) = try timed { [try model.logits(for: rows)] }
    let singles = try rows.map { try model.logits(for: [$0]) }

    // A batch mixing prefixed and text-only rows runs each group apart and keeps row order.
    let textOnly = rows.map { D1Row(ids: $0.ids, markers: $0.markers, questionType: $0.questionType, prefix: nil) }
    let mixed = try model.logits(for: zip(rows, textOnly).flatMap { [$0, $1] })
    let alone = try model.logits(for: textOnly)
    for (index, row) in fixtureRows.enumerated() {
        let count = row.logits.count
        XCTAssertEqual(mixed[2 * index, 0..<count].asArray(Float.self), reused[0][index, 0..<count].asArray(Float.self),
                       "\(label) mixed prefixed row", file: file, line: line)
        XCTAssertEqual(mixed[2 * index + 1, 0..<count].asArray(Float.self), alone[index, 0..<count].asArray(Float.self),
                       "\(label) mixed text-only row", file: file, line: line)
    }

    var worstAgainstFull: Float = 0
    var worstAgainstPython: Float = 0
    for (index, row) in fixtureRows.enumerated() {
        let count = row.logits.count
        let fromReused = reused[0][index, 0..<count].asArray(Float.self)
        let fromFull = full[0][index, 0..<count].asArray(Float.self)
        let fromSingle = singles[index][0, 0..<count].asArray(Float.self)
        for (actual, expected) in zip(fromReused, fromFull) {
            worstAgainstFull = max(worstAgainstFull, abs(actual - expected))
            XCTAssertEqual(actual, expected, accuracy: 1e-3, "\(label) reused vs full", file: file, line: line)
        }
        for (actual, expected) in Array(zip(fromSingle, row.logits)) + Array(zip(fromReused, row.batchedLogits)) {
            worstAgainstPython = max(worstAgainstPython, abs(actual - expected))
            XCTAssertEqual(actual, expected, accuracy: 2e-3, "\(label) vs Python", file: file, line: line)
        }
    }
    print("""
        \(label): \(rows.count) rows over \(prefix.length) prefix positions; \
        reused-vs-full max logit diff \(worstAgainstFull), vs Python \(worstAgainstPython); \
        full \(String(format: "%.3f", fullSeconds)) s, reused \(String(format: "%.3f", stateSeconds)) s \
        for prefix states + \(String(format: "%.3f", rowSeconds)) s for rows
        """)
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
