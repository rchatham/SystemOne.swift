import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// `testdata/lfm3_tiny/fixture.json`, written by `tools/lfm3_fixture.py`.
private struct LFM3Fixture: Decodable {
    struct Rows: Decodable {
        let ids: [[Int]]
        let hidden: [[Float]]
    }

    struct SetRow: Decodable {
        let ids: [Int]
        let visible: [[Bool]]
        let hidden: [Float]
    }

    struct Prefix: Decodable {
        let prefix: [Int]
        let prefixHidden: [Float]
        let rows: [[Int]]
        let hidden: [[Float]]

        enum CodingKeys: String, CodingKey {
            case prefix, rows, hidden
            case prefixHidden = "prefix_hidden"
        }
    }

    struct Readout: Decodable {
        let ids: [Int]
        let rows: [Float]
        let logits: [Float]
    }

    let rows: Rows
    let set: SetRow
    let prefix: Prefix
    let readout: Readout

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/lfm3_tiny")

    static func load() throws -> LFM3Fixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
    }
}

final class LFM3ConfigurationTests: XCTestCase {
    func testReadsTheFixtureConfig() throws {
        let configuration = try LFM3Configuration(modelFolder: LFM3Fixture.folder)
        XCTAssertEqual(configuration.layerTypes, [.convolution, .slidingAttention, .fullAttention, .convolution, .slidingAttention, .fullAttention])
        XCTAssertEqual(configuration.fullRotary, LFM3Configuration.Rotary(theta: 1_000_000, angles: 2))
        XCTAssertEqual(configuration.slidingRotary, LFM3Configuration.Rotary(theta: 10_000, angles: 4))
        XCTAssertEqual(configuration.slidingWindow, 8)
        XCTAssertEqual(configuration.convolutionKernel, 7)
        XCTAssertTrue(configuration.sandwichNorms)
    }

    func testDefaultsWithoutRopeParameters() throws {
        let configuration = try LFM3Configuration(json: [
            "model_type": "lfm3", "vocab_size": 10, "hidden_size": 2048, "intermediate_size": 7680,
            "num_hidden_layers": 2, "layer_types": ["conv", "full_attention"], "num_attention_heads": 16,
            "num_key_value_heads": 4,
        ])
        XCTAssertEqual(configuration.headDimension, 128)
        XCTAssertEqual(configuration.fullRotary, LFM3Configuration.Rotary(theta: 1_000_000, angles: 32))
        XCTAssertEqual(configuration.slidingRotary, LFM3Configuration.Rotary(theta: 10_000, angles: 64))
        XCTAssertThrowsError(try LFM3Configuration(json: ["model_type": "lfm3", "num_hidden_layers": 1, "layer_types": ["mamba"]]))
    }

    func testSlidingMaskKeepsTheLastWindow() {
        // Width 4 after a 2-token prefix, window 3: query 1 (absolute 3) sees keys 1–3.
        let values = LFM3Trunk.slidingMaskValues(lengths: [4], width: 4, prefix: 2, window: 3)
        XCTAssertEqual(Array(values[6..<12]), [false, true, true, true, false, false])
    }
}

/// The trunk in MLX: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class LFM3TrunkRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private func assertClose(_ actual: MLXArray, _ expected: [Float], _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, message, file: file, line: line)
        let worst = zip(values, expected).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThanOrEqual(worst, 1e-4, message, file: file, line: line)
    }

    func testMatchesMLXVLM() throws {
        let fixture = try LFM3Fixture.load()
        try Device.withDefaultDevice(device) {
            let trunk = try LFM3Trunk(modelFolder: LFM3Fixture.folder)

            let width = fixture.rows.ids.map(\.count).max() ?? 0
            let padded = try trunk(
                fixture.rows.ids.map { $0 + Array(repeating: 0, count: width - $0.count) },
                attention: DecoderAttention(lengths: fixture.rows.ids.map(\.count))
            )
            for (row, tokens) in fixture.rows.ids.enumerated() {
                assertClose(padded[row, 0 ..< tokens.count], fixture.rows.hidden[row], "row \(row)")
            }

            let set = try trunk(
                [fixture.set.ids],
                attention: DecoderAttention(lengths: [fixture.set.ids.count], plan: .visible([fixture.set.visible]))
            )
            assertClose(set, fixture.set.hidden, "set row")

            let cache = DecoderCache()
            let first = try trunk.prefill(Array(fixture.prefix.prefix[..<6]), cache: cache)
            let second = try trunk.prefill(Array(fixture.prefix.prefix[6...]), cache: cache)
            assertClose(concatenated([first, second], axis: 1), fixture.prefix.prefixHidden, "prefix")
            let branchWidth = fixture.prefix.rows.map(\.count).max() ?? 0
            let branches = try trunk(
                fixture.prefix.rows.map { $0 + Array(repeating: 0, count: branchWidth - $0.count) },
                attention: DecoderAttention(lengths: fixture.prefix.rows.map(\.count)), prefix: cache
            )
            for (row, tokens) in fixture.prefix.rows.enumerated() {
                assertClose(branches[row, 0 ..< tokens.count], fixture.prefix.hidden[row], "row \(row) after the prefix")
            }

            let rows = try trunk.readoutRows(fixture.readout.ids)
            assertClose(rows, fixture.readout.rows, "readout rows")
            assertClose(matmul(rows, MLXArray(Array(fixture.rows.hidden[0].suffix(32)))), fixture.readout.logits, "readout logits")
        }
    }
}
