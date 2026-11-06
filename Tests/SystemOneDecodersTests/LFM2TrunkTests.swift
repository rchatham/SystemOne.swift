import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// `testdata/lfm2_tiny/fixture.json`, written by `tools/lfm2_fixture.py`.
private struct LFM2Fixture: Decodable {
    struct Rows: Decodable {
        let ids: [[Int]]
        let lengths: [Int]
        let hidden: [[Float]]
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

    let padded: Rows
    let prefix: Prefix
    let readout: Readout

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/lfm2_tiny")

    static func load() throws -> LFM2Fixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
    }
}

final class LFM2ConfigurationTests: XCTestCase {
    func testReadsTheNestedFixtureConfig() throws {
        let configuration = try LFM2Configuration(modelFolder: LFM2Fixture.folder)
        XCTAssertEqual(configuration.layerTypes, [.convolution, .convolution, .fullAttention, .convolution])
        XCTAssertEqual(configuration.headDimension, 8)
        XCTAssertEqual(configuration.keyValueHeads, 2)
        XCTAssertEqual(configuration.intermediateSize, 64)
        XCTAssertEqual(configuration.ropeTheta, 1_000_000)
        XCTAssertTrue(configuration.tieWordEmbeddings)
    }

    func testAdjustsTheFeedForwardWidthAndReadsAttentionIndices() throws {
        // LFM2-1.2B style: block_auto_adjust_ff_dim, full_attn_idxs, rope_theta.
        let configuration = try LFM2Configuration(json: [
            "model_type": "lfm2", "vocab_size": 65_536, "hidden_size": 2048, "intermediate_size": 12_288,
            "block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256,
            "num_hidden_layers": 4, "full_attn_idxs": [1, 3], "num_attention_heads": 32, "num_key_value_heads": 8,
            "rope_theta": 1_000_000.0, "norm_eps": 1e-5, "conv_L_cache": 3,
        ])
        XCTAssertEqual(configuration.intermediateSize, 8192)
        XCTAssertEqual(configuration.layerTypes, [.convolution, .fullAttention, .convolution, .fullAttention])
        XCTAssertEqual(configuration.headDimension, 64)

        XCTAssertThrowsError(try LFM2Configuration(json: ["model_type": "lfm3", "num_hidden_layers": 1]))
        XCTAssertThrowsError(try LFM2Configuration(json: ["model_type": "lfm2", "num_hidden_layers": 1])) {
            XCTAssertEqual($0 as? LFM2ConfigurationError, .missing("layer_types"))
        }
    }
}

/// The trunk in MLX: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class LFM2TrunkRuntimeTests: XCTestCase {
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

    func testMatchesTransformersWithPaddingAndPrefixes() throws {
        let fixture = try LFM2Fixture.load()
        try Device.withDefaultDevice(device) {
            let trunk = try LFM2Trunk(modelFolder: LFM2Fixture.folder)
            let padded = try trunk(fixture.padded.ids, attention: DecoderAttention(lengths: fixture.padded.lengths))
            for (row, length) in fixture.padded.lengths.enumerated() {
                assertClose(padded[row, 0 ..< length], fixture.padded.hidden[row], "padded row \(row)")
            }

            let cache = DecoderCache()
            let first = try trunk.prefill(Array(fixture.prefix.prefix[..<10]), cache: cache)
            let second = try trunk.prefill(Array(fixture.prefix.prefix[10...]), cache: cache)
            assertClose(concatenated([first, second], axis: 1), fixture.prefix.prefixHidden, "prefix")
            let width = fixture.prefix.rows.map(\.count).max() ?? 0
            let hidden = try trunk(
                fixture.prefix.rows.map { $0 + Array(repeating: 0, count: width - $0.count) },
                attention: DecoderAttention(lengths: fixture.prefix.rows.map(\.count)), prefix: cache
            )
            for (row, tokens) in fixture.prefix.rows.enumerated() {
                assertClose(hidden[row, 0 ..< tokens.count], fixture.prefix.hidden[row], "row \(row) after the prefix")
            }

            let rows = try trunk.readoutRows(fixture.readout.ids)
            assertClose(rows, fixture.readout.rows, "readout rows")
            let last = MLXArray(Array(fixture.padded.hidden[0].suffix(32)))
            assertClose(matmul(rows, last), fixture.readout.logits, "readout logits")
        }
    }
}
