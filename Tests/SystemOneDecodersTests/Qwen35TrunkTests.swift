import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// `testdata/qwen35_tiny/fixture.json`, written by `tools/qwen35_fixture.py`.
private struct Qwen35Fixture: Decodable {
    struct Rows: Decodable {
        let ids: [[Int]]
        let lengths: [Int]
        let hidden: [[Float]]
    }

    struct Tree: Decodable {
        let ids: [Int]
        let positions: [Int]
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

    struct Single: Decodable {
        let ids: [Int]
        let hidden: [Float]
    }

    let padded: Rows
    let bidirectional: Rows
    let tree: Tree
    let prefix: Prefix
    let readout: Readout
    let mlx8bit: Single

    enum CodingKeys: String, CodingKey {
        case padded, bidirectional, tree, prefix, readout
        case mlx8bit = "mlx_8bit"
    }

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/qwen35_tiny")

    static func load() throws -> Qwen35Fixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
    }
}

final class Qwen35ConfigurationTests: XCTestCase {
    private func json(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    func testReadsTheNestedFixtureConfig() throws {
        let configuration = try Qwen35Configuration(modelFolder: Qwen35Fixture.root.appendingPathComponent("transformers"))
        XCTAssertEqual(configuration.layerTypes, [.linearAttention, .linearAttention, .fullAttention, .linearAttention, .linearAttention, .fullAttention])
        XCTAssertEqual(configuration.hiddenSize, 32)
        XCTAssertEqual(configuration.headDimension, 16)
        XCTAssertEqual(configuration.rotaryDimension, 4)
        XCTAssertEqual(configuration.ropeTheta, 10_000)
        XCTAssertEqual(configuration.convolutionDimension, 2 * 16 + 32)
        XCTAssertFalse(configuration.tieWordEmbeddings)
        XCTAssertTrue(configuration.attentionOutputGate)
    }

    func testReadsPrunedLayerTypesAndTheIntervalFallback() throws {
        // d3-edge keeps 12 of Qwen3.5-0.8B's layers, not in the 3:1 pattern.
        let edge = try Qwen35Configuration(json: json("""
        {"model_type": "qwen3_5", "tie_word_embeddings": true, "text_config": {
          "model_type": "qwen3_5_text", "vocab_size": 248320, "hidden_size": 1024, "intermediate_size": 3584,
          "num_hidden_layers": 12, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
          "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_key_head_dim": 128,
          "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "tie_word_embeddings": true,
          "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention",
            "linear_attention", "full_attention", "linear_attention", "linear_attention", "full_attention",
            "full_attention", "full_attention", "full_attention"],
          "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": true}}}
        """))
        XCTAssertEqual(edge.layerTypes.filter { $0 == .fullAttention }.count, 6)
        XCTAssertEqual(edge.rotaryDimension, 64)
        XCTAssertEqual(edge.ropeTheta, 10_000_000)
        XCTAssertTrue(edge.tieWordEmbeddings)

        let interval = try Qwen35Configuration(json: json("""
        {"model_type": "qwen3_5_text", "vocab_size": 10, "hidden_size": 8, "intermediate_size": 8,
         "num_hidden_layers": 8, "num_attention_heads": 2, "head_dim": 4, "linear_num_key_heads": 1,
         "linear_num_value_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4,
         "partial_rotary_factor": 0.5, "rope_theta": 1000}
        """))
        XCTAssertEqual(interval.layerTypes.indices.filter { interval.layerTypes[$0] == .fullAttention }, [3, 7])
        XCTAssertEqual(interval.keyValueHeads, 2)
        XCTAssertEqual(interval.rotaryDimension, 2)
        XCTAssertEqual(interval.ropeTheta, 1000)
    }

    func testRejectsUnsupportedConfigs() throws {
        let base = """
        "vocab_size": 10, "hidden_size": 8, "intermediate_size": 8, "num_hidden_layers": 2,
        "num_attention_heads": 2, "head_dim": 4, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
        "linear_key_head_dim": 4, "linear_value_head_dim": 4
        """
        XCTAssertThrowsError(try Qwen35Configuration(json: json("{\(base), \"num_experts\": 8}"))) {
            XCTAssertEqual($0 as? Qwen35ConfigurationError, .unsupported("mixture-of-experts layers"))
        }
        XCTAssertThrowsError(try Qwen35Configuration(json: json("{\(base), \"layer_types\": [\"full_attention\"]}")))
        XCTAssertThrowsError(try Qwen35Configuration(json: json("{\(base), \"rope_parameters\": {\"rope_type\": \"yarn\"}}"))) {
            XCTAssertEqual($0 as? Qwen35ConfigurationError, .unsupported("rope_type yarn"))
        }
        XCTAssertThrowsError(try Qwen35Configuration(json: json("{\(base), \"model_type\": \"llama\"}")))
        XCTAssertThrowsError(try Qwen35Configuration(json: json("{\"hidden_size\": 8}"))) {
            XCTAssertEqual($0 as? Qwen35ConfigurationError, .missing("num_hidden_layers"))
        }
    }

    func testAttentionPlansAndPositions() throws {
        let causal = DecoderAttention(lengths: [3, 2])
        XCTAssertFalse(causal.isPlainCausal(width: 3))
        XCTAssertTrue(DecoderAttention(lengths: [3]).isPlainCausal(width: 3))
        XCTAssertFalse(DecoderAttention(lengths: [3], plan: .bidirectional).isPlainCausal(width: 3))

        // Row 1 (two real tokens) after a one-token prefix: real queries see the
        // prefix and earlier real keys; the padding query sees only itself.
        let values = causal.maskValues(width: 3, prefix: 1)
        let row1 = Array(values[12..<24])
        XCTAssertEqual(row1, [
            true, true, false, false,
            true, true, true, false,
            false, false, false, true,
        ])
        let bidirectional = DecoderAttention(lengths: [2], plan: .bidirectional).maskValues(width: 2, prefix: 0)
        XCTAssertEqual(bidirectional, [true, true, true, true])
        let visible = DecoderAttention(lengths: [2], plan: .visible([[[true, false], [false, true]]]))
        XCTAssertEqual(visible.maskValues(width: 2, prefix: 0), [true, false, false, true])

        XCTAssertEqual(causal.positionIDs(width: 3, prefix: 4), [4, 5, 6, 4, 5, 6])
        let explicit = DecoderAttention(lengths: [2], positions: [[7, 7]])
        XCTAssertEqual(explicit.positionIDs(width: 3, prefix: 0), [7, 7, 2])
        XCTAssertFalse(explicit.hasDefaultPositions(prefix: 0))
        XCTAssertTrue(DecoderAttention(lengths: [2], positions: [[5, 6]]).hasDefaultPositions(prefix: 5))

        XCTAssertThrowsError(try DecoderAttention(lengths: [4]).validate(width: 3))
        XCTAssertThrowsError(try DecoderAttention(lengths: [2], positions: [[0]]).validate(width: 2))
        XCTAssertThrowsError(try DecoderAttention(lengths: [2], plan: .visible([[[false, false], [true, true]]])).validate(width: 2)) {
            XCTAssertEqual($0 as? DecoderError, .invalidVisibility(row: 0))
        }
    }
}

/// The trunk in MLX: only from an Xcode-built test bundle with
/// `SYSTEMONE_MLX_TESTS=1` (`SYSTEMONE_MLX_DEVICE=gpu` for the GPU).
final class Qwen35TrunkRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    /// mlx-lm and transformers themselves differ by up to 6e-4 on this random
    /// model (activations reach about 4); a porting error is far larger.
    private let tolerance: Float = 2e-3

    private func trunk(_ name: String) throws -> Qwen35Trunk {
        try Qwen35Trunk(modelFolder: Qwen35Fixture.root.appendingPathComponent(name))
    }

    private func assertClose(_ actual: MLXArray, _ expected: [Float], _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, message, file: file, line: line)
        let worst = zip(values, expected).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThanOrEqual(worst, tolerance, message, file: file, line: line)
    }

    func testBothLayoutsMatchTransformers() throws {
        let fixture = try Qwen35Fixture.load()
        try Device.withDefaultDevice(device) {
            for name in ["transformers", "mlx-fp32"] {
                let model = try trunk(name)
                XCTAssertEqual(model.layout, name == "transformers" ? .transformers : .mlx)

                let padded = fixture.padded
                let hidden = try model(padded.ids, attention: DecoderAttention(lengths: padded.lengths))
                for (row, length) in padded.lengths.enumerated() {
                    assertClose(hidden[row, 0 ..< length], padded.hidden[row], "\(name) padded row \(row)")
                }

                let bidirectional = fixture.bidirectional
                let both = try model(bidirectional.ids, attention: DecoderAttention(lengths: bidirectional.lengths, plan: .bidirectional))
                for (row, length) in bidirectional.lengths.enumerated() {
                    assertClose(both[row, 0 ..< length], bidirectional.hidden[row], "\(name) bidirectional row \(row)")
                }

                let tree = fixture.tree
                let treeHidden = try model(
                    [tree.ids],
                    attention: DecoderAttention(lengths: [tree.ids.count], positions: [tree.positions], plan: .visible([tree.visible]))
                )
                assertClose(treeHidden, tree.hidden, "\(name) tree")
            }
        }
    }

    func testPrefixCacheMatchesWholeSequences() throws {
        let fixture = try Qwen35Fixture.load().prefix
        try Device.withDefaultDevice(device) {
            let model = try trunk("transformers")
            // Two prefills, the second across the cached convolution window and state.
            let cache = Qwen35Cache()
            let first = try model.prefill(Array(fixture.prefix[..<20]), cache: cache)
            let branch = cache.copy()
            let second = try model.prefill(Array(fixture.prefix[20...]), cache: cache)
            XCTAssertEqual(cache.length, fixture.prefix.count)
            XCTAssertEqual(branch.length, 20)
            assertClose(concatenated([first, second], axis: 1), fixture.prefixHidden, "prefix prefill")

            let width = fixture.rows.map(\.count).max() ?? 0
            let rows = fixture.rows.map { $0 + Array(repeating: 0, count: width - $0.count) }
            let lengths = fixture.rows.map(\.count)
            let hidden = try model(rows, attention: DecoderAttention(lengths: lengths), prefix: cache)
            XCTAssertEqual(cache.length, fixture.prefix.count, "reading a prefix leaves it unchanged")
            for (row, length) in lengths.enumerated() {
                assertClose(hidden[row, 0 ..< length], fixture.hidden[row], "row \(row) after the prefix")
            }

            // The copy taken after 20 tokens still continues from there.
            let rest = try model([Array(fixture.prefix[20...])], prefix: branch)
            assertClose(rest, Array(fixture.prefixHidden[(20 * 32)...]), "branch")
        }
    }

    func testReadoutRowsAndQuantizedExport() throws {
        let fixture = try Qwen35Fixture.load()
        try Device.withDefaultDevice(device) {
            for name in ["transformers", "mlx-fp32"] {
                let model = try trunk(name)
                let rows = try model.readoutRows(fixture.readout.ids)
                assertClose(rows, fixture.readout.rows, "\(name) readout rows")
                let last = MLXArray(Array(fixture.padded.hidden[0].suffix(32)))
                assertClose(matmul(rows, last), fixture.readout.logits, "\(name) readout logits")
                XCTAssertThrowsError(try model.readoutRows([96]))
            }

            let quantized = try trunk("mlx-8bit")
            XCTAssertEqual(quantized.layout, .mlx)
            assertClose(try quantized([fixture.mlx8bit.ids]), fixture.mlx8bit.hidden, "mlx-8bit against mlx-lm")
        }
    }

    func testRejectsMalformedBatches() throws {
        try Device.withDefaultDevice(device) {
            let model = try trunk("transformers")
            XCTAssertThrowsError(try model([])) { XCTAssertEqual($0 as? DecoderError, .emptyBatch) }
            XCTAssertThrowsError(try model([[1, 2], [3]])) { XCTAssertEqual($0 as? DecoderError, .raggedBatch(row: 1)) }
            XCTAssertThrowsError(try model([[1, 96]])) {
                XCTAssertEqual($0 as? DecoderError, .invalidTokenID(row: 0, column: 1, value: 96))
            }
            XCTAssertThrowsError(try model([[1, 2]], attention: DecoderAttention(lengths: [2, 2])))
        }
    }
}
