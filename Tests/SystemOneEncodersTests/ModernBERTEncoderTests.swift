import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneEncoders

/// `testdata/modernbert_tiny/`, written by `tools/modernbert_fixture.py` from
/// transformers' `ModernBertModel`.
struct ModernBERTFixture: Decodable {
    struct Outputs: Decodable {
        let padded: [[Float]]
        let tree: [Float]
    }

    struct Padded: Decodable {
        let ids: [[Int]]
        let lengths: [Int]
    }

    struct Tree: Decodable {
        let ids: [Int]
        let positions: [Int]
        let visible: [[Bool]]
    }

    let models: [String: Outputs]
    let paddedInputs: Padded
    let treeInputs: Tree

    enum CodingKeys: String, CodingKey {
        case models
        case paddedInputs = "padded_inputs"
        case treeInputs = "tree_inputs"
    }

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/modernbert_tiny", isDirectory: true)

    static func load() throws -> ModernBERTFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
    }
}

final class ModernBERTEncoderTests: XCTestCase {
    func testDecodesBothConfigurationForms() throws {
        let saved = try ModernBERTConfiguration(modelFolder: ModernBERTFixture.root.appendingPathComponent("large-rope"))
        XCTAssertEqual(saved.layerTypes, ["full_attention", "sliding_attention", "sliding_attention", "full_attention"])
        XCTAssertEqual(saved.globalRoPETheta, 160_000)
        XCTAssertEqual(saved.localRoPETheta, 10_000)
        XCTAssertEqual(saved.slidingRadius, 4)
        XCTAssertEqual(saved.headDimension, 16)

        // The original ModernBERT-large config (no layer_types, no rope_parameters).
        let legacy = """
        {"vocab_size": 50368, "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
         "intermediate_size": 2624, "global_attn_every_n_layers": 3, "local_attention": 128,
         "global_rope_theta": 160000.0, "local_rope_theta": 10000.0, "max_position_embeddings": 8192,
         "norm_eps": 1e-05, "norm_bias": false, "hidden_activation": "gelu"}
        """
        let original = try JSONDecoder().decode(ModernBERTConfiguration.self, from: Data(legacy.utf8))
        XCTAssertEqual(original.layerTypes.filter { $0 == "full_attention" }.count, 10)
        XCTAssertTrue(original.isFullAttention(27))
        XCTAssertFalse(original.isFullAttention(26))
        XCTAssertEqual(original.ropeTheta(layer: 1), 10_000)
        XCTAssertEqual(original.slidingRadius, 64)

        let wrongTypes = #"{"vocab_size": 8, "hidden_size": 4, "num_hidden_layers": 2, "num_attention_heads": 1, "intermediate_size": 4, "layer_types": ["full_attention"]}"#
        XCTAssertThrowsError(try JSONDecoder().decode(ModernBERTConfiguration.self, from: Data(wrongTypes.utf8)))
    }

    func testVisibilityCombinesLayoutPaddingAndSlidingWindowOnPositions() {
        // Row 0: prefix [0, 1], option A [2], option B [3], both at position 2.
        let attention = EncoderAttention(
            lengths: [4],
            positions: [[0, 1, 2, 2]],
            visible: [[
                [true, true, false, false],
                [true, true, false, false],
                [true, true, true, false],
                [true, true, false, true],
            ]]
        )
        XCTAssertFalse(attention.hasDefaultPositions)
        XCTAssertEqual(attention.positionIDs(width: 5), [0, 1, 2, 2, 4])
        let full = attention.visibility(width: 5, radius: nil)
        XCTAssertEqual(Array(full[15..<20]), [true, true, false, true, false], "option B sees the prefix and itself")
        XCTAssertEqual(Array(full[20..<25]), [false, false, false, false, true], "padding sees only itself")
        // Radius 1 on positions: option B (position 2) no longer sees position 0.
        let sliding = attention.visibility(width: 5, radius: 1)
        XCTAssertEqual(Array(sliding[15..<20]), [false, true, false, true, false])

        let padded = EncoderAttention(lengths: [2, 3])
        XCTAssertTrue(padded.hasDefaultPositions)
        XCTAssertEqual(padded.keyValidity(width: 3), [true, true, false, true, true, true])
        XCTAssertThrowsError(try EncoderAttention(lengths: [4]).validate(width: 3))
        XCTAssertThrowsError(try EncoderAttention(lengths: [2], positions: [[0]]).validate(width: 2))
        XCTAssertThrowsError(try EncoderAttention(lengths: [2], visible: [[[false, true], [true, true]]]).validate(width: 2))
    }

    func testFixtureCheckpointsHoldExactlyTheEncoderSchema() throws {
        for name in ["large-rope", "uniform-rope"] {
            let folder = ModernBERTFixture.root.appendingPathComponent(name)
            let configuration = try ModernBERTConfiguration(modelFolder: folder)
            let file = try SafetensorsFile(contentsOf: folder.appendingPathComponent("model.safetensors"))
            try file.validate(schema: ModernBERTEncoder.schema(for: configuration))
        }
    }
}

/// Runs the encoder in MLX: only from an Xcode-built test bundle with
/// `SYSTEMONE_MLX_TESTS=1` (`SYSTEMONE_MLX_DEVICE=gpu` for the GPU).
final class ModernBERTEncoderRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private func makeEncoder(_ name: String) throws -> ModernBERTEncoder {
        let folder = ModernBERTFixture.root.appendingPathComponent(name)
        return try ModernBERTEncoder(
            store: SafetensorsFile(contentsOf: folder.appendingPathComponent("model.safetensors")),
            configuration: ModernBERTConfiguration(modelFolder: folder)
        )
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
        let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThanOrEqual(worst, 1e-4, message, file: file, line: line)
    }

    func testPaddedBatchMatchesTransformers() throws {
        let fixture = try ModernBERTFixture.load()
        try Device.withDefaultDevice(device) {
            for name in ["large-rope", "uniform-rope"] {
                let encoder = try makeEncoder(name)
                let inputs = fixture.paddedInputs
                let hidden = try encoder(inputs.ids, attention: EncoderAttention(lengths: inputs.lengths))
                let width = encoder.configuration.hiddenSize
                for (row, length) in inputs.lengths.enumerated() {
                    let real = hidden[row, 0..<length].asArray(Float.self)
                    let expected = Array(try XCTUnwrap(fixture.models[name]).padded[row].prefix(length * width))
                    assertClose(real, expected, "\(name) row \(row)")
                }
            }
        }
    }

    func testTreeLayoutMatchesTransformersWithCustomMasksAndPositions() throws {
        let fixture = try ModernBERTFixture.load()
        let tree = fixture.treeInputs
        try Device.withDefaultDevice(device) {
            for name in ["large-rope", "uniform-rope"] {
                let encoder = try makeEncoder(name)
                let hidden = try encoder(
                    [tree.ids],
                    attention: EncoderAttention(lengths: [tree.ids.count], positions: [tree.positions], visible: [tree.visible])
                )
                assertClose(hidden.asArray(Float.self), try XCTUnwrap(fixture.models[name]).tree, "\(name) tree")
            }
        }
    }

    func testExplicitContiguousPositionsMatchTheFusedRoPEPath() throws {
        let fixture = try ModernBERTFixture.load()
        let ids = [fixture.paddedInputs.ids[0]]
        try Device.withDefaultDevice(device) {
            let encoder = try makeEncoder("large-rope")
            let fused = try encoder(ids)
            // Positions equal to their indices take the fused kernel; shifted ones take the
            // explicit rotation. RoPE and the window see only position differences, so the
            // two must agree.
            XCTAssertTrue(EncoderAttention(lengths: [3], positions: [[0, 1, 2]]).hasDefaultPositions)
            let shifted = try encoder(ids, attention: EncoderAttention(lengths: [ids[0].count], positions: [ids[0].indices.map { $0 + 3 }]))
            assertClose(shifted.asArray(Float.self), fused.asArray(Float.self), "positions shifted by 3")
        }
    }
}
