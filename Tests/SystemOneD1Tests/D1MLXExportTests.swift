import Foundation
import MLX
import MLXNN
import SystemOne
import XCTest
@testable import SystemOneD1

/// mlx-vlm exports of d1-omni (`LiquidAI/d1-omni-600M-MLX-*`).
///
/// `testdata/d1_omni_600m_mlx_exports_v1.json` holds the published exports'
/// `config.json` and tensor names; `testdata/d1_tiny/` is a tiny random model in
/// the PyTorch layout plus the float16 and 8-bit exports mlx-vlm's converter
/// writes from it, with mlx-vlm's logits (`tools/d1_mlx_export_fixture.py`).
final class D1MLXExportTests: XCTestCase {
    private struct PublishedExports: Decodable {
        let tensors: [String]
        let quantized8Bit: [String]

        enum CodingKeys: String, CodingKey {
            case tensors
            case quantized8Bit = "quantized_8bit"
        }
    }

    func testExportSchemaNamesTheTensorsOfThePublishedExports() throws {
        let data = try Data(contentsOf: D1Fixture.repositoryRoot.appendingPathComponent(
            "testdata/d1_omni_600m_mlx_exports_v1.json"
        ))
        let exports = try JSONDecoder().decode(PublishedExports.self, from: data)
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let configuration = try JSONDecoder().decode(
            D1Configuration.self, from: JSONSerialization.data(withJSONObject: try XCTUnwrap(root["config_8bit"]))
        )
        XCTAssertEqual(exports.tensors.count, 1_075)
        XCTAssertEqual(exports.quantized8Bit.count, 263)

        let float16 = D1Weights.mlxSchema(for: configuration, floatType: .float16) { _ in nil }
        XCTAssertEqual(float16.map(\.name).sorted(), exports.tensors)
        XCTAssertTrue(float16.allSatisfy { $0.dataType == .float16 })

        let quantized = Set(exports.quantized8Bit)
        let eightBit = D1Weights.mlxSchema(for: configuration, floatType: .float16) {
            quantized.contains($0) ? QuantizationParameters(groupSize: 64, bits: 8) : nil
        }
        let expected = exports.tensors + exports.quantized8Bit.flatMap { ["\($0).scales", "\($0).biases"] }
        XCTAssertEqual(eightBit.map(\.name).sorted(), expected.sorted())
        XCTAssertEqual(eightBit.filter { $0.dataType == .uint32 }.count, quantized.count)
    }

    func testNamesAndLayoutsFollowMLXVLM() {
        let renames = [
            ("vision.tower.vision_model.encoder.layers.0.mlp.fc1", "vision.tower.encoder.layers.0.mlp.fc1"),
            ("head.head.layers.1.self_attn.out_proj", "head.head.layers.1.attention.out_proj"),
            ("head.scorer.3.bias", "head.scorer.layers.3.bias"),
            ("encoder.layers.2.self_attn.out_proj.weight", "encoder.layers.2.self_attn.out_proj.weight"),
            ("audio.encoder.pre_encode.conv.5.weight", "audio.encoder.subsampling.layers.5.weight"),
            ("audio.encoder.pre_encode.out", "audio.encoder.subsampling.linear"),
            ("audio.encoder.layers.3.self_attn.linear_q", "audio.encoder.layers.3.self_attn.q_proj"),
            ("audio.encoder.layers.3.self_attn.linear_pos.weight", "audio.encoder.layers.3.self_attn.relative_k_proj.weight"),
            ("audio.encoder.layers.3.self_attn.pos_bias_u", "audio.encoder.layers.3.self_attn.bias_u"),
            ("audio.encoder.layers.3.conv.batch_norm.running_var", "audio.encoder.layers.3.conv.norm.running_var"),
            ("audio.adapter.linear_1", "audio.adapter.linear_1"),
        ]
        for (pytorch, mlx) in renames {
            XCTAssertEqual(D1Weights.mlxName(pytorch), mlx)
        }
        XCTAssertEqual(D1Weights.convolutionPermutation("encoder.layers.0.conv.conv.weight", rank: 3), [0, 2, 1])
        XCTAssertEqual(D1Weights.convolutionPermutation("audio.encoder.pre_encode.conv.0.weight", rank: 4), [0, 2, 3, 1])
        XCTAssertEqual(D1Weights.convolutionPermutation("audio.encoder.layers.0.conv.depthwise_conv.weight", rank: 3), [0, 2, 1])
        XCTAssertNil(D1Weights.convolutionPermutation("audio.encoder.layers.0.conv.depthwise_conv.bias", rank: 1))
        XCTAssertNil(D1Weights.convolutionPermutation("encoder.layers.0.conv.in_proj.weight", rank: 2))
        XCTAssertEqual(D1Weights.inverse([0, 2, 3, 1]), [0, 3, 1, 2])
    }

    func testTinyCheckpointsValidateInEveryLayout() throws {
        for (name, layout) in [("pytorch", D1WeightLayout.pytorch), ("mlx-fp16", .mlx), ("mlx-8bit", .mlx)] {
            let folder = Self.tinyFolder(name)
            let configuration = try D1Configuration(modelFolder: folder)
            let weights = try D1Weights(modelFolder: folder, configuration: configuration)
            XCTAssertEqual(weights.layout, layout, name)
        }

        let folder = Self.tinyFolder("mlx-8bit")
        let configuration = try D1Configuration(modelFolder: folder)
        let store = try SafetensorsCheckpoint(modelFolder: folder)
        XCTAssertEqual(try MLXQuantizationConfiguration.load(modelFolder: folder)?.defaults, QuantizationParameters(groupSize: 64, bits: 8))
        // Stored shapes say 8-bit groups of 64; a configuration that disagrees is refused.
        XCTAssertThrowsError(try D1Weights(
            store: store, configuration: configuration,
            quantization: MLXQuantizationConfiguration(defaults: QuantizationParameters(groupSize: 32, bits: 8))
        )) { error in
            guard case QuantizationError.parametersMismatch? = error as? QuantizationError else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    static func tinyFolder(_ name: String) -> URL {
        D1Fixture.repositoryRoot.appendingPathComponent("testdata/d1_tiny/\(name)", isDirectory: true)
    }
}

/// Runs the tiny checkpoints through MLX: only from an Xcode-built test bundle
/// with `SYSTEMONE_MLX_TESTS=1` (`SYSTEMONE_MLX_DEVICE=gpu` for the GPU).
final class D1MLXExportRuntimeTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Row: Decodable {
            let ids: [Int]
            let markers: [Int]
            let type: Int
        }

        let rows: [Row]
        let logits: [String: [[Float]]]
    }

    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    /// Swift's logits for every layout against mlx-vlm's. float32 compute must
    /// agree closely, the 8-bit export included (mlx-vlm's quantized and
    /// dequantized float32 paths agree exactly). float16 kernels round
    /// differently: float16 alone moves these random-weight logits by up to
    /// 9e-3, and float16 quantized matmuls by up to 0.035 (mlx-vlm's own
    /// quantized and dequantized float16 paths differ by 0.02), so the float16
    /// 8-bit check is a coarse one, still well below the 0.1 that 8-bit
    /// quantization itself moves them.
    func testTinyLogitsMatchMLXVLMInEveryLayout() throws {
        let fixture = try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: D1Fixture.repositoryRoot.appendingPathComponent("testdata/d1_tiny/fixture.json"))
        )
        let rows = fixture.rows.map {
            D1Row(ids: $0.ids, markers: $0.markers, questionType: D1DecisionHead.QuestionType(rawValue: $0.type)!, prefix: nil)
        }
        let device: Device = ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
        let cases: [(name: String, reference: String, precision: D1Precision, tolerance: Float)] = [
            ("pytorch", "pytorch", .float32, 1e-4), ("mlx-fp16", "mlx-fp16", .float16, 1e-2),
            ("mlx-8bit", "mlx-8bit-float32", .float32, 1e-3), ("mlx-8bit", "mlx-8bit", .float16, 5e-2),
        ]
        try Device.withDefaultDevice(device) {
            for (name, reference, precision, tolerance) in cases {
                let folder = D1MLXExportTests.tinyFolder(name)
                let configuration = try D1Configuration(modelFolder: folder)
                let model = try D1Model(weights: D1Weights(modelFolder: folder, configuration: configuration, precision: precision))
                let logits = try model.logits(for: rows).asType(.float32)
                eval(logits)
                let width = logits.dim(1)
                let flat = logits.asArray(Float.self)
                let expected = try XCTUnwrap(fixture.logits[reference])
                for (index, row) in rows.enumerated() {
                    let actual = Array(flat[(index * width)..<(index * width + row.markers.count)])
                    let worst = zip(actual, expected[index]).map { abs($0 - $1) }.max() ?? 0
                    XCTAssertLessThanOrEqual(worst, tolerance, "\(reference) row \(index): \(actual) vs \(expected[index])")
                }
            }
        }
    }
}
