import Foundation
import MLX
import XCTest
@testable import SystemOne

/// `testdata/mlx_affine_quantization_v1.json`, written by `tools/mlx_quantization_fixture.py`.
struct QuantizationFixture: Decodable {
    struct Case: Decodable {
        let bits: Int
        let groupSize: Int
        let shape: [Int]
        let packedShape: [Int]
        let packed: [UInt32]
        let scales: [Float]
        let biases: [Float]
        let dequantized: [Float]
        let inputs: [Float]
        let output: [Float]

        var rows: Int { shape[0] }
        var columns: Int { shape[1] }
        var groups: Int { columns / groupSize }
    }

    let cases: [Case]

    static func load() throws -> QuantizationFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("testdata/mlx_affine_quantization_v1.json")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(QuantizationFixture.self, from: Data(contentsOf: url))
    }
}

final class QuantizationTests: XCTestCase {
    func testPureSwiftDequantizationMatchesMLXForEveryBitWidth() throws {
        let fixture = try QuantizationFixture.load()
        XCTAssertEqual(Set(fixture.cases.map(\.bits)), AffineQuantization.supportedBits)
        for testCase in fixture.cases {
            XCTAssertEqual(
                testCase.packedShape,
                [testCase.rows, AffineQuantization.packedWidth(columns: testCase.columns, bits: testCase.bits)]
            )
            let values = AffineQuantization.dequantize(
                packed: testCase.packed, scales: testCase.scales, biases: testCase.biases,
                rows: testCase.rows, columns: testCase.columns,
                groupSize: testCase.groupSize, bits: testCase.bits
            )
            let worst = zip(values, testCase.dequantized).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThanOrEqual(worst, 1e-6, "bits \(testCase.bits), group \(testCase.groupSize)")
        }
    }

    func testUnpackReadsValuesThatSpanWords() {
        // 3-bit values 0...7 repeated: value 10 starts at bit 30 and spans words 0 and 1.
        let values: [UInt32] = (0..<32).map { UInt32($0 % 8) }
        var words = [UInt32](repeating: 0, count: 3)
        for (index, value) in values.enumerated() {
            let offset = index * 3
            words[offset / 32] |= value << UInt32(offset % 32)
            if offset % 32 > 29 { words[offset / 32 + 1] |= value >> UInt32(32 - offset % 32) }
        }
        XCTAssertEqual(AffineQuantization.unpack(words[...], bits: 3, count: 32), values)
    }

    func testParsesMLXQuantizationConfigurationAndOverrides() throws {
        let json = """
        {"group_size": 64, "bits": 4, "mode": "affine",
         "model.layers.0.mlp.down_proj": {"group_size": 32, "bits": 8},
         "lm_head": false}
        """
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let configuration = try MLXQuantizationConfiguration(json: object)
        XCTAssertEqual(configuration.defaults, QuantizationParameters(groupSize: 64, bits: 4))
        XCTAssertEqual(configuration.parameters(for: "model.layers.1.mlp.down_proj"), QuantizationParameters(groupSize: 64, bits: 4))
        XCTAssertEqual(
            configuration.parameters(for: "model.layers.0.mlp.down_proj"), QuantizationParameters(groupSize: 32, bits: 8)
        )
        XCTAssertNil(configuration.parameters(for: "lm_head"))

        for invalid in [#"{"bits": 4}"#, #"{"group_size": 64, "bits": 4, "mode": "int3"}"#, #"{"group_size": 64, "bits": 4, "x": 1}"#] {
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(invalid.utf8)) as? [String: Any])
            XCTAssertThrowsError(try MLXQuantizationConfiguration(json: object), invalid)
        }
    }

    func testLoadsQuantizationFromConfigOnlyWhenDeclared() throws {
        let folder = try makeFolder()
        let config = folder.appendingPathComponent("config.json")
        try Data(#"{"model_type": "lfm2"}"#.utf8).write(to: config)
        XCTAssertNil(try MLXQuantizationConfiguration.load(modelFolder: folder))
        try Data(#"{"quantization_config": {"quant_method": "fp8", "bits": 8}}"#.utf8).write(to: config)
        XCTAssertNil(try MLXQuantizationConfiguration.load(modelFolder: folder))
        try Data(#"{"quantization": {"group_size": 64, "bits": 8, "mode": "affine"}}"#.utf8).write(to: config)
        XCTAssertEqual(try MLXQuantizationConfiguration.load(modelFolder: folder)?.defaults.bits, 8)
    }

    func testInfersParametersFromStoredShapes() throws {
        let folder = try makeFolder()
        let url = folder.appendingPathComponent("model.safetensors")
        // A [3, 128] weight at 4 bits in groups of 32: 16 words and 4 groups per row.
        try SafetensorsWriter.write([
            .uint32("q.weight", [3, 16], Array(repeating: 0, count: 48)),
            .halfBits("q.scales", .float16, [3, 4], Array(repeating: 0x3C00, count: 12)),
            .halfBits("q.biases", .float16, [3, 4], Array(repeating: 0, count: 12)),
            .uint32("n.weight", [3, 16], Array(repeating: 0, count: 48)),
            .halfBits("n.scales", .float16, [3, 4], Array(repeating: 0x3C00, count: 12)),
            .float32("d.weight", [2, 2], [1, 2, 3, 4]),
        ], to: url)
        let file = try SafetensorsFile(contentsOf: url)

        XCTAssertTrue(file.isQuantized("q"))
        XCTAssertFalse(file.isQuantized("d"))
        XCTAssertEqual(try file.quantizationParameters(of: "q", inputWidth: 128), QuantizationParameters(groupSize: 32, bits: 4))
        // Read as 64 inputs, the same tensors would be 8-bit groups of 16, which affine MLX never writes.
        XCTAssertThrowsError(try file.quantizationParameters(of: "q", inputWidth: 64))
        XCTAssertThrowsError(
            try file.quantizationParameters(of: "q", inputWidth: 128, configured: QuantizationParameters(groupSize: 64, bits: 4))
        ) { error in
            XCTAssertEqual(
                error as? QuantizationError,
                .parametersMismatch(
                    name: "q.weight", configured: QuantizationParameters(groupSize: 64, bits: 4),
                    stored: QuantizationParameters(groupSize: 32, bits: 4)
                )
            )
        }
        // Without biases the weight is not affine, so the mode must come from configuration.
        XCTAssertThrowsError(try file.quantizationParameters(of: "n", inputWidth: 128))
        XCTAssertEqual(
            try file.quantizationParameters(
                of: "n", inputWidth: 128, configured: QuantizationParameters(groupSize: 32, bits: 4, mode: .mxfp4)
            ),
            QuantizationParameters(groupSize: 32, bits: 4, mode: .mxfp4)
        )
        XCTAssertThrowsError(try file.quantizationParameters(of: "q", inputWidth: 100))
        XCTAssertThrowsError(try file.quantizationParameters(of: "d", inputWidth: 2))
    }

    func testMLXExportSchemaPacksQuantizedWeightsOnly() {
        let schema = [
            WeightSpecification(name: "embed.weight", shape: [10, 128]),
            WeightSpecification(name: "layer.proj.weight", shape: [64, 128]),
            WeightSpecification(name: "layer.proj.bias", shape: [64]),
            WeightSpecification(name: "layer.norm.weight", shape: [128]),
            WeightSpecification(name: "counter", shape: [], dataType: .int64),
        ]
        let exported = schema.mlxExport(floatType: .float16) { module in
            module == "layer.proj" ? QuantizationParameters(groupSize: 64, bits: 8) : nil
        }
        XCTAssertEqual(exported, [
            WeightSpecification(name: "embed.weight", shape: [10, 128], dataType: .float16),
            WeightSpecification(name: "layer.proj.weight", shape: [64, 32], dataType: .uint32),
            WeightSpecification(name: "layer.proj.scales", shape: [64, 2], dataType: .float16),
            WeightSpecification(name: "layer.proj.biases", shape: [64, 2], dataType: .float16),
            WeightSpecification(name: "layer.proj.bias", shape: [64], dataType: .float16),
            WeightSpecification(name: "layer.norm.weight", shape: [128], dataType: .float16),
            WeightSpecification(name: "counter", shape: [], dataType: .int64),
        ])
    }

    private func makeFolder() throws -> URL {
        let url = try URL.temporaryTestDirectory()
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
