import Foundation
import MLX
import MLXNN
import XCTest
@testable import SystemOne

/// Checks that run MLX. `swift test` builds lack MLX's Metal library, so these
/// run only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`
/// (`SYSTEMONE_MLX_DEVICE=gpu` for the GPU; the CPU is the default).
class MLXRuntimeTestCase: XCTestCase {
    static var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    func onDevice(_ body: () throws -> Void) rethrows {
        try Device.withDefaultDevice(Self.device, body)
    }

    func assertClose(
        _ actual: [Float], _ expected: [Float], tolerance: Float, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
        let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThanOrEqual(worst, tolerance, message, file: file, line: line)
    }
}

final class MLXQuantizationRuntimeTests: MLXRuntimeTestCase {
    func testDequantizedMatchesTheFixtureForEveryBitWidth() throws {
        let fixture = try QuantizationFixture.load()
        try onDevice {
            for testCase in fixture.cases {
                let values = MLX.dequantized(
                    MLXArray(testCase.packed, testCase.packedShape),
                    scales: MLXArray(testCase.scales, [testCase.rows, testCase.groups]),
                    biases: MLXArray(testCase.biases, [testCase.rows, testCase.groups]),
                    groupSize: testCase.groupSize, bits: testCase.bits
                )
                assertClose(values.asArray(Float.self), testCase.dequantized, tolerance: 1e-6, "bits \(testCase.bits)")
            }
        }
    }

    func testQuantizedLinearAndEmbeddingLoadFromStoredTensors() throws {
        let fixture = try QuantizationFixture.load()
        let folder = try URL.temporaryTestDirectory()
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        try onDevice {
            for (index, testCase) in fixture.cases.enumerated() {
                let url = folder.appendingPathComponent("case\(index).safetensors")
                let groupsShape = [testCase.rows, testCase.groups]
                try SafetensorsWriter.write([
                    .uint32("layer.weight", testCase.packedShape, testCase.packed),
                    .float32("layer.scales", groupsShape, testCase.scales),
                    .float32("layer.biases", groupsShape, testCase.biases),
                ], to: url)
                let file = try SafetensorsFile(contentsOf: url)

                let linear = try file.linear("layer", inputWidth: testCase.columns, bias: false, dataType: .float32)
                XCTAssertTrue(linear is QuantizedLinear)
                let inputs = MLXArray(testCase.inputs, [testCase.inputs.count / testCase.columns, testCase.columns])
                assertClose(linear(inputs).asArray(Float.self), testCase.output, tolerance: 1e-4, "linear bits \(testCase.bits)")

                let embedding = try file.embedding("layer", width: testCase.columns, dataType: .float32)
                let rows = embedding(MLXArray([Int32(2), Int32(0)]))
                XCTAssertEqual(rows.shape, [2, testCase.columns])
                let expected = Array(testCase.dequantized[(2 * testCase.columns)..<(3 * testCase.columns)])
                    + Array(testCase.dequantized[0..<testCase.columns])
                assertClose(rows.asArray(Float.self), expected, tolerance: 1e-6, "embedding bits \(testCase.bits)")
                assertClose(
                    embedding.asLinear(inputs).asArray(Float.self), testCase.output, tolerance: 1e-4,
                    "embedding as linear bits \(testCase.bits)"
                )
                assertClose(
                    try file.denseWeight("layer", inputWidth: testCase.columns, dataType: .float32).asArray(Float.self),
                    testCase.dequantized, tolerance: 1e-6, "dense bits \(testCase.bits)"
                )
            }
        }
    }

    func testHalfPrecisionTensorsLoadIntoMLX() throws {
        let folder = try URL.temporaryTestDirectory()
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("model.safetensors")
        // 1, -2.5, 65504, smallest subnormal; bfloat16 1, -2.5, 3.0e38.
        try SafetensorsWriter.write([
            .halfBits("h", .float16, [4], [0x3C00, 0xC100, 0x7BFF, 0x0001]),
            .halfBits("b", .bfloat16, [3], [0x3F80, 0xC020, 0x7F62]),
            .float32("dense.weight", [2, 2], [1, 2, 3, 4]),
        ], to: url)
        let file = try SafetensorsFile(contentsOf: url)
        try onDevice {
            let half = try file.requiredTensor("h")
            XCTAssertEqual(half.mlxArray().dtype, .float16)
            XCTAssertEqual(half.mlxArray(.float32).asArray(Float.self), half.floatValues())
            let bfloat = try file.requiredTensor("b")
            XCTAssertEqual(bfloat.mlxArray().dtype, .bfloat16)
            XCTAssertEqual(bfloat.mlxArray(.float32).asArray(Float.self), bfloat.floatValues())
            let (converted, maximum) = bfloat.float16Array()
            XCTAssertEqual(converted.dtype, .float16)
            XCTAssertEqual(maximum, bfloat.floatValues().map(abs).max())
            XCTAssertEqual(Array(converted.asType(.float32).asArray(Float.self).prefix(2)), [1, -2.5])

            let linear = try file.linear("dense", inputWidth: 2, bias: false, dataType: .float16)
            XCTAssertFalse(linear is QuantizedLinear)
            XCTAssertEqual(linear.weight.dtype, .float16)
            XCTAssertEqual(linear(MLXArray([Float(1), 1], [1, 2]).asType(.float16)).asType(.float32).asArray(Float.self), [3, 7])
        }
    }
}
