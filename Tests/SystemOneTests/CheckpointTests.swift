import CryptoKit
import Foundation
import XCTest
@testable import SystemOne

final class CheckpointTests: XCTestCase {
    private let first = "model-00001-of-00002.safetensors"
    private let second = "model-00002-of-00002.safetensors"

    func testOpensShardedCheckpointThroughItsIndex() throws {
        let folder = try makeFolder()
        try SafetensorsWriter.write([.float32("a", [2], [1, 2])], to: folder.appendingPathComponent(first))
        try SafetensorsWriter.write(
            [.float32("b", [1], [3]), .uint32("c", [2], [7, 8])], to: folder.appendingPathComponent(second)
        )
        try writeIndex(["a": first, "b": second, "c": second], in: folder)

        let checkpoint = try SafetensorsCheckpoint(modelFolder: folder)
        XCTAssertEqual(Set(checkpoint.shards.keys), [first, second])
        XCTAssertEqual(Set(checkpoint.descriptors.keys), ["a", "b", "c"])
        XCTAssertEqual(try checkpoint.requiredTensor("a").floatValues(), [1, 2])
        XCTAssertEqual(try checkpoint.requiredTensor("b").floatValues(), [3])
        XCTAssertEqual(checkpoint.tensor(named: "c")?.descriptor.dataType, .uint32)
        XCTAssertNil(checkpoint.tensor(named: "missing"))
        try checkpoint.validate(schema: [
            WeightSpecification(name: "a", shape: [2]),
            WeightSpecification(name: "b", shape: [1]),
            WeightSpecification(name: "c", shape: [2], dataType: .uint32),
        ])
    }

    func testOpensSingleFileWithoutIndex() throws {
        let folder = try makeFolder()
        try SafetensorsWriter.write([.float32("w", [1], [5])], to: folder.appendingPathComponent("model.safetensors"))
        let checkpoint = try SafetensorsCheckpoint(modelFolder: folder)
        XCTAssertEqual(Array(checkpoint.shards.keys), ["model.safetensors"])
        XCTAssertEqual(try checkpoint.requiredTensor("w").floatValues(), [5])
    }

    func testRejectsAnIndexThatDisagreesWithItsShards() throws {
        let folder = try makeFolder()
        try SafetensorsWriter.write([.float32("a", [1], [1])], to: folder.appendingPathComponent(first))
        try SafetensorsWriter.write([.float32("b", [1], [2])], to: folder.appendingPathComponent(second))

        try writeIndex(["a": second, "b": second], in: folder)
        assertCheckpointError(try SafetensorsCheckpoint(modelFolder: folder)) {
            $0 == .tensorNotInIndex(name: "a", shard: self.first)
        }
        try writeIndex(["a": first, "b": second, "z": second], in: folder)
        assertCheckpointError(try SafetensorsCheckpoint(modelFolder: folder)) {
            $0 == .tensorMissingFromShard(name: "z", shard: self.second)
        }
        try writeIndex(["a": first], in: folder)
        // Only the first shard is opened, and it agrees with the index.
        XCTAssertEqual(Set(try SafetensorsCheckpoint(modelFolder: folder).descriptors.keys), ["a"])
    }

    func testRejectsUnsafeShardNamesAndMalformedIndexes() throws {
        for shard in ["../model.safetensors", "nested/model.safetensors", "model.bin", ".hidden.safetensors"] {
            assertCheckpointError(try SafetensorsCheckpoint.validateShardName(shard)) {
                $0 == .invalidShardName(shard)
            }
        }
        for index in [#"{}"#, #"{"weight_map":{}}"#, #"{"weight_map":{"a":1}}"#, #"[1]"#, "not json"] {
            XCTAssertThrowsError(try SafetensorsCheckpoint.weightMap(fromIndex: Data(index.utf8)), index)
        }
        let empty = try makeFolder()
        assertCheckpointError(try SafetensorsCheckpoint(modelFolder: empty)) {
            $0 == .missingWeights(folder: empty.path)
        }
    }

    func testPinsMustCoverEveryShardAndMatchTheirBytes() throws {
        let folder = try makeFolder()
        let firstData = SafetensorsWriter.data([.float32("a", [1], [1])])
        let secondData = SafetensorsWriter.data([.float32("b", [1], [2])])
        try firstData.write(to: folder.appendingPathComponent(first))
        try secondData.write(to: folder.appendingPathComponent(second))
        try writeIndex(["a": first, "b": second], in: folder)

        let pins = [first: pin(firstData), second: pin(secondData)]
        XCTAssertEqual(try SafetensorsCheckpoint(modelFolder: folder, pins: pins).descriptors.count, 2)

        assertCheckpointError(try SafetensorsCheckpoint(modelFolder: folder, pins: [first: pin(firstData)])) {
            $0 == .missingPin(shard: self.second)
        }
        var extra = pins
        extra["other.safetensors"] = pin(firstData)
        assertCheckpointError(try SafetensorsCheckpoint(modelFolder: folder, pins: extra)) {
            $0 == .unusedPin(shard: "other.safetensors")
        }
        var wrong = pins
        wrong[second] = PinnedArtifact(byteCount: secondData.count, sha256: String(repeating: "0", count: 64))
        XCTAssertThrowsError(try SafetensorsCheckpoint(modelFolder: folder, pins: wrong)) { error in
            guard let weightsError = error as? WeightsError, case .artifactSHA256Mismatch = weightsError else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    private func pin(_ data: Data) -> PinnedArtifact {
        PinnedArtifact(byteCount: data.count, sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }

    private func writeIndex(_ map: [String: String], in folder: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: ["metadata": ["total_size": 0], "weight_map": map])
        try data.write(to: folder.appendingPathComponent(SafetensorsCheckpoint.indexFileName))
    }

    private func makeFolder() throws -> URL {
        let url = try URL.temporaryTestDirectory()
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func assertCheckpointError<T>(
        _ expression: @autoclosure () throws -> T,
        file: StaticString = #filePath, line: UInt = #line,
        _ predicate: (CheckpointError) -> Bool
    ) {
        XCTAssertThrowsError(try expression(), file: file, line: line) { error in
            guard let error = error as? CheckpointError else {
                return XCTFail("Unexpected error type: \(error)", file: file, line: line)
            }
            XCTAssertTrue(predicate(error), "Unexpected error: \(error)", file: file, line: line)
        }
    }
}
