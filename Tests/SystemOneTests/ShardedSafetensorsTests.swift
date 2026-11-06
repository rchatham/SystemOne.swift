import CryptoKit
import Foundation
import XCTest
@testable import SystemOne

final class ShardedSafetensorsTests: XCTestCase {
    func testOpensEveryShardAndReadsTensorsAcrossThem() throws {
        let folder = try makeCheckpoint()
        let sharded = try ShardedSafetensors(indexURL: folder.appendingPathComponent("model.safetensors.index.json"))

        XCTAssertEqual(Set(sharded.shards.keys), ["a.safetensors", "b.safetensors"])
        XCTAssertEqual(Set(sharded.descriptors.keys), ["x", "y", "z"])
        XCTAssertNoThrow(try sharded.validate(schema: Self.schema))
        XCTAssertEqual(try sharded.requiredTensor("z").withUnsafeBytes { Array($0) }, [8, 9, 10, 11])
        XCTAssertEqual(try sharded.requiredTensor("y").descriptor.dataType, .bfloat16)
        XCTAssertThrowsError(try sharded.requiredTensor("missing")) {
            XCTAssertEqual($0 as? WeightsError, .tensorSetMismatch(missing: ["missing"], unexpected: []))
        }
        XCTAssertThrowsError(try sharded.validate(schema: Array(Self.schema.dropLast()))) {
            XCTAssertEqual($0 as? WeightsError, .tensorSetMismatch(missing: [], unexpected: ["z"]))
        }
    }

    func testPinsMustCoverExactlyTheShardsAndMatchTheirBytes() throws {
        let folder = try makeCheckpoint()
        let index = folder.appendingPathComponent("model.safetensors.index.json")
        let pins = try ["a.safetensors", "b.safetensors"].reduce(into: [String: PinnedArtifact]()) {
            $0[$1] = try Self.pin(folder.appendingPathComponent($1))
        }

        XCTAssertNoThrow(try ShardedSafetensors(indexURL: index, pins: pins))

        XCTAssertThrowsError(try ShardedSafetensors(indexURL: index, pins: ["a.safetensors": pins["a.safetensors"]!])) {
            XCTAssertEqual($0 as? WeightsError, .shardPinsMismatch(missing: ["b.safetensors"], unexpected: []))
        }
        var extra = pins
        extra["c.safetensors"] = pins["a.safetensors"]
        XCTAssertThrowsError(try ShardedSafetensors(indexURL: index, pins: extra)) {
            XCTAssertEqual($0 as? WeightsError, .shardPinsMismatch(missing: [], unexpected: ["c.safetensors"]))
        }
        var wrong = pins
        wrong["b.safetensors"] = PinnedArtifact(byteCount: pins["b.safetensors"]!.byteCount, sha256: String(repeating: "0", count: 64))
        XCTAssertThrowsError(try ShardedSafetensors(indexURL: index, pins: wrong)) {
            guard case .artifactSHA256Mismatch? = $0 as? WeightsError else { return XCTFail("\($0)") }
        }
    }

    func testRejectsAnIndexThatDisagreesWithAShardHeader() throws {
        let folder = try makeCheckpoint(weightMap: ["x": "a.safetensors", "y": "b.safetensors", "z": "b.safetensors"])
        XCTAssertThrowsError(try ShardedSafetensors(indexURL: folder.appendingPathComponent("model.safetensors.index.json"))) {
            XCTAssertEqual(
                $0 as? WeightsError,
                .shardContentsMismatch(shard: "a.safetensors", missing: [], unexpected: ["y"])
            )
        }
    }

    func testRejectsShardNamesOutsideTheIndexFolder() throws {
        for name in ["../a.safetensors", "sub/a.safetensors", "", ".."] {
            let folder = try makeCheckpoint(weightMap: ["x": name, "y": "a.safetensors", "z": "b.safetensors"])
            XCTAssertThrowsError(try ShardedSafetensors(indexURL: folder.appendingPathComponent("model.safetensors.index.json")), name) {
                guard case .invalidShardIndex? = $0 as? WeightsError else { return XCTFail("\(name): \($0)") }
            }
        }
    }

    func testRejectsMalformedIndexes() throws {
        for body in ["[]", #"{"metadata":{}}"#, #"{"weight_map":{}}"#, #"{"weight_map":{"x":1}}"#] {
            let folder = try makeTemporaryDirectory()
            let index = folder.appendingPathComponent("model.safetensors.index.json")
            try Data(body.utf8).write(to: index)
            XCTAssertThrowsError(try ShardedSafetensors(indexURL: index), body) {
                guard case .invalidShardIndex? = $0 as? WeightsError else { return XCTFail("\(body): \($0)") }
            }
        }
    }
}

private extension ShardedSafetensorsTests {
    static let schema = [
        WeightSpecification(name: "x", shape: [2]),
        WeightSpecification(name: "y", shape: [2], dataType: .bfloat16),
        WeightSpecification(name: "z", shape: [1]),
    ]

    /// Shard `a` holds `x` (F32 [2]) and `y` (BF16 [2]); shard `b` holds `z` (F32 [1]).
    func makeCheckpoint(
        weightMap: [String: String] = ["x": "a.safetensors", "y": "a.safetensors", "z": "b.safetensors"]
    ) throws -> URL {
        let folder = try makeTemporaryDirectory()
        try Self.file(
            header: #"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}}"#,
            payload: Array(0..<12)
        ).write(to: folder.appendingPathComponent("a.safetensors"))
        try Self.file(
            header: #"{"__metadata__":{"format":"pt"},"z":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            payload: Array(8..<12)
        ).write(to: folder.appendingPathComponent("b.safetensors"))
        let index = try JSONSerialization.data(withJSONObject: ["metadata": ["total_size": 16], "weight_map": weightMap])
        try index.write(to: folder.appendingPathComponent("model.safetensors.index.json"))
        return folder
    }

    static func file(header: String, payload: [UInt8]) -> Data {
        let headerBytes = Array(header.utf8)
        var bytes = (0..<8).map { UInt8(truncatingIfNeeded: UInt64(headerBytes.count) >> UInt64($0 * 8)) }
        bytes += headerBytes + payload
        return Data(bytes)
    }

    static func pin(_ url: URL) throws -> PinnedArtifact {
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return PinnedArtifact(byteCount: data.count, sha256: digest)
    }

    func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ShardedSafetensorsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
