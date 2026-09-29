import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import VonSwift

final class VonWeightsTests: XCTestCase {
    func testIndependentProductionSchemaHasExactNamesAndShapes() {
        let expected = productionSpecifications()
        XCTAssertEqual(expected.count, 178)
        XCTAssertEqual(Set(expected.map(\.name)).count, 178)
        XCTAssertEqual(
            expected.filter { $0.name.hasPrefix("encoder.") }.count,
            170
        )
        XCTAssertEqual(
            expected.filter { $0.name.hasPrefix("scorer.") }.count,
            8
        )
        XCTAssertEqual(
            VonWeights.expectedSchema,
            expected.map { VonWeightSpecification(name: $0.name, shape: $0.shape) }
        )
        XCTAssertFalse(expected.contains { $0.name == "encoder.layers.0.attn_norm.weight" })
        for layer in 1...27 {
            XCTAssertTrue(
                expected.contains { $0.name == "encoder.layers.\(layer).attn_norm.weight" }
            )
        }
    }

    func testSchemaOnlyLoadsExactSizeSparseArtifactButProductionVerificationRejectsHash() throws {
        let specifications = productionSpecifications()
        let url = try makeSparseSafetensors(specifications)

        var status = stat()
        XCTAssertEqual(lstat(url.path, &status), 0)
        XCTAssertEqual(Int64(status.st_size), 1_581_259_300)
        XCTAssertLessThan(Int64(status.st_blocks) * 512, 1_000_000)

        let weights = try VonWeights(contentsOf: url)
        XCTAssertEqual(weights.safetensors.descriptors.count, 178)
        XCTAssertEqual(weights.safetensors.payloadByteCount, 1_581_240_324)
        XCTAssertEqual(
            weights.encoder.embeddings.tokenEmbedding.descriptor.name,
            "encoder.embeddings.tok_embeddings.weight"
        )
        XCTAssertEqual(weights.encoder.embeddings.tokenEmbedding.descriptor.shape, [50_368, 1_024])
        XCTAssertEqual(
            weights.encoder.embeddings.norm.descriptor.name,
            "encoder.embeddings.norm.weight"
        )
        XCTAssertEqual(weights.encoder.layers.count, 28)

        for (index, layer) in weights.encoder.layers.enumerated() {
            let prefix = "encoder.layers.\(index)"
            XCTAssertEqual(layer.index, index)
            if index == 0 {
                XCTAssertNil(layer.attentionNorm)
            } else {
                XCTAssertEqual(layer.attentionNorm?.descriptor.name, "\(prefix).attn_norm.weight")
                XCTAssertEqual(layer.attentionNorm?.descriptor.shape, [1_024])
            }
            XCTAssertEqual(layer.attention.queryKeyValue.descriptor.name, "\(prefix).attn.Wqkv.weight")
            XCTAssertEqual(layer.attention.queryKeyValue.descriptor.shape, [3_072, 1_024])
            XCTAssertEqual(layer.attention.output.descriptor.name, "\(prefix).attn.Wo.weight")
            XCTAssertEqual(layer.attention.output.descriptor.shape, [1_024, 1_024])
            XCTAssertEqual(layer.mlp.norm.descriptor.name, "\(prefix).mlp_norm.weight")
            XCTAssertEqual(layer.mlp.norm.descriptor.shape, [1_024])
            XCTAssertEqual(layer.mlp.input.descriptor.name, "\(prefix).mlp.Wi.weight")
            XCTAssertEqual(layer.mlp.input.descriptor.shape, [5_248, 1_024])
            XCTAssertEqual(layer.mlp.output.descriptor.name, "\(prefix).mlp.Wo.weight")
            XCTAssertEqual(layer.mlp.output.descriptor.shape, [1_024, 2_624])
        }

        XCTAssertEqual(weights.encoder.finalNorm.descriptor.name, "encoder.final_norm.weight")
        XCTAssertEqual(weights.scorer.inputNormWeight.descriptor.name, "scorer.input_norm.weight")
        XCTAssertEqual(weights.scorer.inputNormBias.descriptor.name, "scorer.input_norm.bias")
        XCTAssertEqual(weights.scorer.denseWeight.descriptor.shape, [512, 1_024])
        XCTAssertEqual(weights.scorer.denseBias.descriptor.shape, [512])
        XCTAssertEqual(weights.scorer.normWeight.descriptor.shape, [512])
        XCTAssertEqual(weights.scorer.normBias.descriptor.shape, [512])
        XCTAssertEqual(weights.scorer.outputProjectionWeight.descriptor.shape, [1, 512])
        XCTAssertEqual(weights.scorer.outputProjectionBias.descriptor.shape, [1])

        XCTAssertEqual(
            weights.encoder.embeddings.tokenEmbedding.withUnsafeBytes { $0.first },
            0
        )
        XCTAssertEqual(
            weights.scorer.outputProjectionBias.withUnsafeBytes { $0.last },
            0
        )

        XCTAssertThrowsError(try VonWeights.verifiedProduction(contentsOf: url)) { error in
            guard let weightsError = error as? VonWeightsError,
                  case .productionSHA256Mismatch(let expected, let actual) = weightsError
            else {
                return XCTFail("Expected production SHA256 mismatch, got \(error)")
            }
            XCTAssertEqual(
                expected,
                "3cadf5c5f001e859489da2cb035aa015b5b74b8d28a384abedb9d86290419e06"
            )
            XCTAssertNotEqual(actual, expected)
            XCTAssertEqual(actual.count, 64)
        }
    }

    func testVerifiedProductionRejectsSizeBeforeMappingParsingOrHashing() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VonWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("wrong-size.safetensors")
        try Data().write(to: url)

        XCTAssertThrowsError(try VonWeights.verifiedProduction(contentsOf: url)) { error in
            XCTAssertEqual(
                error as? VonWeightsError,
                .productionFileSizeMismatch(expected: 1_581_259_300, actual: 0)
            )
        }
    }

    func testRejectsMissingAndUnexpectedProductionNames() throws {
        var specifications = productionSpecifications()
        let removed = specifications.removeFirst()
        specifications.append(TestSpecification(name: "unexpected.weight", shape: [0]))
        let file = try SafetensorsFile(contentsOf: makeSparseSafetensors(specifications))

        XCTAssertThrowsError(try VonWeights(safetensors: file)) { error in
            XCTAssertEqual(
                error as? VonWeightsError,
                .tensorSetMismatch(
                    missing: [removed.name],
                    unexpected: ["unexpected.weight"]
                )
            )
        }
    }

    func testRejectsProductionShapeMismatchWithSameByteCount() throws {
        var specifications = productionSpecifications()
        let index = try XCTUnwrap(
            specifications.firstIndex { $0.name == "scorer.dense.weight" }
        )
        specifications[index] = TestSpecification(
            name: "scorer.dense.weight",
            shape: [1_024, 512]
        )
        let file = try SafetensorsFile(contentsOf: makeSparseSafetensors(specifications))

        XCTAssertThrowsError(try VonWeights(safetensors: file)) { error in
            XCTAssertEqual(
                error as? VonWeightsError,
                .shapeMismatch(
                    name: "scorer.dense.weight",
                    expected: [512, 1_024],
                    actual: [1_024, 512]
                )
            )
        }
    }

    func testProductionWeightsAgainstManifestWhenConfigured() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["VON_PRODUCTION_WEIGHTS_PATH"], !path.isEmpty else {
            throw XCTSkip("Set VON_PRODUCTION_WEIGHTS_PATH to run the production mmap/hash test.")
        }

        let metadataData = try Data(contentsOf: fixtureURL("option_marker_tensor_metadata_v1.json"))
        let manifestData = try Data(contentsOf: fixtureURL("option_marker_weights_manifest_v1.json"))
        let metadata = try JSONDecoder().decode(TensorMetadataFixture.self, from: metadataData)
        let manifest = try JSONDecoder().decode(WeightsManifest.self, from: manifestData)

        XCTAssertEqual(manifest.schemaVersion, 1)
        XCTAssertEqual(manifest.source.artifactName, "option_marker.pt")
        XCTAssertEqual(manifest.source.repo, "wfzyx/von")
        XCTAssertEqual(manifest.source.revision, "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd")
        XCTAssertEqual(
            manifest.source.sha256,
            "52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2"
        )
        XCTAssertEqual(manifest.output.artifactName, "option_marker.safetensors")
        XCTAssertEqual(manifest.output.format, "safetensors")
        XCTAssertEqual(manifest.output.dtype, "F32")
        XCTAssertEqual(manifest.output.tensorCount, 178)
        XCTAssertEqual(manifest.output.headerLength, 18_968)
        XCTAssertEqual(manifest.output.dataStart, 18_976)
        XCTAssertEqual(manifest.output.payloadBytes, 1_581_240_324)
        XCTAssertEqual(manifest.output.fileSize, 1_581_259_300)
        XCTAssertEqual(manifest.output.dataStart, 8 + manifest.output.headerLength)
        XCTAssertEqual(manifest.tensorMetadata.artifactName, "option_marker_tensor_metadata_v1.json")
        XCTAssertEqual(sha256(metadataData), manifest.tensorMetadata.sha256)

        XCTAssertEqual(metadata.schemaVersion, 1)
        XCTAssertEqual(metadata.artifactName, manifest.output.artifactName)
        XCTAssertEqual(metadata.dtype, manifest.output.dtype)
        XCTAssertEqual(metadata.tensorCount, manifest.output.tensorCount)
        XCTAssertEqual(metadata.encoderTensorCount, 170)
        XCTAssertEqual(metadata.scorerTensorCount, 8)
        XCTAssertEqual(metadata.payloadBytes, manifest.output.payloadBytes)
        XCTAssertEqual(metadata.tensors.count, 178)

        XCTAssertEqual(
            manifest.output.sha256,
            "3cadf5c5f001e859489da2cb035aa015b5b74b8d28a384abedb9d86290419e06"
        )

        let weights = try VonWeights.verifiedProduction(
            contentsOf: URL(fileURLWithPath: path)
        )
        let file = weights.safetensors
        XCTAssertEqual(file.fileByteCount, manifest.output.fileSize)
        XCTAssertEqual(file.headerByteCount, manifest.output.headerLength)
        XCTAssertEqual(file.dataStart, manifest.output.dataStart)
        XCTAssertEqual(file.payloadByteCount, manifest.output.payloadBytes)
        XCTAssertNil(file.metadata)
        XCTAssertEqual(file.descriptors.count, 178)

        XCTAssertEqual(weights.encoder.layers.count, 28)
        XCTAssertEqual(Set(weights.safetensors.descriptors.keys), Set(metadata.tensors.map(\.name)))

        for tensor in metadata.tensors {
            guard tensor.dataOffsets.count == 2 else {
                XCTFail("\(tensor.name) fixture must contain exactly two offsets")
                continue
            }
            let start = tensor.dataOffsets[0]
            let end = tensor.dataOffsets[1]
            guard start >= 0, start <= end else {
                XCTFail("\(tensor.name) fixture offsets must be nonnegative and ordered")
                continue
            }
            let descriptor = try XCTUnwrap(file.descriptors[tensor.name], tensor.name)
            XCTAssertEqual(descriptor.dataType.rawValue, tensor.dtype, tensor.name)
            XCTAssertEqual(descriptor.shape, tensor.shape, tensor.name)
            XCTAssertEqual(descriptor.payloadRange, start..<end, tensor.name)
            XCTAssertEqual(descriptor.byteCount, tensor.byteCount, tensor.name)
            let view = try XCTUnwrap(file.tensor(named: tensor.name), tensor.name)
            XCTAssertEqual(view.withUnsafeBytes(sha256), tensor.sha256, tensor.name)
        }
    }
}

private extension VonWeightsTests {
    struct TestSpecification: Equatable {
        let name: String
        let shape: [Int]

        var byteCount: Int {
            shape.reduce(1, *) * 4
        }
    }

    struct TensorMetadataFixture: Decodable {
        let schemaVersion: Int
        let artifactName: String
        let dtype: String
        let tensorCount: Int
        let encoderTensorCount: Int
        let scorerTensorCount: Int
        let payloadBytes: Int
        let tensors: [TensorRecord]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case artifactName = "artifact_name"
            case dtype
            case tensorCount = "tensor_count"
            case encoderTensorCount = "encoder_tensor_count"
            case scorerTensorCount = "scorer_tensor_count"
            case payloadBytes = "payload_bytes"
            case tensors
        }
    }

    struct TensorRecord: Decodable {
        let name: String
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
        let byteCount: Int
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case name
            case dtype
            case shape
            case dataOffsets = "data_offsets"
            case byteCount = "byte_count"
            case sha256
        }
    }

    struct WeightsManifest: Decodable {
        let schemaVersion: Int
        let source: Source
        let output: Output
        let tensorMetadata: TensorMetadata

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case source
            case output
            case tensorMetadata = "tensor_metadata"
        }

        struct Source: Decodable {
            let artifactName: String
            let repo: String
            let revision: String
            let sha256: String

            enum CodingKeys: String, CodingKey {
                case artifactName = "artifact_name"
                case repo
                case revision
                case sha256
            }
        }

        struct Output: Decodable {
            let artifactName: String
            let dataStart: Int
            let dtype: String
            let fileSize: Int
            let format: String
            let headerLength: Int
            let payloadBytes: Int
            let sha256: String
            let tensorCount: Int

            enum CodingKeys: String, CodingKey {
                case artifactName = "artifact_name"
                case dataStart = "data_start"
                case dtype
                case fileSize = "file_size"
                case format
                case headerLength = "header_length"
                case payloadBytes = "payload_bytes"
                case sha256
                case tensorCount = "tensor_count"
            }
        }

        struct TensorMetadata: Decodable {
            let artifactName: String
            let sha256: String

            enum CodingKeys: String, CodingKey {
                case artifactName = "artifact_name"
                case sha256
            }
        }
    }

    func productionSpecifications() -> [TestSpecification] {
        var specifications = [
            TestSpecification(
                name: "encoder.embeddings.tok_embeddings.weight",
                shape: [50_368, 1_024]
            ),
            TestSpecification(name: "encoder.embeddings.norm.weight", shape: [1_024]),
        ]
        for layer in 0...27 {
            let prefix = "encoder.layers.\(layer)"
            if layer != 0 {
                specifications.append(
                    TestSpecification(name: "\(prefix).attn_norm.weight", shape: [1_024])
                )
            }
            specifications.append(contentsOf: [
                TestSpecification(name: "\(prefix).attn.Wqkv.weight", shape: [3_072, 1_024]),
                TestSpecification(name: "\(prefix).attn.Wo.weight", shape: [1_024, 1_024]),
                TestSpecification(name: "\(prefix).mlp_norm.weight", shape: [1_024]),
                TestSpecification(name: "\(prefix).mlp.Wi.weight", shape: [5_248, 1_024]),
                TestSpecification(name: "\(prefix).mlp.Wo.weight", shape: [1_024, 2_624]),
            ])
        }
        specifications.append(contentsOf: [
            TestSpecification(name: "encoder.final_norm.weight", shape: [1_024]),
            TestSpecification(name: "scorer.input_norm.weight", shape: [1_024]),
            TestSpecification(name: "scorer.input_norm.bias", shape: [1_024]),
            TestSpecification(name: "scorer.dense.weight", shape: [512, 1_024]),
            TestSpecification(name: "scorer.dense.bias", shape: [512]),
            TestSpecification(name: "scorer.norm.weight", shape: [512]),
            TestSpecification(name: "scorer.norm.bias", shape: [512]),
            TestSpecification(name: "scorer.out_proj.weight", shape: [1, 512]),
            TestSpecification(name: "scorer.out_proj.bias", shape: [1]),
        ])
        return specifications
    }

    func makeSparseSafetensors(_ specifications: [TestSpecification]) throws -> URL {
        let productionHeaderByteCount = 18_968
        var cursor = 0
        var entries: [String] = []
        entries.reserveCapacity(specifications.count)
        for specification in specifications.sorted(by: { $0.name < $1.name }) {
            let start = cursor
            cursor += specification.byteCount
            let shape = specification.shape.map(String.init).joined(separator: ",")
            entries.append(
                "\"\(specification.name)\":{\"dtype\":\"F32\",\"shape\":[\(shape)],\"data_offsets\":[\(start),\(cursor)]}"
            )
        }
        var header = Array(("{" + entries.joined(separator: ",") + "}").utf8)
        guard header.count <= productionHeaderByteCount else {
            throw SafetensorsError.headerTooLarge(
                actual: UInt64(header.count),
                maximum: UInt64(productionHeaderByteCount)
            )
        }
        header.append(
            contentsOf: repeatElement(
                UInt8(ascii: " "),
                count: productionHeaderByteCount - header.count
            )
        )

        var prefix = [UInt8](repeating: 0, count: 8)
        let headerLength = UInt64(header.count)
        for index in 0..<8 {
            prefix[index] = UInt8(truncatingIfNeeded: headerLength >> UInt64(index * 8))
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VonWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("weights.safetensors")
        try Data(prefix + header).write(to: url)

        let descriptor = Darwin.open(url.path, O_RDWR)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { Darwin.close(descriptor) }
        let totalByteCount = 8 + productionHeaderByteCount + cursor
        guard ftruncate(descriptor, off_t(totalByteCount)) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return url
    }

    func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("testdata/\(name)", isDirectory: false)
    }

    func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func sha256(_ bytes: UnsafeRawBufferPointer) -> String {
        var hasher = SHA256()
        hasher.update(bufferPointer: bytes)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
