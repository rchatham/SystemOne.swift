import CryptoKit
import Foundation

/// One entry in the independently defined production Von weight schema.
public struct VonWeightSpecification: Equatable, Sendable {
    public let name: String
    public let shape: [Int]

    public init(name: String, shape: [Int]) {
        self.name = name
        self.shape = shape
    }
}

/// Errors raised when validating the Von schema or pinned production artifact.
public enum VonWeightsError: Error, Equatable, Sendable {
    case productionFileSizeMismatch(expected: Int, actual: Int)
    case productionSHA256Mismatch(expected: String, actual: String)
    case tensorSetMismatch(missing: [String], unexpected: [String])
    case dataTypeMismatch(name: String, expected: SafetensorsDataType, actual: SafetensorsDataType)
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
}

extension VonWeightsError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .productionFileSizeMismatch(let expected, let actual):
            return "Production Von artifact has \(actual) bytes; expected exactly \(expected)."
        case .productionSHA256Mismatch(let expected, let actual):
            return "Production Von artifact has SHA256 \(actual); expected \(expected)."
        case .tensorSetMismatch(let missing, let unexpected):
            return "Von tensor names do not match production schema; missing \(missing), unexpected \(unexpected)."
        case .dataTypeMismatch(let name, let expected, let actual):
            return "Von tensor \(name) has dtype \(actual.rawValue); expected \(expected.rawValue)."
        case .shapeMismatch(let name, let expected, let actual):
            return "Von tensor \(name) has shape \(actual); expected \(expected)."
        }
    }
}

/// Typed access to the embedding tensors.
public struct VonEmbeddingWeights: Sendable {
    public let tokenEmbedding: SafetensorsTensorView
    public let norm: SafetensorsTensorView
}

/// Typed access to one encoder layer's attention tensors.
public struct VonAttentionWeights: Sendable {
    public let queryKeyValue: SafetensorsTensorView
    public let output: SafetensorsTensorView
}

/// Typed access to one encoder layer's MLP tensors.
public struct VonMLPWeights: Sendable {
    public let norm: SafetensorsTensorView
    public let input: SafetensorsTensorView
    public let output: SafetensorsTensorView
}

/// Typed access to one of the 28 encoder layers.
public struct VonEncoderLayerWeights: Sendable {
    public let index: Int
    public let attentionNorm: SafetensorsTensorView?
    public let attention: VonAttentionWeights
    public let mlp: VonMLPWeights
}

/// Typed access to all 170 encoder tensors.
public struct VonEncoderWeights: Sendable {
    public let embeddings: VonEmbeddingWeights
    public let layers: [VonEncoderLayerWeights]
    public let finalNorm: SafetensorsTensorView
}

/// Typed access to the eight option scorer tensors.
public struct VonScorerWeights: Sendable {
    public let inputNormWeight: SafetensorsTensorView
    public let inputNormBias: SafetensorsTensorView
    public let denseWeight: SafetensorsTensorView
    public let denseBias: SafetensorsTensorView
    public let normWeight: SafetensorsTensorView
    public let normBias: SafetensorsTensorView
    public let outputProjectionWeight: SafetensorsTensorView
    public let outputProjectionBias: SafetensorsTensorView
}

/// A Von checkpoint with exact schema validation and typed zero-copy groups.
///
/// The regular initializers validate the Von tensor schema but do not establish
/// artifact identity. Use `verifiedProduction(contentsOf:)` when the checkpoint
/// must match the pinned production artifact byte for byte.
public struct VonWeights: Sendable {
    private static let productionFileByteCount = 1_581_259_300
    private static let productionSHA256 =
        "3cadf5c5f001e859489da2cb035aa015b5b74b8d28a384abedb9d86290419e06"
    private static let hashChunkByteCount = 16 * 1_024 * 1_024

    public static let encoderTensorCount = 170
    public static let scorerTensorCount = 8
    public static let tensorCount = encoderTensorCount + scorerTensorCount

    /// The exact production tensor names and shapes, defined without fixtures.
    public static let expectedSchema: [VonWeightSpecification] = {
        var schema: [VonWeightSpecification] = [
            VonWeightSpecification(
                name: "encoder.embeddings.tok_embeddings.weight",
                shape: [50_368, 1_024]
            ),
            VonWeightSpecification(name: "encoder.embeddings.norm.weight", shape: [1_024]),
        ]

        for layer in 0...27 {
            let prefix = "encoder.layers.\(layer)"
            if layer > 0 {
                schema.append(
                    VonWeightSpecification(name: "\(prefix).attn_norm.weight", shape: [1_024])
                )
            }
            schema.append(contentsOf: [
                VonWeightSpecification(
                    name: "\(prefix).attn.Wqkv.weight",
                    shape: [3_072, 1_024]
                ),
                VonWeightSpecification(
                    name: "\(prefix).attn.Wo.weight",
                    shape: [1_024, 1_024]
                ),
                VonWeightSpecification(name: "\(prefix).mlp_norm.weight", shape: [1_024]),
                VonWeightSpecification(
                    name: "\(prefix).mlp.Wi.weight",
                    shape: [5_248, 1_024]
                ),
                VonWeightSpecification(
                    name: "\(prefix).mlp.Wo.weight",
                    shape: [1_024, 2_624]
                ),
            ])
        }

        schema.append(contentsOf: [
            VonWeightSpecification(name: "encoder.final_norm.weight", shape: [1_024]),
            VonWeightSpecification(name: "scorer.input_norm.weight", shape: [1_024]),
            VonWeightSpecification(name: "scorer.input_norm.bias", shape: [1_024]),
            VonWeightSpecification(name: "scorer.dense.weight", shape: [512, 1_024]),
            VonWeightSpecification(name: "scorer.dense.bias", shape: [512]),
            VonWeightSpecification(name: "scorer.norm.weight", shape: [512]),
            VonWeightSpecification(name: "scorer.norm.bias", shape: [512]),
            VonWeightSpecification(name: "scorer.out_proj.weight", shape: [1, 512]),
            VonWeightSpecification(name: "scorer.out_proj.bias", shape: [1]),
        ])

        precondition(schema.count == tensorCount, "Von production schema must contain 178 tensors")
        precondition(
            Set(schema.map(\.name)).count == schema.count,
            "Von production schema contains duplicate names"
        )
        return schema
    }()

    public let safetensors: SafetensorsFile
    public let encoder: VonEncoderWeights
    public let scorer: VonScorerWeights

    /// Opens a safetensors file and validates only the exact Von tensor schema.
    public init(contentsOf url: URL) throws {
        try self.init(safetensors: SafetensorsFile(contentsOf: url))
    }

    /// Verifies the pinned production artifact's whole-file size and SHA256,
    /// then validates its exact Von tensor schema using the same mapped inode.
    public static func verifiedProduction(contentsOf url: URL) throws -> VonWeights {
        let safetensors = try SafetensorsFile(
            contentsOf: url,
            validatingFileByteCount: { byteCount in
                guard byteCount == Self.productionFileByteCount else {
                    throw VonWeightsError.productionFileSizeMismatch(
                        expected: Self.productionFileByteCount,
                        actual: byteCount
                    )
                }
            },
            validatingMappedBytes: { bytes in
                var hasher = SHA256()
                var offset = 0
                while offset < bytes.count {
                    let end = min(offset + Self.hashChunkByteCount, bytes.count)
                    let chunk = UnsafeRawBufferPointer(rebasing: bytes[offset..<end])
                    hasher.update(bufferPointer: chunk)
                    offset = end
                }
                let actualHash = hasher.finalize()
                    .map { String(format: "%02x", $0) }
                    .joined()
                guard actualHash == Self.productionSHA256 else {
                    throw VonWeightsError.productionSHA256Mismatch(
                        expected: Self.productionSHA256,
                        actual: actualHash
                    )
                }
            }
        )
        return try VonWeights(safetensors: safetensors)
    }

    /// Validates only the exact Von tensor schema of an already-open file.
    public init(safetensors: SafetensorsFile) throws {
        let expectedByName = Dictionary(
            uniqueKeysWithValues: Self.expectedSchema.map { ($0.name, $0) }
        )
        let expectedNames = Set(expectedByName.keys)
        let actualNames = Set(safetensors.descriptors.keys)
        let missing = expectedNames.subtracting(actualNames).sorted()
        let unexpected = actualNames.subtracting(expectedNames).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw VonWeightsError.tensorSetMismatch(missing: missing, unexpected: unexpected)
        }

        for specification in Self.expectedSchema {
            guard let descriptor = safetensors.descriptors[specification.name] else {
                throw VonWeightsError.tensorSetMismatch(
                    missing: [specification.name],
                    unexpected: []
                )
            }
            guard descriptor.dataType == .float32 else {
                throw VonWeightsError.dataTypeMismatch(
                    name: specification.name,
                    expected: .float32,
                    actual: descriptor.dataType
                )
            }
            guard descriptor.shape == specification.shape else {
                throw VonWeightsError.shapeMismatch(
                    name: specification.name,
                    expected: specification.shape,
                    actual: descriptor.shape
                )
            }
        }

        func tensor(_ name: String) throws -> SafetensorsTensorView {
            guard let view = safetensors.tensor(named: name) else {
                throw VonWeightsError.tensorSetMismatch(missing: [name], unexpected: [])
            }
            return view
        }

        let embeddings = VonEmbeddingWeights(
            tokenEmbedding: try tensor("encoder.embeddings.tok_embeddings.weight"),
            norm: try tensor("encoder.embeddings.norm.weight")
        )
        let layers = try (0...27).map { layer -> VonEncoderLayerWeights in
            let prefix = "encoder.layers.\(layer)"
            let attentionNorm: SafetensorsTensorView?
            if layer == 0 {
                attentionNorm = nil
            } else {
                attentionNorm = try tensor("\(prefix).attn_norm.weight")
            }
            return VonEncoderLayerWeights(
                index: layer,
                attentionNorm: attentionNorm,
                attention: VonAttentionWeights(
                    queryKeyValue: try tensor("\(prefix).attn.Wqkv.weight"),
                    output: try tensor("\(prefix).attn.Wo.weight")
                ),
                mlp: VonMLPWeights(
                    norm: try tensor("\(prefix).mlp_norm.weight"),
                    input: try tensor("\(prefix).mlp.Wi.weight"),
                    output: try tensor("\(prefix).mlp.Wo.weight")
                )
            )
        }
        let encoder = VonEncoderWeights(
            embeddings: embeddings,
            layers: layers,
            finalNorm: try tensor("encoder.final_norm.weight")
        )
        let scorer = VonScorerWeights(
            inputNormWeight: try tensor("scorer.input_norm.weight"),
            inputNormBias: try tensor("scorer.input_norm.bias"),
            denseWeight: try tensor("scorer.dense.weight"),
            denseBias: try tensor("scorer.dense.bias"),
            normWeight: try tensor("scorer.norm.weight"),
            normBias: try tensor("scorer.norm.bias"),
            outputProjectionWeight: try tensor("scorer.out_proj.weight"),
            outputProjectionBias: try tensor("scorer.out_proj.bias")
        )

        self.safetensors = safetensors
        self.encoder = encoder
        self.scorer = scorer
    }
}
