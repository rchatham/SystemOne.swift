import Foundation
import SystemOne

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
    /// The converted `option_marker.safetensors` produced by
    /// `tools/convert_option_marker.py` from the pinned von checkpoint.
    public static let productionArtifact = PinnedArtifact(
        byteCount: 1_581_259_300,
        sha256: "3cadf5c5f001e859489da2cb035aa015b5b74b8d28a384abedb9d86290419e06"
    )

    public static let encoderTensorCount = 170
    public static let scorerTensorCount = 8
    public static let tensorCount = encoderTensorCount + scorerTensorCount

    /// The exact production tensor names and shapes, defined without fixtures.
    public static let expectedSchema: [WeightSpecification] = {
        var schema: [WeightSpecification] = [
            WeightSpecification(
                name: "encoder.embeddings.tok_embeddings.weight",
                shape: [50_368, 1_024]
            ),
            WeightSpecification(name: "encoder.embeddings.norm.weight", shape: [1_024]),
        ]

        for layer in 0...27 {
            let prefix = "encoder.layers.\(layer)"
            if layer > 0 {
                schema.append(
                    WeightSpecification(name: "\(prefix).attn_norm.weight", shape: [1_024])
                )
            }
            schema.append(contentsOf: [
                WeightSpecification(
                    name: "\(prefix).attn.Wqkv.weight",
                    shape: [3_072, 1_024]
                ),
                WeightSpecification(
                    name: "\(prefix).attn.Wo.weight",
                    shape: [1_024, 1_024]
                ),
                WeightSpecification(name: "\(prefix).mlp_norm.weight", shape: [1_024]),
                WeightSpecification(
                    name: "\(prefix).mlp.Wi.weight",
                    shape: [5_248, 1_024]
                ),
                WeightSpecification(
                    name: "\(prefix).mlp.Wo.weight",
                    shape: [1_024, 2_624]
                ),
            ])
        }

        schema.append(contentsOf: [
            WeightSpecification(name: "encoder.final_norm.weight", shape: [1_024]),
            WeightSpecification(name: "scorer.input_norm.weight", shape: [1_024]),
            WeightSpecification(name: "scorer.input_norm.bias", shape: [1_024]),
            WeightSpecification(name: "scorer.dense.weight", shape: [512, 1_024]),
            WeightSpecification(name: "scorer.dense.bias", shape: [512]),
            WeightSpecification(name: "scorer.norm.weight", shape: [512]),
            WeightSpecification(name: "scorer.norm.bias", shape: [512]),
            WeightSpecification(name: "scorer.out_proj.weight", shape: [1, 512]),
            WeightSpecification(name: "scorer.out_proj.bias", shape: [1]),
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
        try VonWeights(safetensors: SafetensorsFile(contentsOf: url, pinnedTo: productionArtifact))
    }

    /// Validates only the exact Von tensor schema of an already-open file.
    public init(safetensors: SafetensorsFile) throws {
        try safetensors.validate(schema: Self.expectedSchema)
        let tensor = safetensors.requiredTensor

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
