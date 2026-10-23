import Foundation
import MLX
import SystemOne

/// A d1-omni checkpoint validated against the exact tensor schema its
/// configuration implies: trunk, decision head, vision tower, and audio encoder.
public struct D1Weights: Sendable {
    /// `model.safetensors` at `LiquidAI/d1-omni-600M` revision
    /// `02b55d7076f15129e59ab3f94783f32c4b088674`.
    public static let d1Omni600MArtifact = PinnedArtifact(
        byteCount: 2_348_774_500,
        sha256: "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1"
    )

    public let safetensors: SafetensorsFile
    public let configuration: D1Configuration

    /// Opens a checkpoint and validates its schema without pinning its bytes.
    public init(contentsOf url: URL, configuration: D1Configuration) throws {
        try self.init(safetensors: SafetensorsFile(contentsOf: url), configuration: configuration)
    }

    /// Verifies whole-file size and SHA-256 before validating the schema.
    public init(contentsOf url: URL, pinnedTo artifact: PinnedArtifact, configuration: D1Configuration) throws {
        try self.init(safetensors: SafetensorsFile(contentsOf: url, pinnedTo: artifact), configuration: configuration)
    }

    public init(safetensors: SafetensorsFile, configuration: D1Configuration) throws {
        try safetensors.validate(schema: Self.schema(for: configuration))
        self.safetensors = safetensors
        self.configuration = configuration
    }

    /// Copies one float32 tensor into an MLX-owned array.
    func array(_ name: String) -> MLXArray {
        // The schema was validated at load, so every requested name exists.
        safetensors.tensor(named: name)!.mlxArray()
    }

    /// Every tensor name, shape, and dtype the configuration implies.
    public static func schema(for configuration: D1Configuration) -> [WeightSpecification] {
        var schema: [WeightSpecification] = []
        func add(_ name: String, _ shape: [Int], _ dataType: SafetensorsDataType = .float32) {
            schema.append(WeightSpecification(name: name, shape: shape, dataType: dataType))
        }
        func linear(_ name: String, _ output: Int, _ input: Int, bias: Bool = true) {
            add("\(name).weight", [output, input])
            if bias { add("\(name).bias", [output]) }
        }
        func norm(_ name: String, _ width: Int) {
            add("\(name).weight", [width])
            add("\(name).bias", [width])
        }

        let text = configuration.text
        let d = text.hiddenSize
        let headDimension = text.headDimension
        add("encoder.embed_tokens.weight", [text.vocabSize, d])
        add("encoder.embedding_norm.weight", [d])
        for index in 0..<text.numHiddenLayers {
            let prefix = "encoder.layers.\(index)"
            if text.isAttentionLayer(index) {
                linear("\(prefix).self_attn.q_proj", text.numAttentionHeads * headDimension, d, bias: false)
                linear("\(prefix).self_attn.k_proj", text.numKeyValueHeads * headDimension, d, bias: false)
                linear("\(prefix).self_attn.v_proj", text.numKeyValueHeads * headDimension, d, bias: false)
                linear("\(prefix).self_attn.out_proj", d, text.numAttentionHeads * headDimension, bias: false)
                add("\(prefix).self_attn.q_layernorm.weight", [headDimension])
                add("\(prefix).self_attn.k_layernorm.weight", [headDimension])
            } else {
                add("\(prefix).conv.conv.weight", [d, 1, text.convLCache])
                linear("\(prefix).conv.in_proj", 3 * d, d, bias: false)
                linear("\(prefix).conv.out_proj", d, d, bias: false)
            }
            linear("\(prefix).feed_forward.w1", text.feedForwardSize, d, bias: false)
            linear("\(prefix).feed_forward.w3", text.feedForwardSize, d, bias: false)
            linear("\(prefix).feed_forward.w2", d, text.feedForwardSize, bias: false)
            add("\(prefix).operator_norm.weight", [d])
            add("\(prefix).ffn_norm.weight", [d])
        }

        add("head.type_emb.weight", [3, d])
        for index in 0..<configuration.headLayers {
            let prefix = "head.head.layers.\(index)"
            add("\(prefix).self_attn.in_proj_weight", [3 * d, d])
            add("\(prefix).self_attn.in_proj_bias", [3 * d])
            linear("\(prefix).self_attn.out_proj", d, d)
            linear("\(prefix).linear1", 4 * d, d)
            linear("\(prefix).linear2", d, 4 * d)
            norm("\(prefix).norm1", d)
            norm("\(prefix).norm2", d)
        }
        norm("head.scorer.0", d)
        linear("head.scorer.1", d, d)
        linear("head.scorer.3", 1, d)

        let vision = configuration.vision
        let v = vision.hiddenSize
        let tower = "vision.tower.vision_model"
        linear("\(tower).embeddings.patch_embedding", v, vision.numChannels * vision.patchSize * vision.patchSize)
        add("\(tower).embeddings.position_embedding.weight", [vision.numPatches, v])
        for index in 0..<vision.numHiddenLayers {
            let prefix = "\(tower).encoder.layers.\(index)"
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                linear("\(prefix).self_attn.\(projection)", v, v)
            }
            linear("\(prefix).mlp.fc1", vision.intermediateSize, v)
            linear("\(prefix).mlp.fc2", v, vision.intermediateSize)
            norm("\(prefix).layer_norm1", v)
            norm("\(prefix).layer_norm2", v)
        }
        norm("\(tower).post_layernorm", v)
        linear("vision.projector.linear_1", configuration.projectorHiddenSize, 4 * v)
        linear("vision.projector.linear_2", d, configuration.projectorHiddenSize)

        let audio = configuration.audio
        let a = audio.dModel
        let channels = audio.subsamplingConvChannels
        linear("audio.encoder.pre_encode.out", a, channels * audio.subsampledFeatures)
        add("audio.encoder.pre_encode.conv.0.weight", [channels, 1, 3, 3])
        add("audio.encoder.pre_encode.conv.0.bias", [channels])
        for depthwise in [2, 5] {
            add("audio.encoder.pre_encode.conv.\(depthwise).weight", [channels, 1, 3, 3])
            add("audio.encoder.pre_encode.conv.\(depthwise).bias", [channels])
            add("audio.encoder.pre_encode.conv.\(depthwise + 1).weight", [channels, channels, 1, 1])
            add("audio.encoder.pre_encode.conv.\(depthwise + 1).bias", [channels])
        }
        for index in 0..<audio.nLayers {
            let prefix = "audio.encoder.layers.\(index)"
            for feedForward in ["feed_forward1", "feed_forward2"] {
                linear("\(prefix).\(feedForward).linear1", a * audio.ffExpansionFactor, a)
                linear("\(prefix).\(feedForward).linear2", a, a * audio.ffExpansionFactor)
            }
            for name in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] {
                norm("\(prefix).\(name)", a)
            }
            for projection in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                linear("\(prefix).self_attn.\(projection)", a, a)
            }
            linear("\(prefix).self_attn.linear_pos", a, a, bias: false)
            add("\(prefix).self_attn.pos_bias_u", [audio.nHeads, a / audio.nHeads])
            add("\(prefix).self_attn.pos_bias_v", [audio.nHeads, a / audio.nHeads])
            add("\(prefix).conv.pointwise_conv1.weight", [2 * a, a, 1])
            add("\(prefix).conv.pointwise_conv1.bias", [2 * a])
            add("\(prefix).conv.depthwise_conv.weight", [a, 1, audio.convKernelSize])
            add("\(prefix).conv.depthwise_conv.bias", [a])
            add("\(prefix).conv.pointwise_conv2.weight", [a, a, 1])
            add("\(prefix).conv.pointwise_conv2.bias", [a])
            norm("\(prefix).conv.batch_norm", a)
            add("\(prefix).conv.batch_norm.running_mean", [a])
            add("\(prefix).conv.batch_norm.running_var", [a])
            add("\(prefix).conv.batch_norm.num_batches_tracked", [], .int64)
        }
        norm("audio.adapter.norm", a)
        linear("audio.adapter.linear_1", d, a)
        linear("audio.adapter.linear_2", d, d)
        norm("audio.residual.ln", d)
        linear("audio.residual.down", audio.residualWidth, d)
        linear("audio.residual.up", d, audio.residualWidth)
        return schema
    }
}
