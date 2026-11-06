import Foundation
import MLX
import SystemOne

/// The dtype JEV activations compute in. Weights stay in their released
/// bfloat16 either way, so `.float32` costs compute time, not weight memory.
public enum JevPrecision: Sendable {
    /// bfloat16 activations, as the released model serves.
    case bfloat16
    /// float32 activations over the same bfloat16 weights, matching the
    /// float32 reference the golden fixtures were computed with.
    case float32

    var dataType: DType {
        switch self {
        case .bfloat16: return .bfloat16
        case .float32: return .float32
        }
    }
}

/// The exact files a JEV checkpoint is verified against before parsing.
public struct JevArtifacts: Equatable, Sendable {
    /// Backbone shards, by file name.
    public let shards: [String: PinnedArtifact]
    /// `adapter/adapter_model.safetensors`.
    public let adapter: PinnedArtifact
    /// `head.safetensors`.
    public let head: PinnedArtifact

    public init(shards: [String: PinnedArtifact], adapter: PinnedArtifact, head: PinnedArtifact) {
        self.shards = shards
        self.adapter = adapter
        self.head = head
    }

    /// `autotrust/JEV-9B` at revision `7ac669b8fb476ba926491e3d1c39f9330d68dbed`.
    public static let jev9B = JevArtifacts(
        shards: [
            "model-00001-of-00005.safetensors": PinnedArtifact(
                byteCount: 4_234_572_344, sha256: "fdce203bd6767c95690c4695699e9e796fa53caeb06c7f6d43a014f7c513b32c"),
            "model-00002-of-00005.safetensors": PinnedArtifact(
                byteCount: 4_199_125_880, sha256: "64003b88d6e6871f2315e39356188082063ff879994434d5ce8c209b3c736a01"),
            "model-00003-of-00005.safetensors": PinnedArtifact(
                byteCount: 4_215_396_752, sha256: "ef7cf4227fb3f391bb2bcd5d79a3c10cd90d823dbf964f41a0aa34591377ae82"),
            "model-00004-of-00005.safetensors": PinnedArtifact(
                byteCount: 3_224_330_536, sha256: "2265f2fcf900572fa0b7c645a9a64f662d4123daeeeda23e917411e907d66272"),
            "model-00005-of-00005.safetensors": PinnedArtifact(
                byteCount: 2_034_237_568, sha256: "dd63614f1dc80dce2d83be3f3e69af1f8a0ebc8add9b8e0abf3910f563e44344"),
        ],
        adapter: PinnedArtifact(
            byteCount: 160_486_456, sha256: "796386ace560ec5624bbfe83f5c505121e5c9f9654200a847f032cb333cd1f71"),
        head: PinnedArtifact(
            byteCount: 393_464, sha256: "d960a6f5989402c6d08d74d8d435a21a031cc7878f3c4ac274a126c900f627e7")
    )
}

/// A JEV checkpoint in memory: the bfloat16 backbone with the System 1 LoRA
/// merged in, the float32 state-space parameters, and the float32 head.
///
/// The merge is peft's `merge_and_unload` onto a bfloat16 base:
/// `bf16(f32(W) + (alpha / r) * B @ A)`. `lm_head` is validated but never
/// loaded: decisions read only the head's 24 slots.
final class JevWeights {
    let configuration: JevConfiguration
    /// Compute dtype of activations.
    let dataType: DType
    let headWeight: MLXArray
    let headBias: MLXArray
    private let tensors: [String: MLXArray]

    /// Loads `model.safetensors.index.json` and its shards, the adapter named
    /// by `judge_config.json`, and `head.safetensors`. Pass `artifacts: nil`
    /// to skip the byte-for-byte pins, for example for the tiny test model.
    init(
        modelFolder: URL,
        configuration: JevConfiguration,
        artifacts: JevArtifacts?,
        precision: JevPrecision
    ) throws {
        self.configuration = configuration
        dataType = precision.dataType

        let backbone = try ShardedSafetensors(
            indexURL: modelFolder.appendingPathComponent("model.safetensors.index.json"),
            pins: artifacts?.shards
        )
        try backbone.validate(schema: Self.schema(for: configuration.text))

        let adapterURL = modelFolder
            .appendingPathComponent(configuration.judge.adapterSubfolder)
            .appendingPathComponent("adapter_model.safetensors")
        let adapter = try artifacts.map { try SafetensorsFile(contentsOf: adapterURL, pinnedTo: $0.adapter) }
            ?? SafetensorsFile(contentsOf: adapterURL)
        let targets = Self.adapterTargets(for: configuration)
        try adapter.validate(schema: Self.adapterSchema(for: configuration, targets: targets))

        let headURL = modelFolder.appendingPathComponent("head.safetensors")
        let head = try artifacts.map { try SafetensorsFile(contentsOf: headURL, pinnedTo: $0.head) }
            ?? SafetensorsFile(contentsOf: headURL)
        try head.validate(schema: [
            WeightSpecification(name: "proj.weight", shape: [configuration.judge.slots.numSlots, configuration.text.hiddenSize]),
            WeightSpecification(name: "proj.bias", shape: [configuration.judge.slots.numSlots]),
        ])
        headWeight = try head.requiredTensor("proj.weight").mlxArray()
        headBias = try head.requiredTensor("proj.bias").mlxArray()

        let scale = configuration.adapter.scale
        var tensors: [String: MLXArray] = [:]
        for specification in Self.schema(for: configuration.text) where specification.name != "lm_head.weight" {
            let base = try backbone.requiredTensor(specification.name).mlxArray()
            guard let target = targets[specification.name] else {
                tensors[specification.name] = base
                continue
            }
            let a = try adapter.requiredTensor("\(target).lora_A.weight").mlxArray()
            let b = try adapter.requiredTensor("\(target).lora_B.weight").mlxArray()
            let merged = (base.asType(.float32) + matmul(b, a) * scale).asType(.bfloat16)
            // MLX `eval` materialises the lazy array (it runs no code), one
            // tensor at a time, so float32 intermediates never accumulate.
            eval(merged)
            tensors[specification.name] = merged
        }
        eval(headWeight, headBias)
        self.tensors = tensors
    }

    /// A backbone tensor as stored (bfloat16, or float32 for `A_log` and the
    /// gated norm), with the LoRA merged where it targets one.
    func array(_ name: String) -> MLXArray {
        guard let array = tensors[name] else { preconditionFailure("JEV weights have no tensor \(name)") }
        return array
    }

    /// Every backbone tensor of a Qwen3.5 text checkpoint, with released dtypes.
    static func schema(for text: JevConfiguration.Text) -> [WeightSpecification] {
        let hidden = text.hiddenSize
        func bf16(_ name: String, _ shape: [Int]) -> WeightSpecification {
            WeightSpecification(name: name, shape: shape, dataType: .bfloat16)
        }
        var schema = [
            bf16("model.embed_tokens.weight", [text.vocabSize, hidden]),
            bf16("model.norm.weight", [hidden]),
        ]
        if !text.tieWordEmbeddings { schema.append(bf16("lm_head.weight", [text.vocabSize, hidden])) }
        for layer in 0..<text.numHiddenLayers {
            let prefix = "model.layers.\(layer)"
            schema += [
                bf16("\(prefix).input_layernorm.weight", [hidden]),
                bf16("\(prefix).post_attention_layernorm.weight", [hidden]),
                bf16("\(prefix).mlp.gate_proj.weight", [text.intermediateSize, hidden]),
                bf16("\(prefix).mlp.up_proj.weight", [text.intermediateSize, hidden]),
                bf16("\(prefix).mlp.down_proj.weight", [hidden, text.intermediateSize]),
            ]
            if text.isFullAttention(layer) {
                let heads = text.numAttentionHeads * text.headDim
                let keyValue = text.numKeyValueHeads * text.headDim
                schema += [
                    bf16("\(prefix).self_attn.q_proj.weight", [2 * heads, hidden]),
                    bf16("\(prefix).self_attn.k_proj.weight", [keyValue, hidden]),
                    bf16("\(prefix).self_attn.v_proj.weight", [keyValue, hidden]),
                    bf16("\(prefix).self_attn.o_proj.weight", [hidden, heads]),
                    bf16("\(prefix).self_attn.q_norm.weight", [text.headDim]),
                    bf16("\(prefix).self_attn.k_norm.weight", [text.headDim]),
                ]
            } else {
                let heads = text.linearNumValueHeads
                schema += [
                    WeightSpecification(name: "\(prefix).linear_attn.A_log", shape: [heads]),
                    bf16("\(prefix).linear_attn.dt_bias", [heads]),
                    bf16("\(prefix).linear_attn.conv1d.weight", [text.convolutionChannels, 1, text.linearConvKernelDim]),
                    bf16("\(prefix).linear_attn.in_proj_qkv.weight", [text.convolutionChannels, hidden]),
                    bf16("\(prefix).linear_attn.in_proj_z.weight", [text.linearValueDimension, hidden]),
                    bf16("\(prefix).linear_attn.in_proj_b.weight", [heads, hidden]),
                    bf16("\(prefix).linear_attn.in_proj_a.weight", [heads, hidden]),
                    WeightSpecification(name: "\(prefix).linear_attn.norm.weight", shape: [text.linearValueHeadDim]),
                    bf16("\(prefix).linear_attn.out_proj.weight", [hidden, text.linearValueDimension]),
                ]
            }
        }
        return schema
    }

    /// Backbone weight name → peft module prefix, for every linear the adapter targets.
    static func adapterTargets(for configuration: JevConfiguration) -> [String: String] {
        let targets = Set(configuration.adapter.targetModules)
        var map: [String: String] = [:]
        for specification in schema(for: configuration.text) {
            let name = specification.name
            guard name.hasPrefix("model.layers."), name.hasSuffix(".weight") else { continue }
            let module = name.dropLast(".weight".count)
            guard let last = module.split(separator: ".").last, targets.contains(String(last)) else { continue }
            map[name] = "base_model.model.\(module)"
        }
        return map
    }

    /// The adapter's float32 `lora_A` `[r, in]` and `lora_B` `[out, r]` per target.
    static func adapterSchema(
        for configuration: JevConfiguration,
        targets: [String: String]
    ) -> [WeightSpecification] {
        let shapes = Dictionary(uniqueKeysWithValues: schema(for: configuration.text).map { ($0.name, $0.shape) })
        let rank = configuration.adapter.r
        return targets.sorted { $0.key < $1.key }.flatMap { weight, prefix -> [WeightSpecification] in
            let shape = shapes[weight]!
            return [
                WeightSpecification(name: "\(prefix).lora_A.weight", shape: [rank, shape[1]]),
                WeightSpecification(name: "\(prefix).lora_B.weight", shape: [shape[0], rank]),
            ]
        }
    }
}
