// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation
import MLX
import MLXNN
import SystemOne

/// The dtype d1 keeps its weights and runs its activations in.
///
/// The checkpoint is float32 and so is the reference. float16 halves weight
/// memory in MLX (2.24 GB to 1.12 GB) and, per the d1-omni model card, kept the float32
/// top answer on every text, image, and audio row it was checked on; bfloat16
/// did not, so it is not offered. The audio mel front end, the head's logits,
/// and calibration stay float32 either way.
public enum D1Precision: String, Sendable, CaseIterable {
    case float32
    case float16

    var dataType: DType {
        switch self {
        case .float32: return .float32
        case .float16: return .float16
        }
    }
}

/// How a d1 checkpoint names and lays out its tensors.
public enum D1WeightLayout: String, Sendable {
    /// The PyTorch reference (`LiquidAI/d1-omni-600M`): float32, the names
    /// `D1Weights.schema(for:)` lists.
    case pytorch

    /// mlx-vlm's export (`LiquidAI/d1-omni-600M-MLX-fp16`, `-MLX-8bit`):
    /// renamed tensors, the head's fused attention projection split into query,
    /// key, and value, channels-last convolutions, float16, and, in the
    /// quantized exports, affine-quantized linear layers.
    case mlx
}

/// A d1-omni checkpoint validated against the exact tensor schema its
/// configuration implies: trunk, decision head, vision tower, and audio encoder.
///
/// Weights are read by their PyTorch names and in the PyTorch layout whichever
/// layout the checkpoint uses, so the model code is the same for both.
public struct D1Weights: Sendable {
    /// `model.safetensors` at `LiquidAI/d1-omni-600M` revision
    /// `02b55d7076f15129e59ab3f94783f32c4b088674`.
    public static let d1Omni600MArtifact = PinnedArtifact(
        byteCount: 2_348_774_500,
        sha256: "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1"
    )

    public let store: any TensorStore
    public let layout: D1WeightLayout
    public let configuration: D1Configuration
    public let precision: D1Precision
    /// The first weight float16 would turn into infinity, met while converting.
    private let overflow = Float16Overflow()
    /// The input width of every PyTorch linear and embedding weight, by module.
    private let inputWidths: [String: Int]

    /// Opens a checkpoint and validates its schema without pinning its bytes.
    public init(contentsOf url: URL, configuration: D1Configuration, precision: D1Precision = .float32) throws {
        try self.init(store: SafetensorsFile(contentsOf: url), configuration: configuration, precision: precision)
    }

    /// Verifies whole-file size and SHA-256 before validating the schema.
    public init(
        contentsOf url: URL,
        pinnedTo artifact: PinnedArtifact,
        configuration: D1Configuration,
        precision: D1Precision = .float32
    ) throws {
        try self.init(
            store: SafetensorsFile(contentsOf: url, pinnedTo: artifact), configuration: configuration,
            precision: precision
        )
    }

    /// Opens a model folder's checkpoint in either layout: an mlx-vlm export
    /// (it has `model.safetensors.index.json`) or the PyTorch `model.safetensors`.
    ///
    /// `artifact` pins the checkpoint byte for byte. The PyTorch release's pin,
    /// the default elsewhere, is not applied to an export, which has none recorded.
    public init(
        modelFolder: URL,
        configuration: D1Configuration,
        artifact: PinnedArtifact? = nil,
        precision: D1Precision = .float32
    ) throws {
        let indexURL = modelFolder.appendingPathComponent(SafetensorsCheckpoint.indexFileName)
        let isExport = FileManager.default.fileExists(atPath: indexURL.path)
        let pin = isExport && artifact == Self.d1Omni600MArtifact ? nil : artifact
        let store: any TensorStore
        if isExport {
            store = try SafetensorsCheckpoint(
                modelFolder: modelFolder, pins: pin.map { [SafetensorsCheckpoint.singleFileName: $0] }
            )
        } else {
            let url = modelFolder.appendingPathComponent(SafetensorsCheckpoint.singleFileName)
            store = try pin.map { try SafetensorsFile(contentsOf: url, pinnedTo: $0) } ?? SafetensorsFile(contentsOf: url)
        }
        try self.init(
            store: store, configuration: configuration, precision: precision,
            quantization: try MLXQuantizationConfiguration.load(modelFolder: modelFolder)
        )
    }

    /// Validates a checkpoint in either layout. MLX-quantized weights must agree
    /// with `quantization` (the export's `config.json`) when it is given.
    public init(
        store: any TensorStore,
        configuration: D1Configuration,
        precision: D1Precision = .float32,
        quantization: MLXQuantizationConfiguration? = nil
    ) throws {
        let pytorchSchema = Self.schema(for: configuration)
        let layout = Self.layout(of: store)
        switch layout {
        case .pytorch:
            try store.validate(schema: pytorchSchema)
        case .mlx:
            try store.validate(schema: Self.mlxSchema(for: configuration, in: store, quantization: quantization))
        }
        self.store = store
        self.layout = layout
        self.configuration = configuration
        self.precision = precision
        inputWidths = Dictionary(uniqueKeysWithValues: pytorchSchema.compactMap { specification in
            guard specification.name.hasSuffix(".weight"), specification.shape.count == 2 else { return nil }
            return (String(specification.name.dropLast(".weight".count)), specification.shape[1])
        })
    }

    /// The compute dtype every activation should stay in.
    var dataType: DType { precision.dataType }

    /// Copies one weight, by PyTorch name and in the PyTorch layout, into an
    /// MLX-owned array of the compute dtype. At float16, a value float16 cannot
    /// hold is recorded for `checkFloat16Range()`.
    func array(_ name: String) -> MLXArray {
        let (tensor, permutation) = stored(name)
        let array: MLXArray
        if precision == .float16 {
            let (converted, maximum) = tensor.float16Array()
            overflow.record(name: name, maximumMagnitude: maximum)
            array = converted
        } else {
            array = tensor.mlxArray(.float32)
        }
        return permutation.map { array.transposed(axes: $0) } ?? array
    }

    /// Copies one weight at float32, for derived weights folded before the cast.
    func float32Array(_ name: String) -> MLXArray {
        let (tensor, permutation) = stored(name)
        let array = tensor.mlxArray(.float32)
        return permutation.map { array.transposed(axes: $0) } ?? array
    }

    /// A linear layer by PyTorch module name: `QuantizedLinear` when the
    /// checkpoint stores the weight quantized.
    func linear(_ prefix: String, bias: Bool = true) -> Linear {
        guard layout == .mlx else {
            return Linear(weight: array("\(prefix).weight"), bias: bias ? array("\(prefix).bias") : nil)
        }
        // The schema was validated at load, so every module loads.
        return try! store.linear(
            Self.mlxName(prefix), inputWidth: inputWidths[prefix]!, bias: bias, dataType: dataType
        )
    }

    /// An embedding table by PyTorch module name, quantized when stored quantized.
    func embedding(_ prefix: String) -> Embedding {
        guard layout == .mlx else { return Embedding(weight: array("\(prefix).weight")) }
        return try! store.embedding(Self.mlxName(prefix), width: inputWidths[prefix]!, dataType: dataType)
    }

    /// The head's fused `in_proj` (query, key, value stacked on the output axis).
    /// mlx-vlm stores the three projections apart; they are stacked back, packed
    /// words and all, since quantization works row by row.
    func headInProjection(_ prefix: String) -> Linear {
        guard layout == .mlx else {
            return Linear(weight: array("\(prefix).self_attn.in_proj_weight"), bias: array("\(prefix).self_attn.in_proj_bias"))
        }
        let width = configuration.text.hiddenSize
        let parts = ["query_proj", "key_proj", "value_proj"].map {
            try! store.linear("\(prefix).attention.\($0)", inputWidth: width, bias: true, dataType: dataType)
        }
        let bias = concatenated(parts.map { $0.bias! }, axis: 0)
        let quantized = parts.compactMap { $0 as? QuantizedLinear }
        guard !quantized.isEmpty else {
            return Linear(weight: concatenated(parts.map(\.weight), axis: 0), bias: bias)
        }
        let first = quantized[0]
        precondition(
            quantized.count == 3 && quantized.allSatisfy {
                $0.groupSize == first.groupSize && $0.bits == first.bits && $0.mode == first.mode
                    && ($0.biases == nil) == (first.biases == nil)
            },
            "the head's query, key, and value projections are quantized alike"
        )
        return QuantizedLinear(
            weight: concatenated(quantized.map(\.weight), axis: 0), bias: bias,
            scales: concatenated(quantized.map(\.scales), axis: 0),
            biases: first.biases == nil ? nil : concatenated(quantized.map { $0.biases! }, axis: 0),
            groupSize: first.groupSize, bits: first.bits, mode: first.mode
        )
    }

    /// `name`'s head weight as a dense array in PyTorch layout, whatever the
    /// export split (the fused `in_proj`) or quantized: what `D1DecisionHead`
    /// keys its parameters by and head training differentiates.
    func denseArray(_ name: String) -> MLXArray {
        guard layout == .mlx else { return array(name) }
        if let fused = name.range(of: ".self_attn.in_proj_") {
            let linear = headInProjection(String(name[..<fused.lowerBound]))
            return name.hasSuffix("weight") ? Self.denseWeight(of: linear) : linear.bias!
        }
        guard name.hasSuffix(".weight") else { return array(name) }
        let module = String(name.dropLast(".weight".count))
        guard let width = inputWidths[module] else { return array(name) }
        // The schema was validated at load, so every module loads.
        return try! store.denseWeight(Self.mlxName(module), inputWidth: width, dataType: dataType)
    }

    /// A linear layer's weight, dequantized when the export stores it quantized.
    static func denseWeight(of linear: Linear) -> MLXArray {
        guard let quantized = linear as? QuantizedLinear else { return linear.weight }
        return dequantized(
            quantized.weight, scales: quantized.scales, biases: quantized.biases,
            groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode
        )
    }

    /// Casts a weight derived in float32 (folded BatchNorm, say) to the compute
    /// dtype, recording it like `array(_:)` when float16 cannot hold it.
    func cast(_ derived: MLXArray, name: String) -> MLXArray {
        guard precision == .float16 else { return derived }
        checkRange(of: derived, name: name)
        return derived.asType(.float16)
    }

    /// Records a float32 weight that is cast to float16 later, such as a
    /// position grid resampled per image, when float16 cannot hold it.
    func checkRange(of derived: MLXArray, name: String) {
        guard precision == .float16 else { return }
        overflow.record(name: name, maximumMagnitude: abs(derived).max().item(Float.self))
    }

    /// Throws `WeightsError.float16Overflow` for the first weight that would
    /// have loaded as infinity, rather than answering with infinities.
    func checkFloat16Range() throws {
        if let error = overflow.first { throw error }
    }

    /// The stored tensor for a PyTorch name, and the permutation that turns its
    /// stored layout back into PyTorch's (`nil` when they agree).
    private func stored(_ name: String) -> (SafetensorsTensorView, [Int]?) {
        guard layout == .mlx else { return (store.tensor(named: name)!, nil) }
        let tensor = store.tensor(named: Self.mlxName(name))!
        let permutation = Self.convolutionPermutation(name, rank: tensor.descriptor.shape.count)
        return (tensor, permutation.map(Self.inverse))
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

extension D1Weights {
    /// PyTorch checkpoints keep the head's fused attention projection.
    static func layout(of store: any TensorStore) -> D1WeightLayout {
        store.descriptors.keys.contains { $0.hasSuffix(".self_attn.in_proj_weight") } ? .pytorch : .mlx
    }

    /// mlx-vlm's renames (`d1_omni.sanitize`, `audio.sanitize_audio`), for a
    /// tensor or module name. The head's fused `in_proj` has no single name.
    static func mlxName(_ name: String) -> String {
        // Patterns end in "." so they match a module name as well as its tensors.
        var key = name + "."
        if key.hasPrefix("audio.") {
            for (old, new) in audioRenames { key = key.replacingOccurrences(of: old, with: new) }
        } else {
            key = key.replacingOccurrences(of: "vision.tower.vision_model.", with: "vision.tower.")
            if key.hasPrefix("head.head.") {
                key = key.replacingOccurrences(of: ".self_attn.out_proj.", with: ".attention.out_proj.")
            }
            if key.hasPrefix("head.scorer."), !key.hasPrefix("head.scorer.layers.") {
                key = "head.scorer.layers." + key.dropFirst("head.scorer.".count)
            }
        }
        return String(key.dropLast())
    }

    private static let audioRenames: [(String, String)] = [
        ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
        ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
        ("self_attn.linear_q.", "self_attn.q_proj."),
        ("self_attn.linear_k.", "self_attn.k_proj."),
        ("self_attn.linear_v.", "self_attn.v_proj."),
        ("self_attn.linear_out.", "self_attn.o_proj."),
        ("self_attn.linear_pos.", "self_attn.relative_k_proj."),
        ("self_attn.pos_bias_u.", "self_attn.bias_u."),
        ("self_attn.pos_bias_v.", "self_attn.bias_v."),
        ("conv.batch_norm.", "conv.norm."),
    ]

    /// The axis order that turns a PyTorch convolution weight into MLX's
    /// channels-last layout, or `nil` for weights MLX stores as PyTorch does.
    static func convolutionPermutation(_ name: String, rank: Int) -> [Int]? {
        if name.hasSuffix(".conv.conv.weight"), rank == 3 { return [0, 2, 1] }
        guard name.hasPrefix("audio."), name.hasSuffix(".weight") else { return nil }
        if rank == 4 { return [0, 2, 3, 1] }
        if rank == 3, name.contains("pointwise_conv") || name.contains("depthwise_conv") { return [0, 2, 1] }
        return nil
    }

    static func inverse(_ permutation: [Int]) -> [Int] {
        var inverse = [Int](repeating: 0, count: permutation.count)
        for (index, axis) in permutation.enumerated() { inverse[axis] = index }
        return inverse
    }

    /// The PyTorch schema under mlx-vlm's names and layouts, still float32 and unquantized.
    static func renamedSchema(for configuration: D1Configuration) -> [WeightSpecification] {
        schema(for: configuration).flatMap { specification -> [WeightSpecification] in
            let name = specification.name
            if name.hasSuffix("num_batches_tracked") { return [] }
            if name.hasPrefix("head.head."), let range = name.range(of: ".self_attn.in_proj_") {
                let prefix = name[..<range.lowerBound]
                let suffix = name[range.upperBound...]
                let shape = [specification.shape[0] / 3] + specification.shape.dropFirst()
                return ["query_proj", "key_proj", "value_proj"].map {
                    WeightSpecification(name: "\(prefix).attention.\($0).\(suffix)", shape: shape)
                }
            }
            var shape = specification.shape
            if let permutation = convolutionPermutation(name, rank: shape.count) {
                shape = permutation.map { specification.shape[$0] }
            }
            return [WeightSpecification(name: mlxName(name), shape: shape, dataType: specification.dataType)]
        }
    }

    /// Every tensor an mlx-vlm export of `configuration` holds, with floats in
    /// `floatType` and the modules `quantization` names quantized.
    public static func mlxSchema(
        for configuration: D1Configuration, floatType: SafetensorsDataType,
        quantization: (_ module: String) -> QuantizationParameters?
    ) -> [WeightSpecification] {
        renamedSchema(for: configuration).mlxExport(floatType: floatType, quantization: quantization)
    }

    /// The export schema a store must match: its float type, and each quantized
    /// module's group size and bits as stored (checked against `quantization`).
    static func mlxSchema(
        for configuration: D1Configuration, in store: any TensorStore, quantization: MLXQuantizationConfiguration?
    ) throws -> [WeightSpecification] {
        let floatType = store.descriptors["encoder.embedding_norm.weight"]?.dataType ?? .float16
        var parameters: [String: QuantizationParameters] = [:]
        for specification in renamedSchema(for: configuration)
        where specification.name.hasSuffix(".weight") && specification.shape.count == 2 {
            let module = String(specification.name.dropLast(".weight".count))
            guard store.isQuantized(module) else { continue }
            parameters[module] = try store.quantizationParameters(
                of: module, inputWidth: specification.shape[1], configured: quantization?.parameters(for: module)
            )
        }
        return mlxSchema(for: configuration, floatType: floatType) { parameters[$0] }
    }
}

/// The first float16 overflow seen while one model's weights were converted.
private final class Float16Overflow: @unchecked Sendable {
    private let lock = NSLock()
    private var error: WeightsError?

    var first: WeightsError? { lock.withLock { error } }

    func record(name: String, maximumMagnitude: Float) {
        guard maximumMagnitude >= SafetensorsTensorView.float16OverflowMagnitude else { return }
        lock.withLock {
            if error == nil { error = .float16Overflow(name: name, maximumMagnitude: maximumMagnitude) }
        }
    }
}
