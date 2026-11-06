import Foundation
import MLX
import MLXNN
import SystemOne

/// LFM3-VL's image settings: the vision tower (`vision_config`), the
/// multimodal projector, and the image processor (`processor_config.json`).
public struct LFM3VLVisionConfiguration: Equatable, Sendable {
    public enum LayerType: String, Equatable, Sendable {
        /// A gated depthwise 2D convolution over the patch grid.
        case convolution = "conv"
        /// Bidirectional grouped-query attention with axial RoPE.
        case fullAttention = "full_attention"
    }

    // The tower.
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerTypes: [LayerType]
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let mergeSize: Int
    public let convolutionKernel: Int
    public let normEpsilon: Float
    public let ropeTheta: Float

    // The projector.
    public let projectorHiddenSize: Int
    public let projectorBias: Bool
    public let projectorLayerNormEpsilon: Float
    public let textHiddenSize: Int
    /// The placeholder token image features replace (`<|image|>`).
    public let imageTokenID: Int

    // The image processor.
    public let minPixels: Int
    public let maxPixels: Int
    public let resampling: LFM2VLVisionConfiguration.Resampling
    public let imageMean: [Double]
    public let imageStd: [Double]
    public let rescaleFactor: Double

    /// The video processor (`processor_config.json`'s `video_processor`, with
    /// the reference's defaults) and the token its features fill; `nil`
    /// without a `video_token_id`.
    public let video: Video?

    public var headDimension: Int { hiddenSize / attentionHeads }

    /// LFM3-VL's video settings.
    public struct Video: Equatable, Sendable {
        /// The placeholder token video features replace (`<|video|>`).
        public let tokenID: Int
        /// Frames sampled per second of video.
        public let fps: Double
        public let minFrames: Int
        public let maxFrames: Int
        /// The whole clip's pixel budget, frames times frame area (`size`).
        public let minPixels: Int
        public let maxPixels: Int
        /// Caps each frame at `maxVideoTokens` merged patches or its share of the budget.
        public let capPixelsPerFrame: Bool
        public let maxVideoTokens: Int
        public let resampling: LFM2VLVisionConfiguration.Resampling
        public let mean: [Double]
        public let std: [Double]
        public let rescaleFactor: Double
        /// LFM3-Omni's timestamp (`processing_lfm3_omni.py`): the group's first and
        /// last frame indices summed, then divided by twice the clip's rate;
        /// LFM3-VL's (`processing_lfm3_vl.py`) averages the two frames' times.
        public let sumsIndices: Bool
    }

    /// Reads `config.json` and the `image_processor` of `processor_config.json`
    /// (or `preprocessor_config.json`), with the reference's defaults for missing keys.
    public init(modelFolder: URL) throws {
        func object(_ name: String) throws -> [String: Any]? {
            let url = modelFolder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw LFM2ConfigurationError.invalid("\(name) is not a JSON object")
            }
            return value
        }
        guard let config = try object("config.json") else { throw LFM2ConfigurationError.missing("config.json") }
        let processor = try object("processor_config.json")?["image_processor"] as? [String: Any]
            ?? object("preprocessor_config.json") ?? [:]
        let video = try object("processor_config.json")?["video_processor"] as? [String: Any] ?? [:]
        try self.init(json: config, processor: processor, video: video)
    }

    /// `json` is an LFM3-VL `config.json`; `processor` its image processor's settings.
    public init(json: [String: Any], processor: [String: Any], video videoProcessor: [String: Any] = [:]) throws {
        guard let vision = json["vision_config"] as? [String: Any] else {
            throw LFM2ConfigurationError.missing("vision_config")
        }
        let text = json["text_config"] as? [String: Any] ?? [:]
        func int(_ object: [String: Any], _ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = object[key] as? Int { return value }
            if let fallback { return fallback }
            throw LFM2ConfigurationError.missing(key)
        }
        func double(_ object: [String: Any], _ key: String, _ fallback: Double) -> Double {
            (object[key] as? NSNumber)?.doubleValue ?? fallback
        }
        func doubles(_ object: [String: Any], _ key: String) throws -> [Double] {
            guard let values = object[key] else { return [0.5, 0.5, 0.5] }
            guard let list = values as? [NSNumber], list.count == 3 else {
                throw LFM2ConfigurationError.invalid("\(key) needs three channel values")
            }
            return list.map(\.doubleValue)
        }

        hiddenSize = try int(vision, "hidden_size")
        intermediateSize = try int(vision, "intermediate_size")
        let layers = try int(vision, "num_hidden_layers")
        guard let names = vision["layer_types"] as? [String], names.count == layers else {
            throw LFM2ConfigurationError.invalid("vision layer_types must name all \(layers) layers")
        }
        layerTypes = try names.map { name -> LayerType in
            guard let type = LayerType(rawValue: name) else { throw LFM2ConfigurationError.unsupported("vision layer type \(name)") }
            return type
        }
        let heads = try int(vision, "num_attention_heads")
        attentionHeads = heads
        keyValueHeads = try int(vision, "num_key_value_heads", heads)
        patchSize = try int(vision, "patch_size", 16)
        temporalPatchSize = try int(vision, "temporal_patch_size", 2)
        mergeSize = try int(vision, "spatial_merge_size", 2)
        convolutionKernel = try int(vision, "conv_kernel_size", 3)
        normEpsilon = Float(double(vision, "norm_eps", 1e-5))
        let rope = vision["rope_parameters"] as? [String: Any] ?? [:]
        let ropeType = rope["rope_type"] as? String ?? "axial"
        guard ropeType == "axial" else { throw LFM2ConfigurationError.unsupported("vision rope_type \(ropeType)") }
        ropeTheta = Float(double(rope, "rope_theta", 1_000_000))

        projectorHiddenSize = try int(json, "projector_hidden_size", 4096)
        projectorBias = json["projector_bias"] as? Bool ?? true
        projectorLayerNormEpsilon = Float(double(json, "projector_layer_norm_eps", 1e-6))
        textHiddenSize = try int(text, "hidden_size")
        imageTokenID = try int(json, "image_token_id")

        let size = processor["size"] as? [String: Any] ?? [:]
        minPixels = try int(processor, "min_pixels", try int(size, "shortest_edge", 65_536))
        maxPixels = try int(processor, "max_pixels", try int(size, "longest_edge", 16_777_216))
        let resample = try int(processor, "resample", LFM2VLVisionConfiguration.Resampling.bicubic.rawValue)
        guard let resampling = LFM2VLVisionConfiguration.Resampling(rawValue: resample) else {
            throw LFM2ConfigurationError.unsupported("resample \(resample)")
        }
        self.resampling = resampling
        imageMean = try doubles(processor, "image_mean")
        imageStd = try doubles(processor, "image_std")
        rescaleFactor = double(processor, "rescale_factor", 1.0 / 255)

        if let token = json["video_token_id"] as? Int {
            let v = videoProcessor
            let videoSize = v["size"] as? [String: Any] ?? [:]
            let resample = try int(v, "resample", LFM2VLVisionConfiguration.Resampling.bicubic.rawValue)
            guard let videoResampling = LFM2VLVisionConfiguration.Resampling(rawValue: resample) else {
                throw LFM2ConfigurationError.unsupported("video resample \(resample)")
            }
            guard try int(v, "patch_size", patchSize) == patchSize, try int(v, "merge_size", mergeSize) == mergeSize,
                  try int(v, "temporal_patch_size", temporalPatchSize) == temporalPatchSize
            else {
                throw LFM2ConfigurationError.invalid("the video processor's patches and merges differ from the tower's")
            }
            let settings = Video(
                tokenID: token,
                fps: double(v, "fps", 2),
                minFrames: try int(v, "min_frames", 4),
                maxFrames: try int(v, "max_frames", 768),
                minPixels: try int(v, "min_pixels", try int(videoSize, "shortest_edge", 4_096)),
                maxPixels: try int(v, "max_pixels", try int(videoSize, "longest_edge", 25_165_824)),
                capPixelsPerFrame: v["cap_pixels_per_frame"] as? Bool ?? false,
                maxVideoTokens: try int(v, "max_video_tokens", 768),
                resampling: videoResampling,
                mean: try doubles(v, "image_mean"),
                std: try doubles(v, "image_std"),
                rescaleFactor: double(v, "rescale_factor", 1.0 / 255),
                sumsIndices: json["model_type"] as? String == "lfm3_omni" || json["audio_encoder_config"] != nil
            )
            guard settings.fps > 0, settings.minFrames >= 1, settings.minFrames <= settings.maxFrames,
                  settings.minPixels <= settings.maxPixels, settings.maxVideoTokens > 0
            else { throw LFM2ConfigurationError.invalid("video sampling or pixel budget") }
            video = settings
        } else {
            video = nil
        }

        let activation = json["projector_hidden_act"] as? String ?? "gelu"
        guard activation == "gelu" else { throw LFM2ConfigurationError.unsupported("projector_hidden_act \(activation)") }
        let downsample = try int(json, "downsample_factor", mergeSize)
        let processorPatch = try int(processor, "patch_size", patchSize)
        let processorMerge = try int(processor, "merge_size", mergeSize)
        let processorTemporal = try int(processor, "temporal_patch_size", temporalPatchSize)
        guard downsample == mergeSize, processorPatch == patchSize, processorMerge == mergeSize,
              processorTemporal == temporalPatchSize
        else {
            throw LFM2ConfigurationError.invalid("the processor's patches and merges differ from the tower's")
        }
        guard attentionHeads > 0, hiddenSize % attentionHeads == 0, attentionHeads % keyValueHeads == 0,
              headDimension % 4 == 0, convolutionKernel % 2 == 1, minPixels <= maxPixels
        else {
            throw LFM2ConfigurationError.invalid("vision tower of \(hiddenSize) features and \(attentionHeads) heads")
        }
    }
}

/// LFM3-VL's image path, Apache-2.0, following transformers' LFM3-VL code (as
/// vendored with LiquidAI's d1-2B) and mlx-vlm's `lfm3_vl` port:
///
/// - the processor's smart resize to multiples of 32 pixels within
///   `[min_pixels, max_pixels]`, with torchvision's antialiased uint8 kernel;
/// - 16-pixel patches in 2 × 2 merge-block order, each repeated over the
///   temporal patch, through a linear patchifier and an RMSNorm;
/// - gated depthwise 2D convolutions and bidirectional grouped-query
///   attention layers (RMSNorm on query and key heads, axial RoPE over the
///   patch's row and column), each followed by a SwiGLU block;
/// - a layer norm per patch, each merge block's patches side by side, and a
///   two-layer GELU projector into the text trunk's width;
/// - `<|vision_start|>`, one `<|image|>` per feature row, `<|vision_end|>`.
public final class LFM3VLVision {
    public static let imageToken = "<|image|>"
    static let visionStart = "<|vision_start|>"
    static let visionEnd = "<|vision_end|>"

    private struct Attention {
        let query: Linear, key: Linear, value: Linear, output: Linear
        let queryNorm: MLXArray, keyNorm: MLXArray
    }

    private struct Convolution {
        let input: Linear
        /// `[kernel, kernel, hidden]` depthwise taps.
        let taps: MLXArray
        let output: Linear
    }

    private struct Layer {
        let attention: Attention?
        let convolution: Convolution?
        let operatorNorm: MLXArray
        let feedForwardNorm: MLXArray
        let gate: Linear, up: Linear, down: Linear
    }

    public let configuration: LFM3VLVisionConfiguration
    public let dataType: DType
    private let patchifier: Linear
    private let embeddingNorm: MLXArray
    private let layers: [Layer]
    private let projectorNormWeight: MLXArray
    private let projectorNormBias: MLXArray
    private let projectorExpand: Linear
    private let projectorContract: Linear

    public convenience init(modelFolder: URL, dataType: DType = .float32) throws {
        try self.init(
            store: SafetensorsCheckpoint(modelFolder: modelFolder),
            configuration: LFM3VLVisionConfiguration(modelFolder: modelFolder),
            dataType: dataType
        )
    }

    /// Loads the tower and projector under transformers' names
    /// (`model.vision_tower.`, `model.multi_modal_projector.`) or mlx-vlm's
    /// (`vision_tower.`, `multi_modal_projector.`); depthwise kernels may be
    /// `[hidden, 1, k, k]` or MLX's `[hidden, k, k, 1]`.
    public init(store: any TensorStore, configuration: LFM3VLVisionConfiguration, dataType: DType = .float32) throws {
        self.configuration = configuration
        self.dataType = dataType
        let c = configuration
        let d = c.hiddenSize
        guard let tower = ["model.vision_tower.", "vision_tower."].first(where: { store.descriptors["\($0)patchifier.weight"] != nil }) else {
            throw WeightsError.tensorSetMismatch(missing: ["vision_tower.patchifier.weight"], unexpected: [])
        }
        let projectorPrefixes = ["model.multi_modal_projector.", "multi_modal_projector."]
        guard let projector = projectorPrefixes.first(where: { store.descriptors["\($0)linear_1.weight"] != nil }) else {
            throw WeightsError.tensorSetMismatch(missing: ["multi_modal_projector.linear_1.weight"], unexpected: [])
        }

        func check(_ name: String, _ shape: [Int]) throws {
            let module = String(name.dropLast(".weight".count))
            if name.hasSuffix(".weight"), shape.count == 2, store.isQuantized(module) {
                _ = try store.quantizationParameters(of: module, inputWidth: shape[1])
                return
            }
            let actual = try store.requiredTensor(name).descriptor.shape
            guard actual == shape else { throw WeightsError.shapeMismatch(name: name, expected: shape, actual: actual) }
        }
        func linear(_ name: String, _ output: Int, _ input: Int, bias: Bool = false) throws -> Linear {
            try check("\(name).weight", [output, input])
            if bias { try check("\(name).bias", [output]) }
            return try store.linear(name, inputWidth: input, bias: bias, dataType: dataType)
        }
        func vector(_ name: String, _ width: Int) throws -> MLXArray {
            try check(name, [width])
            return try store.requiredTensor(name).mlxArray(dataType)
        }

        let patchWidth = 3 * c.temporalPatchSize * c.patchSize * c.patchSize
        patchifier = try linear("\(tower)patchifier", d, patchWidth)
        embeddingNorm = try vector("\(tower)embedding_norm.weight", d)
        let k = c.convolutionKernel
        layers = try c.layerTypes.enumerated().map { index, type -> Layer in
            let p = "\(tower)layers.\(index)."
            var attention: Attention?
            var convolution: Convolution?
            switch type {
            case .fullAttention:
                let headDimension = c.headDimension
                attention = Attention(
                    query: try linear("\(p)self_attn.q_proj", c.attentionHeads * headDimension, d),
                    key: try linear("\(p)self_attn.k_proj", c.keyValueHeads * headDimension, d),
                    value: try linear("\(p)self_attn.v_proj", c.keyValueHeads * headDimension, d),
                    output: try linear("\(p)self_attn.out_proj", d, c.attentionHeads * headDimension),
                    queryNorm: try vector("\(p)self_attn.q_layernorm.weight", headDimension),
                    keyNorm: try vector("\(p)self_attn.k_layernorm.weight", headDimension)
                )
            case .convolution:
                let name = "\(p)conv.spatial_conv.weight"
                let shape = try store.requiredTensor(name).descriptor.shape
                guard shape == [d, 1, k, k] || shape == [d, k, k, 1] else {
                    throw WeightsError.shapeMismatch(name: name, expected: [d, 1, k, k], actual: shape)
                }
                convolution = Convolution(
                    input: try linear("\(p)conv.in_proj", 3 * d, d),
                    taps: try store.requiredTensor(name).mlxArray(dataType).reshaped(d, k, k).transposed(1, 2, 0),
                    output: try linear("\(p)conv.out_proj", d, d)
                )
            }
            return Layer(
                attention: attention,
                convolution: convolution,
                operatorNorm: try vector("\(p)operator_norm.weight", d),
                feedForwardNorm: try vector("\(p)ffn_norm.weight", d),
                gate: try linear("\(p)feed_forward.w1", c.intermediateSize, d),
                up: try linear("\(p)feed_forward.w3", c.intermediateSize, d),
                down: try linear("\(p)feed_forward.w2", d, c.intermediateSize)
            )
        }
        let merged = d * c.mergeSize * c.mergeSize
        projectorNormWeight = try vector("\(projector)layer_norm.weight", d)
        projectorNormBias = try vector("\(projector)layer_norm.bias", d)
        projectorExpand = try linear("\(projector)linear_1", c.projectorHiddenSize, merged, bias: c.projectorBias)
        projectorContract = try linear("\(projector)linear_2", c.textHiddenSize, c.projectorHiddenSize, bias: c.projectorBias)
    }

    // MARK: - The processor

    /// `smart_resize`: `(height, width)` divisible by `patch · merge`, its area
    /// within `[min_pixels, max_pixels]`. Aspect ratios past 200 are refused.
    public static func smartResize(
        height: Int, width: Int, configuration c: LFM3VLVisionConfiguration
    ) throws -> (height: Int, width: Int) {
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw LFM2ConfigurationError.unsupported("an aspect ratio of \(max(height, width) / min(height, width)) or more")
        }
        let factor = c.patchSize * c.mergeSize
        let (h0, w0) = (Double(height), Double(width))
        var h = Int((h0 / Double(factor)).rounded(.toNearestOrEven)) * factor
        var w = Int((w0 / Double(factor)).rounded(.toNearestOrEven)) * factor
        if h * w > c.maxPixels {
            let beta = (h0 * w0 / Double(c.maxPixels)).squareRoot()
            h = max(factor, Int((h0 / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((w0 / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < c.minPixels {
            let beta = (Double(c.minPixels) / (h0 * w0)).squareRoot()
            h = Int((h0 * beta / Double(factor)).rounded(.up)) * factor
            w = Int((w0 * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// Feature rows (and `<|image|>` tokens) a resized `height × width` picture yields.
    public static func featureCount(height: Int, width: Int, configuration c: LFM3VLVisionConfiguration) -> Int {
        (height / c.patchSize) * (width / c.patchSize) / (c.mergeSize * c.mergeSize)
    }

    /// The chat template's `<|vision_start|><|image|><|vision_end|>` with the
    /// processor's expansion of `<|image|>`.
    public static func markup(featureCount: Int) -> String {
        visionStart + String(repeating: imageToken, count: featureCount) + visionEnd
    }

    public static let videoToken = "<|video|>"

    /// The chat template's `<|vision_start|><|video|><|vision_end|>` with the
    /// processor's expansion of `<|video|>`: per temporal group, its time
    /// (`<{t:.1f} seconds>`, Python's `.1f`) and its `<|video|>` tokens in
    /// their own `<|vision_start|>`…`<|vision_end|>`.
    public static func videoMarkup(_ layout: LFM3VLVideoLayout) -> String {
        visionStart + layout.seconds.map { seconds in
            "<" + String(format: "%.1f", seconds) + " seconds>" + visionStart
                + String(repeating: videoToken, count: layout.tokensPerGroup) + visionEnd
        }.joined() + visionEnd
    }

    /// `image` resized as the processor resizes it: `(height, width, RGB bytes)`.
    public static func resized(_ image: RGBImage, configuration c: LFM3VLVisionConfiguration) throws -> (height: Int, width: Int, pixels: [UInt8]) {
        let size = try smartResize(height: image.height, width: image.width, configuration: c)
        let filter: AntialiasedResize.Filter = c.resampling == .bicubic ? .cubic : .triangle
        let pixels = AntialiasedResize.resize(
            image.pixels, width: image.width, height: image.height, channels: 3,
            toWidth: size.width, toHeight: size.height, filter: filter, kernel: .aten
        )
        return (size.height, size.width, pixels)
    }

    // MARK: - Video

    /// `sample_frames`: `int(frames / fps · rate)` frames, at least
    /// `min_frames` and at most `max_frames` and the clip, spread evenly
    /// (`np.linspace(0, frames - 1, count).round()`). A vanishing `fps`
    /// asks for every frame rather than overflowing `Int`.
    public static func sampledFrames(total: Int, fps: Double, video v: LFM3VLVisionConfiguration.Video) -> [Int] {
        guard total > 0 else { return [] }
        let wanted = Int(max(0, min(Double(total) / fps * v.fps, Double(total))))
        let count = min(max(wanted, v.minFrames), v.maxFrames, total)
        guard count > 1 else { return [0] }
        let step = Double(total - 1) / Double(count - 1)
        return (0..<count).map { $0 == count - 1 ? total - 1 : Int((Double($0) * step).rounded(.toNearestOrEven)) }
    }

    /// The video processor's `smart_resize`: one frame size, multiples of
    /// `patch · merge`, with the clip's frames times frame area within the
    /// pixel budget (each frame capped first with `cap_pixels_per_frame`).
    public static func smartResize(
        frames: Int, height: Int, width: Int, configuration c: LFM3VLVisionConfiguration
    ) throws -> (height: Int, width: Int) {
        guard let v = c.video else { throw LFM2ConfigurationError.missing("video_token_id") }
        let temporal = c.temporalPatchSize
        guard frames >= temporal else { throw LFM3VLVideoError.tooShort(frames: frames, minimum: temporal) }
        let factor = c.patchSize * c.mergeSize
        var maxPixels = v.maxPixels
        if v.capPixelsPerFrame {
            let perFrame = max(min(v.maxVideoTokens * factor * factor, v.maxPixels / frames), Int(Double(v.minPixels) * 1.05))
            maxPixels = perFrame * frames
        }
        var (h0, w0) = (height, width)
        if h0 < factor || w0 < factor {
            let scale = max(Double(factor) / Double(h0), Double(factor) / Double(w0))
            (h0, w0) = (Int(Double(h0) * scale), Int(Double(w0) * scale))
        }
        guard Double(max(h0, w0)) / Double(min(h0, w0)) <= 200 else {
            throw LFM2ConfigurationError.unsupported("an aspect ratio of \(max(h0, w0) / min(h0, w0)) or more")
        }
        var h = Int((Double(h0) / Double(factor)).rounded(.toNearestOrEven)) * factor
        var w = Int((Double(w0) / Double(factor)).rounded(.toNearestOrEven)) * factor
        let t = Int((Double(frames) / Double(temporal)).rounded(.toNearestOrEven)) * temporal
        if t * h * w > maxPixels {
            let beta = (Double(frames * h0 * w0) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(h0) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(w0) / beta / Double(factor)).rounded(.down)) * factor)
        } else if t * h * w < v.minPixels {
            let beta = (Double(v.minPixels) / Double(frames * h0 * w0)).squareRoot()
            h = Int((Double(h0) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(w0) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// How a clip of `total` frames `width × height` at `fps` is read: the
    /// frames sampled, their size, and each temporal group's tokens and time.
    public static func videoLayout(
        total: Int, width: Int, height: Int, fps: Double, configuration c: LFM3VLVisionConfiguration
    ) throws -> LFM3VLVideoLayout {
        guard let v = c.video else { throw LFM2ConfigurationError.missing("video_token_id") }
        guard fps.isFinite, fps > 0 else { throw LFM3VLVideoError.fps(fps) }
        let indices = sampledFrames(total: total, fps: fps, video: v)
        let size = try smartResize(frames: indices.count, height: height, width: width, configuration: c)
        let temporal = c.temporalPatchSize
        var padded = indices
        while padded.count % temporal != 0 { padded.append(padded[padded.count - 1]) }
        // LFM3-Omni sums the group's first and last indices over twice the clip's
        // rate; LFM3-VL averages the two frames' times. They can print different
        // tenths (10 fps, frames 1 and 2: 0.1 and 0.2).
        let seconds = stride(from: 0, to: padded.count, by: temporal).map { start -> Double in
            let (first, last) = (padded[start], padded[start + temporal - 1])
            return v.sumsIndices ? Double(first + last) / (2 * fps) : (Double(first) / fps + Double(last) / fps) / 2
        }
        let tokens = (size.height / c.patchSize) * (size.width / c.patchSize) / (c.mergeSize * c.mergeSize)
        return LFM3VLVideoLayout(indices: indices, height: size.height, width: size.width, tokensPerGroup: tokens, seconds: seconds)
    }

    // MARK: - The tower and projector

    /// `[featureCount, textHidden]` features for one picture, in the order its
    /// `<|image|>` tokens run.
    public func callAsFunction(_ image: RGBImage) throws -> MLXArray {
        let c = configuration
        let view = try Self.resized(image, configuration: c)
        let frame = normalized(view.pixels, height: view.height, width: view.width, mean: c.imageMean, std: c.imageStd, rescale: c.rescaleFactor)
        // The picture is every frame of its temporal patch.
        return encode(Array(repeating: frame, count: c.temporalPatchSize), rows: view.height / c.patchSize, columns: view.width / c.patchSize)
    }

    /// `[groups · tokensPerGroup, textHidden]` features for a video's sampled
    /// frames (`LFM3VLVideoLayout.indices`), group after group: each group of
    /// `temporal_patch_size` frames (the last frame repeated to fill the last
    /// group) runs through the tower as one picture.
    public func callAsFunction(video frames: [RGBImage], layout: LFM3VLVideoLayout) throws -> MLXArray {
        let c = configuration
        guard let v = c.video else { throw LFM2ConfigurationError.missing("video_token_id") }
        guard frames.count == layout.indices.count, let first = frames.first,
              frames.allSatisfy({ $0.width == first.width && $0.height == first.height })
        else { throw LFM3VLVideoError.frames }
        let filter: AntialiasedResize.Filter = v.resampling == .bicubic ? .cubic : .triangle
        var normalizedFrames = frames.map { frame -> MLXArray in
            let pixels = AntialiasedResize.resize(
                frame.pixels, width: frame.width, height: frame.height, channels: 3,
                toWidth: layout.width, toHeight: layout.height, filter: filter, kernel: .aten
            )
            return normalized(pixels, height: layout.height, width: layout.width, mean: v.mean, std: v.std, rescale: v.rescaleFactor)
        }
        let temporal = c.temporalPatchSize
        while normalizedFrames.count % temporal != 0 { normalizedFrames.append(normalizedFrames[normalizedFrames.count - 1]) }
        let (rows, columns) = (layout.height / c.patchSize, layout.width / c.patchSize)
        let groups = stride(from: 0, to: normalizedFrames.count, by: temporal).map { start in
            encode(Array(normalizedFrames[start ..< start + temporal]), rows: rows, columns: columns)
        }
        return groups.count == 1 ? groups[0] : concatenated(groups, axis: 0)
    }

    /// `[height, width, 3]` pixels in the tower's type, rescaled and normalized
    /// as the processor's fused step: `(x - mean / r) / (std / r)`.
    private func normalized(_ pixels: [UInt8], height: Int, width: Int, mean: [Double], std: [Double], rescale: Double) -> MLXArray {
        let means = MLXArray(mean.map { Float($0 / rescale) })
        let stds = MLXArray(std.map { Float($0 / rescale) })
        return ((MLXArray(pixels, [height, width, 3]).asType(.float32) - means) / stds).asType(dataType)
    }

    /// The tower and projector over one temporal patch of `[rows · p, columns · p, 3]` frames.
    private func encode(_ frames: [MLXArray], rows: Int, columns: Int) -> MLXArray {
        let c = configuration
        let (m, p) = (c.mergeSize, c.patchSize)
        let count = rows * columns
        // Patches in merge-block order, each flattened as (channel, time, dy, dx).
        let blocks = frames.map {
            $0.reshaped(rows / m, m, p, columns / m, m, p, 3)
                .transposed(0, 3, 1, 4, 6, 2, 5)
                .reshaped(count, 3, 1, p, p)
        }
        let patches = concatenated(blocks, axis: 2).reshaped(count, 3 * frames.count * p * p)
        var hidden = MLXFast.rmsNorm(patchifier(patches), weight: embeddingNorm, eps: c.normEpsilon)
        let rotation = Self.rotation(rows: rows, columns: columns, configuration: c)
        for layer in layers {
            let input = MLXFast.rmsNorm(hidden, weight: layer.operatorNorm, eps: c.normEpsilon)
            if let block = layer.attention {
                hidden = hidden + attend(input, block, rotation: rotation)
            } else if let block = layer.convolution {
                hidden = hidden + convolve(input, block, rows: rows, columns: columns)
            }
            let feedForwardInput = MLXFast.rmsNorm(hidden, weight: layer.feedForwardNorm, eps: c.normEpsilon)
            hidden = hidden + layer.down(silu(layer.gate(feedForwardInput)) * layer.up(feedForwardInput))
        }
        let normalizedPatches = MLXFast.layerNorm(
            hidden, weight: projectorNormWeight, bias: projectorNormBias, eps: c.projectorLayerNormEpsilon
        )
        let merged = normalizedPatches.reshaped(count / (m * m), c.hiddenSize * m * m)
        return projectorContract(gelu(projectorExpand(merged)))
    }

    /// Axial RoPE `(cos, sin)`, `[patches, headDimension]` in float32: each
    /// patch's row then column, each over `headDimension / 4` frequencies,
    /// the pair repeated over both halves of the head.
    static func rotation(rows: Int, columns: Int, configuration c: LFM3VLVisionConfiguration) -> (cos: MLXArray, sin: MLXArray) {
        let spatial = c.headDimension / 2
        let inverse = stride(from: 0, to: spatial, by: 2).map { 1 / powf(c.ropeTheta, Float($0) / Float(spatial)) }
        let m = c.mergeSize
        var angles: [Float] = []
        angles.reserveCapacity(rows * columns * spatial * 2)
        for blockRow in 0..<(rows / m) {
            for blockColumn in 0..<(columns / m) {
                for i in 0..<m {
                    for j in 0..<m {
                        let row = Float(blockRow * m + i)
                        let column = Float(blockColumn * m + j)
                        let half = inverse.map { row * $0 } + inverse.map { column * $0 }
                        angles += half + half
                    }
                }
            }
        }
        let table = MLXArray(angles, [rows * columns, spatial * 2])
        return (MLX.cos(table), MLX.sin(table))
    }

    private func attend(_ x: MLXArray, _ block: Attention, rotation: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let c = configuration
        let count = x.dim(0)
        let headDimension = c.headDimension
        func rotate(_ t: MLXArray) -> MLXArray {
            let wide = t.asType(.float32)
            let halves = wide.split(parts: 2, axis: -1)
            let turned = concatenated([-halves[1], halves[0]], axis: -1)
            let cos = rotation.cos.expandedDimensions(axis: 1)
            let sin = rotation.sin.expandedDimensions(axis: 1)
            return (wide * cos + turned * sin).asType(t.dtype)
        }
        let queries = rotate(MLXFast.rmsNorm(block.query(x).reshaped(count, c.attentionHeads, headDimension), weight: block.queryNorm, eps: c.normEpsilon))
        let keys = rotate(MLXFast.rmsNorm(block.key(x).reshaped(count, c.keyValueHeads, headDimension), weight: block.keyNorm, eps: c.normEpsilon))
        let values = block.value(x).reshaped(count, c.keyValueHeads, headDimension)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: keys.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: values.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: 1 / Float(headDimension).squareRoot(),
            mask: nil
        )
        return block.output(attended[0].transposed(1, 0, 2).reshaped(count, -1))
    }

    /// The gated depthwise convolution: `out(C · conv(B · x))` over the raster grid, zero-padded.
    private func convolve(_ x: MLXArray, _ block: Convolution, rows: Int, columns: Int) -> MLXArray {
        let c = configuration
        let (m, d, k) = (c.mergeSize, c.hiddenSize, c.convolutionKernel)
        let parts = block.input(x).split(parts: 3, axis: -1)
        let gated = parts[0] * parts[2]
        // Merge-block order to the raster grid.
        let grid = gated.reshaped(rows / m, columns / m, m, m, d).transposed(0, 2, 1, 3, 4).reshaped(rows, columns, d)
        let r = k / 2
        let verticalPad = MLXArray.zeros([r, columns, d], dtype: grid.dtype)
        let tall = concatenated([verticalPad, grid, verticalPad], axis: 0)
        let horizontalPad = MLXArray.zeros([rows + 2 * r, r, d], dtype: grid.dtype)
        let padded = concatenated([horizontalPad, tall, horizontalPad], axis: 1)
        var convolved = padded[0 ..< rows, 0 ..< columns] * block.taps[0, 0]
        for dy in 0..<k {
            for dx in 0..<k where dy > 0 || dx > 0 {
                convolved = convolved + padded[dy ..< (dy + rows), dx ..< (dx + columns)] * block.taps[dy, dx]
            }
        }
        // Back to merge-block order.
        let blocks = convolved.reshaped(rows / m, m, columns / m, m, d).transposed(0, 2, 1, 3, 4).reshaped(rows * columns, d)
        return block.output(parts[1] * blocks)
    }
}

/// How LFM3-VL reads a video: the frames it samples, the size they are
/// resized to, and per temporal group its tokens and its time in seconds.
public struct LFM3VLVideoLayout: Equatable, Sendable {
    /// The sampled frames' indices in the clip.
    public let indices: [Int]
    public let height: Int
    public let width: Int
    /// `<|video|>` tokens (feature rows) per temporal group.
    public let tokensPerGroup: Int
    /// Per temporal group: the time its timestamp prints.
    public let seconds: [Double]

    public var featureCount: Int { tokensPerGroup * seconds.count }
}

public enum LFM3VLVideoError: Error, Equatable, Sendable {
    case tooShort(frames: Int, minimum: Int)
    case fps(Double)
    case frames
}

extension LFM3VLVideoError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .tooShort(let frames, let minimum): return "A video needs at least \(minimum) sampled frames; this one has \(frames)."
        case .fps(let fps): return "A video needs a positive frame rate; this one has \(fps)."
        case .frames: return "A video's frames must all be the same size, one per sampled index."
        }
    }
}
