import Foundation
import MLX
import MLXFFT
import MLXNN
import SystemOne

/// LFM3-Omni's speech settings: the audio encoder (`audio_encoder_config`,
/// whose `text_config` is an LFM2 trunk), the audio projector, and the log-mel
/// feature extractor (`processor_config.json`'s `feature_extractor`).
public struct LFM3AudioConfiguration: Equatable, Sendable {
    // The encoder.
    public let trunk: LFM2Configuration
    public let convolutionChannels: Int
    public let hiddenSize: Int
    public let melBins: Int
    /// Keys each encoder position attends to, itself included.
    public let slidingWindow: Int

    // The projector.
    public let textHiddenSize: Int
    /// The placeholder token audio features replace (`<|audio|>`).
    public let audioTokenID: Int

    // The feature extractor.
    public let sampleRate: Int
    public let fftSize: Int
    public let hop: Int
    public let logFloor: Float
    public let normalizationOffset: Float
    public let normalizationDivisor: Float
    /// Feature frames per encoder position: three stride-2 convolutions.
    public let encoderStride: Int

    /// Reads `config.json` and the `feature_extractor` of
    /// `processor_config.json` (or `preprocessor_config.json`).
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
        let processor = try object("processor_config.json")?["feature_extractor"] as? [String: Any]
            ?? object("preprocessor_config.json") ?? [:]
        try self.init(json: config, processor: processor)
    }

    /// `json` is an LFM3-Omni `config.json`; `processor` its feature extractor's settings.
    public init(json: [String: Any], processor: [String: Any]) throws {
        guard let encoder = json["audio_encoder_config"] as? [String: Any] else {
            throw LFM2ConfigurationError.missing("audio_encoder_config")
        }
        func int(_ object: [String: Any], _ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = object[key] as? Int { return value }
            if let fallback { return fallback }
            throw LFM2ConfigurationError.missing(key)
        }
        func float(_ object: [String: Any], _ key: String, _ fallback: Float) -> Float {
            (object[key] as? NSNumber)?.floatValue ?? fallback
        }

        trunk = try LFM2Configuration(json: encoder)
        convolutionChannels = try int(encoder, "conv_channels", 256)
        hiddenSize = try int(encoder, "hidden_size", 768)
        melBins = try int(encoder, "num_mel_bins", 128)
        slidingWindow = try int(encoder, "sliding_window", 32)
        textHiddenSize = try int(json["text_config"] as? [String: Any] ?? [:], "hidden_size")
        audioTokenID = try int(json, "audio_token_id", 125_031)

        sampleRate = try int(processor, "sampling_rate", 16_000)
        fftSize = try int(processor, "n_fft", 400)
        hop = try int(processor, "hop_length", 160)
        logFloor = float(processor, "log_floor", -12)
        normalizationOffset = float(processor, "normalization_offset", 4)
        normalizationDivisor = float(processor, "normalization_divisor", 4)
        encoderStride = try int(processor, "encoder_stride", 8)

        let window = try int(processor, "win_length", fftSize)
        let features = try int(processor, "feature_size", melBins)
        let side = processor["padding_side"] as? String ?? "right"
        guard window == fftSize, features == melBins, side == "right", encoderStride == 8 else {
            throw LFM2ConfigurationError.unsupported(
                "a feature extractor with win_length \(window), feature_size \(features), padding_side \(side), encoder_stride \(encoderStride)"
            )
        }
        guard trunk.hiddenSize == hiddenSize, melBins % 8 == 0, sampleRate > 0, fftSize > 0, hop > 0,
              slidingWindow > 0, normalizationDivisor != 0
        else {
            throw LFM2ConfigurationError.invalid("audio encoder of \(hiddenSize) features over \(melBins) mel bins")
        }
    }
}

public enum LFM3AudioError: Error, Equatable, Sendable {
    case sampleRate(Int)
}

extension LFM3AudioError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .sampleRate(let rate): return "An audio clip needs a positive sample rate; this one has \(rate) Hz."
        }
    }
}

/// LFM3-Omni's speech path, Apache-2.0, following transformers' LFM3-Omni
/// code (as vendored with LiquidAI's d1-Omni-2B) and mlx-vlm's `lfm3_audio`
/// port:
///
/// - the clip resampled to the extractor's rate (d1's `media.resampled`);
/// - right-padded to whole frames, a periodic-Hann power spectrum every hop,
///   Slaney mel filters, `log10` floored at `log_floor`, then `(x + 4) / 4`;
/// - three stride-2 convolutions over the (mel, time) plane (the last two
///   depthwise, each followed by a pointwise one), SiLU after each stage,
///   and a linear map of each step's channels × mel bins;
/// - a causal LFM2 trunk whose attention sees the last `sliding_window`
///   positions, and its final norm;
/// - a layer norm and a two-layer GELU projector into the text trunk's width;
/// - `<|vision_start|>`, one `<|audio|>` per feature row, `<|vision_end|>`.
public final class LFM3AudioEncoder {
    public static let audioToken = "<|audio|>"

    private struct Convolution {
        /// MLX's `[out, kh, kw, in / groups]`.
        let weight: MLXArray
        let bias: MLXArray
        let stride: Int
        let padding: Int
        let groups: Int

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            conv2d(x, weight, stride: [stride, stride], padding: [padding, padding], groups: groups) + bias
        }
    }

    public let configuration: LFM3AudioConfiguration
    public let dataType: DType
    /// `[mel, fft / 2 + 1]` float32.
    private let filterbank: MLXArray
    /// The periodic Hann window, float32.
    private let window: MLXArray
    private let input: Convolution
    private let middle: Convolution, middlePointwise: Convolution
    private let end: Convolution, endPointwise: Convolution
    private let output: Linear
    private let trunk: LFM2Trunk
    private let projectorNormWeight: MLXArray
    private let projectorNormBias: MLXArray
    private let projectorExpand: Linear
    private let projectorContract: Linear

    public convenience init(modelFolder: URL, dataType: DType = .float32) throws {
        try self.init(
            store: SafetensorsCheckpoint(modelFolder: modelFolder),
            configuration: LFM3AudioConfiguration(modelFolder: modelFolder),
            dataType: dataType
        )
    }

    /// Loads the encoder and projector under transformers' names
    /// (`model.audio_encoder.`, `model.audio_projector.`) or mlx-vlm's
    /// (`audio_tower.`, `audio_projector.`). Convolution kernels may be
    /// torch's `[out, in, kh, kw]` or MLX's `[out, kh, kw, in]`; `lin_out` a
    /// linear layer or a width-1 `Conv1d`.
    public init(store: any TensorStore, configuration: LFM3AudioConfiguration, dataType: DType = .float32) throws {
        self.configuration = configuration
        self.dataType = dataType
        let c = configuration
        let channels = c.convolutionChannels
        guard let tower = ["model.audio_encoder.", "audio_encoder.", "audio_tower."]
            .first(where: { store.descriptors["\($0)downsample.conv_in.weight"] != nil })
        else { throw WeightsError.tensorSetMismatch(missing: ["audio_tower.downsample.conv_in.weight"], unexpected: []) }
        guard let projector = ["model.audio_projector.", "audio_projector."]
            .first(where: { store.descriptors["\($0)linear_1.weight"] != nil })
        else { throw WeightsError.tensorSetMismatch(missing: ["audio_projector.linear_1.weight"], unexpected: []) }

        func check(_ name: String, _ shape: [Int]) throws {
            let module = String(name.dropLast(".weight".count))
            if name.hasSuffix(".weight"), shape.count == 2, store.isQuantized(module) {
                _ = try store.quantizationParameters(of: module, inputWidth: shape[1])
                return
            }
            let actual = try store.requiredTensor(name).descriptor.shape
            guard actual == shape else { throw WeightsError.shapeMismatch(name: name, expected: shape, actual: actual) }
        }
        func vector(_ name: String, _ width: Int) throws -> MLXArray {
            try check(name, [width])
            return try store.requiredTensor(name).mlxArray(dataType)
        }
        func linear(_ name: String, _ output: Int, _ input: Int) throws -> Linear {
            try check("\(name).weight", [output, input])
            try check("\(name).bias", [output])
            return try store.linear(name, inputWidth: input, bias: true, dataType: dataType)
        }
        func convolution(_ name: String, input: Int, kernel: Int, stride: Int, groups: Int) throws -> Convolution {
            let weightName = "\(tower)downsample.\(name).weight"
            let shape = try store.requiredTensor(weightName).descriptor.shape
            let torch = [channels, input / groups, kernel, kernel]
            let mlx = [channels, kernel, kernel, input / groups]
            var weight = try store.requiredTensor(weightName).mlxArray(dataType)
            // The two layouts coincide only when a kernel side equals the input width; mlx-vlm's names hold MLX's.
            if shape == torch, shape != mlx || tower != "audio_tower." {
                weight = weight.transposed(0, 2, 3, 1)
            } else if shape != mlx {
                throw WeightsError.shapeMismatch(name: weightName, expected: torch, actual: shape)
            }
            return Convolution(
                weight: weight, bias: try vector("\(tower)downsample.\(name).bias", channels),
                stride: stride, padding: kernel / 2, groups: groups
            )
        }

        filterbank = MLXArray(
            MelFilterbank.slaney(sampleRate: c.sampleRate, fftSize: c.fftSize, melCount: c.melBins).flatMap { $0 },
            [c.melBins, c.fftSize / 2 + 1]
        )
        // torch.hann_window(n_fft): periodic.
        window = MLXArray((0..<c.fftSize).map { index -> Float in
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(c.fftSize)))
        })

        input = try convolution("conv_in", input: 1, kernel: 3, stride: 2, groups: 1)
        middle = try convolution("conv_mid", input: channels, kernel: 3, stride: 2, groups: channels)
        middlePointwise = try convolution("lin_mid", input: channels, kernel: 1, stride: 1, groups: 1)
        end = try convolution("conv_end", input: channels, kernel: 3, stride: 2, groups: channels)
        endPointwise = try convolution("lin_end", input: channels, kernel: 1, stride: 1, groups: 1)
        let flattened = c.melBins / 8 * channels
        let outputName = "\(tower)downsample.lin_out"
        if let shape = store.descriptors["\(outputName).weight"]?.shape, shape.count == 3 {
            guard shape == [c.hiddenSize, flattened, 1] else {
                throw WeightsError.shapeMismatch(name: "\(outputName).weight", expected: [c.hiddenSize, flattened], actual: shape)
            }
            output = Linear(
                weight: try store.requiredTensor("\(outputName).weight").mlxArray(dataType).squeezed(axis: -1),
                bias: try vector("\(outputName).bias", c.hiddenSize)
            )
        } else {
            output = try linear(outputName, c.hiddenSize, flattened)
        }
        trunk = try LFM2Trunk(store: store, configuration: c.trunk, prefix: "\(tower)lfm.", dataType: dataType)

        projectorNormWeight = try vector("\(projector)layer_norm.weight", c.hiddenSize)
        projectorNormBias = try vector("\(projector)layer_norm.bias", c.hiddenSize)
        projectorExpand = try linear("\(projector)linear_1", c.textHiddenSize, c.hiddenSize)
        projectorContract = try linear("\(projector)linear_2", c.textHiddenSize, c.textHiddenSize)
    }

    // MARK: - Counting

    /// Samples `clip` has at the extractor's rate.
    public static func sampleCount(of clip: AudioClip, configuration c: LFM3AudioConfiguration) throws -> Int {
        guard clip.sampleRate > 0 else { throw LFM3AudioError.sampleRate(clip.sampleRate) }
        return SincResample.outputCount(clip.samples.count, from: clip.sampleRate, to: c.sampleRate)
    }

    /// Feature frames of `sampleCount` samples, right-padded to `n_fft + k · hop`.
    public static func frameCount(sampleCount: Int, configuration c: LFM3AudioConfiguration) -> Int {
        guard sampleCount > c.fftSize else { return 1 }
        return 1 + (sampleCount - c.fftSize + c.hop - 1) / c.hop
    }

    /// Feature rows (and `<|audio|>` tokens) `clip` yields: one per `encoder_stride` frames.
    public static func featureCount(of clip: AudioClip, configuration c: LFM3AudioConfiguration) throws -> Int {
        let frames = frameCount(sampleCount: try sampleCount(of: clip, configuration: c), configuration: c)
        return (frames + c.encoderStride - 1) / c.encoderStride
    }

    /// The chat template's `<|vision_start|><|audio|><|vision_end|>` with the
    /// processor's expansion of `<|audio|>`.
    public static func markup(featureCount: Int) -> String {
        LFM3VLVision.visionStart + String(repeating: audioToken, count: featureCount) + LFM3VLVision.visionEnd
    }

    // MARK: - The encoder and projector

    /// `[frames, mel]` float32 log-mel features of samples at the extractor's rate.
    public func logMel(_ samples: [Float]) -> MLXArray {
        let c = configuration
        let frames = Self.frameCount(sampleCount: samples.count, configuration: c)
        let padded = (frames - 1) * c.hop + c.fftSize
        let signal = MLXArray(samples + [Float](repeating: 0, count: padded - samples.count))
        let offsets = MLXArray((0..<frames).map { Int32($0 * c.hop) }).reshaped(frames, 1)
            + MLXArray(0..<Int32(c.fftSize)).reshaped(1, c.fftSize)
        let spectrum = MLXFFT.rfft(signal[offsets] * window, n: c.fftSize, axis: -1)
        let power = abs(spectrum).square()
        let mel = matmul(power, filterbank.transposed())
        return (maximum(log10(mel), MLXArray(c.logFloor)) + c.normalizationOffset) / c.normalizationDivisor
    }

    /// `[featureCount, textHidden]` features for one clip, in the order its
    /// `<|audio|>` tokens run.
    public func callAsFunction(_ clip: AudioClip) throws -> MLXArray {
        let c = configuration
        guard clip.sampleRate > 0 else { throw LFM3AudioError.sampleRate(clip.sampleRate) }
        let samples = SincResample.resample(clip.samples, from: clip.sampleRate, to: c.sampleRate)
        let features = logMel(samples)
        let frames = features.dim(0)
        // NHWC over the (mel, time) plane, one channel.
        var x = features.transposed().reshaped(1, c.melBins, frames, 1).asType(dataType)
        x = Self.activated(input(x), silu)
        x = Self.activated(middlePointwise(middle(x)), silu)
        x = Self.activated(endPointwise(end(x)), silu)
        // Each step's (channel, mel) features, channel-major, as the reference flattens them.
        let (bins, steps, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let embedded = output(x.transposed(0, 2, 3, 1).reshaped(1, steps, channels * bins))
        let hidden = try trunk(embeddings: embedded, window: c.slidingWindow)[0]
        let normalized = MLXFast.layerNorm(hidden, weight: projectorNormWeight, bias: projectorNormBias, eps: 1e-5)
        return projectorContract(Self.activated(projectorExpand(normalized), gelu))
    }

    /// `activation` in float32, rounded once to `x`'s type, as torch evaluates it for bfloat16 inputs.
    private static func activated(_ x: MLXArray, _ activation: (MLXArray) -> MLXArray) -> MLXArray {
        x.dtype == .float32 ? activation(x) : activation(x.asType(.float32)).asType(x.dtype)
    }
}
