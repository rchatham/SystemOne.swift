import Foundation
import MLX
import MLXFFT
import MLXNN
import SystemOne

/// `audio.py`: a log-mel front end, a FastConformer, an MLP adapter, and a
/// residual correction. One prefix embedding per 80 ms of 16 kHz speech.
final class D1Audio {
    static let sampleRate = 16_000
    static let minimumSamples = 8_000
    static let maximumSeconds = 30
    static let hop = 160
    static let fftSize = 512
    static let windowSize = 400

    private struct Layer {
        let feedForward1: FeedForward, feedForward2: FeedForward
        let normFeedForward1: Norm, normAttention: Norm, normConvolution: Norm, normFeedForward2: Norm, normOut: Norm
        let query: Linear, key: Linear, value: Linear, output: Linear, position: Linear
        let biasU: MLXArray, biasV: MLXArray
        let pointwise1: Linear, depthwiseWeight: MLXArray, depthwiseBias: MLXArray, pointwise2: Linear
        let batchNormScale: MLXArray, batchNormShift: MLXArray
    }

    private struct FeedForward {
        let expand: Linear, contract: Linear
        func callAsFunction(_ x: MLXArray) -> MLXArray { contract(silu(expand(x))) }
    }

    private struct Norm {
        let weight: MLXArray, bias: MLXArray
        func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: weight, bias: bias, eps: 1e-5) }
    }

    private let configuration: D1Configuration.Audio
    private let filterbank: MLXArray
    private let window: MLXArray
    /// `pre_encode.conv` 0, 2, 3, 5, 6 as MLX NHWC weights with biases.
    private let subsampling: [(weight: MLXArray, bias: MLXArray, groups: Int, stride: Int)]
    private let subsamplingOut: Linear
    private let layers: [Layer]
    private let adapterNorm: Norm, adapterExpand: Linear, adapterContract: Linear
    private let residualNorm: Norm, residualDown: Linear, residualUp: Linear

    init(weights: D1Weights) {
        let configuration = weights.configuration.audio
        self.configuration = configuration
        func linear(_ name: String, bias: Bool = true) -> Linear {
            Linear(weight: weights.array("\(name).weight"), bias: bias ? weights.array("\(name).bias") : nil)
        }
        func norm(_ name: String) -> Norm { Norm(weight: weights.array("\(name).weight"), bias: weights.array("\(name).bias")) }
        /// Conv1d `[out, in, 1]` as a Linear over the channel axis.
        func pointwise(_ name: String) -> Linear {
            Linear(weight: weights.array("\(name).weight").squeezed(axis: -1), bias: weights.array("\(name).bias"))
        }

        filterbank = MLXArray(
            MelFilterbank.slaney(sampleRate: Self.sampleRate, fftSize: Self.fftSize, melCount: configuration.featIn)
                .flatMap { $0 },
            [configuration.featIn, Self.fftSize / 2 + 1]
        )
        // torch.hann_window(400, periodic=False), centred in the 512-point frame.
        let hann = (0..<Self.windowSize).map { index -> Float in
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(Self.windowSize - 1)))
        }
        let leading = (Self.fftSize - Self.windowSize) / 2
        window = MLXArray([Float](repeating: 0, count: leading) + hann
            + [Float](repeating: 0, count: Self.fftSize - Self.windowSize - leading))

        let pre = "audio.encoder.pre_encode.conv"
        let channels = configuration.subsamplingConvChannels
        subsampling = [(0, 1, 2), (2, channels, 2), (3, 1, 1), (5, channels, 2), (6, 1, 1)].map { index, groups, stride in
            // torch [out, in/groups, kh, kw] -> MLX [out, kh, kw, in/groups]
            (weights.array("\(pre).\(index).weight").transposed(0, 2, 3, 1), weights.array("\(pre).\(index).bias"),
             groups, stride)
        }
        subsamplingOut = linear("audio.encoder.pre_encode.out")
        layers = (0..<configuration.nLayers).map { index in
            let prefix = "audio.encoder.layers.\(index)"
            let mean = weights.array("\(prefix).conv.batch_norm.running_mean")
            let variance = weights.array("\(prefix).conv.batch_norm.running_var")
            let scale = weights.array("\(prefix).conv.batch_norm.weight") / sqrt(variance + 1e-5)
            return Layer(
                feedForward1: FeedForward(expand: linear("\(prefix).feed_forward1.linear1"),
                                          contract: linear("\(prefix).feed_forward1.linear2")),
                feedForward2: FeedForward(expand: linear("\(prefix).feed_forward2.linear1"),
                                          contract: linear("\(prefix).feed_forward2.linear2")),
                normFeedForward1: norm("\(prefix).norm_feed_forward1"),
                normAttention: norm("\(prefix).norm_self_att"),
                normConvolution: norm("\(prefix).norm_conv"),
                normFeedForward2: norm("\(prefix).norm_feed_forward2"),
                normOut: norm("\(prefix).norm_out"),
                query: linear("\(prefix).self_attn.linear_q"),
                key: linear("\(prefix).self_attn.linear_k"),
                value: linear("\(prefix).self_attn.linear_v"),
                output: linear("\(prefix).self_attn.linear_out"),
                position: linear("\(prefix).self_attn.linear_pos", bias: false),
                biasU: weights.array("\(prefix).self_attn.pos_bias_u"),
                biasV: weights.array("\(prefix).self_attn.pos_bias_v"),
                pointwise1: pointwise("\(prefix).conv.pointwise_conv1"),
                // torch [C, 1, k] -> MLX conv1d [C, k, 1]
                depthwiseWeight: weights.array("\(prefix).conv.depthwise_conv.weight").transposed(0, 2, 1),
                depthwiseBias: weights.array("\(prefix).conv.depthwise_conv.bias"),
                pointwise2: pointwise("\(prefix).conv.pointwise_conv2"),
                batchNormScale: scale,
                batchNormShift: weights.array("\(prefix).conv.batch_norm.bias") - mean * scale
            )
        }
        adapterNorm = norm("audio.adapter.norm")
        adapterExpand = linear("audio.adapter.linear_1")
        adapterContract = linear("audio.adapter.linear_2")
        residualNorm = norm("audio.residual.ln")
        residualDown = linear("audio.residual.down")
        residualUp = linear("audio.residual.up")
    }

    /// Samples after the 30 s cut and the 0.5 s minimum padding.
    static func clampedSampleCount(_ count: Int) -> Int {
        max(min(count, maximumSeconds * sampleRate), minimumSamples)
    }

    /// One subsampling step's output length: `floor((L + 2 - 3) / 2) + 1`.
    static func subsampled(_ length: Int) -> Int {
        Int((Double(length - 1) / 2).rounded(.down)) + 1
    }

    /// Prefix embeddings after three stride-2 subsampling steps over 10 ms frames.
    static func prefixLength(sampleCount: Int) -> Int {
        (0..<3).reduce(clampedSampleCount(sampleCount) / hop) { length, _ in subsampled(length) }
    }

    /// `[P, D]` prefix embeddings for one clip.
    func callAsFunction(_ clip: AudioClip) throws -> MLXArray {
        let (features, frames) = melFeatures(clip.samples)
        let encoded = encoder(features, frames: frames)
        let adapted = adapterContract(gelu(adapterExpand(adapterNorm(encoded))))
        return adapted + residualUp(gelu(residualDown(residualNorm(adapted))))
    }

    /// `MelFrontend`: preemphasis, |STFT|², Slaney mel, log, per-feature
    /// normalization over valid frames. Returns `[T, features]` and the valid count.
    func melFeatures(_ samples: [Float]) -> (MLXArray, Int) {
        var signal = Array(samples.prefix(Self.maximumSeconds * Self.sampleRate))
        if signal.count < Self.minimumSamples {
            signal += [Float](repeating: 0, count: Self.minimumSamples - signal.count)
        }
        let count = signal.count
        let valid = count / Self.hop
        var emphasized = signal
        for index in stride(from: count - 1, to: 0, by: -1) {
            emphasized[index] = signal[index] - 0.97 * signal[index - 1]
        }

        let half = Self.fftSize / 2
        let padded = MLXArray([Float](repeating: 0, count: half) + emphasized + [Float](repeating: 0, count: half))
        let total = 1 + count / Self.hop
        let offsets = MLXArray((0..<total).map { Int32($0 * Self.hop) }).reshaped(total, 1)
            + MLXArray((0..<Int32(Self.fftSize))).reshaped(1, Self.fftSize)
        let spectrum = MLXFFT.rfft(padded[offsets] * window, n: Self.fftSize, axis: -1)
        let power = MLX.pow(abs(spectrum), 2)
        let mel = log(matmul(power, filterbank.transposed()) + Float(pow(2.0, -24.0)))

        let rows = MLXArray(0..<Int32(total)).reshaped(total, 1) .< Int32(valid)
        let validMel = MLX.where(rows, mel, MLXArray(Float(0)))
        let mean = validMel.sum(axis: 0) / Float(valid)
        let deviation = MLX.where(rows, mel - mean, MLXArray(Float(0)))
        let std = sqrt(MLX.pow(deviation, 2).sum(axis: 0) / (Float(valid) - 1))
        let normalized = (mel - mean) / (MLX.where(isNaN(std), MLXArray(Float(0)), std) + 1e-5)
        return (MLX.where(rows, normalized, MLXArray(Float(0))), valid)
    }

    /// `ConvSubsampling` with its per-step time masks, then the conformer over
    /// the valid frames only (equivalent to its attention and convolution masks).
    func encoder(_ features: MLXArray, frames: Int) -> MLXArray {
        var length = frames
        var x = features.expandedDimensions(axes: [0, -1])
        func masked(_ x: MLXArray, _ length: Int) -> MLXArray {
            let keep = MLXArray(0..<Int32(x.dim(1))) .< Int32(length)
            return x * keep.asType(.float32).reshaped(1, -1, 1, 1)
        }
        for (index, conv) in subsampling.enumerated() {
            x = masked(x, length)
            x = conv2d(x, conv.weight, stride: .init(conv.stride), padding: conv.stride == 2 ? 1 : 0,
                       groups: conv.groups) + conv.bias
            if conv.stride == 2 { length = Self.subsampled(length) }
            // ReLU follows the first conv and each pointwise conv.
            if index != 1 && index != 3 {
                x = relu(masked(x, length))
            }
        }
        x = masked(x, length)
        // NHWC [1, T, F, C] -> [1, T, C·F], channel-major as torch flattens.
        let (time, frequency, channels) = (x.dim(1), x.dim(2), x.dim(3))
        var hidden = subsamplingOut(x.transposed(0, 1, 3, 2).reshaped(1, time, channels * frequency))[0, 0..<length]

        let positions = relativePositions(length)
        for layer in layers {
            hidden = hidden + 0.5 * layer.feedForward1(layer.normFeedForward1(hidden))
            hidden = hidden + attention(layer.normAttention(hidden), layer: layer, positions: positions)
            hidden = hidden + convolution(layer.normConvolution(hidden), layer: layer)
            hidden = hidden + 0.5 * layer.feedForward2(layer.normFeedForward2(hidden))
            hidden = layer.normOut(hidden)
        }
        return hidden
    }

    /// Sinusoids for relative positions `t-1 … -(t-1)`: `[2t-1, d]`.
    func relativePositions(_ length: Int) -> MLXArray {
        let model = configuration.dModel
        let positions = MLXArray((0..<(2 * length - 1)).map { Float(length - 1 - $0) }).reshaped(-1, 1)
        let frequencies = exp(MLXArray(stride(from: 0, to: model, by: 2).map(Float.init))
                              * Float(-(log(10_000.0) / Double(model))))
        let angles = positions * frequencies
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(2 * length - 1, model)
    }

    /// Transformer-XL relative-position attention over `[T, d]`.
    private func attention(_ x: MLXArray, layer: Layer, positions: MLXArray) -> MLXArray {
        let (time, heads) = (x.dim(0), configuration.nHeads)
        let headDimension = configuration.dModel / heads
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(-1, heads, headDimension).transposed(1, 0, 2) }
        let q = layer.query(x).reshaped(time, heads, headDimension)
        let k = split(layer.key(x))
        let v = split(layer.value(x))
        let p = split(layer.position(positions))
        let content = matmul((q + layer.biasU).transposed(1, 0, 2), k.transposed(0, 2, 1))
        let full = matmul((q + layer.biasV).transposed(1, 0, 2), p.transposed(0, 2, 1))
        // rel_shift: position term for query i, key j sits at column t-1-i+j.
        let columns = MLXArray(0..<Int32(time)).reshaped(1, time) - MLXArray(0..<Int32(time)).reshaped(time, 1)
            + Int32(time - 1)
        let positional = takeAlong(full, broadcast(columns, to: [heads, time, time]), axis: -1)
        let scores = softmax((content + positional) / Float(headDimension).squareRoot(), axis: -1)
        return layer.output(matmul(scores, v).transposed(1, 0, 2).reshaped(time, -1))
    }

    /// Pointwise → GLU → depthwise (k=9) → BatchNorm → SiLU → pointwise over `[T, d]`.
    private func convolution(_ x: MLXArray, layer: Layer) -> MLXArray {
        let gated = glu(layer.pointwise1(x), axis: -1)
        let pad = (configuration.convKernelSize - 1) / 2
        let depthwise = conv1d(gated.expandedDimensions(axis: 0), layer.depthwiseWeight,
                               padding: pad, groups: configuration.dModel)[0] + layer.depthwiseBias
        return layer.pointwise2(silu(depthwise * layer.batchNormScale + layer.batchNormShift))
    }
}
