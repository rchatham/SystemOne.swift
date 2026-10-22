import Foundation

/// PyTorch's separable antialiased bilinear resize (`interpolate(...,
/// mode="bilinear", antialias=True, align_corners=False)`), the Pillow
/// algorithm torchvision uses for image preprocessing.
///
/// Both passes resample the last axis first (width), then height. The uint8
/// form reproduces ATen's integer kernel bit for bit: int16 weights at a
/// per-axis precision, integer accumulation with a rounding bias, and uint8
/// intermediates between the two passes.
package enum AntialiasedResize {
    /// One axis's taps: for each output index, the first input index and its weights.
    package struct Taps<Weight> {
        package let starts: [Int]
        package let weights: [[Weight]]
    }

    /// ATen `_compute_indices_min_size_weights_aa` with the triangle filter, in double.
    package static func taps(input: Int, output: Int) -> Taps<Double> {
        let scale = Double(input) / Double(output)
        let support = scale >= 1 ? scale : 1
        let maximumTaps = Int(support.rounded(.up)) * 2 + 1
        let inverseScale = scale >= 1 ? 1 / scale : 1
        var starts: [Int] = []
        var weights: [[Double]] = []
        for index in 0..<output {
            let center = scale * (Double(index) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let count = min(max(min(Int(center + support + 0.5), input) - start, 0), maximumTaps)
            var row = (0..<count).map { tap -> Double in
                max(0, 1 - abs((Double(tap + start) - center + 0.5) * inverseScale))
            }
            let total = row.reduce(0, +)
            if total != 0 { row = row.map { $0 / total } }
            starts.append(start)
            weights.append(row)
        }
        return Taps(starts: starts, weights: weights)
    }

    /// The same taps computed with float32 scale and weights, as ATen does for float tensors.
    package static func floatTaps(input: Int, output: Int) -> Taps<Float> {
        let scale = Float(input) / Float(output)
        let support = Float(scale >= 1 ? Double(scale) : 1)
        let maximumTaps = Int(support.rounded(.up)) * 2 + 1
        let inverseScale: Float = scale >= 1 ? 1 / scale : 1
        var starts: [Int] = []
        var weights: [[Float]] = []
        for index in 0..<output {
            let center = Float(Double(scale) * (Double(index) + 0.5))
            let start = max(Int(Double(center) - Double(support) + 0.5), 0)
            let count = min(max(min(Int(Double(center) + Double(support) + 0.5), input) - start, 0), maximumTaps)
            var row = (0..<count).map { tap -> Float in
                let x = Float((Double(tap + start) - Double(center) + 0.5) * Double(inverseScale))
                return max(0, 1 - abs(x))
            }
            let total = row.reduce(0, +)
            if total != 0 { row = row.map { $0 / total } }
            starts.append(start)
            weights.append(row)
        }
        return Taps(starts: starts, weights: weights)
    }

    /// Resizes interleaved 8-bit pixels (`height × width × channels`).
    package static func resize(
        _ pixels: [UInt8],
        width: Int,
        height: Int,
        channels: Int,
        toWidth: Int,
        toHeight: Int
    ) -> [UInt8] {
        var current = pixels
        var currentWidth = width
        if toWidth != width {
            current = resampleUInt8(current, lines: height, inputLength: width, outputLength: toWidth,
                                    lineStride: width * channels, step: channels, channels: channels,
                                    outputLineStride: toWidth * channels, outputStep: channels)
            currentWidth = toWidth
        }
        if toHeight != height {
            current = resampleUInt8(current, lines: currentWidth, inputLength: height, outputLength: toHeight,
                                    lineStride: channels, step: currentWidth * channels, channels: channels,
                                    outputLineStride: channels, outputStep: currentWidth * channels)
        }
        return current
    }

    /// One integer pass along an axis. `lines` independent lines are resampled;
    /// a line starts at `line * lineStride` and advances by `step` per sample.
    private static func resampleUInt8(
        _ input: [UInt8],
        lines: Int,
        inputLength: Int,
        outputLength: Int,
        lineStride: Int,
        step: Int,
        channels: Int,
        outputLineStride: Int,
        outputStep: Int
    ) -> [UInt8] {
        let taps = taps(input: inputLength, output: outputLength)
        let maximumWeight = taps.weights.joined().max() ?? 0
        var precision = 0
        while precision < 22, Int(0.5 + maximumWeight * Double(1 << (precision + 1))) < (1 << 15) {
            precision += 1
        }
        let integerWeights = taps.weights.map { row in
            row.map { weight -> Int32 in
                let value = weight * Double(1 << precision)
                return Int32(value < 0 ? Int(-0.5 + value) : Int(0.5 + value))
            }
        }
        let bias = Int32(1) << Int32(precision - 1)
        var output = [UInt8](repeating: 0, count: lines * outputLength * channels)
        for line in 0..<lines {
            for index in 0..<outputLength {
                let start = taps.starts[index]
                for channel in 0..<channels {
                    var sum = bias
                    for (tap, weight) in integerWeights[index].enumerated() {
                        sum += Int32(input[line * lineStride + (start + tap) * step + channel]) * weight
                    }
                    let value = min(max(sum >> Int32(precision), 0), 255)
                    output[line * outputLineStride + index * outputStep + channel] = UInt8(value)
                }
            }
        }
        return output
    }
}
