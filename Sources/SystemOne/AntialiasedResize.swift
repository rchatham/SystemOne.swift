import Foundation

/// Separable antialiased image resizing with the integer kernels image
/// preprocessing runs: PyTorch's (`interpolate(..., antialias=True,
/// align_corners=False)`, which torchvision uses on uint8 images) and Pillow's
/// (`Image.resize`).
///
/// Both compute the same per-axis taps, convert them to fixed point, and
/// resample one axis at a time with integer accumulation, a rounding bias, and
/// uint8 intermediates between the two passes. They differ in three places,
/// all reproduced here bit for bit:
/// - weights: ATen's int16 weights at a per-axis precision (at most 22 bits)
///   versus Pillow's int32 weights at 22 bits;
/// - the cubic kernel's outer branch, written in two algebraically equal forms;
/// - pass order: ATen always resamples width first; Pillow 13 resamples height
///   first when the height shrinks by more than twice the width's shrink.
package enum AntialiasedResize {
    /// The resampling filter.
    package enum Filter: Sendable {
        /// The triangle filter (`bilinear`), support 1.
        case triangle
        /// Keys' cubic convolution with `a = -0.5` (`bicubic`), support 2.
        case cubic

        var support: Double {
            switch self {
            case .triangle: return 1
            case .cubic: return 2
            }
        }
    }

    /// The integer kernel whose rounding to reproduce.
    package enum Kernel: Sendable {
        /// ATen's uint8 kernel (torchvision on uint8 tensors).
        case aten
        /// Pillow 13's `ImagingResample` on 8-bit bands.
        case pillow
    }

    /// One axis's taps: for each output index, the first input index and its weights.
    package struct Taps<Weight> {
        package let starts: [Int]
        package let weights: [[Weight]]
    }

    /// `filter` at `x` as `kernel` writes it.
    static func evaluate(_ filter: Filter, kernel: Kernel, at x: Double) -> Double {
        let x = abs(x)
        switch filter {
        case .triangle:
            return x < 1 ? 1 - x : 0
        case .cubic:
            let a = -0.5
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            guard x < 2 else { return 0 }
            switch kernel {
            case .aten: return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
            case .pillow: return (((x - 5) * x + 8) * x - 4) * a
            }
        }
    }

    /// The taps ATen's `_compute_indices_min_size_weights_aa` and Pillow's
    /// `precompute_coeffs` compute, in double.
    package static func taps(input: Int, output: Int, filter: Filter = .triangle, kernel: Kernel = .aten) -> Taps<Double> {
        let scale = Double(input) / Double(output)
        let support = filter.support * (scale >= 1 ? scale : 1)
        let maximumTaps = Int(support.rounded(.up)) * 2 + 1
        let inverseScale = scale >= 1 ? 1 / scale : 1
        var starts: [Int] = []
        var weights: [[Double]] = []
        for index in 0..<output {
            let center = scale * (Double(index) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let count = min(max(min(Int(center + support + 0.5), input) - start, 0), maximumTaps)
            var row = (0..<count).map { tap -> Double in
                evaluate(filter, kernel: kernel, at: (Double(tap + start) - center + 0.5) * inverseScale)
            }
            let total = row.reduce(0, +)
            if total != 0 { row = row.map { $0 / total } }
            starts.append(start)
            weights.append(row)
        }
        return Taps(starts: starts, weights: weights)
    }

    /// The triangle filter's taps computed with float32 scale and weights, as ATen does for float tensors.
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
        toHeight: Int,
        filter: Filter = .triangle,
        kernel: Kernel = .aten
    ) -> [UInt8] {
        func alongWidth(_ input: [UInt8], lines: Int) -> [UInt8] {
            resampleUInt8(input, lines: lines, inputLength: width, outputLength: toWidth,
                          lineStride: width * channels, step: channels, channels: channels,
                          outputLineStride: toWidth * channels, outputStep: channels, filter: filter, kernel: kernel)
        }
        func alongHeight(_ input: [UInt8], columns: Int) -> [UInt8] {
            resampleUInt8(input, lines: columns, inputLength: height, outputLength: toHeight,
                          lineStride: channels, step: columns * channels, channels: channels,
                          outputLineStride: channels, outputStep: columns * channels, filter: filter, kernel: kernel)
        }
        let widthChanges = toWidth != width
        let heightChanges = toHeight != height
        let heightFirst = kernel == .pillow && height - toHeight > 0 && height - toHeight > (width - toWidth) * 2
        if heightFirst {
            let shorter = heightChanges ? alongHeight(pixels, columns: width) : pixels
            return widthChanges ? alongWidth(shorter, lines: toHeight) : shorter
        }
        let narrower = widthChanges ? alongWidth(pixels, lines: height) : pixels
        return heightChanges ? alongHeight(narrower, columns: toWidth) : narrower
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
        outputStep: Int,
        filter: Filter,
        kernel: Kernel
    ) -> [UInt8] {
        let taps = taps(input: inputLength, output: outputLength, filter: filter, kernel: kernel)
        var precision = 22
        if kernel == .aten {
            // ATen fits the largest weight in int16.
            let maximumWeight = taps.weights.joined().max() ?? 0
            precision = 0
            while precision < 22, Int(0.5 + maximumWeight * Double(1 << (precision + 1))) < (1 << 15) {
                precision += 1
            }
        }
        // Flat fixed-point weights, `width` taps per output index.
        let width = taps.weights.map(\.count).max() ?? 0
        var integerWeights = [Int32](repeating: 0, count: outputLength * width)
        for (index, row) in taps.weights.enumerated() {
            for (tap, weight) in row.enumerated() {
                let value = weight * Double(1 << precision)
                integerWeights[index * width + tap] = Int32(value < 0 ? Int(-0.5 + value) : Int(0.5 + value))
            }
        }
        let counts = taps.weights.map(\.count)
        let bias = Int32(1) << Int32(precision - 1)
        let shift = Int32(precision)
        var output = [UInt8](repeating: 0, count: lines * outputLength * channels)
        input.withUnsafeBufferPointer { source in
            integerWeights.withUnsafeBufferPointer { weights in
                output.withUnsafeMutableBufferPointer { destination in
                    for line in 0..<lines {
                        let lineStart = line * lineStride
                        let outputLineStart = line * outputLineStride
                        for index in 0..<outputLength {
                            let first = lineStart + taps.starts[index] * step
                            let row = index * width
                            let count = counts[index]
                            for channel in 0..<channels {
                                var sum = bias
                                var position = first + channel
                                for tap in 0..<count {
                                    sum &+= Int32(source[position]) &* weights[row + tap]
                                    position += step
                                }
                                destination[outputLineStart + index * outputStep + channel] =
                                    UInt8(truncatingIfNeeded: min(max(sum >> shift, 0), 255))
                            }
                        }
                    }
                }
            }
        }
        return output
    }
}
