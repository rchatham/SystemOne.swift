// Swift port of `resampled` in the media.py LiquidAI ships with its d1
// checkpoints (LiquidAI/d1-Omni-2B), translated and modified from the original
// Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation

/// Band-limited resampling as torchaudio's `functional.resample` does it by
/// default, in the float32 steps of the NumPy port LiquidAI ships with its d1
/// checkpoints (`media.resampled`): a Hann-windowed sinc over `zeros` zero
/// crossings, cut at `rolloff` of the lower Nyquist rate, one filter per
/// output phase.
package enum SincResample {
    /// Samples a clip of `count` samples has at `target` Hz: `ceil(count · target / rate)`.
    package static func outputCount(_ count: Int, from rate: Int, to target: Int) -> Int {
        guard rate != target else { return count }
        let divisor = gcd(rate, target)
        let (old, new) = (rate / divisor, target / divisor)
        return (count * new + old - 1) / old
    }

    /// `samples` at `rate` Hz, resampled to `target` Hz. Both rates must be positive.
    package static func resample(
        _ samples: [Float], from rate: Int, to target: Int, zeros: Int = 6, rolloff: Double = 0.99
    ) -> [Float] {
        precondition(rate > 0 && target > 0, "sample rates must be positive")
        guard rate != target else { return samples }
        let divisor = gcd(rate, target)
        let (old, new) = (rate / divisor, target / divisor)
        let cut = Double(min(old, new)) * rolloff
        let width = Int((Double(zeros) * Double(old) / cut).rounded(.up))
        let taps = 2 * width + old

        // np.float32(math.pi): the nearest float, where Float.pi rounds toward zero.
        let pi = Float(Double.pi)
        let (cutoff, crossings, scale) = (Float(cut), Float(zeros), Float(cut / Double(old)))
        var kernel = [Float](repeating: 0, count: new * taps)
        for phase in 0..<new {
            let shift = Float(-phase) / Float(new)
            for tap in 0..<taps {
                var t = (shift + Float(tap - width) / Float(old)) * cutoff
                t = min(max(t, -crossings), crossings)
                let c = cos(t * pi / crossings / 2)
                let x = t * pi
                kernel[phase * taps + tap] = (x == 0 ? 1 : sin(x) / x) * (c * c * scale)
            }
        }

        let padded = [Float](repeating: 0, count: width) + samples + [Float](repeating: 0, count: width + old)
        let frames = samples.count / old + 1
        let count = outputCount(samples.count, from: rate, to: target)
        var output = [Float](repeating: 0, count: count)
        padded.withUnsafeBufferPointer { signal in
            kernel.withUnsafeBufferPointer { filters in
                for frame in 0..<frames {
                    let start = frame * old
                    for phase in 0..<new {
                        let index = frame * new + phase
                        guard index < count else { return }
                        let row = phase * taps
                        var sum = 0.0
                        for tap in 0..<taps { sum += Double(signal[start + tap]) * Double(filters[row + tap]) }
                        output[index] = Float(sum)
                    }
                }
            }
        }
        return output
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var (a, b) = (a, b)
        while b != 0 { (a, b) = (b, a % b) }
        return a
    }
}
