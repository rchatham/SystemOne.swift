import Foundation
import MLX
import MLXNN
import SystemOne

/// `audio.py`: a log-mel front end, a FastConformer, an MLP adapter, and a
/// residual correction. One prefix embedding per 80 ms of 16 kHz speech.
final class D1Audio {
    static let sampleRate = 16_000
    static let minimumSamples = 8_000
    static let maximumSeconds = 30
    static let hop = 160

    init(weights: D1Weights) {}

    /// Samples after the 30 s cut and the 0.5 s minimum padding.
    static func clampedSampleCount(_ count: Int) -> Int {
        max(min(count, maximumSeconds * sampleRate), minimumSamples)
    }

    /// Prefix embeddings after three stride-2 subsampling steps over 10 ms frames.
    static func prefixLength(sampleCount: Int) -> Int {
        var length = clampedSampleCount(sampleCount) / hop
        for _ in 0..<3 { length = (length - 1) / 2 + 1 }
        return length
    }

    /// `[P, D]` prefix embeddings for one clip.
    func callAsFunction(_ clip: AudioClip) throws -> MLXArray {
        throw SystemOneEngineError.unsupportedMedia(model: "d1 audio")
    }
}
