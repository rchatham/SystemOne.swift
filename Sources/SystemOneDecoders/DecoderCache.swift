import Foundation
import MLX

/// A cached prefix for a decoder trunk: per layer, attention keys and values,
/// or a convolution window and recurrent state. A trunk's `prefill(_:cache:)`
/// builds it from one row; reading it never changes it, so one prefix can
/// serve many rows. A cache belongs to the trunk that built it.
public final class DecoderCache: @unchecked Sendable {
    /// Tokens the cache holds.
    public internal(set) var length = 0
    var keys: [Int: MLXArray] = [:]
    var values: [Int: MLXArray] = [:]
    var convolution: [Int: MLXArray] = [:]
    var recurrent: [Int: MLXArray] = [:]

    public init() {}

    /// An independent copy to extend along another branch.
    public func copy() -> DecoderCache {
        let copy = DecoderCache()
        copy.length = length
        copy.keys = keys
        copy.values = values
        copy.convolution = convolution
        copy.recurrent = recurrent
        return copy
    }
}
