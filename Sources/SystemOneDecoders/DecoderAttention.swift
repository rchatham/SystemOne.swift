import Foundation
import MLX

/// How the tokens of a right-padded decoder batch see each other, and the
/// positions they take for RoPE.
///
/// A plan governs full-attention layers only. Recurrent layers (Gated DeltaNet)
/// are causal by construction, so a "bidirectional" decoder is causal in its
/// recurrent layers and bidirectional in its attention layers, as d3,
/// Perplexity Decider, and StartLux run.
///
/// A batch may follow a cached prefix. Every token sees the whole prefix, and
/// default positions continue after it.
public struct DecoderAttention: Equatable, Sendable {
    public enum Plan: Equatable, Sendable {
        /// Each token sees the tokens before it and itself.
        case causal
        /// Each token sees every real token of its row.
        case bidirectional
        /// Per row, `visible[query][key]` among the row's real tokens.
        case visible([[[Bool]]])
    }

    /// Real tokens per row; the rest of each row is padding.
    public let lengths: [Int]
    /// Per row, the absolute position of each real token. `nil` means the row's
    /// tokens follow the prefix: `prefix ..< prefix + length`.
    public let positions: [[Int]]?
    public let plan: Plan

    public init(lengths: [Int], positions: [[Int]]? = nil, plan: Plan = .causal) {
        self.lengths = lengths
        self.positions = positions
        self.plan = plan
    }

    /// Whether positions continue the prefix in every row.
    func hasDefaultPositions(prefix: Int) -> Bool {
        positions?.allSatisfy { $0.enumerated().allSatisfy { prefix + $0.offset == $0.element } } ?? true
    }

    /// The `[rows, width]` position ids, padding at its default position.
    func positionIDs(width: Int, prefix: Int) -> [Int32] {
        lengths.indices.flatMap { row -> [Int32] in
            (0..<width).map { column in
                if let positions, column < lengths[row] { return Int32(positions[row][column]) }
                return Int32(prefix + column)
            }
        }
    }

    /// Whether MLX's causal mask mode gives the same mask: a causal plan over
    /// rows without padding.
    func isPlainCausal(width: Int) -> Bool {
        plan == .causal && lengths.allSatisfy { $0 == width }
    }

    /// `[rows, 1, width, prefix + width]` visibility. Real queries see the whole
    /// prefix and the real keys their plan allows; a padding query sees only
    /// itself, so no softmax row is empty.
    func mask(width: Int, prefix: Int) -> MLXArray {
        MLXArray(maskValues(width: width, prefix: prefix), [lengths.count, 1, width, prefix + width])
    }

    /// ``mask(width:prefix:)``'s values, row-major.
    func maskValues(width: Int, prefix: Int) -> [Bool] {
        let keys = prefix + width
        var values = [Bool](repeating: false, count: lengths.count * width * keys)
        for (row, length) in lengths.enumerated() {
            for query in 0..<width {
                let base = (row * width + query) * keys
                guard query < length else {
                    values[base + prefix + query] = true
                    continue
                }
                for key in 0..<prefix { values[base + key] = true }
                for key in 0..<length {
                    let sees: Bool
                    switch plan {
                    case .causal: sees = key <= query
                    case .bidirectional: sees = true
                    case .visible(let visible): sees = visible[row][query][key]
                    }
                    values[base + prefix + key] = sees
                }
            }
        }
        return values
    }

    func validate(width: Int) throws {
        for (row, length) in lengths.enumerated() {
            guard length > 0, length <= width else { throw DecoderError.invalidLength(row: row, length: length) }
            if let positions {
                guard positions.count == lengths.count, positions[row].count == length,
                      positions[row].allSatisfy({ $0 >= 0 })
                else { throw DecoderError.invalidPositions(row: row) }
            }
            if case .visible(let visible) = plan {
                guard visible.count == lengths.count, visible[row].count == length,
                      visible[row].allSatisfy({ $0.count == length }),
                      (0..<length).allSatisfy({ visible[row][$0][$0] })
                else { throw DecoderError.invalidVisibility(row: row) }
            }
        }
    }
}

/// Errors raised before a decoder forward is built.
public enum DecoderError: Error, Equatable, Sendable {
    case emptyBatch
    case raggedBatch(row: Int)
    case invalidLength(row: Int, length: Int)
    case invalidPositions(row: Int)
    case invalidVisibility(row: Int)
    case invalidTokenID(row: Int, column: Int, value: Int)
    case sequenceTooLong(maximum: Int, actual: Int)
    case prefixBatch(Int)
}

extension DecoderError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyBatch:
            return "A decoder batch needs at least one row."
        case .raggedBatch(let row):
            return "Decoder row \(row) is not as wide as the first row."
        case .invalidLength(let row, let length):
            return "Decoder row \(row) has \(length) real tokens, outside 1...width."
        case .invalidPositions(let row):
            return "Decoder row \(row) needs one nonnegative position per real token."
        case .invalidVisibility(let row):
            return "Decoder row \(row) needs a square visibility matrix over its real tokens in which each token sees itself."
        case .invalidTokenID(let row, let column, let value):
            return "Decoder token [\(row), \(column)] is \(value), outside the vocabulary."
        case .sequenceTooLong(let maximum, let actual):
            return "Decoder input has positions up to \(actual); the maximum is \(maximum)."
        case .prefixBatch(let rows):
            return "A cached prefix holds one row; it was built from \(rows)."
        }
    }
}
