import Foundation

/// Which keys each query of a right-padded batch may attend to, and the
/// position each token takes for RoPE and the sliding window.
///
/// The default is a plain bidirectional encoder: every real token sees every
/// real token, at positions `0..<length`. A tree layout gives every row an
/// explicit visibility matrix and position ids, as option-independent models
/// (von 1.2+, Bekko, tasksource-decider-nano) need: options see the shared
/// prefix and themselves, and their positions restart after the prefix.
public struct EncoderAttention: Equatable, Sendable {
    /// Real tokens per row; the rest of each row is padding.
    public let lengths: [Int]
    /// Per row, the position of each real token. `nil` means `0..<length`.
    public let positions: [[Int]]?
    /// Per row, `visible[query][key]` among real tokens. `nil` means all.
    public let visible: [[[Bool]]]?

    public init(lengths: [Int], positions: [[Int]]? = nil, visible: [[[Bool]]]? = nil) {
        self.lengths = lengths
        self.positions = positions
        self.visible = visible
    }

    /// Whether positions run `0..<length` in every row.
    public var hasDefaultPositions: Bool {
        positions?.allSatisfy { $0.enumerated().allSatisfy { $0.offset == $0.element } } ?? true
    }

    /// The position ids of a `[rows, width]` batch, padding at its own index.
    func positionIDs(width: Int) -> [Int32] {
        lengths.indices.flatMap { row -> [Int32] in
            (0..<width).map { column in
                if let positions, column < lengths[row] { return Int32(positions[row][column]) }
                return Int32(column)
            }
        }
    }

    /// Whether the batch needs a query-dependent mask on full-attention layers,
    /// or only key padding.
    var needsQueryMask: Bool { visible != nil }

    /// `[rows, width, width]` visibility, row-major. A query sees a key when both
    /// are real, the layout allows it, and (with `radius`) their positions are at
    /// most `radius` apart. A padding query sees only itself, so no softmax row
    /// is empty.
    func visibility(width: Int, radius: Int?) -> [Bool] {
        var values = [Bool](repeating: false, count: lengths.count * width * width)
        for row in lengths.indices {
            let length = lengths[row]
            for query in 0..<width {
                let base = (row * width + query) * width
                guard query < length else {
                    values[base + query] = true
                    continue
                }
                for key in 0..<length {
                    var sees = visible?[row][query][key] ?? true
                    if sees, let radius {
                        let q = positions?[row][query] ?? query
                        let k = positions?[row][key] ?? key
                        sees = abs(q - k) <= radius
                    }
                    values[base + key] = sees
                }
            }
        }
        return values
    }

    /// `[rows, width]` key validity for padding-only masks.
    func keyValidity(width: Int) -> [Bool] {
        lengths.flatMap { length in (0..<width).map { $0 < length } }
    }

    /// Validates the layout against a batch width.
    func validate(width: Int) throws {
        for (row, length) in lengths.enumerated() {
            guard length > 0, length <= width else { throw EncoderError.invalidLength(row: row, length: length) }
            if let positions {
                guard positions.count == lengths.count, positions[row].count == length,
                      positions[row].allSatisfy({ $0 >= 0 })
                else { throw EncoderError.invalidPositions(row: row) }
            }
            if let visible {
                guard visible.count == lengths.count, visible[row].count == length,
                      visible[row].allSatisfy({ $0.count == length }),
                      (0..<length).allSatisfy({ visible[row][$0][$0] })
                else { throw EncoderError.invalidVisibility(row: row) }
            }
        }
    }
}

/// Errors raised before an encoder forward is built.
public enum EncoderError: Error, Equatable, Sendable {
    case emptyBatch
    case raggedBatch(row: Int)
    case invalidLength(row: Int, length: Int)
    case invalidPositions(row: Int)
    case invalidVisibility(row: Int)
    case invalidTokenID(row: Int, column: Int, value: Int)
    case sequenceTooLong(maximum: Int, actual: Int)
}

extension EncoderError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyBatch:
            return "An encoder batch needs at least one row."
        case .raggedBatch(let row):
            return "Encoder row \(row) is not as wide as the first row."
        case .invalidLength(let row, let length):
            return "Encoder row \(row) has \(length) real tokens, outside 1...width."
        case .invalidPositions(let row):
            return "Encoder row \(row) needs one nonnegative position per real token."
        case .invalidVisibility(let row):
            return "Encoder row \(row) needs a square visibility matrix over its real tokens in which each token sees itself."
        case .invalidTokenID(let row, let column, let value):
            return "Encoder token [\(row), \(column)] is \(value), outside the vocabulary."
        case .sequenceTooLong(let maximum, let actual):
            return "Encoder input has positions up to \(actual); the maximum is \(maximum)."
        }
    }
}
