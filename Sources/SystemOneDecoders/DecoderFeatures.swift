import Foundation
import MLX
import MLXNN

/// Embeddings that stand in for a placeholder token's: each row's placeholder
/// positions take that row's feature rows in order, as transformers'
/// `masked_scatter` of image or audio features does. A row with no
/// placeholders takes `nil`.
public struct DecoderFeatures {
    /// Per decoder row: the placeholder token id (an image or audio token).
    public let tokens: [Int]
    /// Per decoder row: `[placeholders, hidden]` features, or `nil`.
    public let rows: [MLXArray?]

    /// One placeholder token for every row.
    public init(token: Int, rows: [MLXArray?]) {
        self.init(tokens: Array(repeating: token, count: rows.count), rows: rows)
    }

    /// A placeholder token per row: a batch may mix pictures and audio clips.
    public init(tokens: [Int], rows: [MLXArray?]) {
        self.tokens = tokens
        self.rows = rows
    }
}

extension DecoderBatch {
    /// `[rows, width, hidden]` input embeddings of `ids`, with `features`
    /// gathered into their placeholder positions. Every placeholder needs a
    /// feature row and every feature row a placeholder.
    static func embed(_ ids: [[Int]], width: Int, embedding: Embedding, features: DecoderFeatures?) throws -> MLXArray {
        let rows = ids.count
        let tokens = embedding(MLXArray(ids.flatMap { $0.map(Int32.init) }, [rows, width]))
        guard let features else { return tokens }
        guard features.rows.count == rows, features.tokens.count == rows else {
            throw DecoderError.featureCount(row: min(rows, features.rows.count), placeholders: 0, features: 0)
        }
        let hidden = tokens.dim(-1)
        // Gather from [token embeddings; every row's features]: placeholders
        // point past the token table.
        var sources = (0..<(rows * width)).map(Int32.init)
        var parts = [tokens.reshaped(rows * width, hidden)]
        var next = rows * width
        for (row, tokenRow) in ids.enumerated() {
            let slots = tokenRow.indices.filter { tokenRow[$0] == features.tokens[row] }
            let count = features.rows[row].map { $0.ndim == 2 ? $0.dim(0) : -1 } ?? 0
            guard count == slots.count else {
                throw DecoderError.featureCount(row: row, placeholders: slots.count, features: count)
            }
            guard let values = features.rows[row], count > 0 else { continue }
            guard values.dim(1) == hidden else {
                throw DecoderError.featureCount(row: row, placeholders: slots.count, features: count)
            }
            for (offset, slot) in slots.enumerated() { sources[row * width + slot] = Int32(next + offset) }
            parts.append(values.asType(tokens.dtype))
            next += count
        }
        guard parts.count > 1 else { return tokens }
        return concatenated(parts, axis: 0).take(MLXArray(sources), axis: 0).reshaped(rows, width, hidden)
    }
}
