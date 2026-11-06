import Foundation
import MLX
import MLXNN

/// Embeddings that stand in for placeholder tokens': in each row, a
/// placeholder token's positions take that row's feature rows for it in order,
/// as transformers' `masked_scatter` of image or audio features does. A row
/// may fill several placeholder tokens (a picture state and a spoken
/// question), or none.
public struct DecoderFeatures {
    /// The placeholder token ids. In every row, each one's positions must
    /// match its feature rows, none when the row has no features for it.
    public let tokens: [Int]
    /// Per decoder row: `[placeholders, hidden]` features for each placeholder token it fills.
    public let rows: [[Int: MLXArray]]

    /// One placeholder token for every row; a row without placeholders takes `nil`.
    public init(token: Int, rows: [MLXArray?]) {
        self.init(tokens: [token], rows: rows.map { $0.map { [token: $0] } ?? [:] })
    }

    /// Several placeholder tokens: a batch may mix pictures, audio clips and
    /// text, and one row may hold both pictures and speech.
    public init(tokens: [Int], rows: [[Int: MLXArray]]) {
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
        guard features.rows.count == rows else {
            throw DecoderError.featureCount(row: min(rows, features.rows.count), placeholders: 0, features: 0)
        }
        let hidden = tokens.dim(-1)
        // Gather from [token embeddings; every row's features]: placeholders
        // point past the token table.
        var sources = (0..<(rows * width)).map(Int32.init)
        var parts = [tokens.reshaped(rows * width, hidden)]
        var next = rows * width
        for (row, tokenRow) in ids.enumerated() {
            let filled = features.rows[row]
            if let stray = filled.keys.first(where: { !features.tokens.contains($0) }) {
                throw DecoderError.featureCount(row: row, placeholders: 0, features: filled[stray]?.dim(0) ?? 0)
            }
            for token in features.tokens {
                let slots = tokenRow.indices.filter { tokenRow[$0] == token }
                let count = filled[token].map { $0.ndim == 2 ? $0.dim(0) : -1 } ?? 0
                guard count == slots.count else {
                    throw DecoderError.featureCount(row: row, placeholders: slots.count, features: count)
                }
                guard let values = filled[token], count > 0 else { continue }
                guard values.dim(1) == hidden else {
                    throw DecoderError.featureCount(row: row, placeholders: slots.count, features: count)
                }
                for (offset, slot) in slots.enumerated() { sources[row * width + slot] = Int32(next + offset) }
                parts.append(values.asType(tokens.dtype))
                next += count
            }
        }
        guard parts.count > 1 else { return tokens }
        return concatenated(parts, axis: 0).take(MLXArray(sources), axis: 0).reshaped(rows, width, hidden)
    }
}
