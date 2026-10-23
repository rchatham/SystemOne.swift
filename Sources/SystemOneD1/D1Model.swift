// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation
import MLX
import MLXNN
import SystemOne

/// A request's images or audio, encoded once and shared by all of its rows.
final class D1MediaPrefix {
    let media: SystemOneMedia

    /// Prefix positions, known from the media's shape before any encoder runs.
    let length: Int

    private var cached: MLXArray?

    init(media: SystemOneMedia, length: Int) {
        self.media = media
        self.length = length
    }

    /// `[length, D]` embeddings, encoded on first use.
    func embeddings(using model: D1Model) throws -> MLXArray {
        if let cached { return cached }
        let encoded = try model.encode(media)
        precondition(encoded.dim(0) == length, "media encoder produced an unexpected prefix length")
        cached = encoded
        return encoded
    }
}

/// One question over one state: the forward pass the engine schedules.
public struct D1Row: PlannedRow {
    let ids: [Int]
    let markers: [Int]
    let questionType: D1DecisionHead.QuestionType
    let prefix: D1MediaPrefix?

    public var sequenceLength: Int { (prefix?.length ?? 0) + ids.count }
    public var logitCount: Int { markers.count }
}

/// The trunk, decision head, and media encoders of one d1-omni checkpoint.
final class D1Model {
    let configuration: D1Configuration
    let trunk: D1Trunk
    let head: D1DecisionHead
    let vision: D1Vision
    let audio: D1Audio

    init(weights: D1Weights) {
        configuration = weights.configuration
        trunk = D1Trunk(weights: weights)
        head = D1DecisionHead(weights: weights)
        vision = D1Vision(weights: weights)
        audio = D1Audio(weights: weights)
    }

    /// Prefix length a media input will occupy, computed from its shape alone.
    static func prefixLength(of media: SystemOneMedia) -> Int {
        switch media {
        case .images(let images):
            return images.reduce(0) { $0 + D1Vision.prefixLength(width: $1.width, height: $1.height) }
        case .audio(let clip):
            return D1Audio.prefixLength(sampleCount: clip.samples.count)
        }
    }

    /// `[P, D]` prefix embeddings for a media input.
    func encode(_ media: SystemOneMedia) throws -> MLXArray {
        switch media {
        case .images(let images):
            return concatenated(images.map { vision($0) }, axis: 0)
        case .audio(let clip):
            return try audio(clip)
        }
    }

    /// Raw head logits for right-padded rows, before any temperature.
    func logits(for rows: [D1Row]) throws -> MLXArray {
        let width = configuration.text.hiddenSize
        let totals = rows.map(\.sequenceLength)
        let longest = totals.max() ?? 0
        let sequences = try rows.map { row -> MLXArray in
            var text = trunk.embedding(MLXArray(row.ids.map(Int32.init)))
            if let prefix = row.prefix {
                text = concatenated([try prefix.embeddings(using: self), text], axis: 0)
            }
            return padded(text, widths: [[0, longest - text.dim(0)], 0])
        }
        let hidden = trunk(
            stacked(sequences),
            lengths: totals,
            prefixLengths: rows.map { $0.prefix?.length ?? 0 }
        ).output

        let textLengths = rows.map(\.ids.count)
        let longestText = textLengths.max() ?? 0
        let text = stacked(rows.indices.map { index -> MLXArray in
            let start = rows[index].prefix?.length ?? 0
            let slice = hidden[index, start..<(start + textLengths[index])]
            return padded(slice, widths: [[0, longestText - textLengths[index]], 0])
        })
        precondition(text.dim(2) == width)
        return head(
            text,
            lengths: textLengths,
            questionTypes: rows.map(\.questionType),
            markers: rows.map(\.markers)
        )
    }
}
