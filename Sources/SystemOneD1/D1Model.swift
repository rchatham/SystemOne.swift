import Foundation
import MLX
import MLXNN
import SystemOne

/// A request's images or audio, encoded and run through the trunk once and
/// shared by all of its rows.
final class D1MediaPrefix {
    let media: SystemOneMedia

    /// Prefix positions, known from the media's shape before any encoder runs.
    let length: Int

    private var cached: MLXArray?
    private var cachedState: D1Trunk.PrefixState?

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

    /// The trunk states every row's text reads, computed and evaluated on first use.
    func state(using model: D1Model) throws -> D1Trunk.PrefixState {
        if let cachedState { return cachedState }
        let state = model.trunk.prefixState(of: try embeddings(using: model))
        eval(state.arrays)
        cachedState = state
        return state
    }
}

/// One question over one state: the forward pass the engine schedules.
public struct D1Row: PlannedRow {
    let ids: [Int]
    let markers: [Int]
    let questionType: D1DecisionHead.QuestionType
    let prefix: D1MediaPrefix?

    /// Positions the row reads, as the reference's `usage.input_tokens` counts them.
    public var sequenceLength: Int { (prefix?.length ?? 0) + ids.count }
    public var logitCount: Int { markers.count }

    /// Positions the row's own forward pass computes: the prefix's trunk
    /// states are shared by the request, so only the text runs per row.
    var computedLength: Int { ids.count }
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
    ///
    /// Rows that share a media prefix run only their text, attending to the
    /// prefix's cached trunk states. `reusingPrefixStates: false` runs the whole
    /// prefix again with every row, as the reference does.
    func logits(for rows: [D1Row], reusingPrefixStates: Bool = true) throws -> MLXArray {
        let groups = Dictionary(grouping: rows.indices) { rows[$0].prefix.map(ObjectIdentifier.init) }
        if groups.count > 1 {
            // Rows over different prefixes run apart and come back in order.
            let parts = try groups.values.map {
                ($0, try logits(for: $0.map { rows[$0] }, reusingPrefixStates: reusingPrefixStates))
            }
            let width = parts.map { $0.1.dim(1) }.max()!
            var inverse = [Int32](repeating: 0, count: rows.count)
            for (position, index) in parts.flatMap(\.0).enumerated() { inverse[index] = Int32(position) }
            let joined = concatenated(parts.map { padded($0.1, widths: [0, [0, width - $0.1.dim(1)]]) }, axis: 0)
            return joined.take(MLXArray(inverse), axis: 0)
        }

        let textLengths = rows.map(\.ids.count)
        let hidden: MLXArray
        if rows.first?.prefix != nil, !reusingPrefixStates {
            hidden = try textHiddenRunningPrefix(rows)
        } else {
            let longestText = textLengths.max() ?? 0
            let text = stacked(rows.map { row in
                padded(trunk.embedding(MLXArray(row.ids.map(Int32.init))),
                       widths: [[0, longestText - row.ids.count], 0])
            })
            if let prefix = rows.first?.prefix {
                hidden = trunk(text, lengths: textLengths, after: try prefix.state(using: self))
            } else {
                hidden = trunk(text, lengths: textLengths, prefixLengths: rows.map { _ in 0 }).output
            }
        }
        precondition(hidden.dim(2) == configuration.text.hiddenSize)
        return head(
            hidden,
            lengths: textLengths,
            questionTypes: rows.map(\.questionType),
            markers: rows.map(\.markers)
        )
    }

    /// `[B, T, D]` text states from running each row's prefix and text together.
    private func textHiddenRunningPrefix(_ rows: [D1Row]) throws -> MLXArray {
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
        return stacked(rows.indices.map { index -> MLXArray in
            let start = rows[index].prefix?.length ?? 0
            let slice = hidden[index, start..<(start + textLengths[index])]
            return padded(slice, widths: [[0, longestText - textLengths[index]], 0])
        })
    }
}
