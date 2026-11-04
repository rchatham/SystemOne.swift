import Foundation
import MLX
import SystemOne

/// One question's forward pass for marker-based encoder models: token ids and
/// the positions whose states score the options.
public struct EncoderRow: PlannedRow, Equatable, Sendable {
    public let ids: [Int]
    public let markers: [Int]
    /// Question type for heads that embed it: choice 0, score 1, noul 2.
    public let questionType: Int

    public init(ids: [Int], markers: [Int], questionType: Int = 0) {
        self.ids = ids
        self.markers = markers
        self.questionType = questionType
    }

    public var sequenceLength: Int { ids.count }
    public var logitCount: Int { markers.count }
}

/// Text conventions the encoder references share.
public enum EncoderText {
    /// Question type ids: choice 0, score 1, noul 2.
    public static func typeID(_ question: Question) -> Int {
        switch question {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }

    public static func typeName(_ question: Question) -> String {
        switch question {
        case .choice: return "choice"
        case .score: return "score"
        case .noul: return "noul"
        }
    }

    /// `json.dumps(state, ensure_ascii=False)` for fields, `", "` and `": "`
    /// separators; a text state passes through; no state is empty text.
    public static func pythonJSON(_ state: SystemOneState) -> String {
        switch state {
        case .media:
            return ""
        case .text(let text):
            return text
        case .fields(let fields):
            let body = fields.map { "\(PythonSemantics.jsonString($0.key)): \(PythonSemantics.jsonString($0.value))" }
            return "{" + body.joined(separator: ", ") + "}"
        }
    }

    /// The same with compact `","` and `":"` separators.
    public static func compactJSON(_ state: SystemOneState) -> String {
        switch state {
        case .media:
            return ""
        case .text(let text):
            return text
        case .fields(let fields):
            let body = fields.map { "\(PythonSemantics.jsonString($0.key)):\(PythonSemantics.jsonString($0.value))" }
            return "{" + body.joined(separator: ",") + "}"
        }
    }

    /// Python's `%s` of a score level: the text itself, or the dict
    /// `{'what': …, 'examples': […]}` as `str()` prints it.
    public static func pythonString(_ level: ScoreLevel) -> String {
        switch level {
        case .text(let text):
            return text
        case .described(let what, let examples):
            return "{'what': \(pythonRepr(what)), 'examples': [\(examples.map(pythonRepr).joined(separator: ", "))]}"
        }
    }

    /// CPython `repr(str)` for printable text: single quotes unless the text
    /// holds a single quote and no double quote.
    static func pythonRepr(_ text: String) -> String {
        let quote: Character = text.contains("'") && !text.contains("\"") ? "\"" : "'"
        var out = String(quote)
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where Character(scalar) == quote: out += "\\\(quote)"
            case _ where scalar.value < 0x20 || scalar.value == 0x7F:
                out += String(format: "\\x%02x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + String(quote)
    }
}

/// Answer assembly shared by the encoder backends.
public enum EncoderAnswers {
    /// Softmax in double precision after dividing by `temperature`.
    public static func softmax(_ logits: [Float], temperature: Double = 1) -> [Double] {
        let scaled = logits.map { Double($0) / temperature }
        let top = scaled.max() ?? 0
        let weights = scaled.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        return weights.map { $0 / total }
    }

    /// The first index of the largest probability.
    public static func argmax(_ probabilities: [Double]) -> Int {
        probabilities.indices.max { probabilities[$0] < probabilities[$1] || (probabilities[$0] == probabilities[$1] && $0 > $1) } ?? 0
    }

    /// `1 - H(p) / log k`, Jev's confidence (1 for fewer than two options).
    public static func entropyConfidence(_ probabilities: [Double]) -> Double {
        guard probabilities.count >= 2 else { return 1 }
        let entropy = -probabilities.reduce(0) { $0 + $1 * log(min(max($1, 1e-12), 1)) }
        return 1 - entropy / log(Double(probabilities.count))
    }

    public static func choice(keys: [String], probabilities: [Double], confidence: Double) -> Answer {
        .choice(ChoiceAnswer(
            choice: keys[argmax(probabilities)],
            probabilities: zip(keys, probabilities).map { ProbabilityEntry(key: $0, probability: $1) },
            confidence: confidence
        ))
    }

    public static func score(levels: [String], probabilities: [Double], confidence: Double) -> Answer {
        let keys = levels.indices.map(String.init)
        return .score(ScoreAnswer(
            score: probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element },
            confidence: confidence,
            legend: zip(keys, levels).map { LegendEntry(key: $0, description: $1) },
            probabilities: zip(keys, probabilities).map { ProbabilityEntry(key: $0, probability: $1) }
        ))
    }
}

/// Gathers the states at each row's markers from `[rows, width, d]` hidden states.
func gatherMarkers(_ hidden: MLXArray, rows: [EncoderRow]) -> MLXArray {
    let width = hidden.dim(1)
    let indices = rows.enumerated().flatMap { index, row in row.markers.map { Int32(index * width + $0) } }
    return hidden.reshaped(-1, hidden.dim(2)).take(MLXArray(indices), axis: 0)
}

/// Splits row-major marker scores into per-row logits.
func splitLogits(_ scores: MLXArray, rows: [EncoderRow]) -> [[Float]] {
    let values = scores.asType(.float32).asArray(Float.self)
    var offset = 0
    return rows.map { row in
        defer { offset += row.markers.count }
        return Array(values[offset..<(offset + row.markers.count)])
    }
}

/// Right-pads token rows with `pad`.
func paddedIDs(_ rows: [[Int]], pad: Int) -> [[Int]] {
    let width = rows.map(\.count).max() ?? 0
    return rows.map { $0 + Array(repeating: pad, count: width - $0.count) }
}
