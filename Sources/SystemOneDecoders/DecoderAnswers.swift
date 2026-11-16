import Foundation
import SystemOne

/// Answers from the probabilities a decoder readout gives a question's options,
/// as d3's runtime assembles them (`to_answer`, `product_answer`) and as JEV
/// reports them: a choice takes the first maximum with Jev's confidence
/// `1 − H / log k`, a noul is `P(true)`, a score is the expected level index.
enum DecoderAnswers {
    /// Softmax of `logits / temperature` in double precision.
    static func probabilities(_ logits: [Float], temperature: Double) -> [Double] {
        let scaled = logits.map { Double($0) / temperature }
        let top = scaled.max() ?? 0
        let weights = scaled.map { Foundation.exp($0 - top) }
        let total = weights.reduce(0, +)
        return weights.map { $0 / total }
    }

    /// One minus the normalized Shannon entropy, clamped to [0, 1].
    static func confidence(_ probabilities: [Double]) -> Double {
        guard probabilities.count >= 2 else { return 1 }
        let entropy = -probabilities.reduce(0) { $1 > 0 ? $0 + $1 * Foundation.log($1) : $0 }
        return max(0, min(1, 1 - entropy / Foundation.log(Double(probabilities.count))))
    }

    static func answer(_ question: Question, keys: [String], probabilities: [Double]) -> Answer {
        switch question {
        case .noul:
            return .noul(NoulAnswer(noul: probabilities[1]))
        case .choice:
            var best = 0
            for index in probabilities.indices.dropFirst() where probabilities[index] > probabilities[best] { best = index }
            return .choice(ChoiceAnswer(
                choice: keys[best],
                probabilities: zip(keys, probabilities).map { ProbabilityEntry(key: $0, probability: $1) },
                confidence: confidence(probabilities)
            ))
        case .score(let score):
            return .score(ScoreAnswer(
                score: probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element },
                confidence: confidence(probabilities),
                legend: zip(keys, score.criteria).map { LegendEntry(key: $0, description: D3Prompt.canonical($1)) },
                probabilities: zip(keys, probabilities).map { ProbabilityEntry(key: $0, probability: $1) }
            ))
        }
    }
}
