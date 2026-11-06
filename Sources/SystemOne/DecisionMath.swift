import Foundation

/// Model-independent arithmetic that turns a question's calibrated logits into
/// an `Answer`, shared by backends whose answers are a plain softmax.
///
/// Confidence is the winning option's probability. Ties go to the lowest index.
public enum DecisionMath {
    /// float32 `softmax(z / T)`, subtracting the maximum first.
    public static func softmax(_ logits: [Float], temperature: Float = 1) -> [Float] {
        let scaled = logits.map { $0 / temperature }
        let maximum = scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    /// The index of the largest value, preferring the lowest index on ties.
    public static func argmax(_ values: [Double]) -> Int {
        precondition(!values.isEmpty, "argmax needs at least one value")
        var best = 0
        for index in values.indices.dropFirst() where values[index] > values[best] {
            best = index
        }
        return best
    }

    /// `Σ i · p[i]`: the expected level of an ordered scale.
    public static func expectedIndex(_ probabilities: [Double]) -> Double {
        probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
    }

    /// A noul from its `[P(false), P(true)]` distribution.
    public static func noulAnswer(probabilities: [Float]) -> NoulAnswer {
        precondition(probabilities.count == 2, "a noul distribution is [false, true]")
        return NoulAnswer(noul: Double(probabilities[1]))
    }

    /// A choice whose probabilities are aligned with `keys`.
    public static func choiceAnswer(keys: [String], probabilities: [Float]) -> ChoiceAnswer {
        precondition(keys.count == probabilities.count, "one probability per option")
        let p = probabilities.map(Double.init)
        let best = argmax(p)
        return ChoiceAnswer(
            choice: keys[best],
            probabilities: zip(keys, p).map { ProbabilityEntry(key: $0, probability: $1) },
            confidence: p[best]
        )
    }

    /// A score over levels `0..<n`, keyed `"0"`, `"1"`, …, with one legend
    /// description per level. The score is the expected level.
    public static func scoreAnswer(legend: [String], probabilities: [Float]) -> ScoreAnswer {
        precondition(legend.count == probabilities.count, "one probability per level")
        let p = probabilities.map(Double.init)
        let keys = p.indices.map(String.init)
        return ScoreAnswer(
            score: expectedIndex(p),
            confidence: p[argmax(p)],
            legend: zip(keys, legend).map { LegendEntry(key: $0, description: $1) },
            probabilities: zip(keys, p).map { ProbabilityEntry(key: $0, probability: $1) }
        )
    }
}
