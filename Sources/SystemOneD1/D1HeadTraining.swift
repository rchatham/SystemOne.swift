// Swift port of LiquidAI/d1-omni-600M reference code, translated and modified
// from the original Python. Licensed under the LFM Open License v1.0; see
// LICENSES/LFM-Open-License-1.0.txt.

import Foundation
import MLX

/// Head-only fine-tuning over a fixed batch of labelled rows, the first step
/// toward on-device training (docs/FEDERATED.md).
///
/// The trunk runs once, outside any gradient, and its text output is reused
/// by every step. Each decision-head weight is an argument of the loss, so
/// `valueAndGrad` differentiates the head alone.
struct D1HeadTraining {
    /// `DecisionHead`'s fill for option slots past a row's count.
    static let maskedLogit: Float = -1e4

    let model: D1Model
    let rows: [D1Row]
    /// The correct option index of each row.
    let targets: [Int]
    /// Frozen `[B, T, D]` trunk output.
    let text: MLXArray
    /// Head weight names, in the order of the gradient arrays.
    let names: [String]

    init(model: D1Model, rows: [D1Row], targets: [Int]) throws {
        precondition(rows.count == targets.count, "one target per row")
        precondition(zip(rows, targets).allSatisfy { $1 >= 0 && $1 < $0.markers.count }, "target out of range")
        self.model = model
        self.rows = rows
        self.targets = targets
        text = stopGradient(try model.trunkText(for: rows))
        eval(text)
        names = model.head.parameters.keys.sorted()
    }

    /// Mean cross-entropy over rows of the raw logits, before any calibration
    /// temperature, with padded option slots masked as `DecisionHead` masks them.
    static func crossEntropy(logits: MLXArray, targets: [Int], optionCounts: [Int]) -> MLXArray {
        let logits = logits.asType(.float32)
        let slots = MLXArray(0..<Int32(logits.dim(1))).reshaped(1, -1)
        let real = slots .< MLXArray(optionCounts.map(Int32.init)).reshaped(-1, 1)
        let masked = MLX.where(real, logits, MLXArray(maskedLogit))
        let picked = takeAlong(masked, MLXArray(targets.map(Int32.init)).reshaped(-1, 1), axis: 1).squeezed(axis: 1)
        return (logSumExp(masked, axis: 1) - picked).mean()
    }

    /// The loss with the given head weights in place of the loaded ones.
    func loss(_ parameters: [String: MLXArray]) -> MLXArray {
        Self.crossEntropy(
            logits: model.headLogits(for: rows, text: text, headParameters: parameters),
            targets: targets,
            optionCounts: rows.map(\.markers.count)
        )
    }

    /// The loss and its gradient with respect to every head weight.
    func lossAndGradients(_ parameters: [String: MLXArray]) -> (loss: MLXArray, gradients: [String: MLXArray]) {
        let names = names
        let step = valueAndGrad({ arrays in
            [loss(Dictionary(uniqueKeysWithValues: zip(names, arrays)))]
        }, argumentNumbers: names.indices)
        let (values, gradients) = step(names.map { parameters[$0]! })
        return (values[0], Dictionary(uniqueKeysWithValues: zip(names, gradients)))
    }

    /// One plain gradient-descent step: `w - learningRate * g` for every weight.
    static func descend(
        _ parameters: [String: MLXArray],
        along gradients: [String: MLXArray],
        learningRate: Float
    ) -> [String: MLXArray] {
        parameters.merging(gradients) { weight, gradient in
            weight - (learningRate * gradient).asType(weight.dtype)
        }
    }
}
