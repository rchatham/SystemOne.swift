import Foundation
import MLX
import XCTest
@testable import SystemOneD1

/// `testdata/d1_training_golden_v1.json`, written by `tools/d1_training_capture.py`.
struct D1TrainingFixture: Decodable {
    struct Row: Decodable { let ids: [Int]; let markers: [Int]; let qtype: Int }
    struct Gradient: Decodable {
        let name: String, shape: [Int], norm: Double, sum: Double, indices: [Int], values: [Float]
    }
    struct HeadStep: Decodable {
        let id: String
        let rows: [Row]
        let targets: [Int]
        let logits: [[Float]]
        let loss: Float
        let gradients: [Gradient]
        let learningRate: Float
        let lossAfterStep: Float
        enum CodingKeys: String, CodingKey {
            case id, rows, targets, logits, loss, gradients
            case learningRate = "learning_rate", lossAfterStep = "loss_after_step"
        }
    }

    let headStep: [HeadStep]
    enum CodingKeys: String, CodingKey { case headStep = "head_step" }

    static func load() throws -> D1TrainingFixture {
        let url = D1Fixture.repositoryRoot.appendingPathComponent("testdata/d1_training_golden_v1.json")
        return try JSONDecoder().decode(D1TrainingFixture.self, from: Data(contentsOf: url))
    }
}

final class D1HeadTrainingTests: XCTestCase {
    // MARK: No weights: needs D1_DEVICE=gpu

    func testCrossEntropyMasksPaddedOptions() throws {
        guard ProcessInfo.processInfo.environment["D1_DEVICE"] == "gpu" else {
            throw XCTSkip("Set D1_DEVICE=gpu to build MLX arrays from the Xcode test bundle.")
        }
        Device.withDefaultDevice(.gpu) {
            // Row 0 has two real options; its third slot holds garbage that must not count.
            let logits = MLXArray([1, 3, 50, 0.5, -1, 2] as [Float], [2, 3])
            let loss = D1HeadTraining.crossEntropy(logits: logits, targets: [1, 2], optionCounts: [2, 3])
            func rowLoss(_ z: [Float], _ target: Int) -> Float { log(z.map { exp($0) }.reduce(0, +)) - z[target] }
            let expected = (rowLoss([1, 3], 1) + rowLoss([0.5, -1, 2], 2)) / 2
            XCTAssertEqual(loss.item(Float.self), expected, accuracy: 1e-6)
        }
    }

    // MARK: Weights: needs D1_MODEL_FOLDER and D1_MODEL_PARITY=1

    func testHeadLossGradientsAndStepMatchPyTorch() throws {
        let weights = try D1TextTests.loadWeights()
        try Device.withDefaultDevice(D1TextTests.device) { try checkHeadStep(weights) }
    }

    private func checkHeadStep(_ weights: D1Weights) throws {
        let model = try D1Model(weights: weights)
        for testCase in try D1TrainingFixture.load().headStep {
            let rows = testCase.rows.map {
                D1Row(ids: $0.ids, markers: $0.markers, questionType: .init(rawValue: $0.qtype)!, prefix: nil)
            }
            let training = try D1HeadTraining(model: model, rows: rows, targets: testCase.targets)
            let parameters = model.head.parameters
            XCTAssertEqual(Set(parameters.keys), Set(testCase.gradients.map(\.name)), testCase.id)

            let logits = model.headLogits(for: rows, text: training.text)
            for (index, expected) in testCase.logits.enumerated() {
                let actual = logits[index, 0..<expected.count].asArray(Float.self)
                for (a, e) in zip(actual, expected) { XCTAssertEqual(a, e, accuracy: 2e-3, testCase.id) }
            }

            let (loss, gradients) = training.lossAndGradients(parameters)
            XCTAssertEqual(loss.item(Float.self), testCase.loss, accuracy: 1e-4, testCase.id)

            var worstNorm = 0.0, worstValue: Float = 0
            for expected in testCase.gradients {
                let gradient = gradients[expected.name]!.asType(.float32)
                XCTAssertEqual(gradient.shape, expected.shape, expected.name)
                let norm = Double(sqrt((gradient * gradient).sum()).item(Float.self))
                // A gradient that is zero by symmetry (the scorer's output bias) compares absolutely.
                let normError = abs(norm - expected.norm) / max(expected.norm, 1e-3)
                worstNorm = max(worstNorm, normError)
                XCTAssertLessThan(normError, 1e-3, "\(expected.name) norm \(norm) vs \(expected.norm)")

                let flat = gradient.reshaped(-1)
                let sampled = flat[MLXArray(expected.indices.map(Int32.init))].asArray(Float.self)
                // The floor sits below every real head gradient (smallest max |g| ~7e-4) and above
                // the zero gradient's float32 residue (PyTorch 1.9e-9, MLX exactly 0).
                let scale = max(abs(flat).max().item(Float.self), 1e-5)
                for (a, e) in zip(sampled, expected.values) {
                    worstValue = max(worstValue, abs(a - e) / scale)
                    XCTAssertEqual(a, e, accuracy: 1e-3 * scale, expected.name)
                }
            }
            print("d1 head gradient worst norm error \(worstNorm), worst sampled error \(worstValue) of max |g|")

            let stepped = D1HeadTraining.descend(parameters, along: gradients, learningRate: testCase.learningRate)
            let lossAfter = training.loss(stepped).item(Float.self)
            XCTAssertLessThan(lossAfter, loss.item(Float.self), testCase.id)
            XCTAssertEqual(lossAfter, testCase.lossAfterStep, accuracy: 1e-3, testCase.id)
            print("d1 head step loss \(loss.item(Float.self)) -> \(lossAfter) (PyTorch \(testCase.loss) -> \(testCase.lossAfterStep))")
        }
    }
}
