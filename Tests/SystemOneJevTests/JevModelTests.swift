import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneJev

/// Layer-by-layer parity with transformers' Qwen3.5 classes on the committed
/// tiny model (`testdata/jev_tiny_v1`). Needs only `JEV_DEVICE=gpu`.
final class JevModelTests: XCTestCase {
    func testGatedDeltaRuleMatchesTransformers() throws {
        try JevTestDevice.require()
        let vector = try JevTinyFixture.load().deltaRule
        let (t, heads, dk, dv) = (vector.shape[0], vector.shape[1], vector.shape[2], vector.shape[3])
        try Device.withDefaultDevice(.gpu) {
            let q = MLXArray(vector.q, [1, t, heads, dk])
            let k = MLXArray(vector.k, [1, t, heads, dk])
            let v = MLXArray(vector.v, [1, t, heads, dv])
            let g = MLXArray(vector.g, [1, t, heads])
            let beta = MLXArray(vector.beta, [1, t, heads])

            let chunked = JevDeltaRule.chunked(query: q, key: k, value: v, decay: g, beta: beta)
            let recurrent = JevDeltaRule.recurrent(query: q, key: k, value: v, decay: g, beta: beta)
            let worst = [
                assertClose(chunked.output.asArray(Float.self), vector.output, relative: 1e-5, "chunked output"),
                assertClose(chunked.state.asArray(Float.self), vector.finalState, relative: 1e-5, "chunked final state"),
                assertClose(recurrent.output.asArray(Float.self), vector.recurrentOutput, relative: 1e-5,
                            "recurrent output"),
                assertClose(recurrent.state.asArray(Float.self), vector.finalState, relative: 1e-5,
                            "recurrent final state"),
            ].max()!
            print("JEV delta rule worst relative error: \(worst)")
        }
    }

    func testLoRAMergesAsPeftDoes() throws {
        try JevTestDevice.require()
        let fixture = try JevTinyFixture.load()
        try Device.withDefaultDevice(.gpu) {
            let weights = try Self.weights(.float32)
            XCTAssertEqual(weights.configuration.adapter.scale, fixture.loraScale)
            for (name, probe) in fixture.mergeProbes.sorted(by: { $0.key < $1.key }) {
                let merged = weights.array(name)
                XCTAssertEqual(merged.dtype, .bfloat16, name)
                let values = merged.asType(.float32).reshaped(-1).take(MLXArray(probe.indices.map(Int32.init)), axis: 0)
                    .asArray(Float.self)
                for (actual, expected) in zip(values, probe.merged) {
                    // Accumulation order may round the float32 sum to the neighbouring bfloat16.
                    XCTAssertEqual(actual, expected, accuracy: abs(expected) * 0x1p-7 + 1e-9, name)
                }
                XCTAssertNotEqual(probe.base, probe.merged, "\(name): the adapter must change the weight")
            }
            XCTAssertEqual(weights.array("model.layers.0.linear_attn.A_log").dtype, .float32)
            XCTAssertEqual(weights.array("model.layers.0.linear_attn.norm.weight").dtype, .float32)
        }
    }

    func testEveryLayerMatchesTransformersAtFloat32() throws {
        try JevTestDevice.require()
        let fixture = try JevTinyFixture.load()
        try Device.withDefaultDevice(.gpu) {
            let model = JevModel(weights: try Self.weights(.float32))
            let configuration = try JevConfiguration(modelFolder: JevTinyFixture.modelFolder)
            let names = ["embed"] + (0..<4).map { "layer.\($0)" } + ["norm"]
            var worstLayer: Float = 0
            var worstLogit: Float = 0
            for testCase in fixture.cases {
                let label = "length \(testCase.ids.count)"
                let run = model.lastHiddenStates([testCase.ids], captureLayers: true)
                XCTAssertEqual(run.layers.count, names.count)
                for (name, states) in zip(names, run.layers) {
                    let rows = states[0].take(MLXArray(testCase.probeRows.map(Int32.init)), axis: 0)
                    worstLayer = max(worstLayer, assertClose(
                        rows.asType(.float32).asArray(Float.self), testCase.layers[name]!.flatMap { $0 },
                        relative: 1e-4, "\(label) \(name)"))
                }
                assertClose(run.last.asArray(Float.self), testCase.lastHidden, relative: 1e-4, "\(label) last hidden")

                let logits = model.slotLogits([testCase.ids]).asArray(Float.self)
                worstLogit = max(worstLogit, assertClose(logits, testCase.headLogits, relative: 1e-4,
                                                         "\(label) head logits"))
                for (kind, count, expected) in [
                    (JevKind.noul, 2, testCase.probabilities.noul),
                    (.score, 6, testCase.probabilities.score),
                    (.choice, fixture.choiceOptions, testCase.probabilities.choice),
                ] {
                    let slots = configuration.slots(for: kind)
                    let p = DecisionMath.softmax(
                        Array(logits[slots.lowerBound..<(slots.lowerBound + count)]),
                        temperature: configuration.temperatures[kind]!)
                    assertClose(p, expected, relative: 1e-4, "\(label) \(kind) probabilities")
                }
            }
            print("JEV tiny float32 worst relative error: layers \(worstLayer), head logits \(worstLogit)")
        }
    }

    func testRightPaddedBatchesMatchSingleRows() throws {
        try JevTestDevice.require()
        let fixture = try JevTinyFixture.load()
        try Device.withDefaultDevice(.gpu) {
            let model = JevModel(weights: try Self.weights(.float32))
            let rows = fixture.cases.map(\.ids)
            let batched = model.slotLogits(rows).asArray(Float.self)
            for (index, testCase) in fixture.cases.enumerated() {
                assertClose(Array(batched[(index * 24)..<((index + 1) * 24)]), testCase.headLogits,
                            relative: 1e-4, "row \(index) of the padded batch")
            }
        }
    }

    func testBFloat16StaysNearTheFloat32Reference() throws {
        try JevTestDevice.require()
        let fixture = try JevTinyFixture.load()
        try Device.withDefaultDevice(.gpu) {
            let model = JevModel(weights: try Self.weights(.bfloat16))
            XCTAssertEqual(model.dataType, .bfloat16)
            let configuration = try JevConfiguration(modelFolder: JevTinyFixture.modelFolder)
            var worst: Float = 0
            for testCase in fixture.cases {
                let logits = model.slotLogits([testCase.ids]).asArray(Float.self)
                let slots = configuration.slots(for: .noul)
                let p = DecisionMath.softmax(Array(logits[slots]), temperature: configuration.temperatures[.noul]!)
                worst = max(worst, zip(p, testCase.probabilities.noul).map { abs($0 - $1) }.max()!)
            }
            print("JEV tiny bfloat16 worst noul probability error: \(worst)")
            XCTAssertLessThan(worst, 0.05)
        }
    }

    static func weights(_ precision: JevPrecision) throws -> JevWeights {
        let folder = JevTinyFixture.modelFolder
        return try JevWeights(
            modelFolder: folder, configuration: JevConfiguration(modelFolder: folder), artifacts: nil, precision: precision)
    }
}
