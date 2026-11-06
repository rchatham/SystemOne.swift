import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneJev

/// `testdata/jev_golden_v1.json`, written by `tools/jev_golden_capture.py full`.
struct JevFullFixture: Decodable {
    struct TokenizerCase: Decodable { let text: String; let ids: [Int] }
    struct TrunkProbe: Decodable {
        let rows: [Int]
        let dimensions: [Int]
        let entries: [String: [[Float]]]
    }
    struct Case: Decodable {
        let id: String
        let kind: String
        let state: String
        let question: String
        let options: [String]
        let prompt: String
        let ids: [Int]
        let headLogits: [Float]
        let probabilities: [Float]
        let trunkProbe: TrunkProbe?
        enum CodingKeys: String, CodingKey {
            case id, kind, state, question, options, prompt, ids, probabilities
            case headLogits = "head_logits"
            case trunkProbe = "trunk_probe"
        }
    }

    let revision: String
    let temperatures: [String: Float]
    let tokenizerCases: [TokenizerCase]
    let cases: [Case]

    enum CodingKeys: String, CodingKey {
        case revision, temperatures, cases
        case tokenizerCases = "tokenizer_cases"
    }

    static func load() throws -> JevFullFixture {
        let url = JevTinyFixture.repositoryRoot.appendingPathComponent("testdata/jev_golden_v1.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("testdata/jev_golden_v1.json is not generated yet (tools/jev_golden_capture.py full).")
        }
        return try JSONDecoder().decode(JevFullFixture.self, from: Data(contentsOf: url))
    }
}

/// Parity with transformers + peft on the pinned autotrust/JEV-9B snapshot.
/// Tokenizer and prompt checks need `JEV_MODEL_FOLDER`; weight checks also need
/// `JEV_MODEL_PARITY=1` and `JEV_DEVICE=gpu` (about 16 GB of unified memory).
final class JevFullModelTests: XCTestCase {
    func testTokenizerAndPromptsMatchPython() async throws {
        let folder = try Self.modelFolder()
        let fixture = try JevFullFixture.load()
        let tokenizer = try await JevTokenizer(modelFolder: folder)
        for testCase in fixture.tokenizerCases {
            XCTAssertEqual(tokenizer.encode(testCase.text), testCase.ids, testCase.text)
        }
        for testCase in fixture.cases {
            let kind = try XCTUnwrap(JevKind(rawValue: testCase.kind))
            let prompt = JevPrompt.render(kind: kind, state: testCase.state, question: testCase.question,
                                          options: testCase.options)
            XCTAssertEqual(prompt, testCase.prompt, testCase.id)
            XCTAssertEqual(tokenizer.encode(prompt), testCase.ids, testCase.id)
        }
    }

    func testTrunkHeadAndDecisionsMatchTransformersAtFloat32() throws {
        let fixture = try JevFullFixture.load()
        let weights = try Self.weights()
        try Device.withDefaultDevice(.gpu) {
            let model = JevModel(weights: weights, dataType: .float32)
            var worst: Float = 0
            for testCase in fixture.cases {
                let kind = try XCTUnwrap(JevKind(rawValue: testCase.kind))
                if let probe = testCase.trunkProbe {
                    let run = model.lastHiddenStates([testCase.ids], captureLayers: true)
                    let names = ["embed"] + (0..<32).map { "layer.\($0)" } + ["norm"]
                    for (name, states) in zip(names, run.layers) {
                        let rows = states[0].take(MLXArray(probe.rows.map(Int32.init)), axis: 0)
                            .take(MLXArray(probe.dimensions.map(Int32.init)), axis: 1)
                        assertClose(rows.asType(.float32).asArray(Float.self), probe.entries[name]!.flatMap { $0 },
                                    relative: 1e-3, "\(testCase.id) \(name)")
                    }
                }
                let logits = model.slotLogits([testCase.ids]).asArray(Float.self)
                assertClose(logits, testCase.headLogits, relative: 2e-3, "\(testCase.id) head logits")
                let p = Self.probabilities(logits, kind: kind, count: testCase.options.count, weights: weights)
                for (actual, expected) in zip(p, testCase.probabilities) {
                    XCTAssertEqual(actual, expected, accuracy: 2e-3, testCase.id)
                    worst = max(worst, abs(actual - expected))
                }
            }
            print("JEV-9B float32 worst probability error: \(worst)")
        }
    }

    func testBFloat16DecisionsStayCloseAndKeepTheTopAnswer() throws {
        let fixture = try JevFullFixture.load()
        let weights = try Self.weights()
        try Device.withDefaultDevice(.gpu) {
            let model = JevModel(weights: weights, dataType: .bfloat16)
            var worst: Float = 0
            for testCase in fixture.cases {
                let kind = try XCTUnwrap(JevKind(rawValue: testCase.kind))
                let logits = model.slotLogits([testCase.ids]).asArray(Float.self)
                let p = Self.probabilities(logits, kind: kind, count: testCase.options.count, weights: weights)
                let errors = zip(p, testCase.probabilities).map { abs($0 - $1) }
                worst = max(worst, errors.max()!)
                XCTAssertLessThanOrEqual(errors.max()!, 2e-2, testCase.id)
                // The top answer must survive bf16 unless the reference itself is a near tie.
                let expected = testCase.probabilities.map(Double.init)
                let top = DecisionMath.argmax(expected)
                let runnerUp = expected.enumerated().filter { $0.offset != top }.map(\.element).max() ?? 0
                if expected[top] - runnerUp > 0.05 {
                    XCTAssertEqual(DecisionMath.argmax(p.map(Double.init)), top, testCase.id)
                }
            }
            print("JEV-9B bfloat16 worst probability error: \(worst)")
        }
    }

    func testEngineAnswersTheModelCardExamples() async throws {
        let folder = try Self.modelFolder()
        _ = try Self.weights()  // skips unless JEV_MODEL_PARITY=1 and JEV_DEVICE=gpu
        Self.shared = nil
        let engine = try await SystemOneEngine.jev(modelFolder: folder)
        let response = try await engine.systemOne(
            state: .text("Customer says the parcel arrived damaged and wants their money back."),
            questions: [QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Is the customer asking for a refund?")))]
        )
        guard case .noul(let refund)? = response["refund"] else { return XCTFail("refund") }
        // The model card reports {'false': 0.009, 'true': 0.991}.
        XCTAssertEqual(refund.noul, 0.991, accuracy: 0.02)
        XCTAssertEqual(response.model, "JEV-9B")
    }

    // MARK: Helpers

    nonisolated(unsafe) private static var shared: JevWeights?

    static func modelFolder() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["JEV_MODEL_FOLDER"], !path.isEmpty else {
            throw XCTSkip("Set JEV_MODEL_FOLDER to an autotrust/JEV-9B snapshot to run JEV-9B parity.")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// The pinned weights, loaded once for every weight test.
    static func weights() throws -> JevWeights {
        let folder = try modelFolder()
        guard ProcessInfo.processInfo.environment["JEV_MODEL_PARITY"] == "1" else {
            throw XCTSkip("Set JEV_MODEL_PARITY=1 to load the JEV-9B weights.")
        }
        try JevTestDevice.require()
        if let shared { return shared }
        let weights = try Device.withDefaultDevice(.gpu) {
            try JevWeights(modelFolder: folder, configuration: JevConfiguration(modelFolder: folder),
                           artifacts: .jev9B, precision: .bfloat16)
        }
        shared = weights
        return weights
    }

    static func probabilities(_ logits: [Float], kind: JevKind, count: Int, weights: JevWeights) -> [Float] {
        let slots = weights.configuration.slots(for: kind)
        return DecisionMath.softmax(Array(logits[slots.lowerBound..<(slots.lowerBound + count)]),
                                    temperature: weights.configuration.temperatures[kind]!)
    }
}
