import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// `testdata/jev_golden_v1.json`, written by `tools/jev_golden_capture.py full`.
struct JEVFullFixture: Decodable {
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
        let probabilities: [Double]
        let trunkProbe: TrunkProbe?
        enum CodingKeys: String, CodingKey {
            case id, kind, state, question, options, prompt, ids, probabilities
            case headLogits = "head_logits"
            case trunkProbe = "trunk_probe"
        }
    }

    let tokenizerCases: [TokenizerCase]
    let cases: [Case]
    enum CodingKeys: String, CodingKey {
        case cases
        case tokenizerCases = "tokenizer_cases"
    }

    static func load() throws -> JEVFullFixture {
        let url = JEVTinyFixture.testdata.appendingPathComponent("jev_golden_v1.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("testdata/jev_golden_v1.json is not generated yet (tools/jev_golden_capture.py full).")
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

/// Parity with transformers + peft on the pinned autotrust/JEV-9B snapshot.
/// Tokenizer and prompts need `JEV_MODEL_FOLDER`; weights also need an
/// Xcode-built bundle with `SYSTEMONE_MLX_TESTS=1` (bfloat16 holds about
/// 16 GB of weights, float32 about 36 GB).
final class JEVFullModelTests: XCTestCase {
    func testTokenizerAndPromptsMatchPython() throws {
        let folder = try Self.modelFolder()
        let fixture = try JEVFullFixture.load()
        let tokenizer = try HubDecoderTokenizer(folder: folder)
        for testCase in fixture.tokenizerCases {
            XCTAssertEqual(tokenizer.encode(testCase.text), testCase.ids, testCase.text)
        }
        for testCase in fixture.cases {
            let kind = try XCTUnwrap(JEVKind(rawValue: testCase.kind))
            let prompt = JEVPrompt.render(kind: kind, state: testCase.state, question: testCase.question, options: testCase.options)
            XCTAssertEqual(prompt, testCase.prompt, testCase.id)
            XCTAssertEqual(tokenizer.encode(prompt), testCase.ids, testCase.id)
        }
    }

    func testFinalStatesHeadAndDecisionsMatchTransformersAtFloat32() throws {
        let fixture = try JEVFullFixture.load()
        try Device.withDefaultDevice(Self.device) {
            let backend = try Self.backend(.float32)
            var worst = 0.0
            for testCase in fixture.cases {
                let kind = try XCTUnwrap(JEVKind(rawValue: testCase.kind))
                if let probe = testCase.trunkProbe {
                    let states = try backend.trunkStates(testCase.ids)
                        .take(MLXArray(probe.rows.map(Int32.init)), axis: 0)
                        .take(MLXArray(probe.dimensions.map(Int32.init)), axis: 1)
                    Self.assertClose(states.asArray(Float.self), probe.entries["norm"]!.flatMap { $0 }, 1e-3, "\(testCase.id) final norm")
                }
                let logits = try backend.headLogits([testCase.ids])
                Self.assertClose(logits, testCase.headLogits, 2e-3, "\(testCase.id) head logits")
                let p = Self.probabilities(logits, kind: kind, count: testCase.options.count, backend: backend)
                for (actual, expected) in zip(p, testCase.probabilities) {
                    XCTAssertEqual(actual, expected, accuracy: 2e-3, testCase.id)
                    worst = max(worst, abs(actual - expected))
                }
            }
            print("JEV-9B float32 worst probability error: \(worst)")
        }
    }

    func testBFloat16DecisionsStayCloseAndKeepClearAnswers() throws {
        let fixture = try JEVFullFixture.load()
        try Device.withDefaultDevice(Self.device) {
            let backend = try Self.backend(.bfloat16)
            var worst = 0.0
            for testCase in fixture.cases {
                let kind = try XCTUnwrap(JEVKind(rawValue: testCase.kind))
                let p = Self.probabilities(try backend.headLogits([testCase.ids]), kind: kind, count: testCase.options.count, backend: backend)
                let errors = zip(p, testCase.probabilities).map { abs($0 - $1) }
                worst = max(worst, errors.max()!)
                XCTAssertLessThanOrEqual(errors.max()!, 2e-2, testCase.id)
                // The top answer must survive bfloat16 unless the reference is a near tie.
                let expected = testCase.probabilities
                let top = expected.indices.max { expected[$0] < expected[$1] }!
                let runnerUp = expected.enumerated().filter { $0.offset != top }.map(\.element).max() ?? 0
                if expected[top] - runnerUp > 0.05 {
                    XCTAssertEqual(p.indices.max { p[$0] < p[$1] }, top, testCase.id)
                }
            }
            print("JEV-9B bfloat16 worst probability error: \(worst)")
        }
    }

    func testEngineAnswersTheModelCardRefundExample() async throws {
        let folder = try Self.modelFolder()
        try Self.requireMLX()
        let engine = try Device.withDefaultDevice(Self.device) { try SystemOneEngine.jev(modelFolder: folder) }
        let response = try await engine.systemOne(
            state: .text("Customer says the parcel arrived damaged and wants their money back."),
            questions: [QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Is the customer asking for a refund?")))]
        )
        guard case .noul(let refund)? = response["refund"] else { return XCTFail("refund") }
        // The model card reports {'false': 0.009, 'true': 0.991}.
        XCTAssertEqual(refund.noul, 0.991, accuracy: 0.02)
    }

    // MARK: Helpers

    static var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    static func modelFolder() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["JEV_MODEL_FOLDER"], !path.isEmpty else {
            throw XCTSkip("Set JEV_MODEL_FOLDER to an autotrust/JEV-9B snapshot to run JEV-9B parity.")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireMLX() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    static func backend(_ dataType: DType) throws -> JEVBackend {
        let folder = try modelFolder()
        try requireMLX()
        return try JEVBackend(modelFolder: folder, dataType: dataType)
    }

    static func probabilities(_ logits: [Float], kind: JEVKind, count: Int, backend: JEVBackend) -> [Double] {
        let start = backend.configuration.slots[kind]!.lowerBound
        return DecoderAnswers.probabilities(Array(logits[start ..< start + count]), temperature: backend.configuration.temperatures[kind]!)
    }

    static func assertClose(
        _ actual: [Float], _ expected: [Float], _ tolerance: Float, _ message: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
        let scale = max(1, expected.map(abs).max() ?? 0)
        let worst = zip(actual, expected).map { $0.isNaN ? Float.infinity : abs($0 - $1) }.max() ?? 0
        XCTAssertLessThanOrEqual(worst / scale, tolerance, "\(message): worst error \(worst) (scale \(scale))", file: file, line: line)
    }
}
