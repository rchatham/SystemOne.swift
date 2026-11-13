import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// `testdata/d3_tiny/fixture.json`, written by `tools/d3_fixture.py`.
private struct D3Fixture: Decodable {
    struct Reference: Decodable {
        let type: String
        let choice: String?
        let noul: Double?
        let score: Double?
        let probabilities: [String: Double]?
        let confidence: Double?
        let legend: [String: String]?
    }

    struct Entry: Decodable {
        let state: Int
        let question: String
        let text: String
        let tokens: Int
        let answer: Reference
    }

    let answers: [Entry]
    let temperature: Double

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata")

    static func load() throws -> D3Fixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("d3_tiny/fixture.json")))
    }
}

/// The fixture's requests, as `tools/d3_fixture.py` writes them.
private enum D3Requests {
    static let states: [SystemOneState] = [
        .text("Order #4821 was charged twice — I'd like a \"refund\".\nThanks!"),
        .fields([StateField(key: "ticket", value: "Charged twice"), StateField(key: "tier", value: "gold ✨")]),
        .text(""),
    ]
    static let questions: [(id: String, question: Question)] = [
        ("team", .choice(Choice(instructions: "Which team should handle this?", criteria: [
            ChoiceOption(key: "billing", description: "Charges, refunds"),
            ChoiceOption(key: "tech_support"),
            ChoiceOption(key: "a", description: ""),
        ]))),
        ("refund", .noul(Noul(instructions: "Is the customer asking for a refund?"))),
        ("escalate", .noul(Noul(instructions: "Escalate today?", criteria: NoulCriteria(true: "", false: "Routine")))),
        ("urgency", .score(Score(instructions: "How urgent is it?", criteria: [.text("low"), .text("medium"), .text("high")]))),
        ("tone", .score(Score(instructions: "", criteria: [
            .described(what: "calm", examples: ["fine, thanks"]), .described(what: "upset", examples: []),
        ]))),
        ("pick", .choice(Choice(instructions: "", criteria: [ChoiceOption(key: "only")]))),
    ]

    static func question(_ id: String) -> Question {
        questions.first { $0.id == id }!.question
    }

    static func request(state: Int) -> SystemOneRequest {
        SystemOneRequest(state: states[state], questions: questions.map { QuestionEntry(id: $0.id, question: $0.question) })
    }
}

/// One token per UTF-8 byte, `byte % 96`, as the fixture tokenizes.
private struct ByteTokenizer: DecoderTokenizer {
    func encode(_ text: String) -> [Int] { text.utf8.map { Int($0) % 96 } }
}

final class D3Tests: XCTestCase {
    private func configuration() throws -> D3Configuration {
        try D3Configuration(modelFolder: D3Fixture.root.appendingPathComponent("d3_tiny"))
    }

    func testPromptsMatchTheReference() throws {
        let fixture = try D3Fixture.load()
        let codes = try configuration().codes
        XCTAssertEqual(fixture.answers.count, 18)
        for entry in fixture.answers {
            let prompt = D3Prompt.prompt(state: D3Requests.states[entry.state], question: D3Requests.question(entry.question), codes: codes)
            XCTAssertEqual(prompt, entry.text, "state \(entry.state) \(entry.question)")
        }
    }

    func testPlanTokenizesOneRowPerQuestion() throws {
        let fixture = try D3Fixture.load()
        let backend = try D3Backend(modelID: "d3-tiny", configuration: configuration(), tokenizer: ByteTokenizer(), trunk: nil, readout: nil)
        for state in 0..<3 {
            let plan = try backend.plan(D3Requests.request(state: state), media: nil)
            let expected = fixture.answers.filter { $0.state == state }
            XCTAssertEqual(plan.rows.map(\.ids.count), expected.map(\.tokens))
            XCTAssertEqual(plan.rows.map(\.optionCount), [3, 2, 2, 3, 2, 1])
            XCTAssertEqual(plan.usage.inputTokens, expected.map(\.tokens).reduce(0, +))
        }
    }

    func testConfigurationAndValidation() throws {
        let configuration = try configuration()
        XCTAssertEqual(configuration.codes.count, 255)
        XCTAssertEqual(Array(configuration.codes[24...27]), ["Y", "Z", "AA", "AB"])
        XCTAssertEqual(configuration.temperature, 0.7)
        XCTAssertTrue(configuration.bidirectional)
        XCTAssertNil(configuration.maxLength)

        func json(_ changes: [String: Any]) -> [String: Any] {
            var base: [String: Any] = [
                "format_id": "d3-code-readout-v1", "codes": configuration.codes, "token_ids": configuration.tokenIDs,
                "temperature": 1.0, "attention_mode": "noncausal_full_attention", "pooling": "last", "max_length": 512,
            ]
            for (key, value) in changes { base[key] = value }
            return base
        }
        XCTAssertEqual(try D3Configuration(json: json([:])).maxLength, 512)
        XCTAssertFalse(try D3Configuration(json: json(["attention_mode": "causal"])).bidirectional)
        XCTAssertThrowsError(try D3Configuration(json: json(["format_id": "other"])))
        XCTAssertThrowsError(try D3Configuration(json: json(["codes": ["A"]])))
        XCTAssertThrowsError(try D3Configuration(json: json(["temperature": 0])))
        XCTAssertThrowsError(try D3Configuration(json: json(["attention_mode": "block"])))
        XCTAssertThrowsError(try D3Configuration(json: json(["pooling": "mean"])))

        XCTAssertNil(D3Prompt.validate(D3Requests.question("pick")))
        XCTAssertEqual(
            D3Prompt.validate(.score(Score(instructions: "", criteria: [.text("only")]))),
            "score criteria must be an ordered list of 2 to 10 levels"
        )
        XCTAssertEqual(
            D3Prompt.validate(.score(Score(instructions: "", criteria: [.text("a"), .text("a")]))),
            "score levels must be distinct and nonempty"
        )
        XCTAssertNotNil(D3Prompt.validate(.choice(Choice(instructions: "", criteria: (0..<256).map { ChoiceOption(key: "k\($0)") }))))

        let backend = try D3Backend(
            modelID: "d3-tiny",
            configuration: D3Configuration(json: json(["max_length": 100])),
            tokenizer: ByteTokenizer(), trunk: nil, readout: nil
        )
        XCTAssertThrowsError(try backend.plan(D3Requests.request(state: 0), media: nil)) {
            guard case .invalidQuestion(let id, let reason) = $0 as? SystemOneEngineError else { return XCTFail("\($0)") }
            XCTAssertEqual(id, "team")
            XCTAssertTrue(reason.contains("over the maximum context length of 100 tokens"))
        }
    }

    func testAnswersFollowTheRuntime() {
        // Ties keep the first option; confidence is 1 − H / log n.
        guard case .choice(let choice) = D3Answers.answer(D3Requests.question("team"), keys: ["billing", "tech_support", "a"], probabilities: [0.4, 0.4, 0.2]) else {
            return XCTFail("choice")
        }
        XCTAssertEqual(choice.choice, "billing")
        let entropy = -(0.4 * Foundation.log(0.4) * 2 + 0.2 * Foundation.log(0.2))
        XCTAssertEqual(choice.confidence, 1 - entropy / Foundation.log(3.0), accuracy: 1e-12)

        guard case .score(let score) = D3Answers.answer(D3Requests.question("tone"), keys: ["0", "1"], probabilities: [0.25, 0.75]) else {
            return XCTFail("score")
        }
        XCTAssertEqual(score.score, 0.75, accuracy: 1e-12)
        XCTAssertEqual(score.legend.map(\.description), [#"{"examples":["fine, thanks"],"what":"calm"}"#, #"{"examples":[],"what":"upset"}"#])
        XCTAssertEqual(D3Answers.confidence([1]), 1)
        XCTAssertEqual(D3Answers.confidence([1, 0]), 1)
        assertClose(D3Answers.probabilities([0, Float(Foundation.log(3.0))], temperature: 1), [0.25, 0.75], accuracy: 1e-6)
    }
}

private func assertClose(_ actual: [Double], _ expected: [Double], accuracy: Double, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (a, e) in zip(actual, expected) { XCTAssertEqual(a, e, accuracy: accuracy, file: file, line: line) }
}

/// The tiny trunk with the tiny readout: only from an Xcode-built test bundle
/// with `SYSTEMONE_MLX_TESTS=1`.
final class D3RuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testTinyAnswersMatchTheReference() throws {
        let fixture = try D3Fixture.load()
        let tolerance = 5e-3
        try Device.withDefaultDevice(device) {
            let backend = try D3Backend(
                modelID: "d3-tiny",
                configuration: D3Configuration(modelFolder: D3Fixture.root.appendingPathComponent("d3_tiny")),
                tokenizer: ByteTokenizer(),
                trunk: Qwen35Trunk(modelFolder: D3Fixture.root.appendingPathComponent("qwen35_tiny/transformers")),
                readout: SafetensorsFile(contentsOf: D3Fixture.root.appendingPathComponent("d3_tiny/readout.safetensors"))
            )
            for state in 0..<3 {
                let plan = try backend.plan(D3Requests.request(state: state), media: nil)
                // All six questions in one right-padded batch.
                let logits = try backend.optionLogits(for: plan.rows)
                for (decision, entry) in zip(plan.decisions, fixture.answers.filter { $0.state == state }) {
                    let answer = try decision.resolve(decision.rowIndices.map { logits[$0] })
                    let reference = entry.answer
                    let label = "state \(state) \(entry.question)"
                    switch answer {
                    case .choice(let choice):
                        XCTAssertEqual(reference.type, "choice", label)
                        XCTAssertEqual(choice.choice, reference.choice, label)
                        XCTAssertEqual(choice.confidence, reference.confidence ?? .nan, accuracy: tolerance, label)
                        for item in choice.probabilities {
                            XCTAssertEqual(item.probability, reference.probabilities?[item.key] ?? .nan, accuracy: tolerance, label)
                        }
                    case .noul(let noul):
                        XCTAssertEqual(reference.type, "noul", label)
                        XCTAssertEqual(noul.noul, reference.noul ?? .nan, accuracy: tolerance, label)
                    case .score(let score):
                        XCTAssertEqual(reference.type, "score", label)
                        XCTAssertEqual(score.score, reference.score ?? .nan, accuracy: tolerance, label)
                        XCTAssertEqual(score.confidence, reference.confidence ?? .nan, accuracy: tolerance, label)
                        XCTAssertEqual(Dictionary(uniqueKeysWithValues: score.legend.map { ($0.key, $0.description) }), reference.legend, label)
                        for item in score.probabilities {
                            XCTAssertEqual(item.probability, reference.probabilities?[item.key] ?? .nan, accuracy: tolerance, label)
                        }
                    }
                }
            }
        }
    }
}
