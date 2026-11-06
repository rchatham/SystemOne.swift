import Foundation
import MLX
import SystemOne
import SystemOneDecoders
import XCTest
@testable import SystemOneD1

/// `testdata/d1_3b_tiny/fixture.json`, written by `tools/d1_3b_fixture.py`.
private struct D1ChatFixture: Decodable {
    struct Reference: Decodable {
        let type: String
        let choice: String?
        let noul: Double?
        let score: Double?
        let confidence: Double?
        let probabilities: [String: Double]?
        let legend: [String: String]?
    }

    struct Entry: Decodable {
        let id: String
        let suffix: String
        let codes: [String]
        let groups: [[Int]]
        let tokens: Int
        let answer: Reference
    }

    struct Request: Decodable {
        let prefix: String
        let prefixTokens: Int
        let questions: [Entry]

        enum CodingKeys: String, CodingKey {
            case prefix, questions
            case prefixTokens = "prefix_tokens"
        }
    }

    let bos: Int
    let requests: [Request]

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata")

    static func load() throws -> D1ChatFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("d1_3b_tiny/fixture.json")))
    }
}

/// Whole strings with their own ids (yes/no forms, some spaced codes), else
/// one token per UTF-8 byte (`byte % 100`), as the fixture tokenizes.
private struct TableTokenizer: DecoderTokenizer {
    static let table = ["yes": 100, "Yes": 101, "YES": 102, "no": 103, "No": 104, "NO": 105, " A": 106, " B": 107, " C": 108, " x": 109]

    func encode(_ text: String) -> [Int] {
        if let id = Self.table[text] { return [id] }
        return text.utf8.map { Int($0) % 100 }
    }
}

/// The fixture's requests.
private enum D1ChatRequests {
    static let requests: [SystemOneRequest] = [
        SystemOneRequest(state: .text("Order #4821 was charged twice.\nPlease help!"), questions: [
            QuestionEntry(id: "team", question: .choice(Choice(instructions: "Which team should handle this?", criteria: [
                ChoiceOption(key: "billing", description: "Charges"), ChoiceOption(key: "tech_support"),
                ChoiceOption(key: "shipping", description: ""),
            ]))),
            QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Is this a refund request?"))),
            QuestionEntry(id: "escalate", question: .noul(Noul(instructions: "Escalate today?", criteria: NoulCriteria(true: "Needs a human")))),
            QuestionEntry(id: "urgency", question: .score(Score(instructions: "How urgent is it?", criteria: [.text("low"), .text("medium"), .text("high")]))),
            QuestionEntry(id: "letters", question: .choice(Choice(instructions: "Which one?", criteria: [
                ChoiceOption(key: "x", description: "X ray"), ChoiceOption(key: "y"),
            ]))),
        ]),
        SystemOneRequest(
            state: .fields([StateField(key: "ticket", value: "Charged twice"), StateField(key: "tier", value: "gold ✨")]),
            questions: [QuestionEntry(id: "many", question: .choice(Choice(
                instructions: "Pick the topic.",
                criteria: (0..<28).map { ChoiceOption(key: "opt\($0)", description: "Topic \($0)") }
            )))]
        ),
        SystemOneRequest(state: .text(""), questions: [
            QuestionEntry(id: "tone", question: .score(Score(instructions: "Tone?", criteria: [
                .described(what: "calm", examples: ["fine, thanks"]), .described(what: "upset", examples: []),
            ]))),
            QuestionEntry(id: "reply", question: .noul(Noul(instructions: "Reply now?", criteria: NoulCriteria(true: "Yes, now", false: "Later")))),
        ]),
    ]
}

final class D1ChatTests: XCTestCase {
    func testPromptsAndReadoutsMatchTheReference() throws {
        let fixture = try D1ChatFixture.load()
        let tokenizer = TableTokenizer()
        for (request, reference) in zip(D1ChatRequests.requests, fixture.requests) {
            XCTAssertEqual(D1ChatPrompt.prefix(state: request.state), reference.prefix)
            for (entry, expected) in zip(request.questions, reference.questions) {
                let readout = try D1ChatPrompt.readout(entry.question, tokenizer: tokenizer)
                XCTAssertEqual(readout.groups, expected.groups, entry.id)
                XCTAssertEqual(readout.codes, expected.codes, entry.id)
                XCTAssertEqual(D1ChatPrompt.suffix(question: entry.question, codes: readout.codes), expected.suffix, entry.id)
            }
        }
    }

    func testPlanSharesTheStateAcrossQuestions() throws {
        let fixture = try D1ChatFixture.load()
        let backend = D1ChatBackend(modelID: "d1-tiny", tokenizer: TableTokenizer(), bos: fixture.bos, trunk: nil)
        for (request, reference) in zip(D1ChatRequests.requests, fixture.requests) {
            let plan = try backend.plan(request, media: nil)
            XCTAssertEqual(plan.rows.map(\.prefix.count), Array(repeating: reference.prefixTokens, count: plan.rows.count))
            XCTAssertEqual(plan.rows.map(\.ids.count), reference.questions.map(\.tokens))
            XCTAssertEqual(plan.usage.inputTokens, reference.prefixTokens + reference.questions.map(\.tokens).reduce(0, +))
        }
        XCTAssertThrowsError(try backend.plan(SystemOneRequest(state: .text("s"), questions: [
            QuestionEntry(id: "wide", question: .score(Score(instructions: "", criteria: (0..<11).map { .text("\($0)") }))),
        ]), media: nil))
    }

    func testAnswersFollowTheReferenceAPI() {
        let score = Question.score(Score(instructions: "", criteria: [.text("low"), .described(what: "it's", examples: ["a"])]))
        guard case .score(let answer) = D1ChatPrompt.answer(score, probabilities: [0.3, 0.7]) else { return XCTFail("score") }
        XCTAssertEqual(answer.score, 0.7, accuracy: 1e-12)
        XCTAssertEqual(answer.confidence, 0.7, accuracy: 1e-12)
        XCTAssertEqual(answer.legend.map(\.description), ["low", #"{'what': "it's", 'examples': ['a']}"#])
        guard case .noul(let noul) = D1ChatPrompt.answer(.noul(Noul(instructions: "")), probabilities: [0.8, 0.2]) else { return XCTFail("noul") }
        XCTAssertEqual(noul.noul, 0.8, accuracy: 1e-12)
    }
}

/// The tiny LFM2 trunk: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class D1ChatRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testTinyAnswersMatchTheReference() throws {
        let fixture = try D1ChatFixture.load()
        let tolerance = 1e-3
        try Device.withDefaultDevice(device) {
            let backend = D1ChatBackend(
                modelID: "d1-tiny", tokenizer: TableTokenizer(), bos: fixture.bos,
                trunk: try LFM2Trunk(modelFolder: D1ChatFixture.root.appendingPathComponent("lfm2_tiny"))
            )
            for (request, reference) in zip(D1ChatRequests.requests, fixture.requests) {
                let plan = try backend.plan(request, media: nil)
                let logits = try backend.optionLogits(for: plan.rows)
                for (decision, expected) in zip(plan.decisions, reference.questions) {
                    let answer = try decision.resolve(decision.rowIndices.map { logits[$0] })
                    let label = expected.id
                    switch answer {
                    case .choice(let choice):
                        XCTAssertEqual(choice.choice, expected.answer.choice, label)
                        XCTAssertEqual(choice.confidence, expected.answer.confidence ?? .nan, accuracy: tolerance, label)
                        for item in choice.probabilities {
                            XCTAssertEqual(item.probability, expected.answer.probabilities?[item.key] ?? .nan, accuracy: tolerance, label)
                        }
                    case .noul(let noul):
                        XCTAssertEqual(noul.noul, expected.answer.noul ?? .nan, accuracy: tolerance, label)
                    case .score(let score):
                        XCTAssertEqual(score.score, expected.answer.score ?? .nan, accuracy: tolerance, label)
                        XCTAssertEqual(score.confidence, expected.answer.confidence ?? .nan, accuracy: tolerance, label)
                        XCTAssertEqual(Dictionary(uniqueKeysWithValues: score.legend.map { ($0.key, $0.description) }), expected.answer.legend, label)
                    }
                }
            }
        }
    }
}
