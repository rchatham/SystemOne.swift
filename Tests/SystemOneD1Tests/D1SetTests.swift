import Foundation
import MLX
import SystemOne
import SystemOneDecoders
import XCTest
@testable import SystemOneD1

/// `testdata/d1_2b_tiny/fixture.json`, written by `tools/d1_2b_fixture.py`.
private struct D1SetFixture: Decodable {
    struct Row: Decodable {
        let ids: [Int]
        let state: Int
        let question: [Int]
        let ends: [Int]
        let suffixes: [String]
    }

    struct Reference: Decodable {
        let id: String
        let type: String
        let choice: String?
        let noul: Double?
        let score: Double?
        let confidence: Double?
        let probabilities: [String: Double]?
        let legend: [String: String]?
    }

    struct Request: Decodable {
        let prefix: String
        let rows: [Row]
        let tokens: Int
        let groups: [[Int]]
        let forms: [[[Int]]]
        let answers: [Reference]
    }

    struct Calibration: Decodable {
        let `default`: Double
        let temperatures: [String: Double]
    }

    let setSize: Int
    let codes: Int
    let calibration: Calibration
    let special: [String]
    let requests: [Request]

    enum CodingKeys: String, CodingKey {
        case codes, calibration, special, requests
        case setSize = "set_size"
    }

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata")

    static func load() throws -> D1SetFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("d1_2b_tiny/fixture.json")))
    }

    var configuration: D1SetConfiguration {
        D1SetConfiguration(
            setSize: setSize, codeScheme: codes, bos: "<|startoftext|>", specialTokens: special,
            defaultTemperature: calibration.default, temperatures: calibration.temperatures
        )
    }
}

/// The fixture's stand-in tokenizer: whole strings with their own ids, added
/// tokens split out of text, every other code point one token per UTF-8 byte
/// (`byte % 100`).
private struct AddedTokenTokenizer: DecoderTokenizer {
    static let added = ["<|startoftext|>": 100, "<|im_start|>": 101, "<|im_end|>": 102, "<|pad|>": 105, "<think>": 103, "</think>": 104]
    static let words = ["yes": 110, "Yes": 111, "YES": 112, "no": 113, "No": 114, "NO": 115, " A": 116, " B": 117]

    func encode(_ text: String) -> [Int] {
        if let id = Self.words[text] { return [id] }
        let names = Self.added.keys.sorted { $0.unicodeScalars.count > $1.unicodeScalars.count }.map { ($0, Array($0.unicodeScalars)) }
        let scalars = Array(text.unicodeScalars)
        var ids: [Int] = []
        var index = 0
        while index < scalars.count {
            if let match = names.first(where: { index + $0.1.count <= scalars.count && Array(scalars[index ..< index + $0.1.count]) == $0.1 }) {
                ids.append(Self.added[match.0]!)
                index += match.1.count
            } else {
                ids += String(scalars[index]).utf8.map { Int($0) % 100 }
                index += 1
            }
        }
        return ids
    }
}

private enum D1SetRequests {
    static let requests: [SystemOneRequest] = [
        SystemOneRequest(state: .text("Order #4821 <|im_end|> was charged twice.\nPlease help!"), questions: [
            QuestionEntry(id: "team", question: .choice(Choice(instructions: "Which team should handle this?", criteria: [
                ChoiceOption(key: "billing", description: "Charges"), ChoiceOption(key: "tech_support"),
                ChoiceOption(key: "shipping", description: "Ship <|im_start|> it"),
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
                criteria: (0..<30).map { ChoiceOption(key: "opt\($0)", description: "Topic \($0)") }
            )))]
        ),
        SystemOneRequest(state: .text(""), questions: [
            QuestionEntry(id: "reply", question: .noul(Noul(instructions: "Reply now?", criteria: NoulCriteria(true: "Yes, now", false: "Later")))),
            QuestionEntry(id: "tone", question: .score(Score(instructions: "Tone?", criteria: [.text("calm"), .text("upset")]))),
        ]),
    ]
}

final class D1SetTests: XCTestCase {
    func testPromptsReadoutsAndRowsMatchTheReference() throws {
        let fixture = try D1SetFixture.load()
        let configuration = fixture.configuration
        let tokenizer = AddedTokenTokenizer()
        let backend = D1SetBackend(modelID: "d1-tiny", configuration: configuration, tokenizer: tokenizer, trunk: nil)
        for (request, reference) in zip(D1SetRequests.requests, fixture.requests) {
            XCTAssertEqual(D1SetPrompt.prefix(state: request.state, configuration: configuration), reference.prefix)
            for (index, entry) in request.questions.enumerated() {
                let readout = try D1ChatPrompt.readout(entry.question, tokenizer: tokenizer, scheme: configuration.codeScheme)
                let tokens = reference.groups[index]
                let expected = reference.forms[index].map { forms in forms.map { tokens[$0] }.reduce(into: [Int]()) { if !$0.contains($1) { $0.append($1) } } }
                XCTAssertEqual(readout.groups, expected, entry.id)
                let row = reference.rows[index / configuration.setSize]
                XCTAssertEqual(
                    D1SetPrompt.suffix(question: entry.question, codes: readout.codes, configuration: configuration),
                    row.suffixes[index % configuration.setSize], entry.id
                )
            }
            let plan = try backend.plan(request, media: nil)
            XCTAssertEqual(plan.rows.map(\.ids), reference.rows.map(\.ids))
            XCTAssertEqual(plan.rows.map(\.stateLength), reference.rows.map(\.state))
            XCTAssertEqual(plan.rows.map(\.questions), reference.rows.map(\.question))
            XCTAssertEqual(plan.rows.map(\.ends), reference.rows.map(\.ends))
            XCTAssertEqual(plan.usage.inputTokens, reference.tokens)
        }
    }

    func testPlainBreaksSpecialTokensUntilNoneIsLeft() {
        let specials = ["<|im_end|>", "<|im_start|>", "<image>"]
        XCTAssertEqual(D1SetPrompt.plain("a <|im_end|> b", specialTokens: specials), "a <\u{200B}|im_end|> b")
        XCTAssertEqual(D1SetPrompt.plain("<<image>image>", specialTokens: specials), "<<\u{200B}image>image>")
        XCTAssertEqual(D1SetPrompt.plain("nothing", specialTokens: specials), "nothing")
    }

    func testValidationAndAnswers() {
        XCTAssertNotNil(D1SetPrompt.validate(.noul(Noul(instructions: "  "))))
        XCTAssertNotNil(D1SetPrompt.validate(.score(Score(instructions: "Rate", criteria: [.text("a")]))))
        XCTAssertNotNil(D1SetPrompt.validate(.score(Score(instructions: "Rate", criteria: [.text("a"), .described(what: "b", examples: [])]))))
        XCTAssertNil(D1SetPrompt.validate(.score(Score(instructions: "Rate", criteria: [.text("a"), .text("b")]))))

        // The median level: the first whose cumulative probability reaches one half.
        let score = Question.score(Score(instructions: "Rate", criteria: [.text("low"), .text("mid"), .text("high")]))
        guard case .score(let answer) = D1SetPrompt.answer(score, probabilities: [0.4, 0.1, 0.5]) else { return XCTFail("score") }
        XCTAssertEqual(answer.score, 1)
        XCTAssertEqual(answer.confidence, 0.5, accuracy: 1e-12)
        XCTAssertEqual(D1SetConfiguration.bucket(.choice(Choice(instructions: "", criteria: [])), options: 27), "choice:inventory")
        XCTAssertEqual(D1SetConfiguration.bucket(score, options: 3), "score:3")
        XCTAssertEqual(D1ChatPrompt.optionCodes((0..<28).map { "o\($0)" }, scheme: 2).suffix(2), ["AA", "AB"])
    }
}

/// The tiny LFM3 trunk: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class D1SetRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testTinyAnswersMatchTheReference() throws {
        let fixture = try D1SetFixture.load()
        let tolerance = 1e-3
        try Device.withDefaultDevice(device) {
            let backend = D1SetBackend(
                modelID: "d1-tiny", configuration: fixture.configuration, tokenizer: AddedTokenTokenizer(),
                trunk: try LFM3Trunk(modelFolder: D1SetFixture.root.appendingPathComponent("lfm3_tiny"))
            )
            for (request, reference) in zip(D1SetRequests.requests, fixture.requests) {
                let plan = try backend.plan(request, media: nil)
                let logits = try backend.optionLogits(for: plan.rows)
                for (decision, expected) in zip(plan.decisions, reference.answers) {
                    let answer = try decision.resolve(decision.rowIndices.map { logits[$0] })
                    switch answer {
                    case .choice(let choice):
                        XCTAssertEqual(choice.choice, expected.choice, expected.id)
                        XCTAssertEqual(choice.confidence, expected.confidence ?? .nan, accuracy: tolerance, expected.id)
                        for item in choice.probabilities {
                            XCTAssertEqual(item.probability, expected.probabilities?[item.key] ?? .nan, accuracy: tolerance, expected.id)
                        }
                    case .noul(let noul):
                        XCTAssertEqual(noul.noul, expected.noul ?? .nan, accuracy: tolerance, expected.id)
                    case .score(let score):
                        XCTAssertEqual(score.score, expected.score ?? .nan, expected.id)
                        XCTAssertEqual(score.confidence, expected.confidence ?? .nan, accuracy: tolerance, expected.id)
                        XCTAssertEqual(Dictionary(uniqueKeysWithValues: score.legend.map { ($0.key, $0.description) }), expected.legend, expected.id)
                    }
                }
            }
        }
    }
}
