import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneEncoders

/// `testdata/encoder_decisions/fixture.json`, written by
/// `tools/encoder_decisions_fixture.py` from each model's reference code.
private struct DecisionsFixture: Decodable {
    struct Row: Decodable, Equatable {
        let ids: [Int]
        let markers: [Int]
    }

    struct Tree: Decodable {
        let ids: [Int]
        let positions: [Int]
        let kind: [Int]
        let question: [Int]
        let option: [Int]
    }

    struct Prompts: Decodable {
        let decisiontune: [Row]
        let laya: [Row]
        let julia: [Row?]
        let tasksource: [Tree]
        let layaBudget: Row

        enum CodingKeys: String, CodingKey {
            case decisiontune, laya, julia, tasksource
            case layaBudget = "laya_budget"
        }
    }

    struct TinyRow: Decodable {
        let ids: [Int]
        let markers: [Int]
        let type: Int
    }

    struct TinyTree: Decodable {
        let row: Tree
        let logits: [Float]
    }

    struct Tiny: Decodable {
        let rows: [TinyRow]
        let decisiontune: [[Float]]
        let laya: [[Float]]
        let tasksource: TinyTree
    }

    let prompts: Prompts
    let tiny: Tiny

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    static func load() throws -> DecisionsFixture {
        try JSONDecoder().decode(
            Self.self, from: Data(contentsOf: root.appendingPathComponent("testdata/encoder_decisions/fixture.json"))
        )
    }
}

/// The fixture's requests, in the reference's option order.
private enum Requests {
    static let states: [SystemOneState] = [
        .text("I was charged twice for order #4821 and the [MASK] second charge is still pending."),
        .fields([StateField(key: "ticket", value: "Charged twice for one order"), StateField(key: "tier", value: "gold")]),
    ]
    static let questions: [Question] = [
        .choice(Choice(instructions: "Which team should handle this?", criteria: [
            ChoiceOption(key: "billing", description: "Charges, refunds, invoices"),
            ChoiceOption(key: "tech_support", description: ""),
            ChoiceOption(key: "a", description: "App faults"),
            ChoiceOption(key: "shipping", description: "Shipping"),
        ])),
        .noul(Noul(instructions: "Is the customer asking for a refund?")),
        .noul(Noul(instructions: "Escalate today?", criteria: NoulCriteria(true: "Needs a human today", false: "Routine"))),
        .score(Score(instructions: "How urgent is it?", criteria: [.text("low"), .text("medium"), .text("high")])),
    ]
}

final class EncoderDecisionTests: XCTestCase {
    private func tokenizer() throws -> (HubEncoderTokenizer, EncoderSpecialTokens) {
        let tokenizer = try HubEncoderTokenizer(
            folder: DecisionsFixture.root.appendingPathComponent("Tests/SystemOneVonTests/Resources/Tokenizer")
        )
        return (tokenizer, try EncoderSpecialTokens(tokenizer: tokenizer, names: tokenizer.specialTokens))
    }

    func testDecisionTunePromptsMatchTheReference() throws {
        let fixture = try DecisionsFixture.load()
        let (tokenizer, tokens) = try tokenizer()
        var index = 0
        for state in Requests.states {
            for question in Requests.questions {
                let row = DecisionTunePrompt.encode(
                    state: state, instructions: question.instructions,
                    options: DecisionTunePrompt.options(for: question), tokenizer: tokenizer, tokens: tokens
                )
                XCTAssertEqual(DecisionsFixture.Row(ids: row.ids, markers: row.markers), fixture.prompts.decisiontune[index], "request \(index)")
                index += 1
            }
        }
    }

    func testLayaAndJuliaPromptsMatchTheReference() throws {
        let fixture = try DecisionsFixture.load()
        let (tokenizer, tokens) = try tokenizer()
        let laya = MarkerModelConfiguration(style: .laya, maxLength: 512, headLength: 192, headLayers: 2)
        let julia = MarkerModelConfiguration(style: .julia, maxLength: 8_192, headLength: 512, headLayers: 2)
        var index = 0
        for state in Requests.states {
            for question in Requests.questions {
                let layaRow = try MarkerPrompt.encode(
                    state: state, question: question, options: MarkerPrompt.options(for: question, style: .laya),
                    configuration: laya, tokenizer: tokenizer, tokens: tokens
                )
                XCTAssertEqual(DecisionsFixture.Row(ids: layaRow.ids, markers: layaRow.markers), fixture.prompts.laya[index], "laya \(index)")
                XCTAssertEqual(layaRow.questionType, EncoderText.typeID(question))
                if let expected = fixture.prompts.julia[index] {
                    let juliaRow = try MarkerPrompt.encode(
                        state: state, question: question, options: MarkerPrompt.options(for: question, style: .julia),
                        configuration: julia, tokenizer: tokenizer, tokens: tokens
                    )
                    XCTAssertEqual(DecisionsFixture.Row(ids: juliaRow.ids, markers: juliaRow.markers), expected, "julia \(index)")
                } else {
                    XCTAssertThrowsError(try MarkerPrompt.options(for: question, style: .julia), "julia \(index)")
                }
                index += 1
            }
        }

        // Twelve long options shrink evenly into the 192-token head budget.
        let many = Question.choice(Choice(
            instructions: "Pick the closest topic.",
            criteria: (0..<12).map { ChoiceOption(key: "topic_\($0)", description: String(repeating: "a very long description ", count: 6)) }
        ))
        let budget = try MarkerPrompt.encode(
            state: Requests.states[0], question: many, options: MarkerPrompt.options(for: many, style: .laya),
            configuration: laya, tokenizer: tokenizer, tokens: tokens
        )
        XCTAssertEqual(DecisionsFixture.Row(ids: budget.ids, markers: budget.markers), fixture.prompts.layaBudget)
    }

    func testTreeDeciderPacksTheReferenceRow() throws {
        let fixture = try DecisionsFixture.load()
        let (tokenizer, tokens) = try tokenizer()
        let backend = TreeDeciderBackend(modelID: "tree", tokenizer: tokenizer, tokens: tokens, maxLength: 8_192, model: nil)
        for (index, state) in Requests.states.enumerated() {
            let typed = Requests.questions.filter { if case .score = $0 { return false }; return true }
            let (rows, places) = try backend.pack(
                state: state,
                questions: try typed.map { (id: "q", instructions: $0.instructions, options: try TreeDeciderPrompt.options(for: $0)) }
            )
            XCTAssertEqual(rows.count, 1)
            let expected = fixture.prompts.tasksource[index]
            XCTAssertEqual(rows[0].ids, expected.ids, "state \(index)")
            XCTAssertEqual(rows[0].positions, expected.positions)
            XCTAssertEqual(rows[0].kinds, expected.kind)
            XCTAssertEqual(rows[0].questions, expected.question)
            XCTAssertEqual(rows[0].options, expected.option)
            XCTAssertEqual(places.map { $0.map { $0.option } }, [[0, 1, 2, 3], [4, 5], [6, 7]])
        }
        XCTAssertThrowsError(try TreeDeciderPrompt.options(for: Requests.questions[3]))
    }

    func testOptionAndStateRendering() {
        XCTAssertEqual(DecisionTunePrompt.optionText(key: "tech_support", description: nil), "tech support")
        XCTAssertEqual(DecisionTunePrompt.optionText(key: "k", description: "  "), "  ")
        XCTAssertEqual(DecisionTunePrompt.stateText(Requests.states[1]), "ticket: Charged twice for one order | tier: gold")
        XCTAssertEqual(DecisionTunePrompt.clean("a [SEP] b"), "a   b")

        XCTAssertEqual(TreeDeciderPrompt.formatOption(key: "option_2", description: "Billing"), "Billing")
        XCTAssertEqual(TreeDeciderPrompt.formatOption(key: "(b)", description: "Billing"), "Billing")
        XCTAssertEqual(TreeDeciderPrompt.formatOption(key: "billing", description: " Billing "), "billing")
        XCTAssertEqual(TreeDeciderPrompt.formatOption(key: "billing", description: "Charges"), "billing: Charges")
        XCTAssertEqual(TreeDeciderPrompt.formatOption(key: " x ", description: nil), "x")
        XCTAssertEqual(TreeDeciderPrompt.stateText(.fields([])), "(empty)")
        XCTAssertEqual(TreeDeciderPrompt.stateText(Requests.states[1]), #"{"ticket":"Charged twice for one order","tier":"gold"}"#)

        XCTAssertEqual(EncoderText.pythonJSON(Requests.states[1]), #"{"ticket": "Charged twice for one order", "tier": "gold"}"#)
        XCTAssertEqual(EncoderText.pythonString(.described(what: "it's", examples: ["a", "b"])), #"{'what': "it's", 'examples': ['a', 'b']}"#)
        XCTAssertEqual(EncoderText.pythonRepr("tab\there"), #"'tab\there'"#)
    }

    func testLayaCalibrationAndAnswers() throws {
        let configuration = MarkerModelConfiguration(
            style: .laya, maxLength: 512, headLength: 192, headLayers: 2,
            temperatures: [2, 3, 4], temperaturesByOptions: ["choice:2": 0.5, "noul:2": 1]
        )
        XCTAssertEqual(MarkerModelConfiguration.bucket(type: 0, options: 11), "choice:11+")
        XCTAssertEqual(MarkerModelConfiguration.bucket(type: 1, options: 7), "score:6-10")
        XCTAssertEqual(configuration.temperature(type: 0, options: 2), 0.5)
        XCTAssertEqual(configuration.temperature(type: 0, options: 4), 2)
        XCTAssertEqual(configuration.temperature(type: 1, options: 3), 3)

        let backend = MarkerHeadBackend(
            modelID: "laya", configuration: configuration, tokenizer: StubTokenizer(),
            tokens: EncoderSpecialTokens(cls: 1, sep: 2, mask: 3, pad: 0, maskText: "[MASK]"), model: nil
        )
        let plan = try backend.plan(SystemOneRequest(state: .text("state"), questions: [
            QuestionEntry(id: "c", question: Requests.questions[0]),
            QuestionEntry(id: "n", question: Requests.questions[1]),
        ]), media: nil)
        XCTAssertEqual(plan.rows.map(\.markers.count), [4, 2])
        XCTAssertEqual(plan.rows.map(\.questionType), [0, 2])
        guard case .choice(let choice) = try plan.decisions[0].resolve([[2, 0, 0, 0]]) else { return XCTFail("choice") }
        // Temperature 2 (choice:3-5 is not overridden): logits [1, 0, 0, 0].
        let expected = EncoderAnswers.softmax([1, 0, 0, 0])
        XCTAssertEqual(choice.choice, "billing")
        XCTAssertEqual(choice.probabilities.map(\.probability), expected)
        XCTAssertEqual(choice.confidence, EncoderAnswers.entropyConfidence(expected), accuracy: 1e-12)
        guard case .noul(let noul) = try plan.decisions[1].resolve([[0, 1]]) else { return XCTFail("noul") }
        XCTAssertEqual(noul.noul, 1 / (1 + exp(-1)), accuracy: 1e-12)
    }
}

/// A tokenizer that maps every character to one id, for planning tests.
private struct StubTokenizer: EncoderTokenizer {
    func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value % 50) + 10 } }
    func id(of token: String) -> Int? { nil }
}

/// The reference heads on the tiny encoder: only from an Xcode-built test
/// bundle with `SYSTEMONE_MLX_TESTS=1`.
final class EncoderDecisionRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private let tiny = DecisionsFixture.root.appendingPathComponent("testdata/encoder_decisions/tiny")
    private let encoderFolder = DecisionsFixture.root.appendingPathComponent("testdata/modernbert_tiny/large-rope")

    private func assertClose(_ actual: [[Float]], _ expected: [[Float]], _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.map(\.count), expected.map(\.count), message, file: file, line: line)
        let worst = zip(actual.joined(), expected.joined()).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThanOrEqual(worst, 1e-4, "\(message): \(actual) vs \(expected)", file: file, line: line)
    }

    func testTinyHeadsMatchTheirReferences() throws {
        let fixture = try DecisionsFixture.load()
        let configuration = try ModernBERTConfiguration(modelFolder: encoderFolder)
        let rows = fixture.tiny.rows.map { EncoderRow(ids: $0.ids, markers: $0.markers, questionType: $0.type) }
        try Device.withDefaultDevice(device) {
            let tune = try DecisionTuneModel(
                encoder: ModernBERTEncoder(
                    store: SafetensorsFile(contentsOf: encoderFolder.appendingPathComponent("model.safetensors")),
                    configuration: configuration
                ),
                head: SafetensorsFile(contentsOf: tiny.appendingPathComponent("decisiontune/head.safetensors")),
                pad: 0
            )
            assertClose(try tune.logits(rows), fixture.tiny.decisiontune, "DecisionTune")

            let laya = try MarkerHeadModel(
                store: SafetensorsFile(contentsOf: tiny.appendingPathComponent("laya/model.safetensors")),
                encoder: configuration, headLayers: 2, pad: 0
            )
            assertClose(try laya.logits(rows), fixture.tiny.laya, "Laya")

            let tree = fixture.tiny.tasksource.row
            let decider = try TreeDeciderModel(
                store: SafetensorsFile(contentsOf: tiny.appendingPathComponent("tasksource/model.safetensors")),
                encoder: configuration, pad: 0
            )
            let row = TreeRow(
                ids: tree.ids, positions: tree.positions, kinds: tree.kind, questions: tree.question,
                options: tree.option, optionCount: (tree.option.max() ?? -1) + 1
            )
            assertClose(try decider.logits([row]), [fixture.tiny.tasksource.logits], "tasksource")
        }
    }
}
