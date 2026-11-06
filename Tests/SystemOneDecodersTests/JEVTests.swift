import Foundation
import MLX
import MLXNN
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// `testdata/jev_tiny_golden_v1.json`, written by `tools/jev_golden_capture.py tiny`
/// from transformers' Qwen3.5 classes and peft on the committed tiny model.
struct JEVTinyFixture: Decodable {
    struct MergeProbe: Decodable {
        let indices: [Int]
        let base: [Float]
        let merged: [Float]
    }
    struct Probabilities: Decodable {
        let noul: [Float]
        let score: [Float]
        let choice: [Float]
    }
    struct Case: Decodable {
        let ids: [Int]
        let probeRows: [Int]
        let layers: [String: [[Float]]]
        let lastHidden: [Float]
        let headLogits: [Float]
        let probabilities: Probabilities
        enum CodingKeys: String, CodingKey {
            case ids, layers, probabilities
            case probeRows = "probe_rows"
            case lastHidden = "last_hidden"
            case headLogits = "head_logits"
        }
    }

    let loraScale: Float
    let temperatures: [String: Double]
    let choiceOptions: Int
    let mergeProbes: [String: MergeProbe]
    let cases: [Case]

    enum CodingKeys: String, CodingKey {
        case temperatures, cases
        case loraScale = "lora_scale"
        case choiceOptions = "choice_options"
        case mergeProbes = "merge_probes"
    }

    static let testdata = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata")
    static let folder = testdata.appendingPathComponent("jev_tiny_v1")

    static func load() throws -> JEVTinyFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: testdata.appendingPathComponent("jev_tiny_golden_v1.json")))
    }
}

/// One token per Unicode scalar, so prompt lengths are easy to reason about.
private struct ScalarTokenizer: ReversibleDecoderTokenizer {
    func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    func decode(_ ids: [Int]) -> String {
        String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0)) }))
    }
}

final class JEVTests: XCTestCase {
    func testConfigurationReadsTheTinyFolder() throws {
        let configuration = try JEVConfiguration(modelFolder: JEVTinyFixture.folder)
        XCTAssertEqual(configuration.slotCount, 24)
        XCTAssertEqual(configuration.slots, [.noul: 0..<2, .score: 2..<8, .choice: 8..<24])
        XCTAssertEqual(configuration.temperatures, [.noul: 1.25, .choice: 0.8, .score: 1.1])
        XCTAssertEqual(configuration.adapterSubfolder, "adapter")
    }

    func testConfigurationRejectsWhatTheHeadCannotRead() {
        let calibration: [String: Any] = ["per_kind": ["noul": 1.0, "choice": 1.0, "score": 1.0]]
        func judge(template: String = "bare-v1", score: [Int] = [2, 8], adapter: String = "adapter") -> [String: Any] {
            ["slots": ["num_slots": 24, "template_version": template,
                       "ranges": ["noul": [0, 2], "score": score, "choice": [8, 24]]],
             "adapter_subfolder": adapter]
        }
        XCTAssertNoThrow(try JEVConfiguration(judge: judge(), calibration: calibration))
        for (label, judge, calibration) in [
            ("template", judge(template: "chat-v2"), calibration),
            ("score slots", judge(score: [2, 6]), calibration),
            ("slots past num_slots", judge(score: [20, 26]), calibration),
            ("adapter outside", judge(adapter: "../elsewhere"), calibration),
            ("missing temperature", judge(), ["per_kind": ["noul": 1.0, "choice": 1.0]]),
            ("zero temperature", judge(), ["per_kind": ["noul": 1.0, "choice": 0, "score": 1.0]]),
        ] {
            XCTAssertThrowsError(try JEVConfiguration(judge: judge, calibration: calibration), label)
        }
    }

    func testRendersTheModelCardTemplate() {
        XCTAssertEqual(
            JEVPrompt.render(
                kind: .choice,
                state: "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
                question: "Supplier response for this scenario.",
                options: ["issue_warning", "renegotiate", "dual_source", "maintain"]
            ),
            """
            [kind] choice
            [state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
            [question] Supplier response for this scenario.
            [options]
            A) issue_warning
            B) renegotiate
            C) dual_source
            D) maintain
            [decision]:
            """
        )
        XCTAssertEqual(
            JEVPrompt.render(kind: .noul, state: "S", question: "Q?", options: ["false", "true"]),
            "[kind] noul\n[state] S\n[question] Q?\n[options]\nfalse\ntrue\n[decision]:"
        )
        let sixteen = JEVPrompt.render(kind: .choice, state: "", question: "q", options: (0..<16).map { "o\($0)" })
        XCTAssertTrue(sixteen.hasSuffix("O) o14\nP) o15\n[decision]:"))
    }

    func testOptionsAndValidation() {
        let choice = Question.choice(Choice(instructions: "q", criteria: [
            ChoiceOption(key: "billing", description: "Charges and refunds"),
            ChoiceOption(key: "fraud"),
            ChoiceOption(key: "other", description: ""),
        ]))
        let options = JEVPrompt.options(for: choice)
        XCTAssertEqual(options.kind, .choice)
        XCTAssertEqual(options.keys, ["billing", "fraud", "other"])
        XCTAssertEqual(options.texts, ["billing: Charges and refunds", "fraud", "other"])
        XCTAssertEqual(JEVPrompt.options(for: .noul(Noul(instructions: "q"))).texts, ["false", "true"])
        let six = Question.score(Score(instructions: "q", criteria: (0..<6).map { .text("l\($0)") }))
        XCTAssertEqual(JEVPrompt.options(for: six).keys, ["0", "1", "2", "3", "4", "5"])

        XCTAssertNil(JEVPrompt.validate(six))
        XCTAssertNotNil(JEVPrompt.validate(.score(Score(instructions: "q", criteria: (0..<5).map { .text("\($0)") }))))
        let options17 = (0..<17).map { ChoiceOption(key: "o\($0)") }
        XCTAssertNotNil(JEVPrompt.validate(.choice(Choice(instructions: "q", criteria: options17))))
        XCTAssertNil(JEVPrompt.validate(.choice(Choice(instructions: "q", criteria: Array(options17.prefix(16))))))
        XCTAssertNotNil(JEVPrompt.validate(.choice(Choice(instructions: "q", criteria: Array(options17.prefix(9)))), choiceSlots: 8))
    }

    func testFoldedQuestionsStayOnOneLine() {
        let questions: [Question] = [
            .choice(Choice(instructions: "Which team?", criteria: [ChoiceOption(key: "a"), ChoiceOption(key: "b")])),
            .noul(Noul(instructions: "Is it urgent?", criteria: NoulCriteria(true: "needs action today", false: "can wait"))),
            .score(Score(instructions: "How severe?", criteria: [
                .text("none"), .text("cosmetic"), .text("minor"), .text("major"),
                .described(what: "critical", examples: ["outage"]), .text("catastrophic"),
            ])),
        ]
        for question in questions {
            let folded = JEVPrompt.foldedQuestion(question)
            XCTAssertFalse(folded.contains("\n"), folded)
            XCTAssertTrue(folded.hasPrefix(question.instructions), folded)
        }
        XCTAssertEqual(JEVPrompt.foldedQuestion(questions[0]), "Which team?")
    }

    func testPlansOneRowPerQuestionOnItsKindsSlots() throws {
        let backend = try makeBackend()
        let levels: [ScoreLevel] = (0..<6).map { .text("level \($0)") }
        let plan = try backend.plan(SystemOneRequest(state: .text("Ticket"), questions: [
            QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Refund?"))),
            QuestionEntry(id: "team", question: .choice(Choice(instructions: "Team?", criteria: [
                ChoiceOption(key: "billing", description: "money"), ChoiceOption(key: "tech"), ChoiceOption(key: "other"),
            ]))),
            QuestionEntry(id: "urgency", question: .score(Score(instructions: "Urgency?", criteria: levels))),
            QuestionEntry(id: "only", question: .choice(Choice(instructions: "Pick", criteria: [ChoiceOption(key: "x")]))),
        ]), media: nil)

        XCTAssertEqual(plan.rows.map(\.slots), [0..<2, 8..<11, 2..<8])
        XCTAssertEqual(plan.decisions.map(\.rowIndices), [[0], [1], [2], []])
        XCTAssertEqual(
            ScalarTokenizer().decode(plan.rows[1].ids),
            "[kind] choice\n[state] Ticket\n[question] Team?\n[options]\nA) billing: money\nB) tech\nC) other\n[decision]:"
        )
        XCTAssertEqual(plan.usage.inputTokens, plan.rows.reduce(0) { $0 + $1.ids.count })

        // Logits 0, 1, 2 at T = 0.8 for the choice; the one-option choice needs no logits.
        guard case .choice(let team) = try plan.decisions[1].resolve([[0, 1, 2]]) else { return XCTFail("team") }
        let expected = DecoderAnswers.probabilities([0, 1, 2], temperature: 0.8)
        XCTAssertEqual(team.choice, "other")
        XCTAssertEqual(team.probabilities.map(\.probability), expected)
        XCTAssertEqual(team.confidence, DecoderAnswers.confidence(expected))
        guard case .choice(let only) = try plan.decisions[3].resolve([]) else { return XCTFail("only") }
        XCTAssertEqual(only.choice, "x")
        XCTAssertEqual(only.confidence, 1)
        guard case .noul(let refund) = try plan.decisions[0].resolve([[0, 1]]) else { return XCTFail("refund") }
        XCTAssertEqual(refund.noul, DecoderAnswers.probabilities([0, 1], temperature: 1.25)[1])
    }

    func testRejectsQuestionsAndMediaTheHeadCannotRead() throws {
        let backend = try makeBackend()
        func plan(_ question: Question) throws {
            _ = try backend.plan(SystemOneRequest(state: .text("s"), questions: [QuestionEntry(id: "q", question: question)]), media: nil)
        }
        XCTAssertThrowsError(try plan(.choice(Choice(instructions: "q", criteria: (0..<17).map { ChoiceOption(key: "o\($0)") }))))
        XCTAssertThrowsError(try plan(.score(Score(instructions: "q", criteria: (0..<7).map { .text("\($0)") }))))
        XCTAssertThrowsError(
            try backend.plan(SystemOneRequest(state: .text("s"), questions: []),
                             media: .images([RGBImage(width: 1, height: 1, pixels: [0, 0, 0])]))
        ) {
            XCTAssertEqual($0 as? SystemOneEngineError, .unsupportedMedia(model: "jev-test"))
        }
    }

    func testLongStatesKeepTheirHeadAndTail() throws {
        let backend = try makeBackend(maxLength: 120)
        let empty = JEVPrompt.render(kind: .noul, state: "", question: "Q?", options: ["false", "true"]).unicodeScalars.count
        let state = String(repeating: "h", count: 100) + String(repeating: "m", count: 100) + String(repeating: "t", count: 100)
        let entry = QuestionEntry(id: "q", question: .noul(Noul(instructions: "Q?")))

        let plan = try backend.plan(SystemOneRequest(state: .text(state), questions: [entry]), media: nil)
        XCTAssertEqual(plan.rows[0].ids.count, 120)
        let budget = 120 - empty
        let kept = String(repeating: "h", count: budget * 3 / 5) + String(repeating: "t", count: budget - budget * 3 / 5)
        XCTAssertEqual(ScalarTokenizer().decode(plan.rows[0].ids),
                       JEVPrompt.render(kind: .noul, state: kept, question: "Q?", options: ["false", "true"]))

        let tight = try makeBackend(maxLength: empty - 1)
        XCTAssertThrowsError(try tight.plan(SystemOneRequest(state: .text("s"), questions: [entry]), media: nil)) {
            XCTAssertEqual($0 as? JEVError, .promptTooLong(id: "q", tokens: empty, maximum: empty - 1))
        }
    }

    func testAdapterConfigRejectsWhatItCannotMerge() throws {
        let store = try SafetensorsFile(contentsOf: JEVTinyFixture.folder.appendingPathComponent("adapter/adapter_model.safetensors"))
        let base: [String: Any] = ["peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["q_proj"],
                                   "bias": "none", "rank_pattern": [String: Any](), "alpha_pattern": [String: Any]()]
        let adapter = try LoRAAdapter(config: base, store: store)
        XCTAssertEqual(adapter.scale, 2)
        XCTAssertTrue(adapter.targets("model.layers.3.self_attn.q_proj"))
        XCTAssertFalse(adapter.targets("model.layers.3.self_attn.k_proj"))
        for (key, value) in [("peft_type", "IA3"), ("use_dora", true), ("use_rslora", true), ("bias", "all"),
                             ("target_modules", "q_proj|k_proj"), ("rank_pattern", ["q_proj": 8]), ("r", 0)] as [(String, Any)] {
            var config = base
            config[key] = value
            XCTAssertThrowsError(try LoRAAdapter(config: config, store: store), key)
        }
    }

    private func makeBackend(maxLength: Int = 4_096) throws -> JEVBackend {
        try JEVBackend(
            modelID: "jev-test", configuration: JEVConfiguration(modelFolder: JEVTinyFixture.folder),
            tokenizer: ScalarTokenizer(), maxLength: maxLength, trunk: nil, head: nil
        )
    }
}

/// The tiny model through `Qwen35Trunk` with the adapter merged: only from an
/// Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class JEVRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testLoRAMergesAsPeftDoes() throws {
        let fixture = try JEVTinyFixture.load()
        try Device.withDefaultDevice(device) {
            let folder = JEVTinyFixture.folder
            let store = try SafetensorsCheckpoint(modelFolder: folder)
            let adapter = try LoRAAdapter(folder: folder.appendingPathComponent("adapter"))
            XCTAssertEqual(adapter.scale, fixture.loraScale)
            for (name, probe) in fixture.mergeProbes.sorted(by: { $0.key < $1.key }) {
                let module = String(name.dropLast(".weight".count))
                let shape = try store.requiredTensor(name).descriptor.shape
                let linear = try adapter.linear(
                    module, base: store, inputWidth: shape[1], outputWidth: shape[0], bias: false, dataType: .float32)
                let values = linear.weight.reshaped(-1).take(MLXArray(probe.indices.map(Int32.init)), axis: 0)
                    .asArray(Float.self)
                for (actual, expected) in zip(values, probe.merged) {
                    // Summation order may round the float32 sum to the neighbouring bfloat16.
                    XCTAssertEqual(actual, expected, accuracy: abs(expected) * 0x1p-7 + 1e-9, name)
                }
                XCTAssertNotEqual(probe.base, probe.merged, "\(name): the adapter must change the weight")
            }
        }
    }

    func testHiddenStatesAndHeadMatchTransformersAtFloat32() throws {
        let fixture = try JEVTinyFixture.load()
        try Device.withDefaultDevice(device) {
            let backend = try tinyBackend(.float32)
            var worst: Float = 0
            for testCase in fixture.cases {
                let label = "length \(testCase.ids.count)"
                let hidden = try backend.trunkStates(testCase.ids)
                let rows = hidden.take(MLXArray(testCase.probeRows.map(Int32.init)), axis: 0)
                worst = max(worst, assertClose(rows.asArray(Float.self), testCase.layers["norm"]!.flatMap { $0 }, 1e-4, "\(label) final norm"))
                worst = max(worst, assertClose(hidden[testCase.ids.count - 1].asArray(Float.self), testCase.lastHidden, 1e-4, "\(label) last hidden"))

                let logits = try backend.headLogits([testCase.ids])
                worst = max(worst, assertClose(logits, testCase.headLogits, 1e-4, "\(label) head logits"))
                for (kind, count, expected) in [
                    (JEVKind.noul, 2, testCase.probabilities.noul),
                    (.score, 6, testCase.probabilities.score),
                    (.choice, fixture.choiceOptions, testCase.probabilities.choice),
                ] {
                    let start = backend.configuration.slots[kind]!.lowerBound
                    let p = DecoderAnswers.probabilities(
                        Array(logits[start ..< start + count]), temperature: backend.configuration.temperatures[kind]!)
                    assertClose(p.map(Float.init), expected, 1e-4, "\(label) \(kind) probabilities")
                }
            }
            print("JEV tiny float32 worst relative error: \(worst)")
        }
    }

    func testRightPaddedBatchesMatchSingleRows() throws {
        let fixture = try JEVTinyFixture.load()
        try Device.withDefaultDevice(device) {
            let backend = try tinyBackend(.float32)
            let batched = try backend.headLogits(fixture.cases.map(\.ids))
            for (index, testCase) in fixture.cases.enumerated() {
                assertClose(Array(batched[(index * 24) ..< ((index + 1) * 24)]), testCase.headLogits, 1e-4,
                            "row \(index) of the padded batch")
            }
        }
    }

    func testBFloat16StaysNearTheFloat32Reference() throws {
        let fixture = try JEVTinyFixture.load()
        try Device.withDefaultDevice(device) {
            let backend = try tinyBackend(.bfloat16)
            var worst = 0.0
            for testCase in fixture.cases {
                let logits = try backend.headLogits([testCase.ids])
                let p = DecoderAnswers.probabilities(Array(logits[0 ..< 2]), temperature: backend.configuration.temperatures[.noul]!)
                worst = max(worst, zip(p, testCase.probabilities.noul).map { abs($0 - Double($1)) }.max()!)
            }
            print("JEV tiny bfloat16 worst noul probability error: \(worst)")
            XCTAssertLessThan(worst, 0.05)
        }
    }

    func testTheTrunkSkipsTheUnusedHeadAndRequiresTheWholeAdapter() throws {
        try Device.withDefaultDevice(device) {
            let folder = JEVTinyFixture.folder
            let trunk = try Qwen35Trunk(
                store: SafetensorsCheckpoint(modelFolder: folder), configuration: Qwen35Configuration(modelFolder: folder),
                adapter: LoRAAdapter(folder: folder.appendingPathComponent("adapter")), loadsReadout: false
            )
            XCTAssertThrowsError(try trunk.readoutRows([0])) { XCTAssertEqual($0 as? DecoderError, .noReadout) }

            // An adapter targeting fewer modules than its file holds leaves tensors unused.
            let store = try SafetensorsFile(contentsOf: folder.appendingPathComponent("adapter/adapter_model.safetensors"))
            let partial = try LoRAAdapter(config: ["peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]], store: store)
            XCTAssertThrowsError(try Qwen35Trunk(
                store: SafetensorsCheckpoint(modelFolder: folder), configuration: Qwen35Configuration(modelFolder: folder),
                adapter: partial, loadsReadout: false
            )) {
                guard case .unusedTensors? = $0 as? LoRAAdapterError else { return XCTFail("\($0)") }
            }
        }
    }

    private func tinyBackend(_ dataType: DType) throws -> JEVBackend {
        let folder = JEVTinyFixture.folder
        let trunk = try Qwen35Trunk(
            store: SafetensorsCheckpoint(modelFolder: folder), configuration: Qwen35Configuration(modelFolder: folder),
            dataType: dataType, adapter: LoRAAdapter(folder: folder.appendingPathComponent("adapter")), loadsReadout: false
        )
        return try JEVBackend(
            modelID: "jev-tiny", configuration: JEVConfiguration(modelFolder: folder), tokenizer: ScalarTokenizer(),
            maxLength: 4_096, trunk: trunk, head: SafetensorsFile(contentsOf: folder.appendingPathComponent("head.safetensors"))
        )
    }
}

/// Fails when `actual` and `expected` differ by more than `tolerance` times
/// `max(1, max |expected|)`; returns that scaled worst error.
@discardableResult
private func assertClose(
    _ actual: [Float], _ expected: [Float], _ tolerance: Float, _ message: String,
    file: StaticString = #filePath, line: UInt = #line
) -> Float {
    XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
    let scale = max(1, expected.map(abs).max() ?? 0)
    var worst: Float = 0
    for (a, e) in zip(actual, expected) { worst = max(worst, a.isNaN ? .infinity : abs(a - e)) }
    XCTAssertLessThanOrEqual(worst / scale, tolerance, "\(message): worst error \(worst) (scale \(scale))", file: file, line: line)
    return worst / scale
}
