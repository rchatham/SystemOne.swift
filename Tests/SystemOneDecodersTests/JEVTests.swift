import CryptoKit
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

/// One token per UTF-8 byte, as a byte-level BPE falls back to for rare
/// characters: a slice that ends inside a character decodes to U+FFFD.
private struct ByteTokenizer: ReversibleDecoderTokenizer {
    func encode(_ text: String) -> [Int] { text.utf8.map(Int.init) }
    func decode(_ ids: [Int]) -> String { String(decoding: ids.map(UInt8.init), as: UTF8.self) }
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

    func testConfigurationRejectsFieldsThatChangeWhatTheSlotsMean() {
        let calibration: [String: Any] = ["per_kind": ["noul": 1.0, "choice": 1.0, "score": 1.0]]
        let letters = "ABCDEFGHIJKLMNOP".map(String.init)
        let sides = ["false", "true"], levels = ["0", "1", "2", "3", "4", "5"]
        let standard = ["noul": [0, 2], "score": [2, 8], "choice": [8, 24]]
        func judge(
            ranges: [String: [Int]] = standard, slots: Int = 24, verbalizers: Any? = nil, extra: [String: Any] = [:]
        ) -> [String: Any] {
            var object: [String: Any] = ["num_slots": slots, "template_version": "bare-v1", "ranges": ranges]
            object["verbalizers"] = verbalizers
            return ["slots": object].merging(extra) { $1 }
        }

        // JEV-9B's own values, and a config that says nothing.
        XCTAssertNoThrow(try JEVConfiguration(
            judge: judge(verbalizers: sides + levels + letters, extra: ["softcap": NSNull(), "weights_mode": "unmerged"]),
            calibration: calibration))
        XCTAssertNoThrow(try JEVConfiguration(judge: judge(), calibration: calibration))
        // The verbalizers follow the ranges, wherever those put a kind and however many choices they hold.
        let moved = ["choice": [0, 8], "score": [8, 14], "noul": [14, 16]]
        XCTAssertNoThrow(try JEVConfiguration(
            judge: judge(ranges: moved, slots: 16, verbalizers: Array(letters.prefix(8)) + levels + sides),
            calibration: calibration))

        for (label, judge) in [
            ("softcap", judge(extra: ["softcap": 30.0])),
            ("softcap object", judge(extra: ["softcap": ["kind": "tanh", "cap": 30] as [String: Any]])),
            ("merged weights", judge(extra: ["weights_mode": "merged"])),
            ("weights_mode not a string", judge(extra: ["weights_mode": 1])),
            ("noul sides swapped", judge(verbalizers: ["true", "false"] + levels + letters)),
            ("score levels reversed", judge(verbalizers: sides + Array(levels.reversed()) + letters)),
            ("choice letters shuffled", judge(verbalizers: sides + levels + ["B", "A"] + Array(letters.dropFirst(2)))),
            ("lowercase letters", judge(verbalizers: sides + levels + letters.map { $0.lowercased() })),
            ("a slot short", judge(verbalizers: Array((sides + levels + letters).dropLast()))),
            ("not strings", judge(verbalizers: Array(0..<24))),
            ("standard order under moved ranges",
             judge(ranges: moved, slots: 16, verbalizers: sides + levels + Array(letters.prefix(8)))),
        ] {
            XCTAssertThrowsError(try JEVConfiguration(judge: judge, calibration: calibration), label) {
                guard case .invalidConfiguration? = $0 as? JEVError else { return XCTFail("\(label): \($0)") }
            }
        }
    }

    func testArtifactsPinTheFilesBesideTheWeights() throws {
        let folder = JEVTinyFixture.folder
        func pin(_ name: String) throws -> PinnedArtifact {
            let data = try Data(contentsOf: folder.appendingPathComponent(name))
            return PinnedArtifact(
                byteCount: data.count, sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }
        // No weights file has these, so a backend that reached one would fail on it.
        let noWeights = PinnedArtifact(byteCount: 0, sha256: "")
        func artifacts(_ files: [String: PinnedArtifact]) -> JEVArtifacts {
            JEVArtifacts(shards: [:], adapter: noWeights, head: noWeights, files: files)
        }
        let names = ["config.json", "model.safetensors.index.json", "adapter/adapter_config.json",
                     "judge_config.json", "calibration.json"]
        let pins = try Dictionary(uniqueKeysWithValues: names.map { ($0, try pin($0)) })
        XCTAssertNoThrow(try artifacts(pins).verifyFiles(in: folder))
        XCTAssertNoThrow(try artifacts([:]).verifyFiles(in: folder))
        XCTAssertEqual(JEVArtifacts(shards: [:], adapter: noWeights, head: noWeights).files, [:])

        let calibration = pins["calibration.json"]!, judge = pins["judge_config.json"]!
        var resized = pins
        resized["calibration.json"] = PinnedArtifact(byteCount: calibration.byteCount + 1, sha256: calibration.sha256)
        let wrongSize = JEVError.pinnedFile(
            name: "calibration.json",
            reason: "it has \(calibration.byteCount) bytes; expected exactly \(calibration.byteCount + 1)")
        XCTAssertThrowsError(try artifacts(resized).verifyFiles(in: folder)) { XCTAssertEqual($0 as? JEVError, wrongSize) }

        // The same length with other bytes, as a retuned temperature would be.
        let zeros = String(repeating: "0", count: 64)
        var edited = pins
        edited["judge_config.json"] = PinnedArtifact(byteCount: judge.byteCount, sha256: zeros)
        XCTAssertThrowsError(try artifacts(edited).verifyFiles(in: folder)) {
            XCTAssertEqual($0 as? JEVError, .pinnedFile(
                name: "judge_config.json", reason: "its SHA-256 is \(judge.sha256); expected \(zeros)"))
        }
        // The tiny folder has no tokenizer.
        XCTAssertThrowsError(try artifacts(["tokenizer.json": judge]).verifyFiles(in: folder)) {
            XCTAssertEqual($0 as? JEVError, .pinnedFile(name: "tokenizer.json", reason: "it is missing or unreadable"))
        }

        // The backend checks these before it reads a config or opens a weights file.
        XCTAssertThrowsError(try JEVBackend(modelFolder: folder, artifacts: artifacts(resized))) {
            XCTAssertEqual($0 as? JEVError, wrongSize)
        }
    }

    func testJEV9BPinsEveryFileTheBackendReads() {
        let pinned = JEVArtifacts.jev9B
        XCTAssertEqual(Set(pinned.files.keys), [
            "config.json", "model.safetensors.index.json", "adapter/adapter_config.json",
            "judge_config.json", "calibration.json", "tokenizer.json", "tokenizer_config.json",
        ])
        XCTAssertEqual(pinned.shards.count, 5)
        // tools/jev_golden_capture.py pins the same tokenizer.
        XCTAssertEqual(
            pinned.files["tokenizer.json"]?.sha256, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523")
        for (name, pin) in pinned.files {
            XCTAssertGreaterThan(pin.byteCount, 0, name)
            XCTAssertEqual(pin.sha256.count, 64, name)
        }
    }

    func testRendersTheModelCardTemplate() throws {
        XCTAssertEqual(
            try JEVPrompt.render(
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
            try JEVPrompt.render(kind: .noul, state: "S", question: "Q?", options: ["false", "true"]),
            "[kind] noul\n[state] S\n[question] Q?\n[options]\nfalse\ntrue\n[decision]:"
        )
        let sixteen = try JEVPrompt.render(kind: .choice, state: "", question: "q", options: (0..<16).map { "o\($0)" })
        XCTAssertTrue(sixteen.hasSuffix("O) o14\nP) o15\n[decision]:"))

        // There is no seventeenth label: a longer choice is an error, not a trap.
        XCTAssertThrowsError(
            try JEVPrompt.render(kind: .choice, state: "", question: "q", options: (0..<17).map { "o\($0)" })
        ) {
            XCTAssertEqual($0 as? JEVError, .tooManyOptions(count: 17, maximum: 16))
        }
        // Only a choice labels its options.
        XCTAssertNoThrow(try JEVPrompt.render(kind: .score, state: "", question: "q", options: (0..<17).map(String.init)))
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
        // An option stays on its own line, whatever line breaks its text has.
        let broken = Question.choice(Choice(instructions: "q", criteria: [
            ChoiceOption(key: "billing", description: "Charges\n  and refunds"),
            ChoiceOption(key: "two\nlines", description: " \n "),
        ]))
        XCTAssertEqual(JEVPrompt.options(for: broken).keys, ["billing", "two\nlines"])
        XCTAssertEqual(JEVPrompt.options(for: broken).texts, ["billing: Charges and refunds", "two lines"])
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
        // A choice's descriptions are on its option lines already.
        XCTAssertEqual(JEVPrompt.foldedQuestion(questions[0]), "Which team?")
        // The labels are the option lines' own, in their order.
        XCTAssertEqual(JEVPrompt.foldedQuestion(questions[1]), "Is it urgent? (false: can wait; true: needs action today)")
        XCTAssertEqual(
            JEVPrompt.foldedQuestion(questions[2]),
            "How severe? (0: none; 1: cosmetic; 2: minor; 3: major; 4: critical, e.g. outage; 5: catastrophic)")
    }

    func testFoldingSkipsWhatIsMissingAndFlattensLines() {
        let noul = { (criteria: NoulCriteria?) in JEVPrompt.foldedQuestion(.noul(Noul(instructions: "Spam?", criteria: criteria))) }
        XCTAssertEqual(noul(nil), "Spam?")
        XCTAssertEqual(noul(NoulCriteria()), "Spam?")
        XCTAssertEqual(noul(NoulCriteria(true: "sells something", false: " ")), "Spam? (true: sells something)")
        XCTAssertEqual(noul(NoulCriteria(true: "an ad\n  or a\r\nlink")), "Spam? (true: an ad or a link)")

        let score = { (levels: [ScoreLevel]) in JEVPrompt.foldedQuestion(.score(Score(instructions: "Risk?", criteria: levels))) }
        XCTAssertEqual(score((0..<6).map { _ in .text("") }), "Risk?")
        XCTAssertEqual(
            score([.text("low"), .text(""), .described(what: "", examples: ["fraud", "abuse"]),
                   .described(what: "high", examples: []), .text(""), .text("severe\nharm")]),
            "Risk? (0: low; 2: e.g. fraud, abuse; 3: high; 5: severe harm)")
        XCTAssertEqual(JEVPrompt.foldedQuestion(.noul(Noul(instructions: "", criteria: NoulCriteria(true: "yes")))), "(true: yes)")
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
        // Each is the request's fault, with a reason an HTTP server can return.
        XCTAssertThrowsError(try plan(.choice(Choice(instructions: "q", criteria: (0..<17).map { ChoiceOption(key: "o\($0)") })))) {
            XCTAssertEqual(
                ($0 as? any RequestRejection)?.rejectionReason, "Question 'q' is invalid: a JEV choice takes 1 to 16 options")
        }
        XCTAssertThrowsError(try plan(.score(Score(instructions: "q", criteria: (0..<7).map { .text("\($0)") })))) {
            XCTAssertEqual(
                ($0 as? any RequestRejection)?.rejectionReason,
                "Question 'q' is invalid: a JEV score has exactly 6 levels, 0 to 5")
        }
        XCTAssertThrowsError(
            try backend.plan(SystemOneRequest(state: .text("s"), questions: []),
                             media: .images([RGBImage(width: 1, height: 1, pixels: [0, 0, 0])]))
        ) {
            XCTAssertEqual($0 as? SystemOneEngineError, .unsupportedMedia(model: "jev-test"))
            XCTAssertEqual(($0 as? any RequestRejection)?.rejectionReason, "Model jev-test does not accept this media.")
        }
        XCTAssertEqual(
            JEVError.promptTooLong(id: "q", tokens: 5_000, maximum: 4_096).rejectionReason,
            "Question q needs 5000 tokens without its state; the limit is 4096.")
        XCTAssertEqual(
            JEVError.tooManyOptions(count: 17, maximum: 16).rejectionReason,
            "A JEV choice lists at most 16 options; this one has 17.")
        // A broken model folder is the server's fault.
        XCTAssertNil(JEVError.invalidConfiguration("softcap").rejectionReason)
        XCTAssertNil(JEVError.invalidHead("shape").rejectionReason)
        XCTAssertNil(JEVError.pinnedFile(name: "calibration.json", reason: "it is missing or unreadable").rejectionReason)
    }

    func testLongStatesKeepTheirHeadAndTail() throws {
        let backend = try makeBackend(maxLength: 120)
        let empty = try JEVPrompt.render(kind: .noul, state: "", question: "Q?", options: ["false", "true"]).unicodeScalars.count
        let state = String(repeating: "h", count: 100) + String(repeating: "m", count: 100) + String(repeating: "t", count: 100)
        let entry = QuestionEntry(id: "q", question: .noul(Noul(instructions: "Q?")))

        let plan = try backend.plan(SystemOneRequest(state: .text(state), questions: [entry]), media: nil)
        XCTAssertEqual(plan.rows[0].ids.count, 120)
        let budget = 120 - empty
        let kept = String(repeating: "h", count: budget * 3 / 5) + String(repeating: "t", count: budget - budget * 3 / 5)
        XCTAssertEqual(ScalarTokenizer().decode(plan.rows[0].ids),
                       try JEVPrompt.render(kind: .noul, state: kept, question: "Q?", options: ["false", "true"]))

        let tight = try makeBackend(maxLength: empty - 1)
        XCTAssertThrowsError(try tight.plan(SystemOneRequest(state: .text("s"), questions: [entry]), media: nil)) {
            XCTAssertEqual($0 as? JEVError, .promptTooLong(id: "q", tokens: empty, maximum: empty - 1))
        }
    }

    func testLongStatesAreCutBetweenCharacters() throws {
        let tokenizer = ByteTokenizer()
        let empty = try JEVPrompt.render(kind: .noul, state: "", question: "Q?", options: ["false", "true"]).utf8.count
        // Three bytes a character at each end. A budget of 54 tokens keeps 32 at the
        // head and 22 at the tail: both cuts fall inside a character.
        let state = String(repeating: "日", count: 40) + String(repeating: "m", count: 60) + String(repeating: "本", count: 40)
        let stateIDs = tokenizer.encode(state)
        XCTAssertTrue(tokenizer.decode(Array(stateIDs[..<32])).hasSuffix("\u{FFFD}"))
        XCTAssertTrue(tokenizer.decode(Array(stateIDs[(stateIDs.count - 22)...])).hasPrefix("\u{FFFD}"))

        let backend = try makeBackend(maxLength: empty + 54, tokenizer: tokenizer)
        let entry = QuestionEntry(id: "q", question: .noul(Noul(instructions: "Q?")))
        let plan = try backend.plan(SystemOneRequest(state: .text(state), questions: [entry]), media: nil)
        let prompt = tokenizer.decode(plan.rows[0].ids)
        XCTAssertFalse(prompt.unicodeScalars.contains("\u{FFFD}"), prompt)
        // Ten whole characters of the head's 32 bytes, and seven of the tail's 22.
        let kept = String(repeating: "日", count: 10) + String(repeating: "本", count: 7)
        XCTAssertEqual(prompt, try JEVPrompt.render(kind: .noul, state: kept, question: "Q?", options: ["false", "true"]))
        XCTAssertLessThanOrEqual(plan.rows[0].ids.count, empty + 54)

        // A replacement character the state itself holds is not at a cut, so it stays.
        let marked = "\u{FFFD}" + String(repeating: "a", count: 200) + "\u{FFFD}"
        let markedPlan = try backend.plan(SystemOneRequest(state: .text(marked), questions: [entry]), media: nil)
        let markedPrompt = tokenizer.decode(markedPlan.rows[0].ids)
        XCTAssertTrue(markedPrompt.contains("[state] \u{FFFD}a"), markedPrompt)
        XCTAssertTrue(markedPrompt.contains("a\u{FFFD}\n[question]"), markedPrompt)
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

    private func makeBackend(
        maxLength: Int = 4_096, tokenizer: any ReversibleDecoderTokenizer = ScalarTokenizer()
    ) throws -> JEVBackend {
        try JEVBackend(
            modelID: "jev-test", configuration: JEVConfiguration(modelFolder: JEVTinyFixture.folder),
            tokenizer: tokenizer, maxLength: maxLength, trunk: nil, head: nil
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
