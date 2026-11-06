import Foundation
import SystemOne
import XCTest
@testable import SystemOneJev

/// One token per Unicode scalar, so prompt lengths are easy to reason about.
private struct ScalarCoder: JevTextCoding {
    func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    func decode(_ ids: [Int]) -> String {
        String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0)) }))
    }
}

final class JevBackendTests: XCTestCase {
    func testConfigurationReadsTheTinyFolder() throws {
        let configuration = try JevConfiguration(modelFolder: JevTinyFixture.modelFolder)
        XCTAssertEqual(configuration.text.hiddenSize, 64)
        XCTAssertEqual(configuration.text.rotaryDimensions, 8)
        XCTAssertEqual(configuration.text.convolutionChannels, 2 * 32 + 64)
        XCTAssertEqual(configuration.slots(for: .noul), 0..<2)
        XCTAssertEqual(configuration.slots(for: .score), 2..<8)
        XCTAssertEqual(configuration.slots(for: .choice), 8..<24)
        XCTAssertEqual(configuration.temperatures, [.noul: 1.25, .choice: 0.8, .score: 1.1])
        XCTAssertEqual(configuration.adapter.scale, 2)
        XCTAssertEqual(Set(configuration.adapter.targetModules).count, 10)
    }

    func testConfigurationRejectsUnsupportedFiles() throws {
        func folder(editing name: String, _ edit: (inout [String: Any]) -> Void) throws -> URL {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("JevConfig-\(UUID().uuidString)")
            try FileManager.default.copyItem(at: JevTinyFixture.modelFolder, to: copy)
            addTeardownBlock { try? FileManager.default.removeItem(at: copy) }
            let url = copy.appendingPathComponent(name)
            var object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            edit(&object)
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            return copy
        }
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("config.json", { $0["model_type"] = "qwen3_next" }),
            ("config.json", { $0["attn_output_gate"] = false }),
            ("judge_config.json", { $0["slots"] = ["num_slots": 24, "template_version": "chat-v2",
                                                    "ranges": ["noul": [0, 2], "score": [2, 8], "choice": [8, 24]]] }),
            ("judge_config.json", { $0["slots"] = ["num_slots": 24, "template_version": "bare-v1",
                                                    "ranges": ["noul": [0, 2], "score": [2, 6], "choice": [8, 24]]] }),
            ("judge_config.json", { $0["adapter_subfolder"] = "../elsewhere" }),
            ("adapter/adapter_config.json", { $0["use_dora"] = true }),
            ("adapter/adapter_config.json", { $0["target_modules"] = ["q_proj", "lm_head"] }),
            ("calibration.json", { $0["per_kind"] = ["noul": 1.0, "choice": 1.0] }),
        ]
        for (index, (name, edit)) in cases.enumerated() {
            XCTAssertThrowsError(try JevConfiguration(modelFolder: folder(editing: name, edit)), "case \(index)") {
                XCTAssertTrue($0 is JevConfigurationError, "case \(index): \($0)")
            }
        }
    }

    func testPlansOnePromptPerQuestionAndAnswersWithCalibratedSoftmax() async throws {
        var seen: [JevRow] = []
        let backend = try makeBackend { rows in
            // The engine batches rows by length, so they arrive out of request order.
            seen += rows
            return rows.map { row in (0..<row.optionCount).map { Float($0) } }
        }
        let engine = SystemOneEngine(backend: backend)
        let levels: [ScoreLevel] = (0..<6).map { .text("level \($0)") }
        let response = try await engine.systemOne(
            state: .text("Ticket text"),
            questions: [
                QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Refund?"))),
                QuestionEntry(id: "team", question: .choice(Choice(instructions: "Team?", criteria: [
                    ChoiceOption(key: "billing", description: "money"), ChoiceOption(key: "tech"), ChoiceOption(key: "other"),
                ]))),
                QuestionEntry(id: "urgency", question: .score(Score(instructions: "Urgency?", criteria: levels))),
                QuestionEntry(id: "only", question: .choice(Choice(instructions: "Pick", criteria: [ChoiceOption(key: "x")]))),
            ]
        )

        let byKind = Dictionary(uniqueKeysWithValues: seen.map { ($0.kind, $0) })
        XCTAssertEqual(seen.count, 3, "the one-option choice runs no row")
        XCTAssertEqual(byKind.mapValues(\.optionCount), [.noul: 2, .choice: 3, .score: 6])
        XCTAssertEqual(
            ScalarCoder().decode(byKind[.choice]!.ids),
            "[kind] choice\n[state] Ticket text\n[question] Team?\n[options]\nA) billing: money\nB) tech\nC) other\n[decision]:"
        )
        XCTAssertEqual(response.usage.inputTokens, seen.reduce(0) { $0 + $1.ids.count })

        // Logits 0, 1, … divided by the tiny model's per-kind temperatures.
        func expected(_ count: Int, _ temperature: Float) -> [Double] {
            DecisionMath.softmax((0..<count).map(Float.init), temperature: temperature).map(Double.init)
        }
        guard case .noul(let noul)? = response["refund"] else { return XCTFail("refund") }
        XCTAssertEqual(noul.noul, expected(2, 1.25)[1], accuracy: 1e-7)
        guard case .choice(let team)? = response["team"] else { return XCTFail("team") }
        XCTAssertEqual(team.choice, "other")
        XCTAssertEqual(team.probabilities.map(\.probability), expected(3, 0.8))
        guard case .score(let urgency)? = response["urgency"] else { return XCTFail("urgency") }
        XCTAssertEqual(urgency.legend.map(\.description), (0..<6).map { "level \($0)" })
        XCTAssertEqual(urgency.score, DecisionMath.expectedIndex(expected(6, 1.1)), accuracy: 1e-9)
        guard case .choice(let only)? = response["only"] else { return XCTFail("only") }
        XCTAssertEqual(only.choice, "x")
        XCTAssertEqual(only.confidence, 1)
    }

    func testRejectsQuestionsOutsideTheHeadsFixedSlots() throws {
        let backend = try makeBackend()
        func plan(_ question: Question) throws {
            _ = try backend.plan(SystemOneRequest(state: .text("s"), questions: [QuestionEntry(id: "q", question: question)]),
                                 media: nil)
        }
        let seventeen = (0..<17).map { ChoiceOption(key: "o\($0)") }
        XCTAssertThrowsError(try plan(.choice(Choice(instructions: "q", criteria: seventeen))))
        XCTAssertNoThrow(try plan(.choice(Choice(instructions: "q", criteria: Array(seventeen.prefix(16))))))
        for count in [2, 5, 7] {
            XCTAssertThrowsError(try plan(.score(Score(instructions: "q", criteria: (0..<count).map { .text("\($0)") }))))
        }
        XCTAssertThrowsError(
            try backend.plan(SystemOneRequest(state: .text("s"), questions: []),
                             media: .images([RGBImage(width: 1, height: 1, pixels: [0, 0, 0])]))
        ) {
            XCTAssertEqual($0 as? SystemOneEngineError, .unsupportedMedia(model: "JEV-test"))
        }
    }

    func testLongStatesKeepTheirHeadAndTail() throws {
        let backend = try makeBackend(maxLength: 120)
        let question = Question.noul(Noul(instructions: "Q?"))
        let empty = JevPrompt.render(kind: .noul, state: "", question: "Q?", options: ["false", "true"]).unicodeScalars.count
        let state = String(repeating: "h", count: 100) + String(repeating: "m", count: 100) + String(repeating: "t", count: 100)

        let plan = try backend.plan(
            SystemOneRequest(state: .text(state), questions: [QuestionEntry(id: "q", question: question)]), media: nil)
        let prompt = ScalarCoder().decode(plan.rows[0].ids)
        XCTAssertEqual(plan.rows[0].ids.count, 120)
        let budget = 120 - empty
        let kept = String(repeating: "h", count: budget * 3 / 5) + String(repeating: "t", count: budget - budget * 3 / 5)
        XCTAssertEqual(prompt, JevPrompt.render(kind: .noul, state: kept, question: "Q?", options: ["false", "true"]))

        let tiny = try makeBackend(maxLength: empty - 1)
        XCTAssertThrowsError(try tiny.plan(
            SystemOneRequest(state: .text("s"), questions: [QuestionEntry(id: "q", question: question)]), media: nil)
        ) {
            XCTAssertEqual($0 as? JevPromptError, .promptTooLong(id: "q", tokens: empty, maximum: empty - 1))
        }
    }

    func testProfileSketchesTheSamePromptTheBackendRuns() throws {
        let profile = JevModelProfile(maxLength: 100)
        XCTAssertNil(profile.markerText)
        XCTAssertEqual(profile.maximumForwardUTF8Bytes, 800)
        XCTAssertTrue(profile.modelsJSON.contains("\"JEV-9B\""))
        let question = Question.choice(Choice(instructions: "Team?", criteria: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]))
        let sketch = profile.sketch(question, renderedState: profile.renderState(.text("S")))
        XCTAssertEqual(sketch.options, ["a", "b"])
        XCTAssertEqual(sketch.forwards, [JevPrompt.render(kind: .choice, state: "S", question: "Team?", options: ["a", "b"])])
    }

    private func makeBackend(
        maxLength: Int = 4_096,
        logits: @escaping ([JevRow]) async throws -> [[Float]] = { $0.map { Array(repeating: 0, count: $0.optionCount) } }
    ) throws -> JevBackend {
        JevBackend(
            modelID: "JEV-test",
            configuration: try JevConfiguration(modelFolder: JevTinyFixture.modelFolder),
            coder: ScalarCoder(),
            maxLength: maxLength,
            model: nil,
            logitsProvider: logits
        )
    }
}
