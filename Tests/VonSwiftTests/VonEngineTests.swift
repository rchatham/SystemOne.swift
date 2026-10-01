import Foundation
import MLX
import XCTest
@testable import VonSwift

final class VonEngineTests: XCTestCase {
    func testPublicAPIMatchesFourGoldenCasesInFiveSequentialForwards() async throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.cases.count, 4)
        XCTAssertEqual(fixture.forwards.count, 5)

        let recorder = GoldenLogitsRecorder(forwards: fixture.forwards)
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let engine = VonEngine(tokenizer: tokenizer) { input in
            try await recorder.next(input)
        }
        let questions = fixture.cases.map { testCase in
            QuestionEntry(id: testCase.id, question: question(for: testCase))
        }
        let response = try await engine.systemOne(
            state: .text(fixture.cases[0].state),
            questions: [questions[0]]
        )
        XCTAssertEqual(response.model, VonEngine.modelID)
        try assertAnswer(response.answers[0].answer, matches: fixture.cases[0].answerJSON)

        // The remaining cases have different states and exercise the convenience
        // methods while sharing the same ordered five-forward recorder.
        let defaultNoul = try await engine.judge(
            state: .text(fixture.cases[1].state),
            instructions: fixture.cases[1].instructions
        )
        XCTAssertEqual(defaultNoul, answerNumber(fixture.cases[1].answerJSON, key: "noul"))

        let explicitOptions = fixture.cases[2].orderedOptions
        let explicitNoul = try await engine.judge(
            state: .text(fixture.cases[2].state),
            instructions: fixture.cases[2].instructions,
            criteria: NoulCriteria(
                true: explicitOptions[0].description,
                false: explicitOptions[1].description
            )
        )
        XCTAssertEqual(explicitNoul, answerNumber(fixture.cases[2].answerJSON, key: "noul"))

        let score = try await engine.rate(
            state: .text(fixture.cases[3].state),
            criteria: fixture.cases[3].orderedOptions.map { .text($0.description) },
            instructions: fixture.cases[3].instructions
        )
        try assertAnswer(.score(score), matches: fixture.cases[3].answerJSON)

        let goldenCallCount = await recorder.callCount()
        XCTAssertEqual(goldenCallCount, 5)
    }

    func testSystemOneEvaluatesMultipleQuestionsInOrderAndComputesUsage() async throws {
        let recorder = ConstantLogitsRecorder()
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let engine = VonEngine(tokenizer: tokenizer) { input in
            await recorder.logits(for: input)
        }
        let response = try await engine.systemOne(
            state: .fields([
                VonStateField(key: "b", value: "2"),
                VonStateField(key: "a", value: "1"),
            ]),
            questions: [
                QuestionEntry(id: "first", question: .choice(Choice(
                    instructions: "one",
                    criteria: [ChoiceOption(key: "yes"), ChoiceOption(key: "no")]
                ))),
                QuestionEntry(id: "second", question: .noul(Noul(
                    instructions: "two",
                    criteria: NoulCriteria(true: " ")
                ))),
                QuestionEntry(id: "third", question: .score(Score(
                    instructions: "three",
                    criteria: [.text("low"), .text("high")]
                ))),
            ]
        )

        XCTAssertEqual(response.answers.map(\.id), ["first", "second", "third"])
        XCTAssertEqual(response.usage, Usage(inputTokens: 4, outputTokens: 3))
        let texts = await recorder.packedTexts()
        XCTAssertEqual(texts.count, 3, "truthy partial Noul criteria must skip the null forward")
        XCTAssertTrue(texts.allSatisfy { $0.contains("b: 2\na: 1") })
        XCTAssertTrue(texts[1].contains("[MASK]  [MASK] No, condition is false."))
    }

    func testEmptyNoulCriteriaRunsNullForwardButPartialCriteriaDoesNot() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())

        let defaultRecorder = ConstantLogitsRecorder()
        let defaultEngine = VonEngine(tokenizer: tokenizer) { input in
            await defaultRecorder.logits(for: input)
        }
        _ = try await defaultEngine.judge(
            state: .text("state"),
            instructions: "condition",
            criteria: NoulCriteria(true: "", false: "")
        )
        let defaultCallCount = await defaultRecorder.callCount()
        XCTAssertEqual(defaultCallCount, 2)

        let partialRecorder = ConstantLogitsRecorder()
        let partialEngine = VonEngine(tokenizer: tokenizer) { input in
            await partialRecorder.logits(for: input)
        }
        _ = try await partialEngine.judge(
            state: .text("state"),
            instructions: "condition",
            criteria: NoulCriteria(false: "custom")
        )
        let partialCallCount = await partialRecorder.callCount()
        XCTAssertEqual(partialCallCount, 1)
    }

    func testProductionPublicAPIParityWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["VON_PUBLIC_API_PARITY"] == "1" else {
            throw XCTSkip("Set VON_PUBLIC_API_PARITY=1 to run public API parity.")
        }
        guard let weightsPath = environment["VON_PRODUCTION_WEIGHTS_PATH"], !weightsPath.isEmpty else {
            throw XCTSkip("Set VON_PRODUCTION_WEIGHTS_PATH to the verified production artifact.")
        }
        let deviceName = environment["VON_PUBLIC_API_DEVICE"] ?? "cpu"
        let device: Device
        switch deviceName {
        case "cpu": device = .cpu
        case "gpu": device = .gpu
        default: return XCTFail("VON_PUBLIC_API_DEVICE must be 'cpu' or 'gpu', got \(deviceName)")
        }
        let fixture = try loadFixture()

        try await Device.withDefaultDevice(device) {
            let engine = try await VonEngine(
                tokenizerFolder: tokenizerResourceURL(),
                verifiedWeightsURL: URL(fileURLWithPath: weightsPath)
            )

            let choiceCase = fixture.cases[0]
            let choice = try await engine.decide(
                state: .text(choiceCase.state),
                choices: choiceCase.orderedOptions.map {
                    ChoiceOption(key: $0.key, description: $0.description)
                },
                instructions: choiceCase.instructions
            )
            try assertAnswer(.choice(choice), matches: choiceCase.answerJSON)

            let defaultCase = fixture.cases[1]
            let defaultNoul = try await engine.judge(
                state: .text(defaultCase.state),
                instructions: defaultCase.instructions
            )
            XCTAssertEqual(defaultNoul, answerNumber(defaultCase.answerJSON, key: "noul"))

            let explicitCase = fixture.cases[2]
            let explicitNoul = try await engine.judge(
                state: .text(explicitCase.state),
                instructions: explicitCase.instructions,
                criteria: NoulCriteria(
                    true: explicitCase.orderedOptions[0].description,
                    false: explicitCase.orderedOptions[1].description
                )
            )
            XCTAssertEqual(explicitNoul, answerNumber(explicitCase.answerJSON, key: "noul"))

            let scoreCase = fixture.cases[3]
            let score = try await engine.rate(
                state: .text(scoreCase.state),
                criteria: scoreCase.orderedOptions.map { .text($0.description) },
                instructions: scoreCase.instructions
            )
            try assertAnswer(.score(score), matches: scoreCase.answerJSON)
        }
    }

    func testLiteralMaskInjectionIsRejectedBeforeProviderInvocation() async throws {
        let recorder = ConstantLogitsRecorder()
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let engine = VonEngine(tokenizer: tokenizer) { input in
            await recorder.logits(for: input)
        }
        let expected = VonEngineError.unexpectedMaskCount(expected: 2, actual: 3)

        await assertEngineError(expected) {
            _ = try await engine.decide(
                state: .text("state [MASK] injected"),
                choices: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]
            )
        }
        await assertEngineError(expected) {
            _ = try await engine.decide(
                state: .text("state"),
                choices: [ChoiceOption(key: "a"), ChoiceOption(key: "b")],
                instructions: "choose [MASK] injected"
            )
        }
        await assertEngineError(expected) {
            _ = try await engine.decide(
                state: .text("state"),
                choices: [
                    ChoiceOption(key: "a", description: "option [MASK] injected"),
                    ChoiceOption(key: "b"),
                ]
            )
        }
        await assertEngineError(expected) {
            _ = try await engine.rate(
                state: .text("state"),
                criteria: [.text("level [MASK] injected"), .text("other")]
            )
        }

        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 0)
    }

    func testValidationRejectsAmbiguousOrEmptyOrderedInputBeforeInference() async throws {
        let recorder = ConstantLogitsRecorder()
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let engine = VonEngine(tokenizer: tokenizer) { input in
            await recorder.logits(for: input)
        }

        await assertEngineError(.duplicateStateFieldKey("a")) {
            _ = try await engine.systemOne(
                state: .fields([
                    VonStateField(key: "a", value: "1"),
                    VonStateField(key: "a", value: "2"),
                ]),
                questions: []
            )
        }
        await assertEngineError(.emptyChoiceOptions) {
            _ = try await engine.decide(state: .text("x"), choices: [])
        }
        await assertEngineError(.emptyScoreLevels) {
            _ = try await engine.rate(state: .text("x"), criteria: [])
        }
        await assertEngineError(.duplicateChoiceKey("a")) {
            _ = try await engine.decide(
                state: .text("x"),
                choices: [ChoiceOption(key: "a"), ChoiceOption(key: "a")]
            )
        }
        await assertEngineError(.duplicateQuestionID("q")) {
            _ = try await engine.systemOne(
                state: .text("x"),
                questions: [
                    QuestionEntry(id: "q", question: .noul(Noul(instructions: "x"))),
                    QuestionEntry(id: "q", question: .noul(Noul(instructions: "y"))),
                ]
            )
        }
        let invalidCallCount = await recorder.callCount()
        XCTAssertEqual(invalidCallCount, 0)
    }
}

private actor GoldenLogitsRecorder {
    private let forwards: [GoldenForward]
    private var index = 0

    init(forwards: [GoldenForward]) {
        self.forwards = forwards
    }

    func next(_ input: VonPackedInput) throws -> [Float] {
        guard index < forwards.count else {
            XCTFail("Received unexpected forward \(index)")
            return []
        }
        let expected = forwards[index]
        guard input.text == expected.packedText,
              input.inputIDs == expected.inputIDs,
              input.attentionMask == expected.attentionMask,
              input.maskPositions == expected.maskPositions else {
            throw GoldenRecorderError.inputMismatch(index)
        }
        index += 1
        return expected.rawLogits.map(Float.init)
    }

    func callCount() -> Int { index }
}

private enum GoldenRecorderError: Error {
    case inputMismatch(Int)
}

private actor ConstantLogitsRecorder {
    private var texts: [String] = []

    func logits(for input: VonPackedInput) -> [Float] {
        texts.append(input.text)
        return [0, 0]
    }

    func callCount() -> Int { texts.count }
    func packedTexts() -> [String] { texts }
}

private struct GoldenFixture: Decodable, Sendable {
    let cases: [GoldenCase]
    let forwards: [GoldenForward]
}

private struct GoldenCase: Decodable, Sendable {
    let id: String
    let primitive: String
    let state: String
    let instructions: String
    let orderedOptions: [GoldenOption]
    let answerJSON: String

    enum CodingKeys: String, CodingKey {
        case id, primitive, state, instructions
        case orderedOptions = "ordered_options"
        case answerJSON = "answer_model_dump_json"
    }
}

private struct GoldenOption: Decodable, Sendable {
    let key: String
    let description: String
}

private struct GoldenForward: Decodable, Sendable {
    let packedText: String
    let inputIDs: [Int]
    let attentionMask: [Int]
    let maskPositions: [Int]
    let rawLogits: [Double]

    enum CodingKeys: String, CodingKey {
        case packedText = "packed_text"
        case inputIDs = "input_ids"
        case attentionMask = "attention_mask"
        case maskPositions = "mask_positions"
        case rawLogits = "raw_logits"
    }
}

private extension VonEngineTests {
    func tokenizerResourceURL() -> URL {
        (Bundle.module.resourceURL ?? Bundle.module.bundleURL)
            .appendingPathComponent("Tokenizer", isDirectory: true)
    }

    func loadFixture() throws -> GoldenFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("testdata/von_golden_v1.json")
        return try JSONDecoder().decode(GoldenFixture.self, from: Data(contentsOf: url))
    }

    func question(for testCase: GoldenCase) -> Question {
        switch testCase.primitive {
        case "choice":
            return .choice(Choice(
                instructions: testCase.instructions,
                criteria: testCase.orderedOptions.map {
                    ChoiceOption(key: $0.key, description: $0.description)
                }
            ))
        case "noul":
            let criteria: NoulCriteria? = testCase.id == "explicit_noul"
                ? NoulCriteria(
                    true: testCase.orderedOptions[0].description,
                    false: testCase.orderedOptions[1].description
                )
                : nil
            return .noul(Noul(instructions: testCase.instructions, criteria: criteria))
        case "score":
            return .score(Score(
                instructions: testCase.instructions,
                criteria: testCase.orderedOptions.map { .text($0.description) }
            ))
        default:
            preconditionFailure("Unexpected golden primitive \(testCase.primitive)")
        }
    }

    func assertAnswer(_ answer: Answer, matches expectedJSON: String) throws {
        var actualObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(answer)) as? [String: Any]
        )
        // Order metadata is a Swift Codable extension ignored by Python. Compare
        // the production answer fields byte-semantically through JSON objects.
        actualObject.removeValue(forKey: "probability_order")
        actualObject.removeValue(forKey: "legend_order")
        let expectedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(expectedJSON.utf8)) as? [String: Any]
        )
        XCTAssertEqual(actualObject as NSDictionary, expectedObject as NSDictionary)
    }

    func answerNumber(_ json: String, key: String) -> Double {
        let object = try! JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        return object[key] as! Double
    }

    func assertEngineError(
        _ expected: VonEngineError,
        operation: () async throws -> Void
    ) async {
        do {
            try await operation()
            XCTFail("Expected \(expected)")
        } catch {
            XCTAssertEqual(error as? VonEngineError, expected)
        }
    }
}
