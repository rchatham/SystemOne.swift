import Foundation
import MLX
import XCTest
@testable import SystemOne

final class VonEngineTests: XCTestCase {
    func testPublicAPIMatchesFourGoldenCasesInFiveSequentialForwards() async throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.cases.count, 4)
        XCTAssertEqual(fixture.forwards.count, 5)

        let recorder = GoldenLogitsRecorder(forwards: fixture.forwards)
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let engine = SystemOneEngine(tokenizer: tokenizer) { input in
            try await recorder.next(input)
        }
        let questions = fixture.cases.map { testCase in
            QuestionEntry(id: testCase.id, question: question(for: testCase))
        }
        let response = try await engine.systemOne(
            state: .text(fixture.cases[0].state),
            questions: [questions[0]]
        )
        XCTAssertEqual(response.model, SystemOneEngine.modelID)
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
        let engine = SystemOneEngine(tokenizer: tokenizer) { input in
            await recorder.logits(for: input)
        }
        let response = try await engine.systemOne(
            state: .fields([
                StateField(key: "b", value: "2"),
                StateField(key: "a", value: "1"),
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
        let defaultEngine = SystemOneEngine(tokenizer: tokenizer) { input in
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
        let partialEngine = SystemOneEngine(tokenizer: tokenizer) { input in
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

    func testHeterogeneousBatchingMatchesLegacySequentialBaseline() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let legacy = SystemOneEngine(tokenizer: tokenizer) { input in
            deterministicLogits(for: input)
        }
        let recorder = DeterministicBatchRecorder()
        let batched = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let questions = heterogeneousQuestions()
        let baseline = try await legacy.systemOne(
            state: .text("abcd"), questions: questions
        )
        let response = try await batched.systemOne(
            state: .text("abcd"), questions: questions, batchingStrategy: .batched
        )

        XCTAssertEqual(response, baseline)
        XCTAssertEqual(response.answers.map(\.id), ["choice", "default", "partial", "score"])
        let chunks = await recorder.chunks()
        let inputs = chunks.flatMap { $0 }
        let lengths = inputs.map { $0.inputIDs.count }
        XCTAssertEqual(inputs.count, 5, "all primary and default-Noul null rows must be preplanned")
        XCTAssertEqual(lengths, lengths.sorted(by: >))
        let defaultRows = inputs.filter { $0.text.contains("Default") }
        XCTAssertEqual(defaultRows.count, 2)
        XCTAssertEqual(defaultRows.filter { $0.text.contains("abcd") }.count, 1)
        XCTAssertEqual(defaultRows.filter { !$0.text.contains("abcd") }.count, 1)
    }

    func testHomogeneousRowsUseOneBatchProviderCall() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = DeterministicBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let questions = (0..<4).map { index in
            QuestionEntry(
                id: "q\(index)",
                question: .noul(Noul(
                    instructions: "same",
                    criteria: NoulCriteria(true: "yes")
                ))
            )
        }

        let response = try await engine.systemOne(
            state: .text("state"), questions: questions, batchingStrategy: .batched
        )

        XCTAssertEqual(response.answers.map(\.id), ["q0", "q1", "q2", "q3"])
        let chunks = await recorder.chunks()
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].count, 4)
    }

    func testExecutionCallsBatchProviderOncePerExactGreedyBucket() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = DeterministicBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let wordCounts = [1_400, 700, 700, 50]
        let questions = wordCounts.indices.map { index in
            QuestionEntry(
                id: "q\(index)",
                question: .noul(Noul(
                    instructions: "bucket-\(index) " + String(repeating: "word ", count: wordCounts[index]),
                    criteria: NoulCriteria(true: "yes")
                ))
            )
        }

        _ = try await engine.systemOne(
            state: .text("state"), questions: questions, batchingStrategy: .batched
        )

        let chunks = await recorder.chunks()
        let inputsByIndex = try Dictionary(uniqueKeysWithValues: chunks.flatMap { $0 }.map { input in
            let index = try XCTUnwrap(
                wordCounts.indices.first { input.text.contains("bucket-\($0) ") }
            )
            return (index, input)
        })
        let lengths = try wordCounts.indices.map { try XCTUnwrap(inputsByIndex[$0]).inputIDs.count }
        let expected = SystemOneEngine.microbatchIndices(sequenceLengths: lengths, maxAmplification: 2.0)
        let actual = try chunks.map { chunk in
            try chunk.map { input in
                try XCTUnwrap(wordCounts.indices.first { input.text.contains("bucket-\($0) ") })
            }
        }
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertEqual(actual, expected)
    }

    func testMicrobatchIndicesUsesExactSortedGreedyBounds() {
        // batched (maxAmplification 2.0)
        XCTAssertEqual(SystemOneEngine.microbatchIndices(sequenceLengths: [], maxAmplification: 2.0), [])
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [800, 700, 700, 100], maxAmplification: 2.0),
            [[0], [1, 2], [3]]
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [3, 5, 5, 3], maxAmplification: 2.0),
            [[1, 2, 0, 3]],
            "equal lengths must retain their original-index order"
        )

        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [512, 512, 512, 512], maxAmplification: 2.0),
            [[0, 1, 2, 3]],
            "the absolute padded-work bound is inclusive"
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [512, 512, 512, 512, 512], maxAmplification: 2.0),
            [[0, 1, 2, 3], [4]]
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [500, 100, 100], maxAmplification: 2.0),
            [[0, 1], [2]],
            "padding may not exceed twice the useful squared work"
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [2, 1, 1], maxAmplification: 2.0),
            [[0, 1, 2]],
            "the padding-efficiency bound is inclusive"
        )

        let boundedLengths = [2_000, 800, 700, 700, 100, 3, 3]
        for batch in SystemOneEngine.microbatchIndices(sequenceLengths: boundedLengths, maxAmplification: 2.0) {
            let squares = batch.map { boundedLengths[$0] * boundedLengths[$0] }
            let maximumSquared = try! XCTUnwrap(squares.max())
            let paddedWork = batch.count * maximumSquared
            XCTAssertLessThanOrEqual(paddedWork, max(maximumSquared, 1_048_576))
            XCTAssertLessThanOrEqual(paddedWork, 2 * squares.reduce(0, +))
        }
    }

    func testMicrobatchIndicesConservativelySingletonsOverflowAndOversizeRows() {
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [Int.max, Int.max - 1, 2_000, 1], maxAmplification: 2.0),
            [[0], [1], [2], [3]]
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [1_700_000_000, 1_700_000_000], maxAmplification: 2.0),
            [[0], [1]],
            "oversize rows remain singletons even when the useful-work bound overflows"
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [3_000_000_000, 3_000_000_000], maxAmplification: 2.0),
            [[0], [1]],
            "overflowing padded work disallows appending to an oversize singleton"
        )
        let first = SystemOneEngine.microbatchIndices(sequenceLengths: [8, 2, 7, 1, 1, 6], maxAmplification: 2.0)
        let second = SystemOneEngine.microbatchIndices(sequenceLengths: [8, 2, 7, 1, 1, 6], maxAmplification: 2.0)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.flatMap { $0 }.sorted(), Array(0..<6))
    }

    func testEmptyRequestNeverInvokesBatchProvider() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = BatchLogitsRecorder(rowLogits: [])
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            try await recorder.next(inputs)
        }

        let response = try await engine.systemOne(state: .text("state"), questions: [])
        XCTAssertEqual(response.answers, [])
        XCTAssertEqual(response.usage, Usage(inputTokens: 2, outputTokens: 0))
        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 0)
    }

    func testBatchProviderContractRejectsMalformedLogitsAndOffsets() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        func assertRejected(
            output: BatchLogitsOutput,
            expected: SystemOneEngineError
        ) async {
            let engine = SystemOneEngine(tokenizer: tokenizer) { (_: [VonPackedInput]) in output }
            await assertEngineError(expected) {
                _ = try await engine.decide(
                    state: .text("x"),
                    choices: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]
                )
            }
        }

        await assertRejected(
            output: BatchLogitsOutput(logits: [0], rowOffsets: [0, 2]),
            expected: .unexpectedLogitCount(expected: 2, actual: 1)
        )
        await assertRejected(
            output: BatchLogitsOutput(logits: [0, 1, 2], rowOffsets: [0, 2]),
            expected: .unexpectedLogitCount(expected: 2, actual: 3)
        )
        for offsets in [[0], [0, 2, 2], [1, 2], [0, 1], [0, 2, 1], [0, 3]] {
            await assertRejected(
                output: BatchLogitsOutput(logits: [.nan, 0], rowOffsets: offsets),
                expected: .unexpectedBatchOffsets(expected: [0, 2], actual: offsets)
            )
        }

        let twoRowEngine = SystemOneEngine(tokenizer: tokenizer) { (_: [VonPackedInput]) in
            BatchLogitsOutput(logits: [0, 1, 2, 3], rowOffsets: [0, 1, 4])
        }
        await assertEngineError(
            .unexpectedBatchOffsets(expected: [0, 2, 4], actual: [0, 1, 4])
        ) {
            _ = try await twoRowEngine.systemOne(
                state: .text("x"),
                questions: [
                    QuestionEntry(id: "a", question: .noul(Noul(
                        instructions: "a",
                        criteria: NoulCriteria(true: "yes")
                    ))),
                    QuestionEntry(id: "b", question: .noul(Noul(
                        instructions: "b",
                        criteria: NoulCriteria(false: "no")
                    ))),
                ]
            )
        }
    }

    func testLegacyProviderPreservesFailFastDecisionMathBehavior() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = NonFiniteLegacyRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { input in
            await recorder.next(input)
        }
        let questions = [
            QuestionEntry(id: "bad", question: .choice(Choice(
                instructions: "bad",
                criteria: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]
            ))),
            QuestionEntry(id: "later", question: .choice(Choice(
                instructions: "later",
                criteria: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]
            ))),
        ]

        do {
            _ = try await engine.systemOne(state: .text("state"), questions: questions)
            XCTFail("Expected non-finite-logit failure")
        } catch VonDecisionMathError.nonFiniteLogit(let index, let value) {
            XCTAssertEqual(index, 0)
            XCTAssertTrue(value.isNaN)
        } catch {
            XCTFail("Expected nonFiniteLogit, got \(error)")
        }
        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 1, "later questions must not invoke the legacy provider")
    }

    func testLaterBatchProviderErrorStopsAndPropagates() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = FailingBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            try await recorder.next(inputs)
        }

        do {
            let longInstructions = String(repeating: "word ", count: 700)
            let questions = (0..<5).map { index in
                QuestionEntry(
                    id: "q\(index)",
                    question: .noul(Noul(
                        instructions: longInstructions,
                        criteria: NoulCriteria(true: "yes")
                    ))
                )
            }
            _ = try await engine.systemOne(state: .text("x"), questions: questions)
            XCTFail("Expected provider error")
        } catch {
            XCTAssertTrue(error is BatchRecorderError)
        }
        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 2)
    }

    func testCancellationBetweenBatchesStopsBeforeNextProviderCall() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = GatedBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let longInstructions = String(repeating: "word ", count: 700)
        let questions = (0..<5).map { index in
            QuestionEntry(
                id: "q\(index)",
                question: .noul(Noul(
                    instructions: longInstructions,
                    criteria: NoulCriteria(true: "yes")
                ))
            )
        }
        let task = Task { try await engine.systemOne(state: .text("x"), questions: questions) }

        while await recorder.callCount() == 0 { await Task.yield() }
        task.cancel()
        await recorder.releaseFirstCall()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            // Expected between stable sequential chunks.
        } catch {
            XCTFail("Expected CancellationError, got \(error)")
        }
        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 1)
    }

    func testPreCancelledRequestMakesNoProviderCalls() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = BatchLogitsRecorder(rowLogits: [])
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            try await recorder.next(inputs)
        }
        let task = Task {
            while !Task.isCancelled { await Task.yield() }
            return try await engine.decide(
                state: .text("state"),
                choices: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]
            )
        }
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            // Expected before the provider seam is called.
        } catch {
            XCTFail("Expected CancellationError, got \(error)")
        }
        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 0)
    }

    func testIdenticalDefaultNoulRowsScatterDistinctLogitsToPlannedReferences() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = DuplicateDefaultNoulRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let questions = [
            QuestionEntry(id: "first", question: .noul(Noul(instructions: "identical"))),
            QuestionEntry(id: "second", question: .noul(Noul(instructions: "identical"))),
        ]

        let response = try await engine.systemOne(state: .text("same state"), questions: questions)
        guard case .noul(let first) = response.answers[0].answer,
              case .noul(let second) = response.answers[1].answer else {
            return XCTFail("Expected Noul answers")
        }
        XCTAssertGreaterThan(first.noul, 0.5)
        XCTAssertLessThan(second.noul, 0.5)
        let texts = await recorder.texts()
        XCTAssertEqual(texts.count, 4)
        XCTAssertEqual(texts[0], texts[1], "duplicate primary rows must be byte-identical")
        XCTAssertEqual(texts[2], texts[3], "duplicate null rows must be byte-identical")
        XCTAssertNotEqual(texts[0], texts[2])
    }

    func testLateMaskInjectionAndDuplicateValidationMakeNoBatchCalls() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = BatchLogitsRecorder(rowLogits: [])
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            try await recorder.next(inputs)
        }

        await assertEngineError(.unexpectedMaskCount(expected: 2, actual: 3)) {
            _ = try await engine.systemOne(
                state: .text("state"),
                questions: [
                    QuestionEntry(id: "first", question: .noul(Noul(
                        instructions: "valid",
                        criteria: NoulCriteria(true: "yes")
                    ))),
                    QuestionEntry(id: "later", question: .noul(Noul(
                        instructions: "invalid [MASK]",
                        criteria: nil
                    ))),
                ]
            )
        }
        await assertEngineError(.duplicateQuestionID("same")) {
            _ = try await engine.systemOne(
                state: .text("state"),
                questions: [
                    QuestionEntry(id: "same", question: .noul(Noul(instructions: "a"))),
                    QuestionEntry(id: "same", question: .score(Score(instructions: "b", criteria: [.text("one")]))),
                ]
            )
        }
        let callCount = await recorder.callCount()
        XCTAssertEqual(callCount, 0)
    }

    func testBatchGoldenFixtureDecodesHeterogeneousRequestSchema() throws {
        let fixture = try loadBatchFixture()
        let request = fixture.heterogeneousRequest

        XCTAssertEqual(fixture.schemaVersion, 1)
        XCTAssertEqual(request.model, SystemOneEngine.modelID)
        XCTAssertEqual(request.questions.map(\.id), request.questionIDs)
        XCTAssertEqual(request.answers.map(\.questionID), request.questionIDs)
        XCTAssertEqual(request.usage, Usage(inputTokens: 82, outputTokens: 4))
        guard case .choice(let choice) = request.questions[0].question else {
            return XCTFail("Expected the first golden question to be Choice")
        }
        XCTAssertEqual(
            choice.criteria.map(\.key),
            ["infrastructure", "billing", "feature_request", "security"]
        )
    }

    func testMicrobatchIndicesAutomaticConservativeSplitsMoreThanBatched() {
        // automatic (1.5x) is more conservative than batched (2.0x)
        let lengths = [500, 100, 100, 100]
        let batched = SystemOneEngine.microbatchIndices(sequenceLengths: lengths, maxAmplification: 2.0)
        let automatic = SystemOneEngine.microbatchIndices(sequenceLengths: lengths, maxAmplification: 1.5)
        // batched may group [500, 100, 100] and [100]
        // automatic should not pack as aggressively with 1.5x bound
        XCTAssertGreaterThanOrEqual(automatic.count, batched.count,
            "automatic (1.5×) must split at least as much as batched (2.0×)")
    }

    func testMicrobatchIndicesSequentialReturnsAllSingletons() {
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [5, 3, 4], maxAmplification: nil),
            [[0], [2], [1]],
            "sequential (nil maxAmplification) must produce one singleton per row sorted descending"
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [512, 512, 512], maxAmplification: nil),
            [[0], [1], [2]],
            "sequential must singletons even for equal lengths"
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [1], maxAmplification: nil),
            [[0]]
        )
        XCTAssertEqual(
            SystemOneEngine.microbatchIndices(sequenceLengths: [], maxAmplification: nil),
            []
        )
    }

    func testSequentialStrategyInvokesProviderOncePerRow() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = DeterministicBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let questions = heterogeneousQuestions()
        let response = try await engine.systemOne(
            state: .text("abcd"), questions: questions, batchingStrategy: .sequential
        )
        XCTAssertEqual(response.answers.map(\.id), ["choice", "default", "partial", "score"])
        let chunks = await recorder.chunks()
        // 5 rows: choice(1), default-noul(2 primary+null), partial-noul(1), score(1)
        XCTAssertEqual(chunks.count, 5)
        XCTAssertTrue(chunks.allSatisfy { $0.count == 1 }, "each chunk must be a singleton")
    }

    func testAutomaticDefaultIsConservativeOnHeterogeneousInput() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = DeterministicBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        let questions = heterogeneousQuestions()
        let response = try await engine.systemOne(
            state: .text("abcd"), questions: questions
        )
        XCTAssertEqual(response.answers.map(\.id), ["choice", "default", "partial", "score"])
        let chunks = await recorder.chunks()
        let allInputs = chunks.flatMap { $0 }
        // 5 logical rows must all appear
        XCTAssertEqual(allInputs.count, 5)
        XCTAssertGreaterThanOrEqual(chunks.count, 1)
        // Verify the default noul null row is included (no state text)
        let nullRows = allInputs.filter { $0.text.contains("Default") && !$0.text.contains("abcd") }
        XCTAssertEqual(nullRows.count, 1, "null row for default noul must be present")
    }

    func testAllThreeStrategiesYieldIdenticalAnswers() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let questions = heterogeneousQuestions()

        // sequential
        let seqRecorder = DeterministicBatchRecorder()
        let seqEngine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await seqRecorder.next(inputs)
        }
        let seqResponse = try await seqEngine.systemOne(
            state: .text("abcd"), questions: questions, batchingStrategy: .sequential
        )

        // automatic
        let autoRecorder = DeterministicBatchRecorder()
        let autoEngine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await autoRecorder.next(inputs)
        }
        let autoResponse = try await autoEngine.systemOne(
            state: .text("abcd"), questions: questions, batchingStrategy: .automatic
        )

        // batched
        let batRecorder = DeterministicBatchRecorder()
        let batEngine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await batRecorder.next(inputs)
        }
        let batResponse = try await batEngine.systemOne(
            state: .text("abcd"), questions: questions, batchingStrategy: .batched
        )

        XCTAssertEqual(seqResponse, autoResponse)
        XCTAssertEqual(seqResponse, batResponse)
    }

    func testAutomaticHeterogeneousProducesSaferBucketsThanBatched() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let wordCounts = [1_000, 600, 600, 100]
        let questions = wordCounts.indices.map { index in
            QuestionEntry(
                id: "q\(index)",
                question: .noul(Noul(
                    instructions: "bucket-\(index) " + String(repeating: "word ", count: wordCounts[index]),
                    criteria: NoulCriteria(true: "yes")
                ))
            )
        }

        let autoRecorder = DeterministicBatchRecorder()
        let autoEngine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await autoRecorder.next(inputs)
        }
        _ = try await autoEngine.systemOne(
            state: .text("state"), questions: questions, batchingStrategy: .automatic
        )
        let autoChunks = await autoRecorder.chunks()

        let batRecorder = DeterministicBatchRecorder()
        let batEngine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await batRecorder.next(inputs)
        }
        _ = try await batEngine.systemOne(
            state: .text("state"), questions: questions, batchingStrategy: .batched
        )
        let batChunks = await batRecorder.chunks()

        // automatic (1.5×) must not produce fewer batches than batched (2.0×)
        // but both must cover all 4 rows
        let autoRowCount = autoChunks.flatMap { $0 }.count
        let batRowCount = batChunks.flatMap { $0 }.count
        XCTAssertEqual(autoRowCount, 4)
        XCTAssertEqual(batRowCount, 4)
        XCTAssertGreaterThanOrEqual(autoChunks.count, batChunks.count,
            "automatic (1.5×) must not merge more aggressively than batched (2.0×)"
        )
    }

    func testIdenticalNullPairingConsistentAcrossStrategies() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        // Two default Noul questions with same instructions
        let questions = [
            QuestionEntry(id: "first", question: .noul(Noul(instructions: "identical"))),
            QuestionEntry(id: "second", question: .noul(Noul(instructions: "identical"))),
        ]

        for strategy: BatchingStrategy in [.sequential, .automatic, .batched] {
            let recorder = DuplicateDefaultNoulRecorder()
            let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
                await recorder.next(inputs)
            }
            let response = try await engine.systemOne(
                state: .text("same state"), questions: questions, batchingStrategy: strategy
            )
            guard case .noul(let first) = response.answers[0].answer,
                  case .noul(let second) = response.answers[1].answer else {
                return XCTFail("Expected Noul answers for \(strategy)")
            }
            XCTAssertGreaterThan(first.noul, 0.5, "first should be high for \(strategy)")
            XCTAssertLessThan(second.noul, 0.5, "second should be low for \(strategy)")
            let texts = await recorder.texts()
            XCTAssertEqual(texts.count, 4, "all strategies produce 4 rows for \(strategy)")
        }
    }

    func testAutomaticHomogeneousOneBatchWhenConservativeGatePermits() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let recorder = DeterministicBatchRecorder()
        let engine = SystemOneEngine(tokenizer: tokenizer) { inputs in
            await recorder.next(inputs)
        }
        // Small homogeneous questions should fit in one batch even with 1.5× bound
        let questions = (0..<4).map { index in
            QuestionEntry(
                id: "q\(index)",
                question: .noul(Noul(
                    instructions: "same",
                    criteria: NoulCriteria(true: "yes")
                ))
            )
        }
        let response = try await engine.systemOne(
            state: .text("state"), questions: questions, batchingStrategy: .automatic
        )
        XCTAssertEqual(response.answers.map(\.id), ["q0", "q1", "q2", "q3"])
        let chunks = await recorder.chunks()
        XCTAssertEqual(chunks.count, 1, "homogeneous small inputs must batch with automatic")
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
        let batchFixture = try loadBatchFixture()

        try await Device.withDefaultDevice(device) {
            let engine = try await SystemOneEngine(
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

            let golden = batchFixture.heterogeneousRequest
            let batched = try await engine.systemOne(
                SystemOneRequest(
                    model: golden.model,
                    state: .text(golden.state),
                    questions: golden.questions.map {
                        QuestionEntry(id: $0.id, question: $0.question)
                    }
                )
            )
            XCTAssertEqual(batchFixture.schemaVersion, 1)
            XCTAssertEqual(batched.model, golden.model)
            XCTAssertEqual(batched.answers.map(\.id), golden.questionIDs)
            XCTAssertEqual(batched.usage, golden.usage)
            XCTAssertEqual(batched.answers.count, golden.answers.count)
            for (actual, expected) in zip(batched.answers, golden.answers) {
                XCTAssertEqual(actual.id, expected.questionID)
                try assertAnswer(actual.answer, matches: expected.answerJSON)
            }

            var sequentialAnswers: [AnswerEntry] = []
            for question in golden.questions {
                let response = try await engine.systemOne(
                    SystemOneRequest(
                        model: golden.model,
                        state: .text(golden.state),
                        questions: [QuestionEntry(
                            id: question.id,
                            question: question.question
                        )]
                    )
                )
                sequentialAnswers.append(try XCTUnwrap(response.answers.first))
            }
            XCTAssertEqual(
                batched.answers,
                sequentialAnswers,
                "batched and sequential production inference must round identically"
            )
        }
    }

    func testLiteralMaskInjectionIsRejectedBeforeProviderInvocation() async throws {
        let recorder = ConstantLogitsRecorder()
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let engine = SystemOneEngine(tokenizer: tokenizer) { input in
            await recorder.logits(for: input)
        }
        let expected = SystemOneEngineError.unexpectedMaskCount(expected: 2, actual: 3)

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
        let engine = SystemOneEngine(tokenizer: tokenizer) { input in
            await recorder.logits(for: input)
        }

        await assertEngineError(.duplicateStateFieldKey("a")) {
            _ = try await engine.systemOne(
                state: .fields([
                    StateField(key: "a", value: "1"),
                    StateField(key: "a", value: "2"),
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

private actor NonFiniteLegacyRecorder {
    private var calls = 0

    func next(_ input: VonPackedInput) -> [Float] {
        calls += 1
        if calls == 1 {
            return [.nan] + [Float](repeating: 0, count: input.maskPositions.count - 1)
        }
        return [Float](repeating: 0, count: input.maskPositions.count)
    }

    func callCount() -> Int { calls }
}

private func deterministicLogits(for input: VonPackedInput) -> [Float] {
    if input.text.contains("Choose") { return [0, 1, 5] }
    if input.text.contains("Default") { return input.text.contains("abcd") ? [5, 0] : [0, 0] }
    if input.text.contains("Partial") { return [0, 5] }
    if input.text.contains("Score") { return [0, 0, 0, 5] }
    return [Float](repeating: 0, count: input.maskPositions.count)
}

private actor DeterministicBatchRecorder {
    private var recordedChunks: [[VonPackedInput]] = []

    func next(_ inputs: [VonPackedInput]) -> BatchLogitsOutput {
        recordedChunks.append(inputs)
        let rows = inputs.map(deterministicLogits)
        let offsets = rows.reduce(into: [0]) { result, row in
            result.append(result[result.count - 1] + row.count)
        }
        return BatchLogitsOutput(logits: rows.flatMap { $0 }, rowOffsets: offsets)
    }

    func chunks() -> [[VonPackedInput]] { recordedChunks }
}

private actor DuplicateDefaultNoulRecorder {
    private var primaryCount = 0
    private var recordedTexts: [String] = []

    func next(_ inputs: [VonPackedInput]) -> BatchLogitsOutput {
        let rows = inputs.map { input -> [Float] in
            recordedTexts.append(input.text)
            guard input.text.contains("same state") else { return [0, 0] }
            defer { primaryCount += 1 }
            return primaryCount == 0 ? [5, 0] : [0, 5]
        }
        let offsets = rows.reduce(into: [0]) { result, row in
            result.append(result[result.count - 1] + row.count)
        }
        return BatchLogitsOutput(logits: rows.flatMap { $0 }, rowOffsets: offsets)
    }

    func texts() -> [String] { recordedTexts }
}

private actor BatchLogitsRecorder {
    private let rowLogits: [[Float]]
    private var nextRow = 0
    private var recordedChunks: [[VonPackedInput]] = []

    init(rowLogits: [[Float]]) {
        self.rowLogits = rowLogits
    }

    func next(_ inputs: [VonPackedInput]) throws -> BatchLogitsOutput {
        recordedChunks.append(inputs)
        var logits: [Float] = []
        var offsets = [0]
        for input in inputs {
            guard nextRow < rowLogits.count else { throw BatchRecorderError.unexpectedCall }
            let row = rowLogits[nextRow]
            guard row.count == input.maskPositions.count else {
                throw BatchRecorderError.rowWidthMismatch
            }
            logits.append(contentsOf: row)
            offsets.append(logits.count)
            nextRow += 1
        }
        return BatchLogitsOutput(logits: logits, rowOffsets: offsets)
    }

    func chunks() -> [[VonPackedInput]] { recordedChunks }
    func callCount() -> Int { recordedChunks.count }
}

private enum BatchRecorderError: Error {
    case unexpectedCall
    case rowWidthMismatch
    case plannedFailure
}

private actor FailingBatchRecorder {
    private var calls = 0

    func next(_ inputs: [VonPackedInput]) throws -> BatchLogitsOutput {
        calls += 1
        if calls == 2 { throw BatchRecorderError.plannedFailure }
        let widths = inputs.map(\.maskPositions.count)
        let offsets = widths.reduce(into: [0]) { result, width in
            result.append(result[result.count - 1] + width)
        }
        return BatchLogitsOutput(
            logits: widths.flatMap { [Float](repeating: 0, count: $0) },
            rowOffsets: offsets
        )
    }

    func callCount() -> Int { calls }
}

private actor GatedBatchRecorder {
    private var calls = 0
    private var firstCallContinuation: CheckedContinuation<Void, Never>?

    func next(_ inputs: [VonPackedInput]) async -> BatchLogitsOutput {
        calls += 1
        if calls == 1 {
            await withCheckedContinuation { continuation in
                firstCallContinuation = continuation
            }
        }
        let widths = inputs.map(\.maskPositions.count)
        let offsets = widths.reduce(into: [0]) { result, width in
            result.append(result[result.count - 1] + width)
        }
        return BatchLogitsOutput(
            logits: widths.flatMap { [Float](repeating: 0, count: $0) },
            rowOffsets: offsets
        )
    }

    func releaseFirstCall() {
        firstCallContinuation?.resume()
        firstCallContinuation = nil
    }

    func callCount() -> Int { calls }
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

private struct BatchGoldenFixture: Decodable, Sendable {
    let schemaVersion: Int
    let heterogeneousRequest: BatchGoldenRequest

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case heterogeneousRequest = "heterogeneous_request"
    }
}

private struct BatchGoldenRequest: Decodable, Sendable {
    let model: String
    let state: String
    let questionIDs: [String]
    let questions: [BatchGoldenQuestion]
    let answers: [BatchGoldenAnswer]
    let usage: Usage

    enum CodingKeys: String, CodingKey {
        case model, state, questions, answers, usage
        case questionIDs = "question_ids"
    }
}

private struct BatchGoldenAnswer: Decodable, Sendable {
    let questionID: String
    let answerJSON: String

    enum CodingKeys: String, CodingKey {
        case questionID = "question_id"
        case answerJSON = "answer_model_dump_json"
    }
}

private struct BatchGoldenQuestion: Decodable, Sendable {
    let id: String
    let question: Question

    private struct OrderedChoiceOption: Decodable, Sendable {
        let key: String
        let description: String
    }

    private enum CodingKeys: String, CodingKey {
        case id, type, instructions, criteria
        case orderedCriteria = "ordered_criteria"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        let instructions = try container.decode(String.self, forKey: .instructions)
        switch try container.decode(String.self, forKey: .type) {
        case "choice":
            let ordered = try container.decode(
                [OrderedChoiceOption].self,
                forKey: .orderedCriteria
            )
            question = .choice(Choice(
                instructions: instructions,
                criteria: ordered.map {
                    ChoiceOption(key: $0.key, description: $0.description)
                }
            ))
        case "noul":
            question = .noul(Noul(
                instructions: instructions,
                criteria: try container.decodeIfPresent(NoulCriteria.self, forKey: .criteria)
            ))
        case "score":
            question = .score(Score(
                instructions: instructions,
                criteria: try container.decode([ScoreLevel].self, forKey: .criteria)
            ))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unsupported batch golden question type"
            )
        }
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

    func loadBatchFixture() throws -> BatchGoldenFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("testdata/von_batch_golden_v1.json")
        return try JSONDecoder().decode(BatchGoldenFixture.self, from: Data(contentsOf: url))
    }

    func heterogeneousQuestions() -> [QuestionEntry] {
        [
            QuestionEntry(id: "choice", question: .choice(Choice(
                instructions: "Choose",
                criteria: [
                    ChoiceOption(key: "a"),
                    ChoiceOption(key: "b"),
                    ChoiceOption(key: "c"),
                ]
            ))),
            QuestionEntry(id: "default", question: .noul(Noul(
                instructions: "Default"
            ))),
            QuestionEntry(id: "partial", question: .noul(Noul(
                instructions: "Partial",
                criteria: NoulCriteria(true: "yes")
            ))),
            QuestionEntry(id: "score", question: .score(Score(
                instructions: "Score",
                criteria: [.text("zero"), .text("one"), .text("two"), .text("three")]
            ))),
        ]
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
        _ expected: SystemOneEngineError,
        operation: () async throws -> Void
    ) async {
        do {
            try await operation()
            XCTFail("Expected \(expected)")
        } catch {
            XCTAssertEqual(error as? SystemOneEngineError, expected)
        }
    }
}
