import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneD1

final class D1TextTests: XCTestCase {
    // MARK: Offline: fixture only

    func testEscapeMatchesPython() throws {
        for testCase in try D1Fixture.load().escapeCases {
            XCTAssertEqual(D1Prompt.escape(testCase.text), testCase.escaped, testCase.text)
        }
    }

    func testFieldStateSerializesLikeJSONDumps() {
        let state = SystemOneState.fields([
            StateField(key: "author", value: "Zoë \"Z\" O'Neil"),
            StateField(key: "body", value: "line\nbreak\ttab\u{01}\\"),
        ])
        XCTAssertEqual(
            D1Prompt.serialize(state),
            #"{"author": "Zoë \"Z\" O'Neil", "body": "line\nbreak\ttab\u0001\\"}"#
        )
        XCTAssertEqual(D1Prompt.serialize(.fields([])), "{}")
    }

    func testAnswersFromGoldenLogitsMatchPython() throws {
        let fixture = try D1Fixture.load()
        for testCase in fixture.textCases {
            let entries = testCase.request["questions"]!.array.map(\.questionEntry)
            let expected = testCase.result["answers"]!.array
            for (index, (entry, row)) in zip(entries, testCase.rows).enumerated() {
                let type = row.temperatureKey.split(separator: ":")[0]
                let temperature = fixture.temperatures[row.temperatureKey] ?? fixture.temperatures[String(type)] ?? 1
                let answer = D1Backend.answer(
                    entry.question,
                    probabilities: D1Backend.softmax(row.batchedLogits, temperature: temperature)
                )
                assert(answer, matches: expected[index], tolerance: 1e-6, "\(testCase.id).\(entry.id)")
            }
        }
    }

    func testTemperatureBuckets() {
        XCTAssertEqual(D1Prompt.temperatureKey(type: "noul", optionCount: 2), "noul:2")
        XCTAssertEqual(D1Prompt.temperatureKey(type: "choice", optionCount: 5), "choice:3-5")
        XCTAssertEqual(D1Prompt.temperatureKey(type: "score", optionCount: 6), "score:6-10")
        XCTAssertEqual(D1Prompt.temperatureKey(type: "choice", optionCount: 11), "choice:11+")
    }

    // MARK: Tokenizer: needs D1_MODEL_FOLDER

    func testTokenizerMatchesPython() async throws {
        let tokenizer = try await D1Tokenizer(modelFolder: try D1Fixture.modelFolder())
        let fixture = try D1Fixture.load()
        let special = tokenizer.special
        XCTAssertEqual(
            [special.bos, special.pad, special.mask, special.state, special.question, special.option,
             special.optionEnd, special.decide],
            ["bos", "pad", "mask", "state", "q", "opt", "opt_end", "decide"].map { fixture.specialIDs[$0]! }
        )
        for testCase in fixture.tokenizerCases {
            XCTAssertEqual(tokenizer.encode(testCase.text), testCase.ids, testCase.text)
        }
    }

    func testPromptEncodingMatchesPython() async throws {
        let tokenizer = try await D1Tokenizer(modelFolder: try D1Fixture.modelFolder())
        for testCase in try D1Fixture.load().encodeCases {
            let mode: D1PromptMode = testCase.mode == "image" ? .image : testCase.mode == "audio" ? .audio : .text
            let entry = testCase.question.questionEntry
            XCTAssertEqual(D1Prompt.renderOptions(entry.question, mode: mode), testCase.options, testCase.id)
            let encoded = try D1Prompt.encode(
                state: D1Prompt.serialize(testCase.state.state),
                question: entry.question,
                instructions: D1Backend.describe(entry.question).instructions,
                id: entry.id,
                maxLength: testCase.maxLen,
                mode: mode,
                tokenizer: tokenizer
            )
            XCTAssertEqual(encoded.ids, testCase.ids, testCase.id)
            XCTAssertEqual(encoded.markers, testCase.markers, testCase.id)
        }
    }

    func testEngineWithGoldenLogitsMatchesPython() async throws {
        let folder = try D1Fixture.modelFolder()
        let fixture = try D1Fixture.load()
        let backend = D1Backend(
            modelID: "d1-omni-600M",
            configuration: try D1Configuration(modelFolder: folder),
            tokenizer: try await D1Tokenizer(modelFolder: folder),
            model: nil,
            logitsProvider: { rows in
                rows.map { row in
                    fixture.textCases.flatMap(\.rows).first { $0.ids == row.ids }!.batchedLogits
                }
            }
        )
        let engine = SystemOneEngine(backend: backend)
        for testCase in fixture.textCases {
            let response = try await engine.systemOne(
                state: testCase.request["state"]!.state,
                questions: testCase.request["questions"]!.array.map(\.questionEntry)
            )
            XCTAssertEqual(response.usage.inputTokens, Int(testCase.result["usage"]!["input_tokens"]!.number!))
            XCTAssertEqual(response.usage.outputTokens, 0)
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                XCTAssertEqual(answer.id, expected["id"]!.string)
                assert(answer.answer, matches: expected, tolerance: 1e-6, "\(testCase.id).\(answer.id)")
            }
        }
    }

    // MARK: Weights: needs D1_MODEL_FOLDER and D1_MODEL_PARITY=1

    func testTrunkProbesAndHeadLogitsMatchPython() throws {
        let weights = try Self.loadWeights()
        try Device.withDefaultDevice(Self.device) { try checkTrunkAndHead(weights) }
    }

    private func checkTrunkAndHead(_ weights: D1Weights) throws {
        let model = D1Model(weights: weights)
        let fixture = try D1Fixture.load()

        let probe = fixture.trunkProbe
        let embeddings = model.trunk.embedding(MLXArray(probe.ids.map(Int32.init))).expandedDimensions(axis: 0)
        let (_, layers) = model.trunk(embeddings, lengths: [probe.ids.count], prefixLengths: [0], captureLayers: true)
        let names = ["embeddings"] + (0..<weights.configuration.text.numHiddenLayers).map { "layer_\($0)" } + ["final_norm"]
        let dimensions = MLXArray(probe.dimensions.map(Int32.init))
        var worst: Float = 0
        for (name, hidden) in zip(names, layers) {
            let actual = hidden[0].take(dimensions, axis: 1)
            let expected = MLXArray(probe.entries[name]!.flatMap { $0 }, actual.shape)
            let error = abs(actual - expected).max().item(Float.self)
            let scale = max(1, abs(expected).max().item(Float.self))
            worst = max(worst, error / scale)
            XCTAssertLessThan(error / scale, 1e-3, name)
        }
        print("d1 trunk probe worst relative error: \(worst)")

        var worstLogit: Float = 0
        for testCase in fixture.textCases {
            let rows = testCase.rows.map {
                D1Row(ids: $0.ids, markers: $0.markers, questionType: .init(rawValue: $0.qtype)!, prefix: nil)
            }
            let batched = try model.logits(for: rows)
            for (index, row) in testCase.rows.enumerated() {
                let single = try model.logits(for: [rows[index]])[0, 0..<row.logits.count].asArray(Float.self)
                let fromBatch = batched[index, 0..<row.logits.count].asArray(Float.self)
                for (actual, expected) in Array(zip(single, row.logits)) + Array(zip(fromBatch, row.batchedLogits)) {
                    worstLogit = max(worstLogit, abs(actual - expected))
                    XCTAssertEqual(actual, expected, accuracy: 2e-3, testCase.id)
                }
            }
        }
        print("d1 head logit worst absolute error: \(worstLogit)")
    }

    static let probeDimensions = [0, 1, 2, 3, 63, 64, 127, 255, 256, 511, 512, 767, 768, 1021, 1022, 1023]

    /// `D1_DEVICE=gpu` needs the Metal library of an Xcode-built test bundle.
    static var device: Device {
        ProcessInfo.processInfo.environment["D1_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    static func loadWeights() throws -> D1Weights {
        let folder = try D1Fixture.modelFolder()
        guard ProcessInfo.processInfo.environment["D1_MODEL_PARITY"] == "1" else {
            throw XCTSkip("Set D1_MODEL_PARITY=1 to load the d1 weights.")
        }
        return try D1Weights(
            contentsOf: folder.appendingPathComponent("model.safetensors"),
            pinnedTo: D1Weights.d1Omni600MArtifact,
            configuration: try D1Configuration(modelFolder: folder)
        )
    }
}

/// Compares a Swift answer with the fixture's ordered Python answer.
func assert(
    _ answer: Answer,
    matches expected: JSONValue,
    tolerance: Double,
    _ message: String,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    func pairs(_ key: String) -> [(String, Double)] {
        expected[key]!.array.map { ($0.pair.0, $0.pair.1.number!) }
    }
    switch answer {
    case .noul(let noul):
        XCTAssertEqual(expected["type"]?.string, "noul", message, file: file, line: line)
        XCTAssertEqual(noul.noul, expected["noul"]!.number!, accuracy: tolerance, message, file: file, line: line)
    case .choice(let choice):
        XCTAssertEqual(choice.choice, expected["choice"]?.string, message, file: file, line: line)
        XCTAssertEqual(choice.confidence, expected["confidence"]!.number!, accuracy: tolerance, message, file: file, line: line)
        let probabilities = pairs("probabilities")
        XCTAssertEqual(choice.probabilities.map(\.key), probabilities.map(\.0), message, file: file, line: line)
        for (actual, (_, value)) in zip(choice.probabilities, probabilities) {
            XCTAssertEqual(actual.probability, value, accuracy: tolerance, message, file: file, line: line)
        }
    case .score(let score):
        XCTAssertEqual(score.score, expected["score"]!.number!, accuracy: tolerance * 10, message, file: file, line: line)
        XCTAssertEqual(score.confidence, expected["confidence"]!.number!, accuracy: tolerance, message, file: file, line: line)
        XCTAssertEqual(score.legend.map(\.key), expected["legend"]!.array.map { $0.pair.0 }, message, file: file, line: line)
        XCTAssertEqual(score.legend.map(\.description), expected["legend"]!.array.map { $0.pair.1.string! },
                       message, file: file, line: line)
        for (actual, (_, value)) in zip(score.probabilities, pairs("probabilities")) {
            XCTAssertEqual(actual.probability, value, accuracy: tolerance, message, file: file, line: line)
        }
    }
}
