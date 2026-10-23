import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneD1

/// `precision: .float16` against the float32 Python fixture. Half precision is
/// not bit-faithful, so these check what the d1-omni model card promises: the
/// same top answer on every row, with probabilities close to float32's.
final class D1Float16Tests: XCTestCase {
    /// Largest probability difference allowed from the float32 reference.
    static let probabilityTolerance = 2e-2

    // MARK: Offline

    func testBlockedMaskValueFitsTheComputeDType() {
        XCTAssertEqual(D1Trunk.blocked(.float32), -1e9)
        XCTAssertEqual(D1Trunk.blocked(.float16), -65_504)
    }

    // MARK: Weights: needs D1_MODEL_FOLDER and D1_MODEL_PARITY=1

    /// Every stage stays float16: a float32 constant would silently promote and
    /// bring back the float32 activation memory.
    func testActivationsStayFloat16() throws {
        let weights = try D1TextTests.loadWeights(precision: .float16)
        try Device.withDefaultDevice(D1TextTests.device) {
            let model = D1Model(weights: weights)
            let fixture = try D1Fixture.load()
            let image = fixture.vision.requests[0].images[0]
            let audio = fixture.audio.requests[0].audio
            let mediaPrefixes: [SystemOneMedia] = [
                .images([syntheticImage(width: image.width, height: image.height, seed: image.seed)]),
                .audio(AudioClip(pcm16: syntheticAudio(samples: audio.samples, seed: audio.seed))),
            ]
            for media in mediaPrefixes {
                XCTAssertEqual(try model.encode(media).dtype, .float16, "\(media)")
            }

            let row = fixture.textCases[0].rows[0]
            let prefix = D1MediaPrefix(media: mediaPrefixes[1], length: D1Model.prefixLength(of: mediaPrefixes[1]))
            let rows = [
                D1Row(ids: row.ids, markers: row.markers, questionType: .init(rawValue: row.qtype)!, prefix: nil),
                D1Row(ids: row.ids, markers: row.markers, questionType: .init(rawValue: row.qtype)!, prefix: prefix),
            ]
            let embeddings = model.trunk.embedding(MLXArray(row.ids.map(Int32.init))).expandedDimensions(axis: 0)
            let padded = padded(embeddings, widths: [0, [0, 3], 0])
            XCTAssertEqual(model.trunk(padded, lengths: [row.ids.count], prefixLengths: [0]).output.dtype, .float16)
            XCTAssertEqual(try model.logits(for: rows).dtype, .float16)
        }
    }

    func testTextRequestsKeepTopAnswers() async throws {
        let engine = try await Self.engine()
        var worst = 0.0
        for testCase in try D1Fixture.load().textCases {
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry)
                )
            }
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                worst = max(worst, assertTopAnswer(answer.answer, matches: expected, "\(testCase.id).\(answer.id)"))
            }
        }
        print("d1 float16 text worst probability error: \(worst)")
    }

    func testImageRequestsKeepTopAnswers() async throws {
        let engine = try await Self.engine()
        var worst = 0.0
        for testCase in try D1Fixture.load().vision.requests {
            let images = testCase.images.map { syntheticImage(width: $0.width, height: $0.height, seed: $0.seed) }
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry),
                    media: .images(images)
                )
            }
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                worst = max(worst, assertTopAnswer(answer.answer, matches: expected, "\(testCase.id).\(answer.id)"))
            }
        }
        print("d1 float16 image worst probability error: \(worst)")
    }

    func testAudioRequestsKeepTopAnswers() async throws {
        let engine = try await Self.engine()
        var worst = 0.0
        for testCase in try D1Fixture.load().audio.requests {
            let clip = AudioClip(pcm16: syntheticAudio(samples: testCase.audio.samples, seed: testCase.audio.seed))
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry),
                    media: .audio(clip)
                )
            }
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                worst = max(worst, assertTopAnswer(answer.answer, matches: expected, "\(testCase.id).\(answer.id)"))
            }
        }
        print("d1 float16 audio worst probability error: \(worst)")
    }

    private static func engine() async throws -> SystemOneEngine {
        let weights = try D1TextTests.loadWeights(precision: .float16)
        return SystemOneEngine(backend: D1Backend(
            modelID: "d1-omni-600M",
            configuration: weights.configuration,
            tokenizer: try await D1Tokenizer(modelFolder: try D1Fixture.modelFolder()),
            model: D1Model(weights: weights),
            logitsProvider: nil
        ))
    }
}

/// Checks the top answer against the fixture and returns the largest
/// probability difference, failing past `D1Float16Tests.probabilityTolerance`.
@discardableResult
func assertTopAnswer(
    _ answer: Answer,
    matches expected: JSONValue,
    _ message: String,
    file: StaticString = #filePath,
    line: UInt = #line
) -> Double {
    func pairs(_ key: String) -> [Double] { expected[key]!.array.map { $0.pair.1.number! } }
    func argmax(_ p: [Double]) -> Int { p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) }! }
    let (actual, reference): ([Double], [Double])
    switch answer {
    case .noul(let noul):
        (actual, reference) = ([1 - noul.noul, noul.noul], [1 - expected["noul"]!.number!, expected["noul"]!.number!])
    case .choice(let choice):
        XCTAssertEqual(choice.choice, expected["choice"]?.string, message, file: file, line: line)
        (actual, reference) = (choice.probabilities.map(\.probability), pairs("probabilities"))
    case .score(let score):
        (actual, reference) = (score.probabilities.map(\.probability), pairs("probabilities"))
    }
    XCTAssertEqual(argmax(actual), argmax(reference), "\(message): top answer changed", file: file, line: line)
    let error = zip(actual, reference).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(error, D1Float16Tests.probabilityTolerance, message, file: file, line: line)
    return error
}
