import Foundation
import MLX
import SystemOne
import SystemOneDecoders
import XCTest
@testable import SystemOneD1

/// `testdata/d1_omni_2b_audio_tiny/fixture.json`, written by `tools/d1_omni_2b_audio_fixture.py`.
private struct D1SetAudioFixture: Decodable {
    struct Row: Decodable {
        let ids: [Int]
        let state: Int
        let question: [Int]
        let ends: [Int]
    }

    struct Reference: Decodable {
        let id: String
        let choice: String?
        let noul: Double?
        let score: Double?
        let confidence: Double?
        let probabilities: [String: Double]?
    }

    struct Request: Decodable {
        let clip: String?
        let picture: String?
        let rows: [Row]
        let tokens: Int
        let answers: [Reference]
    }

    struct Clip: Decodable {
        let source: [Int]
    }

    struct Calibration: Decodable {
        let `default`: Double
        let temperatures: [String: Double]
    }

    let setSize: Int
    let codes: Int
    let maxPixels: Int
    let calibration: Calibration
    let special: [String]
    let added: [String: Int]
    let words: [String: Int]
    let clips: [String: Clip]
    let pictures: [String: [Int]]
    let requests: [Request]

    enum CodingKeys: String, CodingKey {
        case codes, calibration, special, added, words, clips, pictures, requests
        case setSize = "set_size"
        case maxPixels = "max_pixels"
    }

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_omni_2b_audio_tiny")

    static func load() throws -> D1SetAudioFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
    }

    var configuration: D1SetConfiguration {
        D1SetConfiguration(
            setSize: setSize, codeScheme: codes, bos: "<|startoftext|>", specialTokens: special,
            defaultTemperature: calibration.default, temperatures: calibration.temperatures, visionMaxPixels: maxPixels
        )
    }

    /// The fixture's 16-bit clip: a ramp in five loudness steps plus an LCG's noise.
    static func clip(_ source: [Int]) -> AudioClip {
        let (rate, count, salt) = (source[0], source[1], source[2])
        var state: Int = salt
        var samples: [Int16] = []
        samples.reserveCapacity(count)
        for index in 0..<count {
            state = (state * 1_103_515_245 + 12_345) % 2_147_483_648
            let ramp: Int = (index * (37 + salt)) % 200
            let loudness: Int = (index / 800) % 5 + 1
            let noise: Int = (state >> 16) % 8_192 - 4_096
            let value: Int = ramp * loudness * 24 - 12_000 + noise
            samples.append(Int16(value))
        }
        return AudioClip(pcm16: samples, sampleRate: rate)
    }

    /// `tools/resize_fixture.py`'s synthetic picture.
    static func synthetic(_ spec: [Int]) -> RGBImage {
        let (width, height, salt) = (spec[0], spec[1], spec[2])
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                for channel in 0..<3 {
                    let ramp = (x * 3 + y * 5 + channel * 85 + salt) & 255
                    let texture = ((x * y * 7 + channel * 13) >> 3) & 63
                    pixels[(y * width + x) * 3 + channel] = UInt8(ramp ^ texture)
                }
            }
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    func media(_ request: Request) -> SystemOneMedia {
        if let clip = request.clip { return .audio(Self.clip(clips[clip]!.source)) }
        return .images([Self.synthetic(pictures[request.picture!]!)])
    }
}

/// The fixture's stand-in tokenizer: added tokens (the media markup's among
/// them) split out of text, whole strings with their own ids, else one token
/// per UTF-8 byte (`byte % 100`).
private struct AudioAddedTokenizer: DecoderTokenizer {
    let added: [String: Int]
    let words: [String: Int]
    let names: [(String, [Unicode.Scalar])]

    init(added: [String: Int], words: [String: Int]) {
        self.added = added
        self.words = words
        names = added.keys.sorted { $0.unicodeScalars.count > $1.unicodeScalars.count }.map { ($0, Array($0.unicodeScalars)) }
    }

    func encode(_ text: String) -> [Int] {
        if let id = words[text] { return [id] }
        let scalars = Array(text.unicodeScalars)
        var ids: [Int] = []
        var index = 0
        while index < scalars.count {
            if let match = names.first(where: { index + $0.1.count <= scalars.count && Array(scalars[index ..< index + $0.1.count]) == $0.1 }) {
                ids.append(added[match.0]!)
                index += match.1.count
            } else {
                ids += String(scalars[index]).utf8.map { Int($0) % 100 }
                index += 1
            }
        }
        return ids
    }
}

private enum D1SetAudioRequests {
    static let requests: [SystemOneRequest] = [
        SystemOneRequest(state: .text("Caller says <|audio|> twice."), questions: [
            QuestionEntry(id: "refund", question: .noul(Noul(instructions: "Does the caller want a refund?"))),
            QuestionEntry(id: "topic", question: .choice(Choice(instructions: "What is the call about?", criteria: [
                ChoiceOption(key: "billing", description: "Billing"), ChoiceOption(key: "outage"),
                ChoiceOption(key: "sales", description: "A new plan"),
            ]))),
            QuestionEntry(id: "mood", question: .score(Score(instructions: "How upset is the caller?", criteria: [.text("calm"), .text("annoyed"), .text("angry")]))),
        ]),
        SystemOneRequest(
            state: .fields([StateField(key: "ticket", value: "Voicemail"), StateField(key: "line", value: "2")]),
            questions: [QuestionEntry(id: "urgent", question: .noul(Noul(instructions: "Is it urgent?", criteria: NoulCriteria(true: "Needs a call back today"))))]
        ),
        SystemOneRequest(state: .text(""), questions: [
            QuestionEntry(id: "kind", question: .choice(Choice(instructions: "What is it?", criteria: [
                ChoiceOption(key: "speech"), ChoiceOption(key: "noise"),
            ]))),
            QuestionEntry(id: "loud", question: .score(Score(instructions: "How loud?", criteria: [.text("quiet"), .text("loud")]))),
        ]),
        SystemOneRequest(state: .text("A clip."), questions: [
            QuestionEntry(id: "lang", question: .choice(Choice(instructions: "Which language?", criteria: [
                ChoiceOption(key: "en", description: "English"), ChoiceOption(key: "fr", description: "French"),
            ]))),
        ]),
        SystemOneRequest(state: .text("A photo of the shelf <|image|> here."), questions: [
            QuestionEntry(id: "stocked", question: .noul(Noul(instructions: "Is the shelf stocked?"))),
        ]),
    ]
}

final class D1SetAudioTests: XCTestCase {
    private func planner(_ fixture: D1SetAudioFixture) throws -> D1SetBackend {
        D1SetBackend(
            modelID: "d1-omni-tiny", configuration: fixture.configuration,
            tokenizer: AudioAddedTokenizer(added: fixture.added, words: fixture.words), trunk: nil,
            visionConfiguration: try LFM3VLVisionConfiguration(modelFolder: D1SetAudioFixture.folder),
            audioConfiguration: try LFM3AudioConfiguration(modelFolder: D1SetAudioFixture.folder)
        )
    }

    func testRowsMatchTheReferencePacking() throws {
        let fixture = try D1SetAudioFixture.load()
        let backend = try planner(fixture)
        XCTAssertEqual(fixture.requests.count, D1SetAudioRequests.requests.count)
        for (request, reference) in zip(D1SetAudioRequests.requests, fixture.requests) {
            let media = fixture.media(reference)
            let plan = try backend.plan(request, media: media)
            XCTAssertEqual(plan.rows.map(\.ids), reference.rows.map(\.ids))
            XCTAssertEqual(plan.rows.map(\.stateLength), reference.rows.map(\.state))
            XCTAssertEqual(plan.rows.map(\.questions), reference.rows.map(\.question))
            XCTAssertEqual(plan.rows.map(\.ends), reference.rows.map(\.ends))
            XCTAssertEqual(plan.usage.inputTokens, reference.tokens)
            for row in plan.rows {
                if case .audio(let clip) = media {
                    XCTAssertEqual(row.audio, clip)
                    XCTAssertTrue(row.images.isEmpty)
                } else {
                    XCTAssertNil(row.audio)
                    XCTAssertEqual(row.images.count, 1)
                }
            }
        }
    }

    func testReadsTheOmniCheckpointAndRejectsAudioElsewhere() throws {
        let fixture = try D1SetAudioFixture.load()
        XCTAssertEqual(try D1SetConfiguration(modelFolder: D1SetAudioFixture.folder).visionMaxPixels, fixture.maxPixels)
        let backend = try planner(fixture)
        XCTAssertThrowsError(try backend.plan(D1SetAudioRequests.requests[1], media: .audio(AudioClip(samples: [0, 0.5], sampleRate: 0))))
        let visionOnly = D1SetBackend(
            modelID: "d1-2b", configuration: fixture.configuration,
            tokenizer: AudioAddedTokenizer(added: fixture.added, words: fixture.words), trunk: nil,
            visionConfiguration: try LFM3VLVisionConfiguration(modelFolder: D1SetAudioFixture.folder)
        )
        XCTAssertThrowsError(try visionOnly.plan(D1SetAudioRequests.requests[1], media: fixture.media(fixture.requests[1]))) { error in
            XCTAssertEqual(error as? SystemOneEngineError, .unsupportedMedia(model: "d1-2b"))
        }
    }
}

/// The tiny LFM3-Omni end to end: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class D1SetAudioRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private func backend(_ fixture: D1SetAudioFixture, folder: URL) throws -> D1SetBackend {
        let store = try SafetensorsCheckpoint(modelFolder: folder)
        return D1SetBackend(
            modelID: "d1-omni-tiny", configuration: fixture.configuration,
            tokenizer: AudioAddedTokenizer(added: fixture.added, words: fixture.words),
            trunk: try LFM3Trunk(store: store, configuration: LFM3Configuration(modelFolder: folder)),
            vision: try LFM3VLVision(store: store, configuration: LFM3VLVisionConfiguration(modelFolder: folder)),
            audio: try LFM3AudioEncoder(store: store, configuration: LFM3AudioConfiguration(modelFolder: folder))
        )
    }

    func testTinyAudioAnswersMatchMLXVLM() throws {
        let fixture = try D1SetAudioFixture.load()
        let tolerance = 1e-3
        try Device.withDefaultDevice(device) {
            for folder in [D1SetAudioFixture.folder, D1SetAudioFixture.folder.appendingPathComponent("mlx")] {
                let backend = try backend(fixture, folder: folder)
                for (request, reference) in zip(D1SetAudioRequests.requests, fixture.requests) {
                    let plan = try backend.plan(request, media: fixture.media(reference))
                    let logits = try backend.optionLogits(for: plan.rows)
                    for (decision, expected) in zip(plan.decisions, reference.answers) {
                        let answer = try decision.resolve(decision.rowIndices.map { logits[$0] })
                        let label = "\(folder.lastPathComponent) \(expected.id)"
                        switch answer {
                        case .choice(let choice):
                            XCTAssertEqual(choice.choice, expected.choice, label)
                            XCTAssertEqual(choice.confidence, expected.confidence ?? .nan, accuracy: tolerance, label)
                            for item in choice.probabilities {
                                XCTAssertEqual(item.probability, expected.probabilities?[item.key] ?? .nan, accuracy: tolerance, label)
                            }
                        case .noul(let noul):
                            XCTAssertEqual(noul.noul, expected.noul ?? .nan, accuracy: tolerance, label)
                        case .score(let score):
                            XCTAssertEqual(score.score, expected.score ?? .nan, label)
                            XCTAssertEqual(score.confidence, expected.confidence ?? .nan, accuracy: tolerance, label)
                        }
                    }
                }
            }
        }
    }

    /// Rows of a clip and of a picture in one forward read as they do apart.
    func testMixedMediaRowsShareAForward() throws {
        let fixture = try D1SetAudioFixture.load()
        try Device.withDefaultDevice(device) {
            let backend = try backend(fixture, folder: D1SetAudioFixture.folder.appendingPathComponent("mlx"))
            let heard = try backend.plan(D1SetAudioRequests.requests[1], media: fixture.media(fixture.requests[1]))
            let seen = try backend.plan(D1SetAudioRequests.requests[4], media: fixture.media(fixture.requests[4]))
            let text = try backend.plan(D1SetAudioRequests.requests[3], media: nil)
            let apart = try backend.optionLogits(for: heard.rows) + backend.optionLogits(for: seen.rows)
                + backend.optionLogits(for: text.rows)
            let together = try backend.optionLogits(for: heard.rows + seen.rows + text.rows)
            XCTAssertEqual(together.count, apart.count)
            for (joined, alone) in zip(together, apart) {
                XCTAssertEqual(joined.count, alone.count)
                for (a, b) in zip(joined, alone) { XCTAssertEqual(a, b, accuracy: 1e-4) }
            }
        }
    }
}
