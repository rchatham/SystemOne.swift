import Foundation
import MLX
import SystemOne
import SystemOneDecoders
import XCTest
@testable import SystemOneD1

/// `testdata/d1_2b_vision_tiny/fixture.json`, written by `tools/d1_2b_vision_fixture.py`.
private struct D1SetVisionFixture: Decodable {
    struct Row: Decodable {
        let ids: [Int]
        let state: Int
        let question: [Int]
        let ends: [Int]
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
        let images: [String]
        let rows: [Row]
        let tokens: Int
        let answers: [Reference]
    }

    struct Picture: Decodable {
        let source: [Int]
        let capped: [Int]
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
    let images: [String: Picture]
    let requests: [Request]

    enum CodingKeys: String, CodingKey {
        case codes, calibration, special, added, words, images, requests
        case setSize = "set_size"
        case maxPixels = "max_pixels"
    }

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_2b_vision_tiny")

    static func load() throws -> D1SetVisionFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
    }

    var configuration: D1SetConfiguration {
        D1SetConfiguration(
            setSize: setSize, codeScheme: codes, bos: "<|startoftext|>", specialTokens: special,
            defaultTemperature: calibration.default, temperatures: calibration.temperatures, visionMaxPixels: maxPixels
        )
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
        .images(request.images.map { Self.synthetic(images[$0]!.source) })
    }
}

/// The fixture's stand-in tokenizer: added tokens (the vision markup's among
/// them) split out of text, whole strings with their own ids, else one token
/// per UTF-8 byte (`byte % 100`).
private struct VisionAddedTokenizer: DecoderTokenizer {
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

private enum D1SetVisionRequests {
    static let requests: [SystemOneRequest] = [
        SystemOneRequest(state: .text("A photo of the shelf <|image|> here."), questions: [
            QuestionEntry(id: "stocked", question: .noul(Noul(instructions: "Is the shelf stocked?"))),
            QuestionEntry(id: "aisle", question: .choice(Choice(instructions: "Which aisle?", criteria: [
                ChoiceOption(key: "dairy", description: "Dairy"), ChoiceOption(key: "bakery"),
                ChoiceOption(key: "frozen", description: "Frozen food"),
            ]))),
            QuestionEntry(id: "tidy", question: .score(Score(instructions: "How tidy is it?", criteria: [.text("messy"), .text("fine"), .text("neat")]))),
        ]),
        SystemOneRequest(
            state: .fields([StateField(key: "ticket", value: "Broken lamp"), StateField(key: "photos", value: "2")]),
            questions: [QuestionEntry(id: "broken", question: .noul(Noul(instructions: "Is it broken?", criteria: NoulCriteria(true: "Visible damage"))))]
        ),
        SystemOneRequest(state: .text(""), questions: [
            QuestionEntry(id: "kind", question: .choice(Choice(instructions: "What is it?", criteria: [
                ChoiceOption(key: "lamp"), ChoiceOption(key: "chair"),
            ]))),
            QuestionEntry(id: "light", question: .score(Score(instructions: "How bright?", criteria: [.text("dark"), .text("lit")]))),
        ]),
    ]
}

final class D1SetVisionTests: XCTestCase {
    private func planner(_ fixture: D1SetVisionFixture) throws -> D1SetBackend {
        D1SetBackend(
            modelID: "d1-tiny", configuration: fixture.configuration,
            tokenizer: VisionAddedTokenizer(added: fixture.added, words: fixture.words), trunk: nil,
            visionConfiguration: try LFM3VLVisionConfiguration(modelFolder: D1SetVisionFixture.folder)
        )
    }

    func testRowsMatchTheReferencePacking() throws {
        let fixture = try D1SetVisionFixture.load()
        let backend = try planner(fixture)
        XCTAssertEqual(fixture.requests.count, D1SetVisionRequests.requests.count)
        for (request, reference) in zip(D1SetVisionRequests.requests, fixture.requests) {
            let plan = try backend.plan(request, media: fixture.media(reference))
            XCTAssertEqual(plan.rows.map(\.ids), reference.rows.map(\.ids))
            XCTAssertEqual(plan.rows.map(\.stateLength), reference.rows.map(\.state))
            XCTAssertEqual(plan.rows.map(\.questions), reference.rows.map(\.question))
            XCTAssertEqual(plan.rows.map(\.ends), reference.rows.map(\.ends))
            XCTAssertEqual(plan.usage.inputTokens, reference.tokens)
            for row in plan.rows {
                XCTAssertEqual(row.images.map { [$0.width, $0.height] }, reference.images.map { fixture.images[$0]!.capped })
            }
        }
    }

    func testReadsTheVisionBudgetAndRejectsOtherMedia() throws {
        let fixture = try D1SetVisionFixture.load()
        XCTAssertEqual(try D1SetConfiguration(modelFolder: D1SetVisionFixture.folder).visionMaxPixels, fixture.maxPixels)
        let backend = try planner(fixture)
        XCTAssertThrowsError(try backend.plan(D1SetVisionRequests.requests[1], media: .audio(AudioClip(samples: [0, 0.5]))))
        let textOnly = D1SetBackend(
            modelID: "d1-text", configuration: fixture.configuration,
            tokenizer: VisionAddedTokenizer(added: fixture.added, words: fixture.words), trunk: nil
        )
        XCTAssertThrowsError(try textOnly.plan(D1SetVisionRequests.requests[1], media: fixture.media(fixture.requests[1])))
    }
}

/// The tiny LFM3-VL end to end: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class D1SetVisionRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testTinyImageAnswersMatchMLXVLM() throws {
        let fixture = try D1SetVisionFixture.load()
        let tolerance = 1e-3
        try Device.withDefaultDevice(device) {
            for folder in [D1SetVisionFixture.folder, D1SetVisionFixture.folder.appendingPathComponent("mlx")] {
                let store = try SafetensorsCheckpoint(modelFolder: folder)
                let backend = D1SetBackend(
                    modelID: "d1-tiny", configuration: fixture.configuration,
                    tokenizer: VisionAddedTokenizer(added: fixture.added, words: fixture.words),
                    trunk: try LFM3Trunk(store: store, configuration: LFM3Configuration(modelFolder: folder)),
                    vision: try LFM3VLVision(store: store, configuration: LFM3VLVisionConfiguration(modelFolder: folder))
                )
                for (request, reference) in zip(D1SetVisionRequests.requests, fixture.requests) {
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
}
