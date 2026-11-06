import Foundation
import MLX
import SystemOne
import SystemOneDecoders
import XCTest
@testable import SystemOneD1

/// `testdata/d1_3b_vision_tiny/fixture.json`, written by `tools/d1_3b_vision_fixture.py`.
private struct D1VisionFixture: Decodable {
    struct Reference: Decodable {
        let type: String
        let choice: String?
        let noul: Double?
        let score: Double?
        let confidence: Double?
        let probabilities: [String: Double]?
        let legend: [String: String]?
    }

    struct Entry: Decodable {
        let id: String
        let codes: [String]
        let groups: [[Int]]
        let ids: [Int]
        let answer: Reference
    }

    struct Request: Decodable {
        let images: [String]
        let prefix: [Int]
        let questions: [Entry]
    }

    struct Picture: Decodable {
        let source: [Int]
        let capped: [Int]
    }

    let bos: Int
    let images: [String: Picture]
    let requests: [Request]

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_3b_vision_tiny")

    static func load() throws -> D1VisionFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
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

/// The fixture's stand-in tokenizer: the image markup's special tokens split
/// out of text, whole strings with their own ids, else one token per UTF-8
/// byte (`byte % 100`).
private struct VisionTokenizer: DecoderTokenizer {
    static let table = ["yes": 100, "Yes": 101, "YES": 102, "no": 103, "No": 104, "NO": 105, " A": 106, " B": 107, " C": 108, " x": 109]
    static let special: [String: Int] = {
        var special = ["<image>": 120, "<|image_start|>": 121, "<|image_end|>": 122, "<|img_thumbnail|>": 123]
        for row in 1...10 {
            for column in 1...10 { special["<|img_row_\(row)_col_\(column)|>"] = 130 + 10 * (row - 1) + (column - 1) }
        }
        return special
    }()
    static let names: [(String, [Unicode.Scalar])] = special.keys
        .sorted { $0.unicodeScalars.count > $1.unicodeScalars.count }
        .map { ($0, Array($0.unicodeScalars)) }

    func encode(_ text: String) -> [Int] {
        if let id = Self.table[text] { return [id] }
        let scalars = Array(text.unicodeScalars)
        var ids: [Int] = []
        var index = 0
        while index < scalars.count {
            if let match = Self.names.first(where: { index + $0.1.count <= scalars.count && Array(scalars[index ..< index + $0.1.count]) == $0.1 }) {
                ids.append(Self.special[match.0]!)
                index += match.1.count
            } else {
                ids += String(scalars[index]).utf8.map { Int($0) % 100 }
                index += 1
            }
        }
        return ids
    }
}

private enum D1VisionRequests {
    static let requests: [SystemOneRequest] = [
        SystemOneRequest(state: .text("Is the sign readable?"), questions: [
            QuestionEntry(id: "readable", question: .noul(Noul(instructions: "Can the sign be read?"))),
            QuestionEntry(id: "kind", question: .choice(Choice(instructions: "What is shown?", criteria: [
                ChoiceOption(key: "sign", description: "A street sign"), ChoiceOption(key: "poster"),
                ChoiceOption(key: "screen", description: "A screen"),
            ]))),
        ]),
        SystemOneRequest(
            state: .fields([StateField(key: "ticket", value: "Damaged parcel"), StateField(key: "photos", value: "2")]),
            questions: [QuestionEntry(id: "damage", question: .score(Score(
                instructions: "How bad is the damage?", criteria: [.text("none"), .text("some"), .text("severe")]
            )))]
        ),
        SystemOneRequest(state: .text(""), questions: [
            QuestionEntry(id: "same", question: .noul(Noul(instructions: "Same place?", criteria: NoulCriteria(true: "Yes, same", false: "No")))),
            QuestionEntry(id: "first", question: .choice(Choice(instructions: "Which is brighter?", criteria: [
                ChoiceOption(key: "a"), ChoiceOption(key: "b"),
            ]))),
            QuestionEntry(id: "count", question: .score(Score(
                instructions: "How many people?", criteria: [.text("0"), .text("1"), .text("2"), .text("3+")]
            ))),
        ]),
    ]
}

final class D1ChatVisionTests: XCTestCase {
    private func planner() throws -> D1ChatBackend {
        let fixture = try D1VisionFixture.load()
        return D1ChatBackend(
            modelID: "d1-tiny", tokenizer: VisionTokenizer(), bos: fixture.bos, trunk: nil,
            visionConfiguration: try LFM2VLVisionConfiguration(modelFolder: D1VisionFixture.folder)
        )
    }

    func testPlansMatchTheProcessorPrompts() throws {
        let fixture = try D1VisionFixture.load()
        let backend = try planner()
        XCTAssertEqual(fixture.requests.count, D1VisionRequests.requests.count)
        for (request, reference) in zip(D1VisionRequests.requests, fixture.requests) {
            let plan = try backend.plan(request, media: fixture.media(reference))
            XCTAssertEqual(plan.rows.count, reference.questions.count)
            for (row, entry) in zip(plan.rows, reference.questions) {
                XCTAssertEqual(row.prefix, reference.prefix, entry.id)
                XCTAssertEqual(row.ids, entry.ids, entry.id)
                XCTAssertEqual(row.groups, entry.groups, entry.id)
                XCTAssertEqual(row.images.map { [$0.width, $0.height] }, reference.images.map { fixture.images[$0]!.capped }, entry.id)
            }
            XCTAssertEqual(plan.usage.inputTokens, reference.prefix.count + reference.questions.reduce(0) { $0 + $1.ids.count })
        }
    }

    func testRejectsStrayImageTokensAndOtherMedia() throws {
        let fixture = try D1VisionFixture.load()
        let backend = try planner()
        let picture = D1VisionFixture.synthetic(fixture.images["small"]!.source)
        let stray = SystemOneRequest(state: .text("see <image> here"), questions: D1VisionRequests.requests[0].questions)
        XCTAssertThrowsError(try backend.plan(stray, media: .images([picture]))) { error in
            guard case .imagePlaceholders(let expected, let found)? = error as? D1ChatError else { return XCTFail("\(error)") }
            XCTAssertEqual(found, expected + 1)
        }
        let audio = AudioClip(samples: [0, 0.5])
        XCTAssertThrowsError(try backend.plan(D1VisionRequests.requests[0], media: .audio(audio)))
        let textOnly = D1ChatBackend(modelID: "d1-text", tokenizer: VisionTokenizer(), bos: 1, trunk: nil)
        XCTAssertThrowsError(try textOnly.plan(D1VisionRequests.requests[0], media: .images([picture])))
        // No pictures: the text prompt, unchanged.
        let text = try textOnly.plan(D1VisionRequests.requests[0], media: nil)
        XCTAssertEqual(text.rows.first?.prefix, [1] + VisionTokenizer().encode(D1ChatPrompt.prefix(state: .text("Is the sign readable?"))))
    }

    func testCapPixelsKeepsSmallPicturesAndBoundsLargeOnes() throws {
        let fixture = try D1VisionFixture.load()
        let small = D1VisionFixture.synthetic(fixture.images["small"]!.source)
        XCTAssertEqual(D1ChatPrompt.capPixels(small), small)
        let large = D1ChatPrompt.capPixels(D1VisionFixture.synthetic(fixture.images["large"]!.source))
        XCTAssertEqual([large.width, large.height], fixture.images["large"]!.capped)
        XCTAssertLessThanOrEqual(large.width * large.height, D1ChatPrompt.maximumPixels)
    }
}

/// The tiny LFM2-VL end to end: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class D1ChatVisionRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testTinyImageAnswersMatchTransformers() throws {
        let fixture = try D1VisionFixture.load()
        let tolerance = 1e-3
        try Device.withDefaultDevice(device) {
            for folder in [D1VisionFixture.folder, D1VisionFixture.folder.appendingPathComponent("mlx")] {
                let store = try SafetensorsCheckpoint(modelFolder: folder)
                let backend = D1ChatBackend(
                    modelID: "d1-tiny", tokenizer: VisionTokenizer(), bos: fixture.bos,
                    trunk: try LFM2Trunk(store: store, configuration: LFM2Configuration(modelFolder: folder)),
                    vision: try LFM2VLVision(store: store, configuration: LFM2VLVisionConfiguration(modelFolder: folder))
                )
                for (request, reference) in zip(D1VisionRequests.requests, fixture.requests) {
                    let plan = try backend.plan(request, media: fixture.media(reference))
                    let logits = try backend.optionLogits(for: plan.rows)
                    for (decision, expected) in zip(plan.decisions, reference.questions) {
                        let answer = try decision.resolve(decision.rowIndices.map { logits[$0] })
                        let label = "\(folder.lastPathComponent) \(expected.id)"
                        switch answer {
                        case .choice(let choice):
                            XCTAssertEqual(choice.choice, expected.answer.choice, label)
                            XCTAssertEqual(choice.confidence, expected.answer.confidence ?? .nan, accuracy: tolerance, label)
                            for item in choice.probabilities {
                                XCTAssertEqual(item.probability, expected.answer.probabilities?[item.key] ?? .nan, accuracy: tolerance, label)
                            }
                        case .noul(let noul):
                            XCTAssertEqual(noul.noul, expected.answer.noul ?? .nan, accuracy: tolerance, label)
                        case .score(let score):
                            XCTAssertEqual(score.score, expected.answer.score ?? .nan, accuracy: tolerance, label)
                            XCTAssertEqual(score.confidence, expected.answer.confidence ?? .nan, accuracy: tolerance, label)
                        }
                    }
                }
            }
        }
    }
}
