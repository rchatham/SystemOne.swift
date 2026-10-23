import Foundation
import SystemOne
import XCTest
@testable import SystemOneD1

/// `testdata/d1_golden_v1.json`, written by `tools/d1_golden_capture.py`.
struct D1Fixture: Decodable {
    struct TokenizerCase: Decodable { let text: String; let ids: [Int] }
    struct EscapeCase: Decodable { let text: String; let escaped: String }
    struct EncodeCase: Decodable {
        let id: String
        let state: JSONValue
        let question: JSONValue
        let maxLen: Int
        let mode: String
        let ids: [Int]
        let markers: [Int]
        let options: [String]
        enum CodingKeys: String, CodingKey { case id, state, question, mode, ids, markers, options, maxLen = "max_len" }
    }
    struct TrunkProbe: Decodable {
        let ids: [Int]
        let dimensions: [Int]
        let entries: [String: [[Float]]]
    }
    struct Row: Decodable {
        let ids: [Int]
        let markers: [Int]
        let qtype: Int
        let temperatureKey: String
        let logits: [Float]
        let batchedLogits: [Float]
        enum CodingKeys: String, CodingKey {
            case ids, markers, qtype, logits
            case temperatureKey = "temperature_key"
            case batchedLogits = "batched_logits"
        }
    }
    struct RequestCase: Decodable {
        let id: String
        let mode: String
        let request: JSONValue
        let rows: [Row]
        let probabilities: [[Double]]
        let result: JSONValue
    }

    struct Vision: Decodable {
        struct Layout: Decodable { let width: Int; let height: Int; let grid: [Int]; let thumbnail: [Int]; let tiled: Bool }
        struct Crop: Decodable {
            let height: Int, width: Int, sha256: String, sum: Int
            let positionProbe: [[Float]], towerProbe: [[Float]], probeRows: [Int]
            enum CodingKeys: String, CodingKey {
                case height, width, sha256, sum
                case positionProbe = "position_probe", towerProbe = "tower_probe", probeRows = "probe_rows"
            }
        }
        struct Image: Decodable {
            let name: String, width: Int, height: Int, seed: Int
            let crops: [Crop]
            let probeDimensions: [Int], prefixLength: Int, prefixRows: [Int], prefixProbe: [[Float]]
            enum CodingKeys: String, CodingKey {
                case name, width, height, seed, crops
                case probeDimensions = "probe_dimensions", prefixLength = "prefix_length"
                case prefixRows = "prefix_rows", prefixProbe = "prefix_probe"
            }
        }
        struct Spec: Decodable { let width: Int; let height: Int; let seed: Int }
        struct Request: Decodable {
            let id: String, request: JSONValue, rows: [Row], result: JSONValue, images: [Spec]
        }
        let layouts: [Layout]
        let images: [Image]
        let requests: [Request]
    }

    struct Audio: Decodable {
        struct Clip: Decodable {
            let name: String, samples: Int, seed: Int, clipSHA256: String, frames: Int
            let melFrames: [Int], melBins: [Int], melProbe: [[Float]]
            let encoderRows: [Int], encoderDimensions: [Int], encoderProbe: [[Float]]
            let prefixLength: Int, prefixRows: [Int], prefixProbe: [[Float]]
            enum CodingKeys: String, CodingKey {
                case name, samples, seed, frames
                case clipSHA256 = "clip_sha256", melFrames = "mel_frames", melBins = "mel_bins", melProbe = "mel_probe"
                case encoderRows = "encoder_rows", encoderDimensions = "encoder_dimensions"
                case encoderProbe = "encoder_probe", prefixLength = "prefix_length"
                case prefixRows = "prefix_rows", prefixProbe = "prefix_probe"
            }
        }
        struct Spec: Decodable { let samples: Int; let seed: Int }
        struct Request: Decodable {
            let id: String, request: JSONValue, rows: [Row], result: JSONValue, audio: Spec
        }
        let clips: [Clip]
        let requests: [Request]
    }

    let specialIDs: [String: Int]
    let temperatures: [String: Double]
    let tokenizerCases: [TokenizerCase]
    let escapeCases: [EscapeCase]
    let encodeCases: [EncodeCase]
    let trunkProbe: TrunkProbe
    let textCases: [RequestCase]
    let vision: Vision
    let audio: Audio

    enum CodingKeys: String, CodingKey {
        case temperatures
        case specialIDs = "special_ids"
        case tokenizerCases = "tokenizer_cases"
        case escapeCases = "escape_cases"
        case encodeCases = "encode_cases"
        case trunkProbe = "trunk_probe"
        case textCases = "text_cases"
        case vision, audio
    }

    static func load() throws -> D1Fixture {
        let url = repositoryRoot.appendingPathComponent("testdata/d1_golden_v1.json")
        return try JSONDecoder().decode(D1Fixture.self, from: Data(contentsOf: url))
    }

    static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    /// The local d1 snapshot, or a skip when `D1_MODEL_FOLDER` is unset.
    static func modelFolder() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["D1_MODEL_FOLDER"], !path.isEmpty else {
            throw XCTSkip("Set D1_MODEL_FOLDER to a LiquidAI/d1-omni-600M snapshot to run d1 parity.")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}

/// `synthetic_image` in `tools/d1_golden_capture.py`: integer-only RGB.
func syntheticImage(width: Int, height: Int, seed: Int) -> RGBImage {
    var pixels = [UInt8]()
    pixels.reserveCapacity(width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let r = (x * 255 / max(1, width - 1) + seed * 37) % 256
            let g = (y * 255 / max(1, height - 1) + seed * 91) % 256
            let b = ((x / 16 + y / 16) % 2) * 160 + ((x * y + seed) % 64)
            pixels += [UInt8(r), UInt8(g), UInt8(b % 256)]
        }
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

/// `synthetic_audio`: int16 PCM from an LCG and triangle waves, with Python's floor division.
func syntheticAudio(samples: Int, seed: Int) -> [Int16] {
    func floorDivide(_ a: Int, _ b: Int) -> Int { Int((Double(a) / Double(b)).rounded(.down)) }
    var state = UInt64(seed) & 0xFFFF_FFFF
    var out: [Int16] = []
    out.reserveCapacity(samples)
    for n in 0..<samples {
        state = (state &* 1_103_515_245 &+ 12_345) & 0x7FFF_FFFF
        let noise = Int((state >> 16) % 2001) - 1000
        let periodA = 73 + seed % 11, periodB = 211
        let triA = abs((n % periodA) * 2 * 6000 / periodA - 6000) - 3000
        let triB = abs((n % periodB) * 2 * 9000 / periodB - 9000) - 4500
        let envelope = (n % 4000) * 4 / 4000 + 1
        out.append(Int16(max(-32_768, min(32_767, floorDivide((triA + triB) * envelope, 2) + noise))))
    }
    return out
}

/// Fixture JSON. Every order-bearing mapping is emitted as an array of pairs,
/// so objects here are only ever looked up by key.
indirect enum JSONValue: Decodable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if single.decodeNil() { self = .null }
        else if let bool = try? single.decode(Bool.self) { self = .bool(bool) }
        else if let number = try? single.decode(Double.self) { self = .number(number) }
        else if let string = try? single.decode(String.self) { self = .string(string) }
        else if let array = try? single.decode([JSONValue].self) { self = .array(array) }
        else { self = .object(try single.decode([String: JSONValue].self)) }
    }

    subscript(key: String) -> JSONValue? {
        guard case .object(let members) = self else { return nil }
        return members[key]
    }

    var string: String? { if case .string(let s) = self { return s }; return nil }
    var number: Double? { if case .number(let n) = self { return n }; return nil }
    var array: [JSONValue] { if case .array(let a) = self { return a }; return [] }
    var pair: (String, JSONValue) { (array[0].string!, array[1]) }
}

extension JSONValue {
    /// `{"text": s}` or `{"fields": [[k, v], ...]}`.
    var state: SystemOneState {
        if let text = self["text"]?.string { return .text(text) }
        return .fields(self["fields"]!.array.map { StateField(key: $0.pair.0, value: $0.pair.1.string!) })
    }

    /// `{"id", "type", "instructions", "criteria"}` with ordered criteria.
    var questionEntry: QuestionEntry {
        let instructions = self["instructions"]!.string!
        let criteria = self["criteria"]
        let question: Question
        switch self["type"]!.string! {
        case "choice":
            question = .choice(Choice(instructions: instructions, criteria: criteria!.array.map {
                ChoiceOption(key: $0.pair.0, description: $0.pair.1.string)
            }))
        case "score":
            question = .score(Score(instructions: instructions, criteria: criteria!.array.map { level in
                if let text = level.string { return .text(text) }
                return .described(what: level["what"]!.string!, examples: level["examples"]?.array.map { $0.string! } ?? [])
            }))
        default:
            let noul = criteria.flatMap { c -> NoulCriteria? in
                guard case .object = c else { return nil }
                return NoulCriteria(true: c["true"]?.string, false: c["false"]?.string)
            }
            question = .noul(Noul(instructions: instructions, criteria: noul))
        }
        return QuestionEntry(id: self["id"]!.string!, question: question)
    }
}
