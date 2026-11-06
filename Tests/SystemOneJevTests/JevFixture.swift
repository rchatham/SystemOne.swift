import Foundation
import MLX
import XCTest
@testable import SystemOneJev

/// `testdata/jev_tiny_golden_v1.json`, written by `tools/jev_golden_capture.py tiny`.
struct JevTinyFixture: Decodable {
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
    struct DeltaRule: Decodable {
        let shape: [Int]
        let q: [Float], k: [Float], v: [Float], g: [Float], beta: [Float]
        let output: [Float]
        let finalState: [Float]
        let recurrentOutput: [Float]
        enum CodingKeys: String, CodingKey {
            case shape, q, k, v, g, beta, output
            case finalState = "final_state"
            case recurrentOutput = "recurrent_output"
        }
    }

    let loraScale: Float
    let temperatures: [String: Float]
    let choiceOptions: Int
    let mergeProbes: [String: MergeProbe]
    let cases: [Case]
    let deltaRule: DeltaRule

    enum CodingKeys: String, CodingKey {
        case temperatures, cases
        case loraScale = "lora_scale"
        case choiceOptions = "choice_options"
        case mergeProbes = "merge_probes"
        case deltaRule = "delta_rule"
    }

    static func load() throws -> JevTinyFixture {
        let url = repositoryRoot.appendingPathComponent("testdata/jev_tiny_golden_v1.json")
        return try JSONDecoder().decode(JevTinyFixture.self, from: Data(contentsOf: url))
    }

    static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    static var modelFolder: URL { repositoryRoot.appendingPathComponent("testdata/jev_tiny_v1", isDirectory: true) }
}

enum JevTestDevice {
    /// MLX needs the Metal library of an Xcode-built test bundle, even on the CPU.
    static func require() throws {
        guard ProcessInfo.processInfo.environment["JEV_DEVICE"] == "gpu" else {
            throw XCTSkip("Set JEV_DEVICE=gpu to build MLX arrays from the Xcode test bundle.")
        }
    }
}

/// Fails when `actual` and `expected` differ by more than `tolerance` times
/// `max(1, max |expected|)`, reporting the worst element. Returns that scaled error.
@discardableResult
func assertClose(
    _ actual: [Float],
    _ expected: [Float],
    relative tolerance: Float,
    _ message: String,
    file: StaticString = #filePath,
    line: UInt = #line
) -> Float {
    XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
    let scale = max(1, expected.map(abs).max() ?? 0)
    var worst: (error: Float, index: Int) = (0, 0)
    for (index, (a, e)) in zip(actual, expected).enumerated() where abs(a - e) > worst.error || a.isNaN {
        worst = (a.isNaN ? .infinity : abs(a - e), index)
    }
    XCTAssertLessThanOrEqual(
        worst.error / scale, tolerance,
        "\(message): worst error \(worst.error) at \(worst.index) (scale \(scale))", file: file, line: line)
    return worst.error / scale
}
