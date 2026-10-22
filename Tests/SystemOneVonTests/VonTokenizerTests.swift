import CryptoKit
import Foundation
import XCTest
@testable import SystemOne
@testable import SystemOneVon

final class VonTokenizerTests: XCTestCase {
    func testGoldenPackedInputsAndStateCounts() async throws {
        let fixture = try loadGoldenFixture()
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())
        let casesByID = Dictionary(uniqueKeysWithValues: fixture.cases.map { ($0.id, $0) })

        XCTAssertEqual(fixture.forwards.count, 5)
        for forward in fixture.forwards {
            guard let testCase = casesByID[forward.caseID] else {
                XCTFail("Missing case for forward \(forward.index): \(forward.caseID)")
                continue
            }
            let state = forward.role == "null" ? "" : testCase.state
            let packed = tokenizer.packedInput(
                state: state,
                question: testCase.instructions,
                options: testCase.orderedOptions.map(\.description)
            )

            XCTAssertEqual(packed.text, forward.packedText, "forward \(forward.index) text")
            XCTAssertEqual(packed.inputIDs, forward.inputIDs, "forward \(forward.index) ids")
            XCTAssertEqual(
                packed.attentionMask,
                forward.attentionMask,
                "forward \(forward.index) attention mask"
            )
            XCTAssertEqual(
                packed.maskPositions,
                forward.maskPositions,
                "forward \(forward.index) mask positions"
            )
        }

        XCTAssertEqual(fixture.cases.count, 4)
        for testCase in fixture.cases {
            XCTAssertEqual(
                tokenizer.stateTokenCount(testCase.state),
                testCase.calibration.stateTokenCount,
                "state token count for \(testCase.id)"
            )
        }
    }

    func testSpecialTokensWhitespaceAndNFCParity() async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerResourceURL())

        XCTAssertEqual(tokenizer.encode("", addSpecialTokens: false), [])
        XCTAssertEqual(tokenizer.encode(""), [50_281, 50_282])
        XCTAssertEqual(tokenizer.stateTokenCount(""), 1)
        XCTAssertEqual(tokenizer.encode(" ", addSpecialTokens: false), [209])
        XCTAssertEqual(tokenizer.encode("\t\n", addSpecialTokens: false), [186, 187])

        let composed = tokenizer.encode(" café ", addSpecialTokens: false)
        let decomposed = tokenizer.encode(" cafe\u{0301} ", addSpecialTokens: false)
        XCTAssertEqual(composed, [36_241, 209])
        XCTAssertEqual(decomposed, composed)

        XCTAssertEqual(
            tokenizer.encode("prefix [MASK] value", addSpecialTokens: false),
            [15_445, 50_284, 1_318]
        )
        XCTAssertEqual(
            tokenizer.encode("prefix[MASK] value", addSpecialTokens: false),
            [15_445, 50_284, 1_318]
        )

        XCTAssertEqual(
            VonTokenizer.packSequence(
                state: " \tstate\u{001C}",
                question: "",
                options: ["  yes\n", "\u{3000}no\u{0085}"]
            ),
            "state [SEP] [MASK] yes [MASK] no"
        )
        XCTAssertEqual(
            VonTokenizer.packSequence(state: "", question: "", options: ["yes"]),
            " [SEP] [MASK] yes"
        )
    }

    func testTokenizerResourceProvenanceAndHashes() throws {
        let resourceURL = tokenizerResourceURL()
        let sourceData = try Data(contentsOf: resourceURL.appendingPathComponent("source.json"))
        let source = try JSONDecoder().decode(TokenizerSource.self, from: sourceData)
        let fixture = try loadGoldenFixture()
        let fixtureHashes = Dictionary(
            uniqueKeysWithValues: fixture.metadata.artifacts.map { ($0.name, $0.sha256) }
        )

        XCTAssertEqual(source.source, "https://huggingface.co/wfzyx/von")
        XCTAssertEqual(source.revision, fixture.metadata.hfRevision)
        XCTAssertEqual(source.license, "Apache-2.0")
        XCTAssertEqual(Set(source.files.keys), ["tokenizer.json", "tokenizer_config.json"])

        for (name, expectedHash) in source.files {
            let fileURL = resourceURL.appendingPathComponent(name)
            XCTAssertEqual(try sha256(fileURL), expectedHash, name)
            XCTAssertEqual(expectedHash, fixtureHashes[name], "fixture metadata for \(name)")
        }

        let license = try String(
            contentsOf: resourceURL.appendingPathComponent("LICENSE.md"),
            encoding: .utf8
        )
        XCTAssertTrue(license.contains("Apache License"))
        XCTAssertTrue(license.contains("Version 2.0"))
    }
}

private extension VonTokenizerTests {
    struct GoldenFixture: Decodable {
        let cases: [GoldenCase]
        let forwards: [GoldenForward]
        let metadata: GoldenMetadata
    }

    struct GoldenCase: Decodable {
        let id: String
        let instructions: String
        let state: String
        let orderedOptions: [GoldenOption]
        let calibration: GoldenCalibration

        enum CodingKeys: String, CodingKey {
            case id
            case instructions
            case state
            case orderedOptions = "ordered_options"
            case calibration
        }
    }

    struct GoldenOption: Decodable {
        let description: String
    }

    struct GoldenCalibration: Decodable {
        let stateTokenCount: Int

        enum CodingKeys: String, CodingKey {
            case stateTokenCount = "state_token_count"
        }
    }

    struct GoldenForward: Decodable {
        let attentionMask: [Int]
        let caseID: String
        let index: Int
        let inputIDs: [Int]
        let maskPositions: [Int]
        let packedText: String
        let role: String

        enum CodingKeys: String, CodingKey {
            case attentionMask = "attention_mask"
            case caseID = "case_id"
            case index
            case inputIDs = "input_ids"
            case maskPositions = "mask_positions"
            case packedText = "packed_text"
            case role
        }
    }

    struct GoldenMetadata: Decodable {
        let artifacts: [Artifact]
        let hfRevision: String

        enum CodingKeys: String, CodingKey {
            case artifacts
            case hfRevision = "hf_revision"
        }
    }

    struct Artifact: Decodable {
        let name: String
        let sha256: String
    }

    struct TokenizerSource: Decodable {
        let source: String
        let revision: String
        let license: String
        let files: [String: String]
    }

    func tokenizerResourceURL() -> URL {
        (Bundle.module.resourceURL ?? Bundle.module.bundleURL)
            .appendingPathComponent("Tokenizer", isDirectory: true)
    }

    func loadGoldenFixture() throws -> GoldenFixture {
        let fixtureURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("testdata/von_golden_v1.json", isDirectory: false)
        return try JSONDecoder().decode(GoldenFixture.self, from: Data(contentsOf: fixtureURL))
    }

    func sha256(_ url: URL) throws -> String {
        let digest = SHA256.hash(data: try Data(contentsOf: url))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
