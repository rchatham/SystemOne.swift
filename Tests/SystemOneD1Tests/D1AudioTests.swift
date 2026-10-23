import CryptoKit
import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneD1

final class D1AudioTests: XCTestCase {
    // MARK: Offline: fixture only

    func testSyntheticClipsAndPrefixLengthsMatchPython() throws {
        for expected in try D1Fixture.load().audio.clips {
            let clip = syntheticAudio(samples: expected.samples, seed: expected.seed)
            let bytes = clip.withUnsafeBytes { Data($0) }
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, expected.clipSHA256, expected.name)
            XCTAssertEqual(D1Audio.prefixLength(sampleCount: expected.samples), expected.prefixLength, expected.name)
        }
        // 30 s cap and 0.5 s floor.
        XCTAssertEqual(D1Audio.prefixLength(sampleCount: 1), D1Audio.prefixLength(sampleCount: 8_000))
        XCTAssertEqual(D1Audio.prefixLength(sampleCount: 900_000), D1Audio.prefixLength(sampleCount: 480_000))
    }

    // MARK: Weights: needs D1_MODEL_FOLDER and D1_MODEL_PARITY=1

    func testFrontEndEncoderAndPrefixMatchPython() throws {
        let weights = try D1TextTests.loadWeights()
        try Device.withDefaultDevice(D1TextTests.device) {
            let audio = D1Audio(weights: weights)
            for expected in try D1Fixture.load().audio.clips {
                let clip = AudioClip(pcm16: syntheticAudio(samples: expected.samples, seed: expected.seed))
                let (mel, frames) = audio.melFeatures(clip.samples)
                XCTAssertEqual(frames, expected.frames, expected.name)
                let melProbe = mel[MLXArray(expected.melFrames.map(Int32.init))]
                    .take(MLXArray(expected.melBins.map(Int32.init)), axis: 1)
                assertClose(melProbe, expected.melProbe, 1e-3, "\(expected.name) mel")

                let encoded = audio.encoder(mel, frames: frames)
                assertClose(encoded[MLXArray(expected.encoderRows.map(Int32.init))]
                                .take(MLXArray(expected.encoderDimensions.map(Int32.init)), axis: 1),
                            expected.encoderProbe, 2e-3, "\(expected.name) encoder")

                let prefix = try audio(clip)
                XCTAssertEqual(prefix.dim(0), expected.prefixLength, expected.name)
                assertClose(prefix[MLXArray(expected.prefixRows.map(Int32.init))]
                                .take(MLXArray(D1TextTests.probeDimensions.map(Int32.init)), axis: 1),
                            expected.prefixProbe, 2e-3, "\(expected.name) prefix")
            }
        }
    }

    func testAudioRequestsMatchPython() async throws {
        let weights = try D1TextTests.loadWeights()
        let folder = try D1Fixture.modelFolder()
        let engine = SystemOneEngine(backend: D1Backend(
            modelID: "d1-omni-600M",
            configuration: weights.configuration,
            tokenizer: try await D1Tokenizer(modelFolder: folder),
            model: D1Model(weights: weights),
            logitsProvider: nil
        ))
        for testCase in try D1Fixture.load().audio.requests {
            let clip = AudioClip(pcm16: syntheticAudio(samples: testCase.audio.samples, seed: testCase.audio.seed))
            let response = try await Device.withDefaultDevice(D1TextTests.device) {
                try await engine.systemOne(
                    state: testCase.request["state"]!.state,
                    questions: testCase.request["questions"]!.array.map(\.questionEntry),
                    media: .audio(clip)
                )
            }
            XCTAssertEqual(response.usage.inputTokens, Int(testCase.result["usage"]!["input_tokens"]!.number!))
            for (answer, expected) in zip(response.answers, testCase.result["answers"]!.array) {
                assert(answer.answer, matches: expected, tolerance: 2e-3, "\(testCase.id).\(answer.id)")
            }
        }
    }
}
