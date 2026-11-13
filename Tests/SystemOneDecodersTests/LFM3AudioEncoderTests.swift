import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// The clip part of `testdata/d1_omni_2b_audio_tiny/fixture.json`, written by
/// `tools/d1_omni_2b_audio_fixture.py`.
private struct LFM3AudioFixture: Decodable {
    /// A summary whose first and last entries are rows.
    struct Rows: Decodable {
        let count: Int
        let sum: Double
        let absSum: Double
        let first: [Float]
        let last: [Float]

        enum CodingKeys: String, CodingKey {
            case count, sum, first, last
            case absSum = "abs_sum"
        }
    }

    /// A summary whose first and last entries are samples.
    struct Samples: Decodable {
        let count: Int
        let sum: Double
        let first: Float
        let last: Float
    }

    struct Clip: Decodable {
        let source: [Int]
        let resampled: Samples
        let frames: Int
        let mel: Rows
        let features: Rows
    }

    let clips: [String: Clip]

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_omni_2b_audio_tiny")

    static func load() throws -> LFM3AudioFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
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
}

final class LFM3AudioEncoderTests: XCTestCase {
    func testReadsTheEncoderAndFeatureExtractorSettings() throws {
        let c = try LFM3AudioConfiguration(modelFolder: LFM3AudioFixture.folder)
        XCTAssertEqual(c.convolutionChannels, 8)
        XCTAssertEqual(c.hiddenSize, 16)
        XCTAssertEqual(c.melBins, 128)
        XCTAssertEqual(c.slidingWindow, 4)
        XCTAssertEqual(c.trunk.layerTypes, [.convolution, .fullAttention, .convolution])
        XCTAssertEqual(c.trunk.intermediateSize, 32)
        XCTAssertEqual(c.trunk.keyValueHeads, 2)
        XCTAssertEqual(c.textHiddenSize, 32)
        XCTAssertEqual(c.audioTokenID, 124)
        XCTAssertEqual(c.sampleRate, 16_000)
        XCTAssertEqual(c.fftSize, 400)
        XCTAssertEqual(c.hop, 160)
        XCTAssertEqual(c.logFloor, -12)
        XCTAssertEqual(c.encoderStride, 8)

        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: LFM3AudioFixture.folder.appendingPathComponent("config.json"))) as! [String: Any]
        XCTAssertThrowsError(try LFM3AudioConfiguration(json: config, processor: ["win_length": 320]))
        XCTAssertThrowsError(try LFM3AudioConfiguration(json: config, processor: ["encoder_stride": 4]))
        XCTAssertThrowsError(try LFM3AudioConfiguration(json: config, processor: ["padding_side": "left"]))
        XCTAssertThrowsError(try LFM3AudioConfiguration(json: [:], processor: [:]))
    }

    func testCountsFramesAndTokensAsTheExtractor() throws {
        let fixture = try LFM3AudioFixture.load()
        let c = try LFM3AudioConfiguration(modelFolder: LFM3AudioFixture.folder)
        XCTAssertEqual(fixture.clips.count, 6)
        for (name, info) in fixture.clips {
            let clip = LFM3AudioFixture.clip(info.source)
            let samples = try LFM3AudioEncoder.sampleCount(of: clip, configuration: c)
            XCTAssertEqual(samples, info.resampled.count, name)
            XCTAssertEqual(LFM3AudioEncoder.frameCount(sampleCount: samples, configuration: c), info.frames, name)
            let count = try LFM3AudioEncoder.featureCount(of: clip, configuration: c)
            XCTAssertEqual(count, info.features.count, name)
            let markup = LFM3AudioEncoder.markup(featureCount: count)
            XCTAssertTrue(markup.hasPrefix("<|vision_start|><|audio|>") && markup.hasSuffix("<|audio|><|vision_end|>"), name)
            XCTAssertEqual(markup.components(separatedBy: LFM3AudioEncoder.audioToken).count - 1, count, name)
        }
        // Whole frames: 400 samples is one, 401 two; a clip under one window is padded to it.
        XCTAssertEqual(LFM3AudioEncoder.frameCount(sampleCount: 0, configuration: c), 1)
        XCTAssertEqual(LFM3AudioEncoder.frameCount(sampleCount: 400, configuration: c), 1)
        XCTAssertEqual(LFM3AudioEncoder.frameCount(sampleCount: 401, configuration: c), 2)
        XCTAssertEqual(LFM3AudioEncoder.frameCount(sampleCount: 560, configuration: c), 2)
        XCTAssertThrowsError(try LFM3AudioEncoder.featureCount(of: AudioClip(samples: [0], sampleRate: 0), configuration: c))
    }

    func testResamplesAsD1Media() throws {
        let fixture = try LFM3AudioFixture.load()
        for (name, info) in fixture.clips {
            let clip = LFM3AudioFixture.clip(info.source)
            let samples = SincResample.resample(clip.samples, from: clip.sampleRate, to: 16_000)
            XCTAssertEqual(samples.count, info.resampled.count, name)
            XCTAssertEqual(samples.reduce(0.0) { $0 + Double($1) }, info.resampled.sum, accuracy: 1e-3, name)
            XCTAssertEqual(samples.first ?? .nan, info.resampled.first, accuracy: 1e-5, name)
            XCTAssertEqual(samples.last ?? .nan, info.resampled.last, accuracy: 1e-5, name)
        }
        XCTAssertEqual(SincResample.outputCount(441, from: 44_100, to: 16_000), 160)
        XCTAssertEqual(SincResample.outputCount(442, from: 44_100, to: 16_000), 161)
        XCTAssertEqual(SincResample.resample([], from: 8_000, to: 16_000), [])
    }
}

/// The front end, encoder and projector in MLX: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class LFM3AudioEncoderRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    private func assertRows(
        _ array: MLXArray, _ expected: LFM3AudioFixture.Rows, width: Int, element: Float, label: String
    ) {
        XCTAssertEqual(array.shape, [expected.count, width], label)
        let values = array.asType(.float32).asArray(Float.self)
        let sum = values.reduce(0.0) { $0 + Double($1) }
        let absSum = values.reduce(0.0) { $0 + Double(abs($1)) }
        XCTAssertEqual(sum, expected.sum, accuracy: 1e-3 * max(1, abs(expected.sum)), label)
        XCTAssertEqual(absSum, expected.absSum, accuracy: 1e-3 * max(1, expected.absSum), label)
        XCTAssertLessThanOrEqual(zip(values.prefix(width), expected.first).map { abs($0 - $1) }.max() ?? .infinity, element, label)
        XCTAssertLessThanOrEqual(zip(values.suffix(width), expected.last).map { abs($0 - $1) }.max() ?? .infinity, element, label)
    }

    func testLogMelMatchesTheExtractor() throws {
        let fixture = try LFM3AudioFixture.load()
        try Device.withDefaultDevice(device) {
            let encoder = try LFM3AudioEncoder(modelFolder: LFM3AudioFixture.folder)
            for (name, info) in fixture.clips {
                let clip = LFM3AudioFixture.clip(info.source)
                let samples = SincResample.resample(clip.samples, from: clip.sampleRate, to: 16_000)
                // Upsampled 8 kHz audio leaves near-silent bins, where float32 rounding shows.
                assertRows(encoder.logMel(samples), info.mel, width: 128, element: 3e-3, label: name)
            }
        }
    }

    func testFeaturesMatchMLXVLMInBothLayouts() throws {
        let fixture = try LFM3AudioFixture.load()
        try Device.withDefaultDevice(device) {
            for folder in [LFM3AudioFixture.folder, LFM3AudioFixture.folder.appendingPathComponent("mlx")] {
                let encoder = try LFM3AudioEncoder(modelFolder: folder)
                for (name, info) in fixture.clips {
                    let features = try encoder(LFM3AudioFixture.clip(info.source))
                    assertRows(features, info.features, width: 32, element: 3e-3, label: "\(folder.lastPathComponent) \(name)")
                }
            }
        }
    }
}
