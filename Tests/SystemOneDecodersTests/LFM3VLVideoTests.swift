#if canImport(CryptoKit)
import CryptoKit
import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// The video part of `testdata/d1_omni_2b_audio_tiny/fixture.json`, written by
/// `tools/d1_omni_2b_audio_fixture.py` with d1-Omni-2B's vendored video processor.
private struct LFM3VLVideoFixture: Decodable {
    struct Video: Decodable {
        /// width, height, salt, frames, fps.
        let source: [Double]
        let indices: [Int]
        let resized: [Int]
        let resizedSHA256: [String]
        let groups: Int
        let tokens: Int
        let seconds: [Double]
        let markup: String
        let features: LFM2VLFixture.Features

        enum CodingKeys: String, CodingKey {
            case source, indices, resized, groups, tokens, seconds, markup, features
            case resizedSHA256 = "resized_sha256"
        }

        var fps: Double { source[4] }

        /// Frame `i` is `tools/resize_fixture.py`'s picture with salt `salt + 7 i`.
        var clip: VideoClip {
            let (width, height, salt, count) = (Int(source[0]), Int(source[1]), Int(source[2]), Int(source[3]))
            return VideoClip(frames: (0..<count).map { LFM2VLFixture.synthetic([width, height, salt + 7 * $0]) }, fps: fps)
        }
    }

    let videos: [String: Video]

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_omni_2b_audio_tiny")
    static let visionFolder = folder.deletingLastPathComponent().appendingPathComponent("d1_2b_vision_tiny")

    static func load() throws -> LFM3VLVideoFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
    }
}

final class LFM3VLVideoTests: XCTestCase {
    func testReadsTheVideoProcessor() throws {
        let omni = try XCTUnwrap(try LFM3VLVisionConfiguration(modelFolder: LFM3VLVideoFixture.folder).video)
        XCTAssertEqual(omni.tokenID, 123)
        XCTAssertEqual(omni.fps, 2)
        XCTAssertEqual(omni.minFrames, 4)
        XCTAssertEqual(omni.maxFrames, 384)
        XCTAssertEqual(omni.minPixels, 8_192)
        XCTAssertEqual(omni.maxPixels, 200_000)
        XCTAssertFalse(omni.capPixelsPerFrame)
        XCTAssertTrue(omni.sumsIndices)
        // d1-2B's tiny checkpoint has a video token but no video processor: the reference's defaults.
        let vl = try XCTUnwrap(try LFM3VLVisionConfiguration(modelFolder: LFM3VLVideoFixture.visionFolder).video)
        XCTAssertEqual(vl.maxFrames, 768)
        XCTAssertEqual(vl.maxPixels, 25_165_824)
        XCTAssertFalse(vl.sumsIndices)

        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: LFM3VLVideoFixture.folder.appendingPathComponent("config.json"))) as! [String: Any]
        XCTAssertThrowsError(try LFM3VLVisionConfiguration(json: config, processor: [:], video: ["temporal_patch_size": 4]))
        XCTAssertThrowsError(try LFM3VLVisionConfiguration(json: config, processor: [:], video: ["fps": 0]))
        var silent = config
        silent["video_token_id"] = nil
        XCTAssertNil(try LFM3VLVisionConfiguration(json: silent, processor: [:]).video)
    }

    func testSamplesResizesAndMarksUpAsTheProcessor() throws {
        let fixture = try LFM3VLVideoFixture.load()
        let c = try LFM3VLVisionConfiguration(modelFolder: LFM3VLVideoFixture.folder)
        XCTAssertEqual(fixture.videos.count, 3)
        for (name, video) in fixture.videos {
            let clip = video.clip
            let layout = try LFM3VLVision.videoLayout(
                total: clip.frames.count, width: clip.frames[0].width, height: clip.frames[0].height, fps: clip.fps, configuration: c
            )
            XCTAssertEqual(layout.indices, video.indices, name)
            XCTAssertEqual([layout.height, layout.width], video.resized, name)
            XCTAssertEqual(layout.seconds.count, video.groups, name)
            XCTAssertEqual(layout.tokensPerGroup, video.tokens, name)
            XCTAssertEqual(layout.seconds, video.seconds, name)
            XCTAssertEqual(LFM3VLVision.videoMarkup(layout), video.markup, name)
            let digests = layout.indices.map { index -> String in
                let frame = clip.frames[index]
                return LFM2VLFixture.digest(AntialiasedResize.resize(
                    frame.pixels, width: frame.width, height: frame.height, channels: 3,
                    toWidth: layout.width, toHeight: layout.height, filter: .cubic, kernel: .aten
                ))
            }
            XCTAssertEqual(digests, video.resizedSHA256, name)
        }
        // Too few frames, or no rate, is refused.
        XCTAssertThrowsError(try LFM3VLVision.videoLayout(total: 1, width: 64, height: 64, fps: 2, configuration: c))
        XCTAssertThrowsError(try LFM3VLVision.videoLayout(total: 8, width: 64, height: 64, fps: 0, configuration: c))
        // A vanishing rate wants more frames than `Int` holds: it takes every frame.
        XCTAssertEqual(LFM3VLVision.sampledFrames(total: 8, fps: 1e-300, video: try XCTUnwrap(c.video)), Array(0..<8))
    }

    /// LFM3-Omni and LFM3-VL time a group differently, which can print other
    /// tenths: 26 frames at 10 fps sample 0, 6, 12, 19, 25, and the second
    /// group is 1.55 s, `(12 + 19) / 20` against `(12 / 10 + 19 / 10) / 2`.
    func testOmniAndVLTimestampsDifferWhereTheReferencesDo() throws {
        let omni = try LFM3VLVisionConfiguration(modelFolder: LFM3VLVideoFixture.folder)
        let vl = try LFM3VLVisionConfiguration(modelFolder: LFM3VLVideoFixture.visionFolder)
        func stamps(_ c: LFM3VLVisionConfiguration) throws -> [String] {
            try LFM3VLVision.videoLayout(total: 26, width: 64, height: 64, fps: 10, configuration: c).seconds
                .map { String(format: "%.1f", $0) }
        }
        XCTAssertEqual(try stamps(omni), ["0.3", "1.6", "2.5"])
        XCTAssertEqual(try stamps(vl), ["0.3", "1.5", "2.5"])
        // Python's `.1f` breaks exact ties to even.
        let tiny = try LFM3VLVision.videoLayout(total: 4, width: 20, height: 20, fps: 2, configuration: omni)
        XCTAssertTrue(LFM3VLVision.videoMarkup(tiny).hasPrefix("<|vision_start|><0.2 seconds><|vision_start|><|video|>"))
        XCTAssertTrue(LFM3VLVision.videoMarkup(tiny).contains("<|vision_end|><1.2 seconds><|vision_start|>"))
    }
}

/// Video features in MLX: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class LFM3VLVideoRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testVideoFeaturesMatchMLXVLMInBothLayouts() throws {
        let fixture = try LFM3VLVideoFixture.load()
        try Device.withDefaultDevice(device) {
            for folder in [LFM3VLVideoFixture.folder, LFM3VLVideoFixture.folder.appendingPathComponent("mlx")] {
                let vision = try LFM3VLVision(modelFolder: folder)
                for (name, video) in fixture.videos {
                    let label = "\(folder.lastPathComponent) \(name)"
                    let clip = video.clip
                    let layout = try LFM3VLVision.videoLayout(
                        total: clip.frames.count, width: clip.frames[0].width, height: clip.frames[0].height,
                        fps: clip.fps, configuration: vision.configuration
                    )
                    let features = try vision(video: layout.indices.map { clip.frames[$0] }, layout: layout).asType(.float32)
                    XCTAssertEqual(features.shape, [video.features.count, 32], label)
                    let values = features.asArray(Float.self)
                    let sum = values.reduce(0.0) { $0 + Double($1) }
                    let absSum = values.reduce(0.0) { $0 + Double(abs($1)) }
                    XCTAssertEqual(sum, video.features.sum, accuracy: 1e-3 * max(1, abs(video.features.sum)), label)
                    XCTAssertEqual(absSum, video.features.absSum, accuracy: 1e-4 * video.features.absSum, label)
                    XCTAssertLessThanOrEqual(zip(values.prefix(32), video.features.first).map { abs($0 - $1) }.max() ?? .infinity, 1e-4, label)
                    XCTAssertLessThanOrEqual(zip(values.suffix(32), video.features.last).map { abs($0 - $1) }.max() ?? .infinity, 1e-4, label)
                }
            }
        }
    }
}
#endif
