#if canImport(CryptoKit)
import CryptoKit
import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// The image part of `testdata/d1_2b_vision_tiny/fixture.json`, written by
/// `tools/d1_2b_vision_fixture.py`.
private struct LFM3VLFixture: Decodable {
    struct Picture: Decodable {
        let source: [Int]
        let capped: [Int]
        let cappedSHA256: String
        let resized: [Int]
        let resizedSHA256: String
        let features: LFM2VLFixture.Features

        enum CodingKeys: String, CodingKey {
            case source, capped, resized, features
            case cappedSHA256 = "capped_sha256"
            case resizedSHA256 = "resized_sha256"
        }
    }

    let maxPixels: Int
    let images: [String: Picture]

    enum CodingKeys: String, CodingKey {
        case images
        case maxPixels = "max_pixels"
    }

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_2b_vision_tiny")

    static func load() throws -> LFM3VLFixture {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: folder.appendingPathComponent("fixture.json")))
    }

    /// d1-2B's cap at `limit` pixels (Pillow's bicubic), as `D1ChatPrompt.capPixels` writes it.
    static func capped(_ image: RGBImage, limit: Int) -> RGBImage {
        guard image.width * image.height > limit else { return image }
        let scale = (Double(limit) / Double(image.width * image.height)).squareRoot()
        let width = max(1, Int(Double(image.width) * scale))
        let height = max(1, Int(Double(image.height) * scale))
        return RGBImage(width: width, height: height, pixels: AntialiasedResize.resize(
            image.pixels, width: image.width, height: image.height, channels: 3,
            toWidth: width, toHeight: height, filter: .cubic, kernel: .pillow
        ))
    }
}

final class LFM3VLVisionTests: XCTestCase {
    func testReadsTheConfigAndProcessorSettings() throws {
        let c = try LFM3VLVisionConfiguration(modelFolder: LFM3VLFixture.folder)
        XCTAssertEqual(c.hiddenSize, 32)
        XCTAssertEqual(c.layerTypes, [.convolution, .fullAttention, .convolution])
        XCTAssertEqual(c.attentionHeads, 4)
        XCTAssertEqual(c.keyValueHeads, 2)
        XCTAssertEqual(c.headDimension, 8)
        XCTAssertEqual(c.temporalPatchSize, 2)
        XCTAssertEqual(c.mergeSize, 2)
        XCTAssertEqual(c.minPixels, 65_536)
        XCTAssertEqual(c.maxPixels, 16_777_216)
        XCTAssertEqual(c.resampling, .bicubic)
        XCTAssertEqual(c.imageTokenID, 120)
        XCTAssertEqual(c.projectorHiddenSize, 48)

        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: LFM3VLFixture.folder.appendingPathComponent("config.json"))) as! [String: Any]
        XCTAssertThrowsError(try LFM3VLVisionConfiguration(json: config, processor: ["merge_size": 4]))
        XCTAssertThrowsError(try LFM3VLVisionConfiguration(json: config, processor: ["resample": 0]))
        XCTAssertThrowsError(try LFM3VLVision.smartResize(height: 10, width: 2_100, configuration: c))
    }

    func testResizesAndMarksUpAsTheProcessor() throws {
        let fixture = try LFM3VLFixture.load()
        let c = try LFM3VLVisionConfiguration(modelFolder: LFM3VLFixture.folder)
        XCTAssertEqual(fixture.images.count, 3)
        for (name, picture) in fixture.images {
            let image = LFM3VLFixture.capped(LFM2VLFixture.synthetic(picture.source), limit: fixture.maxPixels)
            XCTAssertEqual([image.width, image.height], picture.capped, name)
            XCTAssertEqual(LFM2VLFixture.digest(image.pixels), picture.cappedSHA256, name)
            let view = try LFM3VLVision.resized(image, configuration: c)
            XCTAssertEqual([view.height, view.width], picture.resized, name)
            XCTAssertEqual(LFM2VLFixture.digest(view.pixels), picture.resizedSHA256, name)
            let count = LFM3VLVision.featureCount(height: view.height, width: view.width, configuration: c)
            XCTAssertEqual(count, picture.features.count, name)
            let markup = LFM3VLVision.markup(featureCount: count)
            XCTAssertTrue(markup.hasPrefix("<|vision_start|><|image|>") && markup.hasSuffix("<|image|><|vision_end|>"), name)
            XCTAssertEqual(markup.components(separatedBy: LFM3VLVision.imageToken).count - 1, count, name)
        }
    }
}

/// The tower and projector in MLX: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class LFM3VLVisionRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testAxialRotationRunsRowThenColumnInMergeOrder() throws {
        let c = try LFM3VLVisionConfiguration(modelFolder: LFM3VLFixture.folder)
        // 2 × 4 patches: merge blocks (0, 0) then (0, 1); the fourth patch is row 1, column 1.
        let angles = Device.withDefaultDevice(device) { () -> [Float] in
            let rotation = LFM3VLVision.rotation(rows: 2, columns: 4, configuration: c)
            XCTAssertEqual(rotation.cos.shape, [8, 8])
            return MLX.acos(MLX.clip(rotation.cos, min: Float(-1), max: Float(1))).asArray(Float.self)
        }
        // Head dimension 8: two frequencies per axis, [row f0, row f1, column f0, column f1] twice.
        XCTAssertEqual(angles[3 * 8 + 0], 1, accuracy: 1e-5)
        XCTAssertEqual(angles[3 * 8 + 2], 1, accuracy: 1e-5)
        XCTAssertEqual(angles[4 * 8 + 0], 0, accuracy: 1e-5)
        XCTAssertEqual(angles[4 * 8 + 2], 2, accuracy: 1e-5)
        XCTAssertEqual(angles[4 * 8 + 6], 2, accuracy: 1e-5)
    }

    func testFeaturesMatchMLXVLMInBothLayouts() throws {
        let fixture = try LFM3VLFixture.load()
        try Device.withDefaultDevice(device) {
            for folder in [LFM3VLFixture.folder, LFM3VLFixture.folder.appendingPathComponent("mlx")] {
                let vision = try LFM3VLVision(modelFolder: folder)
                for (name, picture) in fixture.images {
                    let label = "\(folder.lastPathComponent) \(name)"
                    let image = LFM3VLFixture.capped(LFM2VLFixture.synthetic(picture.source), limit: fixture.maxPixels)
                    let features = try vision(image).asType(.float32)
                    XCTAssertEqual(features.shape, [picture.features.count, 32], label)
                    let values = features.asArray(Float.self)
                    let sum = values.reduce(0.0) { $0 + Double($1) }
                    let absSum = values.reduce(0.0) { $0 + Double(abs($1)) }
                    XCTAssertEqual(sum, picture.features.sum, accuracy: 1e-3 * max(1, abs(picture.features.sum)), label)
                    XCTAssertEqual(absSum, picture.features.absSum, accuracy: 1e-4 * picture.features.absSum, label)
                    XCTAssertLessThanOrEqual(zip(values.prefix(32), picture.features.first).map { abs($0 - $1) }.max() ?? .infinity, 1e-4, label)
                    XCTAssertLessThanOrEqual(zip(values.suffix(32), picture.features.last).map { abs($0 - $1) }.max() ?? .infinity, 1e-4, label)
                }
            }
        }
    }
}
#endif
