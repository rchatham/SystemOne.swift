#if canImport(CryptoKit)
import CryptoKit
import Foundation
import MLX
import SystemOne
import XCTest
@testable import SystemOneDecoders

/// The image part of `testdata/d1_3b_vision_tiny/fixture.json`, written by
/// `tools/d1_3b_vision_fixture.py` from transformers' LFM2-VL.
struct LFM2VLFixture: Decodable {
    struct View: Decodable {
        let size: [Int]
        let sha256: String
    }

    struct Features: Decodable {
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

    struct Picture: Decodable {
        let source: [Int]
        let capped: [Int]
        let cappedSHA256: String
        let rows: Int
        let columns: Int
        let resized: [Int]
        let markupTokens: Int
        let views: [View]
        let features: Features

        enum CodingKeys: String, CodingKey {
            case source, capped, rows, columns, resized, views, features
            case cappedSHA256 = "capped_sha256"
            case markupTokens = "markup_tokens"
        }
    }

    let images: [String: Picture]

    static let folder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("testdata/d1_3b_vision_tiny")

    static func load() throws -> LFM2VLFixture {
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

    /// d1-3B's one-megapixel cap (Pillow's bicubic), as `D1ChatPrompt.capPixels` writes it.
    static func capped(_ image: RGBImage) -> RGBImage {
        let limit = 1024 * 1024
        guard image.width * image.height > limit else { return image }
        let scale = (Double(limit) / Double(image.width * image.height)).squareRoot()
        let width = max(1, Int(Double(image.width) * scale))
        let height = max(1, Int(Double(image.height) * scale))
        return RGBImage(width: width, height: height, pixels: AntialiasedResize.resize(
            image.pixels, width: image.width, height: image.height, channels: 3,
            toWidth: width, toHeight: height, filter: .cubic, kernel: .pillow
        ))
    }

    static func digest(_ bytes: [UInt8]) -> String {
        SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }
}

final class LFM2VLVisionTests: XCTestCase {
    func testReadsTheConfigAndProcessorSettings() throws {
        let c = try LFM2VLVisionConfiguration(modelFolder: LFM2VLFixture.folder)
        XCTAssertEqual(c.hiddenSize, 32)
        XCTAssertEqual(c.layers, 2)
        XCTAssertEqual(c.positionSide, 16)
        XCTAssertEqual(c.textHiddenSize, 32)
        XCTAssertEqual(c.projectorHiddenSize, 48)
        XCTAssertFalse(c.projectorLayerNorm)
        XCTAssertEqual(c.imageTokenID, 120)
        XCTAssertEqual(c.resampling, .bicubic)
        XCTAssertEqual(c.minImageTokens, 64)
        XCTAssertEqual(c.maxImageTokens, 256)
        XCTAssertTrue(c.imageSplitting && c.useThumbnail && c.imageSpecialTokens)

        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: LFM2VLFixture.folder.appendingPathComponent("config.json"))) as! [String: Any]
        XCTAssertThrowsError(try LFM2VLVisionConfiguration(json: config, processor: ["min_tiles": 1, "max_tiles": 4]))
        XCTAssertThrowsError(try LFM2VLVisionConfiguration(json: config, processor: ["resample": 1]))
        XCTAssertNoThrow(try LFM2VLVisionConfiguration(json: config, processor: ["do_image_splitting": false, "max_tiles": 4]))
    }

    func testLayoutsViewsAndMarkupMatchTheProcessor() throws {
        let fixture = try LFM2VLFixture.load()
        let c = try LFM2VLVisionConfiguration(modelFolder: LFM2VLFixture.folder)
        XCTAssertEqual(fixture.images.count, 3)
        for (name, picture) in fixture.images {
            let image = LFM2VLFixture.capped(LFM2VLFixture.synthetic(picture.source))
            XCTAssertEqual([image.width, image.height], picture.capped, name)
            XCTAssertEqual(LFM2VLFixture.digest(image.pixels), picture.cappedSHA256, name)
            let layout = LFM2VLVision.layout(width: image.width, height: image.height, configuration: c)
            XCTAssertEqual(layout.rows, picture.rows, name)
            XCTAssertEqual(layout.columns, picture.columns, name)
            XCTAssertEqual([layout.height, layout.width], picture.resized, name)
            XCTAssertEqual(layout.crops, picture.views.map(\.size), name)
            let markup = LFM2VLVision.markup(layout, configuration: c)
            XCTAssertEqual(markup.components(separatedBy: LFM2VLVision.imageToken).count - 1, picture.markupTokens, name)
            XCTAssertEqual(LFM2VLVision.featureCount(layout, configuration: c), picture.features.count, name)
            XCTAssertTrue(markup.hasPrefix("<|image_start|>") && markup.hasSuffix("<|image_end|>"), name)
            XCTAssertEqual(markup.contains("<|img_thumbnail|>"), layout.tiled, name)
            let views = LFM2VLVision.crops(of: image, configuration: c)
            XCTAssertEqual(views.map { [$0.height, $0.width] }, picture.views.map(\.size), name)
            XCTAssertEqual(views.map { LFM2VLFixture.digest($0.pixels) }, picture.views.map(\.sha256), name)
        }
    }

    func testTiledMarkupNamesEveryTile() throws {
        let c = try LFM2VLVisionConfiguration(modelFolder: LFM2VLFixture.folder)
        let layout = LFM2VLImageLayout(columns: 3, rows: 2, height: 416, width: 576, tileSize: 512, thumbnail: true)
        let markup = LFM2VLVision.markup(layout, configuration: c)
        let tags = ["<|img_row_1_col_1|>", "<|img_row_1_col_2|>", "<|img_row_1_col_3|>",
                    "<|img_row_2_col_1|>", "<|img_row_2_col_2|>", "<|img_row_2_col_3|>", "<|img_thumbnail|>"]
        var cursor = markup.startIndex
        for tag in tags {
            guard let range = markup.range(of: tag, range: cursor..<markup.endIndex) else { return XCTFail("missing \(tag)") }
            cursor = range.upperBound
        }
        XCTAssertEqual(LFM2VLVision.featureCount(layout, configuration: c), 6 * 256 + 13 * 18)
    }
}

/// The tower and projector in MLX: only from an Xcode-built test bundle with `SYSTEMONE_MLX_TESTS=1`.
final class LFM2VLVisionRuntimeTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["SYSTEMONE_MLX_TESTS"] == "1" else {
            throw XCTSkip("Set SYSTEMONE_MLX_TESTS=1 in an Xcode-built test bundle to run MLX.")
        }
    }

    private var device: Device {
        ProcessInfo.processInfo.environment["SYSTEMONE_MLX_DEVICE"] == "gpu" ? .gpu : .cpu
    }

    func testFeaturesMatchTransformersInBothLayouts() throws {
        let fixture = try LFM2VLFixture.load()
        try Device.withDefaultDevice(device) {
            for folder in [LFM2VLFixture.folder, LFM2VLFixture.folder.appendingPathComponent("mlx")] {
                let vision = try LFM2VLVision(modelFolder: folder)
                for (name, picture) in fixture.images {
                    let label = "\(folder.lastPathComponent) \(name)"
                    let features = vision(LFM2VLFixture.capped(LFM2VLFixture.synthetic(picture.source))).asType(.float32)
                    XCTAssertEqual(features.shape, [picture.features.count, 32], label)
                    let values = features.asArray(Float.self)
                    let sum = values.reduce(0.0) { $0 + Double($1) }
                    let absSum = values.reduce(0.0) { $0 + Double(abs($1)) }
                    XCTAssertEqual(sum, picture.features.sum, accuracy: 1e-3 * max(1, abs(picture.features.sum)), label)
                    XCTAssertEqual(absSum, picture.features.absSum, accuracy: 1e-4 * picture.features.absSum, label)
                    let first = Array(values.prefix(32))
                    let last = Array(values.suffix(32))
                    XCTAssertLessThanOrEqual(zip(first, picture.features.first).map { abs($0 - $1) }.max() ?? .infinity, 1e-4, label)
                    XCTAssertLessThanOrEqual(zip(last, picture.features.last).map { abs($0 - $1) }.max() ?? .infinity, 1e-4, label)
                }
            }
        }
    }

    func testTrunkPlacesFeaturesAtPlaceholders() throws {
        try Device.withDefaultDevice(device) {
            let trunk = try LFM2Trunk(modelFolder: LFM2VLFixture.folder)
            let features = MLXArray((0..<64).map { Float($0) / 16 }, [2, 32])
            let ids = [5, 120, 7, 120, 9]
            let placed = try trunk([ids], features: DecoderFeatures(token: 120, rows: [features])).asType(.float32)
            let plain = try trunk([ids]).asType(.float32)
            XCTAssertEqual(placed.shape, [1, 5, 32])
            // Positions before the first placeholder are untouched; the rest differ.
            let before = abs(placed[0, 0] - plain[0, 0]).max().item(Float.self)
            let after = abs(placed[0, 4] - plain[0, 4]).max().item(Float.self)
            XCTAssertLessThan(before, 1e-6)
            XCTAssertGreaterThan(after, 1e-4)
            XCTAssertThrowsError(try trunk([ids], features: DecoderFeatures(token: 120, rows: [features[0 ..< 1]])))
            XCTAssertThrowsError(try trunk([ids], features: DecoderFeatures(token: 120, rows: [nil])))
        }
    }
}
#endif
