#if canImport(CryptoKit)
import CryptoKit
import Foundation
import XCTest
@testable import SystemOne

/// `testdata/antialiased_resize_v1.json`, written by `tools/resize_fixture.py`.
private struct ResizeFixture: Decodable {
    struct Record: Decodable {
        let sha256: String
        let sum: Int
    }

    struct Case: Decodable {
        let input: [Int]
        let output: [Int]
        let source: Record
        let atenCubic: Record
        let atenTriangle: Record
        let pillowCubic: Record
        let pillowTriangle: Record

        enum CodingKeys: String, CodingKey {
            case input, output, source
            case atenCubic = "aten_cubic"
            case atenTriangle = "aten_triangle"
            case pillowCubic = "pillow_cubic"
            case pillowTriangle = "pillow_triangle"
        }
    }

    let cases: [Case]

    static func load() throws -> ResizeFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("testdata/antialiased_resize_v1.json")
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

final class AntialiasedResizeTests: XCTestCase {
    /// `tools/resize_fixture.py`'s synthetic picture, interleaved RGB.
    static func synthetic(width: Int, height: Int, salt: Int) -> [UInt8] {
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
        return pixels
    }

    static func digest(_ bytes: [UInt8]) -> String {
        SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }

    func testMatchesTorchvisionAndPillowBitForBit() throws {
        let fixture = try ResizeFixture.load()
        XCTAssertFalse(fixture.cases.isEmpty)
        for item in fixture.cases {
            let (width, height, salt) = (item.input[0], item.input[1], item.input[2])
            let (toWidth, toHeight) = (item.output[0], item.output[1])
            let label = "\(width)x\(height) to \(toWidth)x\(toHeight)"
            let source = Self.synthetic(width: width, height: height, salt: salt)
            XCTAssertEqual(Self.digest(source), item.source.sha256, "source \(label)")
            let expected: [(AntialiasedResize.Filter, AntialiasedResize.Kernel, ResizeFixture.Record, String)] = [
                (.cubic, .aten, item.atenCubic, "torchvision bicubic"),
                (.triangle, .aten, item.atenTriangle, "torchvision bilinear"),
                (.cubic, .pillow, item.pillowCubic, "Pillow bicubic"),
                (.triangle, .pillow, item.pillowTriangle, "Pillow bilinear"),
            ]
            for (filter, kernel, record, name) in expected {
                let resized = AntialiasedResize.resize(
                    source, width: width, height: height, channels: 3, toWidth: toWidth, toHeight: toHeight,
                    filter: filter, kernel: kernel
                )
                XCTAssertEqual(resized.reduce(0) { $0 + Int($1) }, record.sum, "\(name) \(label)")
                XCTAssertEqual(Self.digest(resized), record.sha256, "\(name) \(label)")
            }
        }
    }

    func testCubicKernelFormsAgreeAndVanishOutsideSupport() {
        for x in stride(from: -2.5, through: 2.5, by: 0.125) {
            let aten = AntialiasedResize.evaluate(.cubic, kernel: .aten, at: x)
            let pillow = AntialiasedResize.evaluate(.cubic, kernel: .pillow, at: x)
            XCTAssertEqual(aten, pillow, accuracy: 1e-12, "x = \(x)")
        }
        XCTAssertEqual(AntialiasedResize.evaluate(.cubic, kernel: .aten, at: 0), 1)
        XCTAssertEqual(AntialiasedResize.evaluate(.cubic, kernel: .aten, at: 2), 0)
        XCTAssertEqual(AntialiasedResize.evaluate(.triangle, kernel: .aten, at: 0.25), 0.75)
    }
}
#endif
