import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
#endif

/// Non-text input that accompanies a state. A request carries images or one
/// audio clip, never both, matching the models trained on them.
public enum SystemOneMedia: Equatable, Sendable {
    /// One or more images, encoded in order.
    case images([RGBImage])

    /// One mono speech clip.
    case audio(AudioClip)
}

/// An 8-bit RGB image in row-major, interleaved `RGBRGB…` order.
public struct RGBImage: Equatable, Sendable {
    public let width: Int
    public let height: Int
    public let pixels: [UInt8]

    /// Wraps interleaved RGB bytes; `pixels.count` must equal `width * height * 3`.
    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(width > 0 && height > 0, "RGBImage dimensions must be positive")
        precondition(pixels.count == width * height * 3, "RGBImage needs width * height * 3 bytes")
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

#if canImport(CoreGraphics)
extension RGBImage {
    /// Draws a Core Graphics image into sRGB and drops alpha, as PIL's
    /// `convert("RGB")` does for opaque images.
    public init?(cgImage: CGImage) {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0,
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        self.init(width: width, height: height, pixels: rgb)
    }
}
#endif

/// Mono audio samples in `[-1, 1]` at `sampleRate` Hz.
public struct AudioClip: Equatable, Sendable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = 16_000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    /// Converts 16-bit PCM by dividing by 32768, as NumPy does for `int16` input.
    public init(pcm16 samples: [Int16], sampleRate: Int = 16_000) {
        self.init(samples: samples.map { Float($0) / 32_768 }, sampleRate: sampleRate)
    }
}
