import AVFoundation
import SystemOne
import UIKit

/// A photo or recording attached to the next request, with a preview for the UI.
enum HarnessMedia {
    case image(preview: UIImage, image: RGBImage)
    case audio(seconds: Double, clip: AudioClip)

    var media: SystemOneMedia {
        switch self {
        case .image(_, let image): .images([image])
        case .audio(_, let clip): .audio(clip)
        }
    }

    var summary: String {
        switch self {
        case .image(_, let image): "Photo \(image.width)×\(image.height)"
        case .audio(let seconds, _): "Recording \(seconds.formatted(.number.precision(.fractionLength(1)))) s"
        }
    }
}

extension RGBImage {
    /// Renders a photo upright at its full pixel size, so EXIF orientation is
    /// applied before the model sees it (`CGImage` alone ignores orientation).
    init?(photo: UIImage) {
        let size = CGSize(width: photo.size.width * photo.scale, height: photo.size.height * photo.scale)
        guard size.width >= 1, size.height >= 1 else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let upright = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            photo.draw(in: CGRect(origin: .zero, size: size))
        }
        guard let cgImage = upright.cgImage else { return nil }
        self.init(cgImage: cgImage)
    }
}

extension RGBImage {
    /// The pixels as an opaque sRGB `CGImage`.
    var cgImage: CGImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0..<(width * height) {
            rgba[pixel * 4] = pixels[pixel * 3]
            rgba[pixel * 4 + 1] = pixels[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = pixels[pixel * 3 + 2]
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )!
    }
}

/// Records one mono 16 kHz clip (d1's input rate), capped at 30 seconds.
@MainActor
@Observable
final class AudioRecorder: NSObject, AVAudioRecorderDelegate {
    enum RecorderError: LocalizedError {
        case permissionDenied
        case unreadable

        var errorDescription: String? {
            switch self {
            case .permissionDenied: "Microphone access is off. Allow it in Settings › Privacy › Microphone."
            case .unreadable: "The recording could not be read."
            }
        }
    }

    static let maximumSeconds: TimeInterval = 30

    private(set) var isRecording = false
    private var recorder: AVAudioRecorder?
    private var finished: CheckedContinuation<Void, Never>?

    private var fileURL: URL {
        URL.temporaryDirectory.appendingPathComponent("harness-recording.wav")
    }

    func start() async throws {
        guard await AVAudioApplication.requestRecordPermission() else { throw RecorderError.permissionDenied }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)
        let recorder = try AVAudioRecorder(url: fileURL, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: D1AudioFormat.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ])
        recorder.delegate = self
        guard recorder.record(forDuration: Self.maximumSeconds) else { throw RecorderError.unreadable }
        self.recorder = recorder
        isRecording = true
    }

    /// Stops (or waits for the 30 s cap) and returns the clip as 16 kHz samples.
    func stop() async throws -> (seconds: Double, clip: AudioClip) {
        if let recorder, recorder.isRecording {
            await withCheckedContinuation { continuation in
                finished = continuation
                recorder.stop()
            }
        }
        recorder = nil
        isRecording = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        let samples = try D1AudioFormat.samples(from: fileURL)
        return (Double(samples.count) / D1AudioFormat.sampleRate, AudioClip(samples: samples))
    }

    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task { @MainActor in
            isRecording = false
            finished?.resume()
            finished = nil
        }
    }
}

/// Reads any audio file as mono float samples at 16 kHz.
enum D1AudioFormat {
    static let sampleRate: Double = 16_000

    /// 16-bit PCM read as float is divided by 32768, matching the reference's
    /// `int16 / 32768`. Other rates or channel counts are converted first.
    static func samples(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1,
                                   interleaved: false)!
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0,
              let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames)
        else { throw AudioRecorder.RecorderError.unreadable }
        try file.read(into: input)

        var output = input
        if file.processingFormat.sampleRate != sampleRate || file.processingFormat.channelCount != 1 {
            guard let converter = AVAudioConverter(from: file.processingFormat, to: target),
                  let converted = AVAudioPCMBuffer(
                      pcmFormat: target,
                      frameCapacity: AVAudioFrameCount(Double(frames) * sampleRate / file.processingFormat.sampleRate) + 1
                  )
            else { throw AudioRecorder.RecorderError.unreadable }
            var consumed = false
            var conversionError: NSError?
            converter.convert(to: converted, error: &conversionError) { _, status in
                if consumed {
                    status.pointee = .endOfStream
                    return nil
                }
                consumed = true
                status.pointee = .haveData
                return input
            }
            if let conversionError { throw conversionError }
            output = converted
        }
        guard let channel = output.floatChannelData?[0] else { throw AudioRecorder.RecorderError.unreadable }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }
}
