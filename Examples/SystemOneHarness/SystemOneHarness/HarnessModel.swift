import AVFoundation
import Foundation
import MLX
import Observation
import os
import SystemOne
import SystemOneD1
import UIKit

/// Owns the on-device d1 `SystemOneEngine`, the attached media, and the
/// harness's load/run state.
@MainActor
@Observable
final class HarnessModel {
    enum Phase: Equatable {
        case missingModel
        case idle
        case importing
        case loading
        case ready
        case running
        case failed(String)
    }

    struct Measurement {
        let label: String
        let milliseconds: Double
    }

    /// The d1 snapshot folder the harness loads, inside the app's Documents.
    static let modelFolderName = "d1-omni-600M"
    static let modelFiles = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]

    /// Bounds MLX's reusable buffer cache so it cannot grow toward the jetsam limit.
    private static let mlxCacheLimitBytes = 64 * 1024 * 1024

    /// Headroom float32 needs before load. On an iPhone 14 Pro the largest image
    /// prefix (2560×1024: 5×2 tiles plus a thumbnail, 2,810 of at most 2,816
    /// positions; the `-selfTest` run) peaks at 3,569 MB of MLX memory with
    /// automatic batching, and the app adds about 135 MB; 3,900 MB leaves about
    /// 200 MB of margin. Text states of thousands of tokens can exceed it: the
    /// trunk accepts up to 16,384 positions.
    static let float32RequiredBytes = 3_900 * 1_048_576

    private(set) var phase: Phase = .idle
    private(set) var measurements: [Measurement] = []
    private(set) var answerLines: [String] = []
    private(set) var memory = MemoryReport.current()
    /// The MLX peak before the self-test last reset it to measure one run.
    private var earlierPeak = 0
    private(set) var media: HarnessMedia?
    private(set) var usage: String?

    var sample: HarnessSample = .text {
        didSet { state = sample.state }
    }
    var state = HarnessSample.text.state
    /// An optional extra yes/no question asked alongside the sample's.
    var customQuestion = ""
    var batching: HarnessBatching = .automatic
    /// Defaults to `-precision float32|float16` from the launch arguments, else automatic.
    var precision = HarnessPrecision(rawValue: UserDefaults.standard.string(forKey: "precision") ?? "") ?? .automatic
    /// The precision the current engine was loaded at.
    private(set) var loadedPrecision: D1Precision?
    let recorder = AudioRecorder()

    private var engine: SystemOneEngine?

    var modelFolder: URL {
        URL.documentsDirectory.appendingPathComponent(Self.modelFolderName, isDirectory: true)
    }

    var missingModelFiles: [String] {
        Self.modelFiles.filter {
            !FileManager.default.fileExists(atPath: modelFolder.appendingPathComponent($0).path)
        }
    }

    init() {
        MLX.Memory.cacheLimit = Self.mlxCacheLimitBytes
        refreshPhase()
    }

    /// Re-checks for the model without disturbing an in-flight or failed operation.
    func refreshPhase() {
        if phase == .idle || phase == .missingModel {
            phase = missingModelFiles.isEmpty ? .idle : .missingModel
        }
        refreshMemory()
    }

    func refreshMemory() {
        memory = MemoryReport.current(earlierPeak: earlierPeak)
    }

    /// Copies the four snapshot files from a user-picked folder into Documents.
    /// The weights are verified against their pinned SHA-256 when loaded.
    func importModel(from source: URL) async {
        phase = .importing
        let destination = modelFolder
        do {
            try await Task.detached(priority: .userInitiated) {
                let scoped = source.startAccessingSecurityScopedResource()
                defer { if scoped { source.stopAccessingSecurityScopedResource() } }
                let fileManager = FileManager.default
                try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
                for name in Self.modelFiles {
                    let target = destination.appendingPathComponent(name)
                    if fileManager.fileExists(atPath: target.path) { try fileManager.removeItem(at: target) }
                    try fileManager.copyItem(at: source.appendingPathComponent(name), to: target)
                }
            }.value
            phase = .idle
        } catch {
            phase = .failed("Import failed: \(error.localizedDescription)")
        }
        refreshPhase()
    }

    /// Verifies the pinned weights' SHA-256 and copies them into MLX arrays.
    func loadEngine() async {
        phase = .loading
        let folder = modelFolder
        let resolved = precision.resolved(availableBytes: MemoryReport.current().availableToProcess)
        let start = ContinuousClock.now
        do {
            engine = try await Task.detached(priority: .userInitiated) {
                try await SystemOneEngine.d1(modelFolder: folder, precision: resolved)
            }.value
            loadedPrecision = resolved
            record("Load \(resolved.rawValue) (verify + copy)", since: start)
            phase = .ready
        } catch {
            phase = .failed("Load failed: \(error.localizedDescription)")
        }
        refreshMemory()
    }

    func attachPhoto(_ photo: UIImage) async {
        let start = ContinuousClock.now
        let image = await Task.detached(priority: .userInitiated) { RGBImage(photo: photo) }.value
        guard let image else {
            phase = .failed("The photo could not be decoded.")
            return
        }
        record("Decode photo", since: start)
        media = .image(preview: photo, image: image)
        if sample == .text { sample = .photo }
    }

    func toggleRecording() async {
        do {
            if recorder.isRecording {
                let recording = try await recorder.stop()
                media = .audio(seconds: recording.seconds, clip: recording.clip)
                if sample == .text { sample = .voice }
            } else {
                try await recorder.start()
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func clearMedia() {
        media = nil
    }

    /// Runs the selected questions once; the first run includes Metal kernel warmup.
    func run() async {
        guard let engine else { return }
        phase = .running
        var questions = sample.questions
        let custom = customQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        if !custom.isEmpty {
            questions.append(QuestionEntry(id: "custom", question: .noul(Noul(instructions: custom))))
        }
        let start = ContinuousClock.now
        do {
            let response = try await engine.systemOne(
                state: .text(state),
                questions: questions,
                media: media?.media,
                batchingStrategy: batching.strategy
            )
            record("systemOne \(media?.summary ?? "text") (\(batching.label))", since: start)
            answerLines = response.answers.map(HarnessSample.describe)
            usage = "\(response.usage.inputTokens) positions read"
            phase = .ready
        } catch {
            phase = .failed("Inference failed: \(error.localizedDescription)")
        }
        refreshMemory()
    }

    func unloadEngine() {
        engine = nil
        loadedPrecision = nil
        MLX.Memory.clearCache()
        answerLines = []
        usage = nil
        phase = .idle
        refreshPhase()
    }

    /// `-selfTest`: loads d1 and runs text, photo, and voice requests through the
    /// same paths as the buttons, printing results so a device run can be checked
    /// from `devicectl device process launch --console`.
    func selfTest() async {
        func log(_ line: String) { print("[selfTest] \(line)") }
        log("available before load: \(MemoryReport.current().availableToProcess / 1_048_576) MB")
        await loadEngine()
        guard phase == .ready, let loadedPrecision else { return log("FAILED \(phase)") }
        log("precision: \(loadedPrecision.rawValue) (\(precision.label))")
        // Exercise the same decode paths as the photo picker and the recorder:
        // a UIImage through RGBImage(photo:), and WAV files through D1AudioFormat.
        let synthetic = Self.syntheticImage(width: 1_200, height: 900)
        let photo = UIImage(cgImage: synthetic.cgImage)
        // 5×2 tiles plus a thumbnail: 2,810 prefix positions, near the 2,816
        // maximum, so this run sets the memory bound `float32RequiredBytes` uses.
        let largest = Self.syntheticImage(width: 2_560, height: 1_024)
        guard let image = RGBImage(photo: photo),
              image == synthetic
        else { return log("FAILED photo round trip") }
        let runs: [(HarnessSample, HarnessMedia?)]
        do {
            let native = try Self.toneFile(sampleRate: 16_000, channels: 1, name: "tone-16k.wav")
            let resampled = try Self.toneFile(sampleRate: 44_100, channels: 2, name: "tone-44k.wav")
            let clip = AudioClip(samples: try D1AudioFormat.samples(from: native))
            let converted = try D1AudioFormat.samples(from: resampled)
            log("decoded \(clip.samples.count) samples at 16 kHz; 44.1 kHz stereo -> \(converted.count)")
            runs = [
                (.text, nil),
                (.photo, .image(preview: photo, image: image)),
                // Again, once Metal kernels for these shapes are compiled.
                (.photo, .image(preview: photo, image: image)),
                (.voice, .audio(seconds: 3, clip: clip)),
                (.photo, .image(preview: UIImage(cgImage: largest.cgImage), image: largest)),
            ]
        } catch {
            return log("FAILED audio decode: \(error)")
        }
        for (sample, attachment) in runs {
            self.sample = sample
            media = attachment
            earlierPeak = max(earlierPeak, MLX.Memory.peakMemory)
            MLX.Memory.peakMemory = 0
            await run()
            guard phase == .ready else { return log("FAILED \(sample.label): \(phase)") }
            let peak = MLX.Memory.peakMemory / 1_048_576
            log("\(sample.label): \(answerLines.joined(separator: " | ")) [\(usage ?? ""); MLX peak \(peak) MB]")
        }
        for measurement in measurements.reversed() {
            log("\(measurement.label): \(Int(measurement.milliseconds)) ms")
        }
        let memory = MemoryReport.current()
        log("MLX active \(memory.mlxActive / 1_048_576) MB, available \(memory.availableToProcess / 1_048_576) MB")
        log("PASSED")
    }

    /// A gradient with a checkerboard, so every crop has structure.
    private static func syntheticImage(width: Int, height: Int) -> RGBImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x) * 3
                pixels[index] = UInt8(x * 255 / width)
                pixels[index + 1] = UInt8(y * 255 / height)
                pixels[index + 2] = (x / 60 + y / 60) % 2 == 0 ? 220 : 30
            }
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    /// Writes a 3 s 220 Hz tone as 16-bit PCM WAV.
    private static func toneFile(sampleRate: Double, channels: AVAudioChannelCount, name: String) throws -> URL {
        let url = URL.temporaryDirectory.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ])
        let frames = AVAudioFrameCount(sampleRate * 3)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0..<Int(channels) {
            for frame in 0..<Int(frames) {
                buffer.floatChannelData![channel][frame] = Float(sin(Double(frame) * 2 * .pi * 220 / sampleRate) * 0.3)
            }
        }
        try file.write(from: buffer)
        return url
    }

    private func record(_ label: String, since start: ContinuousClock.Instant) {
        let elapsed = ContinuousClock.now - start
        let milliseconds = Double(elapsed.components.seconds) * 1_000
            + Double(elapsed.components.attoseconds) / 1e15
        measurements.insert(Measurement(label: label, milliseconds: milliseconds), at: 0)
    }
}

/// How the harness picks d1's precision at load.
enum HarnessPrecision: String, CaseIterable, Hashable {
    case automatic
    case float32
    case float16

    var label: String {
        switch self {
        case .automatic: "Automatic"
        case .float32: "float32"
        case .float16: "float16"
        }
    }

    /// Automatic keeps the reference float32 when the process can hold it and
    /// falls back to float16 (half the weight memory) on smaller devices.
    /// `os_proc_available_memory()` is 0 when the process has no jetsam limit,
    /// as for an iOS app on a Mac, so 0 keeps float32.
    func resolved(availableBytes: Int) -> D1Precision {
        switch self {
        case .automatic:
            availableBytes == 0 || availableBytes >= HarnessModel.float32RequiredBytes ? .float32 : .float16
        case .float32: .float32
        case .float16: .float16
        }
    }
}

/// The engine's batching strategies, in a form SwiftUI can use as `Picker` tags.
enum HarnessBatching: String, CaseIterable, Hashable {
    case sequential
    case automatic
    case batched

    var label: String { rawValue.capitalized }

    var strategy: BatchingStrategy {
        switch self {
        case .sequential: .sequential
        case .automatic: .automatic
        case .batched: .batched
        }
    }
}

/// Process and MLX memory, read at a single point in time.
struct MemoryReport {
    /// Bytes the process can still allocate before jetsam terminates it.
    let availableToProcess: Int
    let mlxActive: Int
    let mlxCache: Int
    let mlxPeak: Int

    /// `earlierPeak` carries the MLX peak across the self-test's per-run resets.
    static func current(earlierPeak: Int = 0) -> MemoryReport {
        let snapshot = MLX.Memory.snapshot()
        return MemoryReport(
            availableToProcess: Int(os_proc_available_memory()),
            mlxActive: snapshot.activeMemory,
            mlxCache: snapshot.cacheMemory,
            mlxPeak: max(snapshot.peakMemory, earlierPeak)
        )
    }
}
