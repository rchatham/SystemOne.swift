import Foundation
import MLX
import Observation
import os
import SystemOne
import SystemOneVon

/// Owns the on-device `SystemOneEngine` and the harness's load/run state.
@MainActor
@Observable
final class HarnessModel {
    enum Phase: Equatable {
        case missingWeights
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

    /// The only weights file the harness loads, in the app's Documents folder.
    static let weightsFileName = "option_marker.safetensors"

    /// Bounds MLX's reusable buffer cache so it cannot grow toward the jetsam limit.
    private static let mlxCacheLimitBytes = 64 * 1024 * 1024

    private(set) var phase: Phase = .idle
    private(set) var measurements: [Measurement] = []
    private(set) var answerLines: [String] = []
    private(set) var memory = MemoryReport.current()

    var state = HarnessSample.state
    var batching: HarnessBatching = .automatic

    private var engine: SystemOneEngine?

    var weightsURL: URL {
        URL.documentsDirectory.appendingPathComponent(Self.weightsFileName, isDirectory: false)
    }

    var weightsArePresent: Bool {
        FileManager.default.fileExists(atPath: weightsURL.path)
    }

    init() {
        MLX.Memory.cacheLimit = Self.mlxCacheLimitBytes
        refreshPhase()
    }

    /// Re-checks for weights without disturbing an in-flight or failed operation.
    func refreshPhase() {
        if phase == .idle || phase == .missingWeights {
            phase = weightsArePresent ? .idle : .missingWeights
        }
        refreshMemory()
    }

    func refreshMemory() {
        memory = MemoryReport.current()
    }

    /// Copies a user-picked safetensors file into Documents. The file is verified on load.
    func importWeights(from source: URL) async {
        phase = .importing
        let destination = weightsURL
        do {
            try await Task.detached(priority: .userInitiated) {
                let scoped = source.startAccessingSecurityScopedResource()
                defer { if scoped { source.stopAccessingSecurityScopedResource() } }
                let fileManager = FileManager.default
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.copyItem(at: source, to: destination)
            }.value
            phase = .idle
        } catch {
            phase = .failed("Import failed: \(error.localizedDescription)")
        }
        refreshMemory()
    }

    /// Verifies the pinned weights' SHA-256 and copies them into MLX arrays.
    func loadEngine() async {
        guard let tokenizerFolder = Bundle.main.url(forResource: "Tokenizer", withExtension: nil) else {
            phase = .failed("The Tokenizer folder is missing from the app bundle.")
            return
        }
        phase = .loading
        let weightsURL = weightsURL
        let start = ContinuousClock.now
        do {
            let engine = try await Task.detached(priority: .userInitiated) {
                try await SystemOneEngine.von(tokenizerFolder: tokenizerFolder, verifiedWeightsURL: weightsURL)
            }.value
            self.engine = engine
            record("Load (verify + copy)", since: start)
            phase = .ready
        } catch {
            phase = .failed("Load failed: \(error.localizedDescription)")
        }
        refreshMemory()
    }

    /// Runs the four-question sample once; the first run includes Metal kernel warmup.
    func runSample() async {
        guard let engine else { return }
        phase = .running
        let request = SystemOneRequest(
            model: VonBackend.modelID,
            state: .text(state),
            questions: HarnessSample.questions
        )
        let start = ContinuousClock.now
        do {
            let response = try await engine.systemOne(request, batchingStrategy: batching.strategy)
            record("systemOne (\(batching.label))", since: start)
            answerLines = response.answers.map(HarnessSample.describe)
            phase = .ready
        } catch {
            phase = .failed("Inference failed: \(error.localizedDescription)")
        }
        refreshMemory()
    }

    func unloadEngine() {
        engine = nil
        MLX.Memory.clearCache()
        answerLines = []
        phase = .idle
        refreshPhase()
    }

    private func record(_ label: String, since start: ContinuousClock.Instant) {
        let elapsed = ContinuousClock.now - start
        let milliseconds = Double(elapsed.components.seconds) * 1_000
            + Double(elapsed.components.attoseconds) / 1e15
        measurements.insert(Measurement(label: label, milliseconds: milliseconds), at: 0)
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

    static func current() -> MemoryReport {
        let snapshot = MLX.Memory.snapshot()
        return MemoryReport(
            availableToProcess: Int(os_proc_available_memory()),
            mlxActive: snapshot.activeMemory,
            mlxCache: snapshot.cacheMemory,
            mlxPeak: snapshot.peakMemory
        )
    }
}
