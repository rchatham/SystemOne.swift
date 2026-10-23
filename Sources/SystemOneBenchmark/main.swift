import CryptoKit
import Dispatch
import Foundation
import MLX
import SystemOne
import SystemOneVon

// MARK: - CLI

private enum BenchmarkError: Error, LocalizedError {
    case missingTokenizer(String)
    case missingWeights(String)
    case usage(String)

    var errorDescription: String? {
        switch self {
        case .missingTokenizer(let path): return "Tokenizer folder not found: \(path)"
        case .missingWeights(let path): return "Verified safetensors not found: \(path)"
        case .usage(let message): return message
        }
    }
}

private let usage = """
    Usage:
      von-benchmark --tokenizer <folder> --weights <safetensors> [--device gpu|cpu] \\
                    [--strategy sequential|automatic|batched] [--workload fixture|homogeneous] \\
                    [--questions N] [--iterations N] [--warmup N] [--json]

    Reports wall-clock latency for a single strategy. Use --strategy to select the
    batching policy (default: automatic). Use --workload fixture for the batch golden
    fixture or homogeneous for N identical Noul questions (default: fixture).
    """

// MARK: - Results model

private struct BenchmarkResults: Encodable {
    let device: String
    let strategy: String
    let workload: String
    let iterations: Int
    let warmup: Int
    let questionCount: Int
    let latency: LatencyStats
    let fixtureSha256: String
    let buildConfiguration: String
    let swiftVersion: String
    let hardware: String

    struct LatencyStats: Encodable {
        let min: Double
        let max: Double
        let mean: Double
        let p50: Double
        let p90: Double
        let p95: Double
        let p99: Double
        let stddev: Double
    }
}

private func percentile(_ values: [Double], _ pct: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let index = pct / 100.0 * Double(sorted.count - 1)
    let lower = Int(index)
    let upper = min(lower + 1, sorted.count - 1)
    let fraction = index - Double(lower)
    return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
}

private func latencyStats(_ values: [Double]) -> BenchmarkResults.LatencyStats {
    let mean = values.reduce(0, +) / Double(values.count)
    let variance = values.count >= 2
        ? values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count - 1)
        : 0.0
    return BenchmarkResults.LatencyStats(
        min: values.min() ?? 0,
        max: values.max() ?? 0,
        mean: mean,
        p50: percentile(values, 50),
        p90: percentile(values, 90),
        p95: percentile(values, 95),
        p99: percentile(values, 99),
        stddev: variance.squareRoot()
    )
}

private func formatSeconds(_ seconds: Double) -> String {
    if seconds < 0.001 { return String(format: "%.1f µs", seconds * 1_000_000) }
    if seconds < 1.0 { return String(format: "%.1f ms", seconds * 1000) }
    return String(format: "%.3f s", seconds)
}

private func padding(_ text: String, _ width: Int) -> String {
    text.count >= width ? String(text.prefix(width)) : text + String(repeating: " ", count: width - text.count)
}

private func printReport(_ results: BenchmarkResults) {
    let stats = results.latency
    print()
    print("  Von Latency Benchmark — \(results.device.uppercased()) (\(results.strategy))")
    print("  \(String(repeating: "─", count: 55))")
    print("  Workload:           \(results.workload)")
    print("  Questions:          \(results.questionCount)")
    print("  Iterations:         \(results.iterations)")
    print("  Warmup passes:      \(results.warmup)")
    print("  Fixture SHA-256:    \(results.fixtureSha256.prefix(16))...")
    print("  Build:              \(results.buildConfiguration)")
    print("  Swift:              \(results.swiftVersion)")
    print("  Hardware:           \(results.hardware)")
    print()
    print("  \(padding("Metric", 8)) \(padding("Time", 14))")
    print("  \(String(repeating: "─", count: 8)) \(String(repeating: "─", count: 14))")
    for (label, value) in [
        ("Mean", stats.mean), ("Min", stats.min), ("Max", stats.max),
        ("P50", stats.p50), ("P90", stats.p90), ("P95", stats.p95), ("P99", stats.p99),
    ] {
        print("  \(padding(label, 8)) \(padding(formatSeconds(value), 14))")
    }
    print()
    print("  stddev:  \(formatSeconds(stats.stddev))")
    print("  Throughput: \(String(format: "%.1f", Double(results.questionCount) / stats.mean)) questions/s")
    print()
}

// MARK: - Workload

private struct Workload: Sendable {
    let state: String
    let questions: [QuestionEntry]
    let questionIDs: [String]
}

private func fixtureWorkload() throws -> Workload {
    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("testdata/von_batch_golden_v1.json")
    let data = try Data(contentsOf: url)
    let fixture = try JSONDecoder().decode(BatchGoldenFixture.self, from: data)
    let request = fixture.heterogeneousRequest
    let questions = try buildOrderedQuestions(questions: request.questions)
    return Workload(
        state: request.state,
        questions: questions,
        questionIDs: request.questionIDs
    )
}

private func homogeneousWorkload(count: Int) -> Workload {
    let ids = (0..<count).map { "homogeneous-q\($0)" }
    let questions = ids.map { id in
        QuestionEntry(
            id: id,
            question: .noul(Noul(
                instructions: "Is this statement true?",
                criteria: NoulCriteria(true: "Yes, it is true.")
            ))
        )
    }
    return Workload(
        state: "The project is on track.",
        questions: questions,
        questionIDs: ids
    )
}

// MARK: - Batch fixture decoding

private struct BatchGoldenFixture: Decodable {
    let schemaVersion: Int  // unused but present
    let heterogeneousRequest: BatchGoldenRequest

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case heterogeneousRequest = "heterogeneous_request"
    }
}

private struct BatchGoldenRequest: Decodable {
    let state: String
    let questionIDs: [String]
    let questions: [BatchGoldenQuestion]
    let answers: [BatchGoldenAnswer]

    enum CodingKeys: String, CodingKey {
        case state, questions, answers
        case questionIDs = "question_ids"
    }
}

private struct BatchGoldenAnswer: Decodable {
    let questionID: String
    let answerJSON: String

    enum CodingKeys: String, CodingKey {
        case questionID = "question_id"
        case answerJSON = "answer_model_dump_json"
    }
}

private struct BatchGoldenQuestion: Decodable {
    let id: String
    let question: Question

    private struct OrderedChoiceOption: Decodable {
        let key: String
        let description: String
    }

    private enum CodingKeys: String, CodingKey {
        case id, type, instructions, criteria
        case orderedCriteria = "ordered_criteria"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        let instructions = try container.decode(String.self, forKey: .instructions)
        switch try container.decode(String.self, forKey: .type) {
        case "choice":
            let ordered = try container.decode(
                [OrderedChoiceOption].self,
                forKey: .orderedCriteria
            )
            question = .choice(Choice(
                instructions: instructions,
                criteria: ordered.map {
                    ChoiceOption(key: $0.key, description: $0.description)
                }
            ))
        case "noul":
            question = .noul(Noul(
                instructions: instructions,
                criteria: try container.decodeIfPresent(
                    NoulCriteria.self, forKey: .criteria)))
        case "score":
            question = .score(Score(
                instructions: instructions,
                criteria: try container.decode([ScoreLevel].self, forKey: .criteria)))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container,
                debugDescription: "Unsupported batch golden question type")
        }
    }
}

private func buildOrderedQuestions(
    questions: [BatchGoldenQuestion]
) throws -> [QuestionEntry] {
    questions.map { QuestionEntry(id: $0.id, question: $0.question) }
}

// MARK: - Benchmark execution

private func runBenchmark(
    engine: SystemOneEngine,
    workload: Workload,
    strategy: BatchingStrategy,
    iterations: Int,
    warmup: Int
) async throws -> BenchmarkResults.LatencyStats {
    // Warmup
    for _ in 0..<warmup {
        _ = try await engine.systemOne(
            SystemOneRequest(model: VonBackend.modelID, state: .text(workload.state), questions: workload.questions),
            batchingStrategy: strategy
        )
    }
    eval(MLXArray(0))

    // One full systemOne call per iteration — no emulation.
    var latencies: [Double] = []
    latencies.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        _ = try await engine.systemOne(
            SystemOneRequest(model: VonBackend.modelID, state: .text(workload.state), questions: workload.questions),
            batchingStrategy: strategy
        )
        eval(MLXArray(0))
        latencies.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000)
    }

    return latencyStats(latencies)
}

// MARK: - Hardware info

private func hardwareInfo() -> String {
    var size = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    var name = [CChar](repeating: 0, count: size)
    sysctlbyname("machdep.cpu.brand_string", &name, &size, nil, 0)
    return String(cString: name)
}

private func swiftVersion() -> String {
    #if swift(>=6.2)
    return "6.2"
    #elseif swift(>=6.0)
    return "6.0"
    #elseif swift(>=5.10)
    return "5.10"
    #else
    return "<5.10"
    #endif
}

private func buildConfiguration() -> String {
    #if DEBUG
    return "debug"
    #else
    return "release"
    #endif
}

private func fixtureSha256() -> String {
    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("testdata/von_batch_golden_v1.json")
    guard let data = try? Data(contentsOf: url) else { return "unknown" }
    return SHA256.hash(data: data).compactMap { String(format: "%02x", $0) }.joined()
}

// MARK: - Main

private func run() async throws {
    var tokenizerPath: String?
    var weightsPath: String?
    var device: DeviceType = .gpu
    var strategy: BatchingStrategy = .automatic
    var workload: String = "fixture"
    var questionCount = 4
    var iterations = 50
    var warmup = 3
    var jsonOutput = false
    var args = Array(CommandLine.arguments.dropFirst())
    var seen = Set<String>()
    let booleanFlags: Set<String> = ["--json"]

    while !args.isEmpty {
        let option = args.removeFirst()
        guard ["--tokenizer", "--weights", "--device", "--strategy", "--workload",
               "--questions", "--iterations", "--warmup", "--json"].contains(option)
        else { throw BenchmarkError.usage("Unknown argument '\(option)'.\n\n\(usage)") }
        guard seen.insert(option).inserted else {
            throw BenchmarkError.usage("Duplicate option '\(option)'.\n\n\(usage)")
        }
        if booleanFlags.contains(option) {
            if option == "--json" { jsonOutput = true }
            continue
        }
        guard !args.isEmpty else {
            throw BenchmarkError.usage("Missing value for '\(option)'.\n\n\(usage)")
        }
        let value = args.removeFirst()
        switch option {
        case "--tokenizer": tokenizerPath = value
        case "--weights": weightsPath = value
        case "--device":
            switch value {
            case "cpu": device = .cpu
            case "gpu": device = .gpu
            default: throw BenchmarkError.usage("--device must be 'gpu' or 'cpu', got '\(value)'")
            }
        case "--strategy":
            switch value {
            case "sequential": strategy = .sequential
            case "automatic": strategy = .automatic
            case "batched": strategy = .batched
            default: throw BenchmarkError.usage("--strategy must be 'sequential', 'automatic', or 'batched'")
            }
        case "--workload":
            switch value {
            case "fixture": workload = "fixture"
            case "homogeneous": workload = "homogeneous"
            default: throw BenchmarkError.usage("--workload must be 'fixture' or 'homogeneous'")
            }
        case "--questions":
            guard let n = Int(value), n >= 1 else {
                throw BenchmarkError.usage("--questions must be >= 1")
            }
            questionCount = n
        case "--iterations":
            guard let n = Int(value), n >= 30 else {
                throw BenchmarkError.usage("--iterations must be at least 30")
            }
            iterations = n
        case "--warmup":
            guard let n = Int(value), n >= 0 else {
                throw BenchmarkError.usage("--warmup must be non-negative")
            }
            warmup = n
        default: preconditionFailure("validated option")
        }
    }

    guard let tokenizerPath else { throw BenchmarkError.usage("--tokenizer is required.\n\n\(usage)") }
    guard let weightsPath else { throw BenchmarkError.usage("--weights is required.\n\n\(usage)") }
    let tokenizerURL = URL(fileURLWithPath: tokenizerPath).standardizedFileURL
    let weightsURL = URL(fileURLWithPath: weightsPath).standardizedFileURL

    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: tokenizerURL.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else { throw BenchmarkError.missingTokenizer(tokenizerPath) }
    guard FileManager.default.fileExists(atPath: weightsURL.path) else {
        throw BenchmarkError.missingWeights(weightsPath)
    }

    try await Device.withDefaultDevice(.init(device)) {
        let engine = try await SystemOneEngine.von(tokenizerFolder: tokenizerURL, verifiedWeightsURL: weightsURL)

        let wl: Workload
        let wlLabel: String
        if workload == "fixture" {
            wl = try fixtureWorkload()
            wlLabel = "fixture"
        } else {
            wl = homogeneousWorkload(count: questionCount)
            wlLabel = "homogeneous-n\(questionCount)"
        }

        let stats = try await runBenchmark(
            engine: engine,
            workload: wl,
            strategy: strategy,
            iterations: iterations,
            warmup: warmup
        )

        let results = BenchmarkResults(
            device: device == .gpu ? "gpu" : "cpu",
            strategy: String(describing: strategy),
            workload: wlLabel,
            iterations: iterations,
            warmup: warmup,
            questionCount: wl.questionIDs.count,
            latency: stats,
            fixtureSha256: fixtureSha256(),
            buildConfiguration: buildConfiguration(),
            swiftVersion: swiftVersion(),
            hardware: hardwareInfo())

        if jsonOutput {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(results), as: UTF8.self))
        } else {
            printReport(results)
        }
    }
}

do {
    try await run()
} catch {
    FileHandle.standardError.write(
        Data(("von-benchmark: \(error.localizedDescription)\n").utf8))
    exit(EXIT_FAILURE)
}