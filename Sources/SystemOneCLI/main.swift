import Darwin
import Dispatch
import Foundation
import SystemOneHTTP
import SystemOne
import SystemOneD1
import SystemOneVon

private enum CLIError: Error, LocalizedError {
    case usage(String)
    case invalidPort(String)
    case missingPath(option: String, path: String)
    case wrongPathType(option: String, path: String, expected: String)
    case missingModelFile(String)
    case invalidCORSOrigin(String)
    case insecureBinding(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message): return message
        case .invalidPort(let value):
            return "--port must be an integer from 1 through 65535 (received '\(value)')."
        case .missingPath(let option, let path): return "\(option) path does not exist: \(path)"
        case .wrongPathType(let option, let path, let expected):
            return "\(option) must name \(expected): \(path)"
        case .missingModelFile(let path):
            return "Model folder is missing required regular file: \(path)"
        case .invalidCORSOrigin(let origin):
            return
                "SYSTEMONE_CORS_ORIGINS must be an explicit comma-separated allowlist; invalid origin '\(origin)'."
        case .insecureBinding(let host):
            return "Refusing non-loopback host '\(host)' without a non-empty SYSTEMONE_API_KEY."
        }
    }
}

private enum ServedModel {
    /// LiquidAI d1-omni from a local snapshot folder.
    case d1(folder: URL)
    /// von from its tokenizer folder and converted, verified safetensors.
    case von(tokenizerFolder: URL, weightsFile: URL)
}

private struct ServeOptions {
    let host: String
    let port: Int
    let model: ServedModel
}

private let usage = """
    Usage:
      system-one serve --port <port> --model-folder <local-d1-snapshot> [--host 127.0.0.1]
      system-one serve --port <port> --tokenizer <local-folder> --weights <local-safetensors> [--host 127.0.0.1]

    The first form serves d1-omni-600M; the second serves the deprecated von model.

    Environment:
      SYSTEMONE_API_KEY          Bearer token required by POST /v1/systemone when set.
      SYSTEMONE_CORS_ORIGINS     Comma-separated explicit origin allowlist. Wildcards are rejected.
    """

private func parseServeOptions(_ arguments: [String]) throws -> ServeOptions {
    guard arguments.first == "serve" else { throw CLIError.usage(usage) }

    var host = "127.0.0.1"
    var portText: String?
    var tokenizerPath: String?
    var weightsPath: String?
    var modelFolderPath: String?
    var index = 1
    var seen = Set<String>()

    while index < arguments.count {
        let option = arguments[index]
        guard ["--host", "--port", "--tokenizer", "--weights", "--model-folder"].contains(option) else {
            throw CLIError.usage("Unknown argument '\(option)'.\n\n\(usage)")
        }
        guard seen.insert(option).inserted else {
            throw CLIError.usage("Duplicate option '\(option)'.\n\n\(usage)")
        }
        index += 1
        guard index < arguments.count else {
            throw CLIError.usage("Missing value for '\(option)'.\n\n\(usage)")
        }
        let value = arguments[index]
        switch option {
        case "--host": host = value
        case "--port": portText = value
        case "--tokenizer": tokenizerPath = value
        case "--weights": weightsPath = value
        case "--model-folder": modelFolderPath = value
        default: preconditionFailure("validated option")
        }
        index += 1
    }

    guard let portText else {
        throw CLIError.usage("--port is required; no default port is assigned.\n\n\(usage)")
    }
    guard let port = Int(portText), (1...65_535).contains(port) else {
        throw CLIError.invalidPort(portText)
    }
    let fileManager = FileManager.default
    func requireFiles(_ names: [String], in folder: URL) throws {
        for name in names {
            let file = folder.appendingPathComponent(name)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: file.path, isDirectory: &isDirectory),
                !isDirectory.boolValue
            else {
                throw CLIError.missingModelFile(file.path)
            }
        }
    }

    let model: ServedModel
    switch (modelFolderPath, tokenizerPath, weightsPath) {
    case (let folderPath?, nil, nil):
        let folder = URL(fileURLWithPath: folderPath).standardizedFileURL
        try requireDirectory(folder, option: "--model-folder", fileManager: fileManager)
        try requireFiles(["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"], in: folder)
        model = .d1(folder: folder)
    case (nil, let tokenizerPath?, let weightsPath?):
        let tokenizerURL = URL(fileURLWithPath: tokenizerPath).standardizedFileURL
        let weightsURL = URL(fileURLWithPath: weightsPath).standardizedFileURL
        try requireDirectory(tokenizerURL, option: "--tokenizer", fileManager: fileManager)
        try requireFiles(["tokenizer.json", "tokenizer_config.json"], in: tokenizerURL)
        try requireRegularFile(weightsURL, option: "--weights", fileManager: fileManager)
        model = .von(tokenizerFolder: tokenizerURL, weightsFile: weightsURL)
    default:
        throw CLIError.usage(
            "Pass either --model-folder, or both --tokenizer and --weights.\n\n\(usage)")
    }

    let apiKey = normalizedAPIKey(ProcessInfo.processInfo.environment["SYSTEMONE_API_KEY"])
    guard isLoopback(host) || apiKey != nil else { throw CLIError.insecureBinding(host) }
    return ServeOptions(host: host, port: port, model: model)
}

private func requireDirectory(_ url: URL, option: String, fileManager: FileManager) throws {
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
        throw CLIError.missingPath(option: option, path: url.path)
    }
    guard isDirectory.boolValue else {
        throw CLIError.wrongPathType(option: option, path: url.path, expected: "a local folder")
    }
}

private func requireRegularFile(_ url: URL, option: String, fileManager: FileManager) throws {
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
        throw CLIError.missingPath(option: option, path: url.path)
    }
    guard !isDirectory.boolValue else {
        throw CLIError.wrongPathType(option: option, path: url.path, expected: "a local file")
    }
}

private func normalizedAPIKey(_ value: String?) -> String? {
    value.flatMap { $0.isEmpty ? nil : $0 }
}

private func isLoopback(_ host: String) -> Bool {
    ["127.0.0.1", "::1", "localhost"].contains(host)
}

private func corsConfiguration(environment: [String: String]) throws -> CORSConfiguration {
    guard let raw = environment["SYSTEMONE_CORS_ORIGINS"], !raw.isEmpty else { return .disabled }
    let origins = raw.split(separator: ",", omittingEmptySubsequences: false).map {
        String($0).trimmingCharacters(in: .whitespacesAndNewlines)
    }
    for origin in origins where origin.isEmpty || origin == "*" {
        throw CLIError.invalidCORSOrigin(origin)
    }
    return CORSConfiguration(allowedOrigins: origins)
}

private final class SignalState: @unchecked Sendable {
    private let lock = NSLock()
    private var received = false

    func markReceived() {
        lock.lock()
        received = true
        lock.unlock()
    }

    var hasReceivedSignal: Bool {
        lock.lock()
        defer { lock.unlock() }
        return received
    }
}

private final class SignalWaiter: @unchecked Sendable {
    private let stream: AsyncStream<Int32>
    private let continuation: AsyncStream<Int32>.Continuation
    private let sources: [DispatchSourceSignal]
    private let state: SignalState

    init() {
        var captured: AsyncStream<Int32>.Continuation!
        stream = AsyncStream(bufferingPolicy: .bufferingNewest(1)) { captured = $0 }
        let streamContinuation = captured!
        continuation = streamContinuation
        let signalState = SignalState()
        state = signalState

        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let createdSources = [SIGINT, SIGTERM].map { signalNumber in
            let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .global())
            source.setEventHandler { [signalState, streamContinuation] in
                signalState.markReceived()
                streamContinuation.yield(signalNumber)
            }
            source.activate()
            return source
        }
        sources = createdSources
    }

    var receivedSignal: Bool { state.hasReceivedSignal }

    func wait() async {
        var iterator = stream.makeAsyncIterator()
        _ = await iterator.next()
    }
}

private func writeStandardError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run() async throws {
    let options = try parseServeOptions(Array(CommandLine.arguments.dropFirst()))
    let environment = ProcessInfo.processInfo.environment
    let apiKey = normalizedAPIKey(environment["SYSTEMONE_API_KEY"])
    let cors = try corsConfiguration(environment: environment)
    let signalWaiter = SignalWaiter()

    let engine: SystemOneEngine
    let profile: any ModelProfile
    switch options.model {
    case .d1(let folder):
        engine = try await SystemOneEngine.d1(modelFolder: folder)
        let configuration = try D1Configuration(modelFolder: folder)
        profile = D1ModelProfile(modelID: engine.modelID, maxLength: configuration.maxLength)
    case .von(let tokenizerFolder, let weightsFile):
        engine = try await SystemOneEngine.von(
            tokenizerFolder: tokenizerFolder, verifiedWeightsURL: weightsFile)
        profile = VonModelProfile()
    }
    guard !signalWaiter.receivedSignal else { return }

    let server = SystemOneHTTPServer(
        configuration: .init(
            host: options.host,
            port: options.port,
            apiKey: apiKey,
            cors: cors
        ),
        codec: SystemOneWireCodec(profile: profile)
    ) { request in
        try await engine.systemOne(request)
    }

    do {
        try await server.start()
        print(
            "System One server [\(engine.modelID)] listening on http://\(options.host):\(options.port)"
        )
        await signalWaiter.wait()
        try await server.shutdown()
    } catch {
        try? await server.shutdown()
        throw error
    }
}

do {
    try await run()
} catch {
    writeStandardError("system-one: \(error.localizedDescription)")
    exit(EXIT_FAILURE)
}
