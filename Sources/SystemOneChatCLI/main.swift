import Foundation
import SystemOne
import SystemOneChat
import SystemOneD1

private let usage = """
    Usage:
      system-one-chat --model-folder <local-d1-snapshot> [--float16] [--quiet]

    A terminal chatbot. d1-omni picks every step (which tool to call, and when to
    reply); Apple Intelligence writes the text. --model-folder defaults to
    $D1_MODEL_FOLDER. --quiet hides the routing trace. Type /quit to exit.
    """

private struct Options {
    var modelFolder: URL?
    var precision: D1Precision = .float32
    var quiet = false
}

private struct UsageError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private func parseOptions(_ arguments: [String]) throws -> Options {
    var options = Options()
    if let folder = ProcessInfo.processInfo.environment["D1_MODEL_FOLDER"], !folder.isEmpty {
        options.modelFolder = URL(fileURLWithPath: folder)
    }
    var index = 0
    while index < arguments.count {
        switch arguments[index] {
        case "--model-folder":
            index += 1
            guard index < arguments.count else { throw UsageError(message: "Missing value for '--model-folder'.\n\n\(usage)") }
            options.modelFolder = URL(fileURLWithPath: arguments[index])
        case "--float16": options.precision = .float16
        case "--quiet": options.quiet = true
        case "-h", "--help": throw UsageError(message: usage)
        default: throw UsageError(message: "Unknown argument '\(arguments[index])'.\n\n\(usage)")
        }
        index += 1
    }
    guard options.modelFolder != nil else {
        throw UsageError(message: "Pass --model-folder or set D1_MODEL_FOLDER.\n\n\(usage)")
    }
    return options
}

private func dim(_ text: String) -> String {
    "\u{1B}[2m\(text)\u{1B}[0m"
}

private func trace(_ event: ChatAgent.Event) {
    switch event {
    case .chose(let tool, let confidence?):
        print(dim("  → \(tool) (\(String(format: "%.2f", confidence)))"))
    case .chose(let tool, nil):
        print(dim("  → \(tool) (forced)"))
    case .observed(_, let observation):
        let lines = observation.split(separator: "\n", omittingEmptySubsequences: false)
        print(dim(lines.map { "    \($0)" }.joined(separator: "\n")))
    }
}

private func run() async throws {
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
    guard #available(macOS 26.0, *) else {
        throw ChatError.textGeneratorUnavailable("FoundationModels needs macOS 26 or later.")
    }
    try AppleIntelligenceTextGenerator.requireAvailable()

    print(dim("Loading d1-omni…"))
    let engine = try await SystemOneEngine.d1(modelFolder: options.modelFolder!, precision: options.precision)
    let notes = NoteStore()
    let agent = try ChatAgent(
        decider: engine,
        textGenerator: AppleIntelligenceTextGenerator(),
        tools: [
            CurrentTimeTool(),
            SaveNoteTool(store: notes),
            ListNotesTool(store: notes),
            SummarizeConversationTool(),
        ],
        onEvent: { [quiet = options.quiet] event in
            if !quiet { trace(event) }
        }
    )
    print("Ready. Router: \(engine.modelID). Text: Apple Intelligence. /quit to exit.\n")

    while true {
        print("you › ", terminator: "")
        fflush(stdout)
        guard let line = readLine() else { break }
        let message = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if message.isEmpty { continue }
        if message == "/quit" || message == "/exit" { break }
        do {
            let turn = try await agent.send(message)
            print("bot › \(turn.reply)\n")
        } catch {
            print("error: \(error.localizedDescription)\n")
        }
    }
}

do {
    try await run()
} catch {
    FileHandle.standardError.write(Data("system-one-chat: \(error.localizedDescription)\n".utf8))
    exit(EXIT_FAILURE)
}
