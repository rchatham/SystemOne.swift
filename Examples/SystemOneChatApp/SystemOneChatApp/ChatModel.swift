import Foundation
import Observation
import SystemOne
import SystemOneChat
import SystemOneD1
#if os(iOS) && !targetEnvironment(simulator)
import MLX
#endif

/// Where routing decisions come from.
enum RouterKind: String, CaseIterable, Identifiable {
    /// d1 loaded into this process with MLX.
    case onDevice
    /// `system-one serve` over HTTP, for the iOS Simulator, where MLX cannot run.
    case server

    var id: Self { self }

    var label: String {
        switch self {
        case .onDevice: return "d1 on this device"
        case .server: return "system-one serve"
        }
    }

    /// MLX needs Metal GPU features the iOS Simulator does not provide.
    static var supportsOnDevice: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }
}

/// One row of the transcript.
struct ChatEntry: Identifiable {
    enum Role {
        case user
        case assistant
        case error
    }

    let id = UUID()
    let role: Role
    let text: String
    /// The tools that ran for an assistant reply, ending with the one that wrote it.
    var calls: [ToolCall] = []
}

/// Owns the `ChatAgent`, its router, and the transcript.
///
/// Settings live in `UserDefaults`, so launch arguments override them for a run:
/// `-router server -serverURL http://127.0.0.1:8083/v1/systemone` or
/// `-router onDevice -modelFolder /path/to/d1-omni-600M`.
@MainActor
@Observable
final class ChatModel {
    enum Phase: Equatable {
        case stopped
        case starting
        case ready
        case sending
        case failed(String)
    }

    static let modelFolderName = "d1-omni-600M"
    static let defaultServerURL = "http://127.0.0.1:8083/v1/systemone"

    #if os(iOS) && !targetEnvironment(simulator)
    /// Bounds MLX's reusable buffer cache so it cannot grow toward the jetsam limit.
    private static let mlxCacheLimitBytes = 64 * 1024 * 1024
    #endif

    private(set) var phase: Phase = .stopped
    private(set) var entries: [ChatEntry] = []
    /// What the running agent routes with, for the header.
    private(set) var routerSummary: String?
    var draft = ""

    var routerKind: RouterKind {
        didSet { defaults.set(routerKind.rawValue, forKey: "router") }
    }
    var serverURL: String {
        didSet { defaults.set(serverURL, forKey: "serverURL") }
    }
    var precision: D1Precision {
        didSet { defaults.set(precision.rawValue, forKey: "precision") }
    }
    private(set) var modelFolder: URL?

    private let defaults = UserDefaults.standard
    private var agent: ChatAgent?

    init() {
        let storedKind = RouterKind(rawValue: defaults.string(forKey: "router") ?? "")
        routerKind = RouterKind.supportsOnDevice ? (storedKind ?? .onDevice) : .server
        serverURL = defaults.string(forKey: "serverURL") ?? Self.defaultServerURL
        #if os(iOS)
        let defaultPrecision = D1Precision.float16
        #else
        let defaultPrecision = D1Precision.float32
        #endif
        precision = D1Precision(rawValue: defaults.string(forKey: "precision") ?? "") ?? defaultPrecision
        modelFolder = Self.resolveModelFolder(defaults)
        #if os(iOS) && !targetEnvironment(simulator)
        MLX.Memory.cacheLimit = Self.mlxCacheLimitBytes
        #endif
    }

    var canSend: Bool {
        phase == .ready && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Builds the router and the agent. Loading d1 verifies and copies 2.35 GB of
    /// weights, which takes several seconds.
    func start() async {
        phase = .starting
        agent = nil
        do {
            try AppleIntelligenceTextGenerator.requireAvailable()
            let decider: any ActionDecider
            switch routerKind {
            case .server:
                guard let url = URL(string: serverURL), url.scheme != nil else {
                    throw ChatError.routerUnavailable("'\(serverURL)' is not a URL.")
                }
                let server = HTTPDecider(endpoint: url)
                // Fail now, not on the first message, when the server is down.
                _ = try await server.chooseAction(
                    state: .text("hello"), options: [ChoiceOption(key: "ok")], instructions: "Ready?"
                )
                decider = server
                routerSummary = "d1 via system-one serve at \(url.host ?? "?"):\(url.port.map(String.init) ?? "")"
            case .onDevice:
                guard let folder = modelFolder else {
                    throw ChatError.routerUnavailable("Choose the d1-omni-600M folder in Settings.")
                }
                let precision = self.precision
                decider = try await Task.detached(priority: .userInitiated) {
                    let scoped = folder.startAccessingSecurityScopedResource()
                    defer { if scoped { folder.stopAccessingSecurityScopedResource() } }
                    return try await SystemOneEngine.d1(modelFolder: folder, precision: precision)
                }.value
                routerSummary = "d1 on this device, \(precision.rawValue)"
            }
            let notes = NoteStore()
            agent = try ChatAgent(
                decider: decider,
                textGenerator: AppleIntelligenceTextGenerator(),
                tools: [
                    CurrentTimeTool(),
                    SaveNoteTool(store: notes),
                    ListNotesTool(store: notes),
                    SummarizeConversationTool(),
                ]
            )
            entries = []
            phase = .ready
        } catch {
            routerSummary = nil
            phase = .failed(error.localizedDescription)
        }
    }

    func send() async {
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, let agent else { return }
        draft = ""
        entries.append(ChatEntry(role: .user, text: message))
        phase = .sending
        do {
            let turn = try await agent.send(message)
            entries.append(ChatEntry(role: .assistant, text: turn.reply, calls: turn.calls))
        } catch {
            entries.append(ChatEntry(role: .error, text: error.localizedDescription))
        }
        phase = .ready
    }

    /// Remembers a folder picked in the file importer across launches.
    func chooseModelFolder(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            defaults.set(try url.bookmarkData(), forKey: "modelFolderBookmark")
            defaults.removeObject(forKey: "modelFolder")
            modelFolder = url
        } catch {
            phase = .failed("Could not remember the folder: \(error.localizedDescription)")
        }
    }

    /// A `-modelFolder` path wins, then a picked folder, then `D1_MODEL_FOLDER`,
    /// then `Documents/d1-omni-600M` (where Finder's Files tab copies to on iOS).
    private static func resolveModelFolder(_ defaults: UserDefaults) -> URL? {
        if let path = defaults.string(forKey: "modelFolder"), !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let bookmark = defaults.data(forKey: "modelFolderBookmark") {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: bookmark, bookmarkDataIsStale: &stale) {
                return url
            }
        }
        if let path = ProcessInfo.processInfo.environment["D1_MODEL_FOLDER"], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        let documents = URL.documentsDirectory.appendingPathComponent(modelFolderName, isDirectory: true)
        return FileManager.default.fileExists(atPath: documents.path) ? documents : nil
    }
}
