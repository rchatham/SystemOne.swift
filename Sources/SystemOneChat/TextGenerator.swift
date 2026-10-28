import Foundation

/// A language model that writes text. System One only decides, so every word the
/// user reads, and any text a tool extracts, comes from a `TextGenerator`.
public protocol TextGenerator: Sendable {
    /// Returns the model's response to `prompt` under `instructions`.
    /// Each call is independent: callers pass all the context the model needs.
    func generate(instructions: String, prompt: String) async throws -> String
}

#if canImport(FoundationModels)
import FoundationModels

/// Apple Intelligence's on-device model via the FoundationModels framework.
@available(macOS 26.0, iOS 26.0, *)
public struct AppleIntelligenceTextGenerator: TextGenerator {
    public init() {}

    /// Throws `ChatError.textGeneratorUnavailable` with the system's reason when
    /// the on-device model cannot run (unsupported device, Apple Intelligence
    /// turned off, or the model still downloading).
    public static func requireAvailable() throws {
        switch SystemLanguageModel.default.availability {
        case .available:
            return
        case .unavailable(let reason):
            throw ChatError.textGeneratorUnavailable(String(describing: reason))
        }
    }

    public func generate(instructions: String, prompt: String) async throws -> String {
        // A fresh session per call: the agent rebuilds context every time, and
        // the on-device model's 4k-token window would fill fast otherwise.
        let session = LanguageModelSession(instructions: instructions)
        return try await session.respond(to: prompt).content
    }
}
#endif
