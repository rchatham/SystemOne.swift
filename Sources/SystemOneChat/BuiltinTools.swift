import Foundation

/// The reply tool: writes the assistant's message from the conversation and
/// whatever the turn's tools reported.
public struct GenerateTextTool: ChatTool {
    public let name = "generate_text"
    public let description = "Anything else: chat, thanks, a general question, or writing"

    public static let defaultInstructions = """
        You are a concise, friendly assistant. Answer the user's latest message in \
        full sentences. "Tool results" are actions you just took and facts you just \
        looked up for this message: tell the user what you did or found, and never \
        invent a result no tool produced.
        """

    private let instructions: String

    public init(instructions: String = GenerateTextTool.defaultInstructions) {
        self.instructions = instructions
    }

    public func run(_ context: ToolContext) async throws -> ToolOutcome {
        var sections: [String] = []
        if !context.history.isEmpty {
            sections.append("Conversation so far:\n\(Transcript.render(context.history.suffix(8)))")
        }
        if !context.calls.isEmpty {
            sections.append("Tool results:\n\(Transcript.render(context.calls))")
        }
        sections.append("User: \(context.userMessage)")
        let text = try await context.textGenerator.generate(
            instructions: instructions,
            prompt: sections.joined(separator: "\n\n")
        )
        return .reply(text.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

/// Reports the local date and time.
public struct CurrentTimeTool: ChatTool {
    public let name = "get_current_time"
    public let description = "The current date, day, or time"

    private let now: @Sendable () -> Date
    private let timeZone: TimeZone

    public init(timeZone: TimeZone = .current, now: @escaping @Sendable () -> Date = { Date() }) {
        self.timeZone = timeZone
        self.now = now
    }

    public func run(_ context: ToolContext) async throws -> ToolOutcome {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a zzz"
        return .observation("It is \(formatter.string(from: now())).")
    }
}

/// In-memory notes shared by `SaveNoteTool` and `ListNotesTool`.
public actor NoteStore {
    public private(set) var notes: [String] = []

    public init() {}

    func add(_ note: String) {
        notes.append(note)
    }
}

/// Saves something the user asked to remember. The note text comes from the
/// text generator, since System One cannot copy text out of the message.
public struct SaveNoteTool: ChatTool {
    public let name = "save_note"
    public let description = "To remember, save, or note something new"

    private let store: NoteStore

    public init(store: NoteStore) {
        self.store = store
    }

    public func run(_ context: ToolContext) async throws -> ToolOutcome {
        let note = try await context.textGenerator.generate(
            instructions: "Extract what the user wants remembered as one short note. Output only the note.",
            prompt: context.userMessage
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        await store.add(note)
        return .observation("Saved note: \(note)")
    }
}

/// Reads back every saved note.
public struct ListNotesTool: ChatTool {
    public let name = "list_notes"
    public let description = "To hear notes or reminders they saved before"

    private let store: NoteStore

    public init(store: NoteStore) {
        self.store = store
    }

    public func run(_ context: ToolContext) async throws -> ToolOutcome {
        let notes = await store.notes
        guard !notes.isEmpty else { return .observation("No notes are saved.") }
        let list = notes.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
        return .observation("Saved notes:\n\(list)")
    }
}

/// A task that talks to the user itself: it writes the summary and ends the turn.
public struct SummarizeConversationTool: ChatTool {
    public let name = "summarize_conversation"
    public let description = "A recap or summary of this conversation"

    public init() {}

    public func run(_ context: ToolContext) async throws -> ToolOutcome {
        guard !context.history.isEmpty else {
            return .reply("We haven't talked about anything yet.")
        }
        let summary = try await context.textGenerator.generate(
            instructions: "Summarize this conversation in two or three sentences, addressed to the user.",
            prompt: Transcript.render(context.history)
        )
        return .reply(summary.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
