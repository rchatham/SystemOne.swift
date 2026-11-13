import Foundation

/// One finished chat message.
public struct ChatMessage: Equatable, Sendable {
    public enum Role: String, Sendable {
        case user
        case assistant
    }

    public let role: Role
    public let text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}

/// A tool that ran during the current turn and what it reported back.
public struct ToolCall: Equatable, Sendable {
    public let tool: String
    /// The router's probability for this tool, or `nil` when the agent called it
    /// without asking (the action budget ran out or only one option remained).
    public let confidence: Double?
    public let observation: String

    public init(tool: String, confidence: Double?, observation: String) {
        self.tool = tool
        self.confidence = confidence
        self.observation = observation
    }
}

/// What a tool hands back to the agent.
public enum ToolOutcome: Equatable, Sendable {
    /// A private result. The agent records it and asks the router what to do next.
    case observation(String)
    /// Text for the user. It ends the turn.
    case reply(String)
}

/// Everything a tool can read while it runs.
public struct ToolContext: Sendable {
    /// The user's latest message.
    public let userMessage: String
    /// Earlier finished messages, oldest first, without `userMessage`.
    public let history: [ChatMessage]
    /// Tools already run this turn, in order.
    public let calls: [ToolCall]
    /// The language model for tools that need to write or extract text.
    public let textGenerator: any TextGenerator

    public init(
        userMessage: String,
        history: [ChatMessage],
        calls: [ToolCall],
        textGenerator: any TextGenerator
    ) {
        self.userMessage = userMessage
        self.history = history
        self.calls = calls
        self.textGenerator = textGenerator
    }
}

/// A function the router can call.
///
/// `name` and `description` become one option of the router's choice question,
/// "What is the user asking for?", so the description names the request the tool
/// handles ("The current date, day, or time"), not what the tool does.
public protocol ChatTool: Sendable {
    var name: String { get }
    var description: String { get }

    func run(_ context: ToolContext) async throws -> ToolOutcome
}
