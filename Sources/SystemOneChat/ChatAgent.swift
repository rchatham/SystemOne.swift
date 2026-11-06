import Foundation
import SystemOne

public enum ChatError: Error, Equatable, LocalizedError, Sendable {
    case duplicateToolName(String)
    case textGeneratorUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .duplicateToolName(let name):
            return "Two tools share the name '\(name)'; the router needs unique option keys."
        case .textGeneratorUnavailable(let reason):
            return "Apple Intelligence is unavailable: \(reason)"
        }
    }
}

/// Picks the next tool. `SystemOneEngine` is the production router; tests swap in a script.
public protocol ActionDecider: Sendable {
    func chooseAction(
        state: SystemOneState,
        options: [ChoiceOption],
        instructions: String
    ) async throws -> ChoiceAnswer
}

extension SystemOneEngine: ActionDecider {
    public func chooseAction(
        state: SystemOneState,
        options: [ChoiceOption],
        instructions: String
    ) async throws -> ChoiceAnswer {
        try await decide(state: state, choices: options, instructions: instructions)
    }
}

/// The result of one user message.
public struct ChatTurn: Equatable, Sendable {
    public let reply: String
    /// Every tool that ran, ending with the one that replied.
    public let calls: [ToolCall]
}

/// A chatbot whose control flow is System One decisions.
///
/// Each step asks the router one choice question about the user's message alone:
/// what is the user asking for? Every option is a tool, described as the request
/// it handles, and the reply tool (`generate_text` by default) comes last as the
/// catch-all. Tools that return an observation loop back to the router; the first
/// tool that returns a reply ends the turn. The agent forces the reply tool once
/// `maxActionsPerTurn` tools have run, so every turn terminates.
///
/// d1-omni routes this question well but cannot tell when a turn's requests are
/// all handled: with the finished tool removed it picks the nearest remaining
/// option instead of the catch-all. Hence one action per turn by default.
public actor ChatAgent {
    public enum Event: Sendable {
        case chose(tool: String, confidence: Double?)
        case observed(tool: String, observation: String)
    }

    public static let routingInstructions = "What is the user asking for?"

    public private(set) var history: [ChatMessage] = []

    private let decider: any ActionDecider
    private let textGenerator: any TextGenerator
    private let replyTool: any ChatTool
    private let tools: [any ChatTool]
    private let maxActionsPerTurn: Int
    private let onEvent: @Sendable (Event) -> Void

    /// - Parameters:
    ///   - tools: Task tools, offered in this order. Each runs at most once per turn.
    ///   - replyTool: The tool that talks to the user; always offered, last.
    ///   - maxActionsPerTurn: Task tools allowed before the reply tool is forced.
    public init(
        decider: any ActionDecider,
        textGenerator: any TextGenerator,
        tools: [any ChatTool],
        replyTool: any ChatTool = GenerateTextTool(),
        maxActionsPerTurn: Int = 1,
        onEvent: @escaping @Sendable (Event) -> Void = { _ in }
    ) throws {
        var names = Set<String>()
        for tool in [replyTool] + tools where !names.insert(tool.name).inserted {
            throw ChatError.duplicateToolName(tool.name)
        }
        self.decider = decider
        self.textGenerator = textGenerator
        self.replyTool = replyTool
        self.tools = tools
        self.maxActionsPerTurn = maxActionsPerTurn
        self.onEvent = onEvent
    }

    public func send(_ message: String) async throws -> ChatTurn {
        var calls: [ToolCall] = []
        while true {
            let available = tools.filter { tool in !calls.contains { $0.tool == tool.name } }
            let tool: any ChatTool
            let confidence: Double?
            if calls.count >= maxActionsPerTurn || available.isEmpty {
                (tool, confidence) = (replyTool, nil)
            } else {
                let options = (available + [replyTool]).map {
                    ChoiceOption(key: $0.name, description: $0.description)
                }
                let answer = try await decider.chooseAction(
                    state: .text(message),
                    options: options,
                    instructions: Self.routingInstructions
                )
                tool = Self.select(answer, replyTool: replyTool, available: available)
                confidence = answer.confidence
            }
            onEvent(.chose(tool: tool.name, confidence: confidence))

            let context = ToolContext(
                userMessage: message, history: history, calls: calls, textGenerator: textGenerator
            )
            switch try await tool.run(context) {
            case .reply(let text):
                calls.append(ToolCall(tool: tool.name, confidence: confidence, observation: text))
                history.append(ChatMessage(role: .user, text: message))
                history.append(ChatMessage(role: .assistant, text: text))
                return ChatTurn(reply: text, calls: calls)
            case .observation(let text):
                calls.append(ToolCall(tool: tool.name, confidence: confidence, observation: text))
                onEvent(.observed(tool: tool.name, observation: text))
            }
        }
    }

    /// Turns the router's answer into the tool to run.
    ///
    /// `answer.choice` is the top option and `answer.confidence` its probability;
    /// `answer.probabilities` holds every option's probability in offered order.
    static func select(
        _ answer: ChoiceAnswer,
        replyTool: any ChatTool,
        available: [any ChatTool]
    ) -> any ChatTool {
        available.first { $0.name == answer.choice } ?? replyTool
    }
}

/// Plain-text renderings for the text generator prompts.
enum Transcript {
    static func render(_ messages: some Sequence<ChatMessage>) -> String {
        messages.map { "\($0.role == .user ? "User" : "Assistant"): \($0.text)" }
            .joined(separator: "\n")
    }

    static func render(_ calls: some Sequence<ToolCall>) -> String {
        calls.map { "\($0.tool): \($0.observation)" }.joined(separator: "\n")
    }
}
