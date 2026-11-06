import SystemOne
import XCTest

@testable import SystemOneChat

/// Answers each routing question with the next scripted tool name and records what it was offered.
private actor ScriptedDecider: ActionDecider {
    private var script: [String]
    private(set) var offered: [[String]] = []
    private(set) var states: [SystemOneState] = []

    init(_ script: [String]) {
        self.script = script
    }

    func chooseAction(
        state: SystemOneState,
        options: [ChoiceOption],
        instructions: String
    ) async throws -> ChoiceAnswer {
        offered.append(options.map(\.key))
        states.append(state)
        let choice = script.removeFirst()
        return ChoiceAnswer(
            choice: choice,
            probabilities: options.map { ProbabilityEntry(key: $0.key, probability: $0.key == choice ? 0.9 : 0.1) },
            confidence: 0.9
        )
    }
}

/// Returns canned text and records every prompt.
private actor FakeGenerator: TextGenerator {
    private let response: String
    private(set) var prompts: [String] = []

    init(_ response: String) {
        self.response = response
    }

    func generate(instructions: String, prompt: String) async throws -> String {
        prompts.append(prompt)
        return response
    }
}

private struct ObservingTool: ChatTool {
    let name: String
    let description = "test tool"
    let observation: String

    func run(_ context: ToolContext) async throws -> ToolOutcome {
        .observation(observation)
    }
}

final class ChatAgentTests: XCTestCase {
    func testReplyToolEndsTheTurnImmediately() async throws {
        let decider = ScriptedDecider(["generate_text"])
        let generator = FakeGenerator(" Hi there! \n")
        let agent = try ChatAgent(decider: decider, textGenerator: generator, tools: [])

        let turn = try await agent.send("hello")

        XCTAssertEqual(turn.reply, "Hi there!")
        XCTAssertEqual(turn.calls.map(\.tool), ["generate_text"])
        let history = await agent.history
        XCTAssertEqual(history, [
            ChatMessage(role: .user, text: "hello"),
            ChatMessage(role: .assistant, text: "Hi there!"),
        ])
    }

    func testObservationsFeedTheRouterAndTheReply() async throws {
        let decider = ScriptedDecider(["get_current_time", "generate_text"])
        let generator = FakeGenerator("It's noon.")
        let agent = try ChatAgent(
            decider: decider,
            textGenerator: generator,
            tools: [
                ObservingTool(name: "get_current_time", observation: "It is 12:00 PM."),
                ObservingTool(name: "list_notes", observation: "No notes are saved."),
            ],
            maxActionsPerTurn: 2
        )

        let turn = try await agent.send("what time is it?")

        XCTAssertEqual(turn.calls.map(\.tool), ["get_current_time", "generate_text"])
        // A tool that already ran is no longer offered.
        let offered = await decider.offered
        XCTAssertEqual(offered, [
            ["get_current_time", "list_notes", "generate_text"],
            ["list_notes", "generate_text"],
        ])
        // The router reads the message alone.
        let states = await decider.states
        XCTAssertEqual(states, [.text("what time is it?"), .text("what time is it?")])
        let prompts = await generator.prompts
        XCTAssertEqual(prompts.count, 1)
        XCTAssertTrue(prompts[0].contains("Tool results:\nget_current_time: It is 12:00 PM."))
    }

    func testRouterIsSkippedOnceEveryTaskHasRun() async throws {
        let decider = ScriptedDecider(["a"])
        let agent = try ChatAgent(
            decider: decider,
            textGenerator: FakeGenerator("ok"),
            tools: [ObservingTool(name: "a", observation: "1")]
        )

        let turn = try await agent.send("go")

        XCTAssertEqual(turn.calls.map(\.tool), ["a", "generate_text"])
        let offered = await decider.offered
        XCTAssertEqual(offered.count, 1)
    }

    func testOneActionPerTurnByDefault() async throws {
        let decider = ScriptedDecider(["a"])
        let generator = FakeGenerator("done")
        let agent = try ChatAgent(
            decider: decider,
            textGenerator: generator,
            tools: [ObservingTool(name: "a", observation: "1"), ObservingTool(name: "b", observation: "2")]
        )

        let turn = try await agent.send("go")

        XCTAssertEqual(turn.calls.map(\.tool), ["a", "generate_text"])
        XCTAssertEqual(turn.calls.map(\.confidence), [0.9, nil])
        let offered = await decider.offered
        XCTAssertEqual(offered.count, 1)
    }

    func testATaskCanReplyDirectly() async throws {
        let decider = ScriptedDecider(["generate_text", "summarize_conversation"])
        let generator = FakeGenerator("We said hello.")
        let agent = try ChatAgent(
            decider: decider, textGenerator: generator, tools: [SummarizeConversationTool()]
        )

        _ = try await agent.send("hello")
        let turn = try await agent.send("recap please")

        XCTAssertEqual(turn.reply, "We said hello.")
        XCTAssertEqual(turn.calls.map(\.tool), ["summarize_conversation"])
        let prompts = await generator.prompts
        XCTAssertEqual(prompts.last, "User: hello\nAssistant: We said hello.")
    }

    func testNotesRoundTripThroughTheGenerator() async throws {
        let store = NoteStore()
        let generator = FakeGenerator("Buy milk")
        let context = ToolContext(
            userMessage: "remind me to buy milk", history: [], calls: [], textGenerator: generator
        )

        let saved = try await SaveNoteTool(store: store).run(context)
        let listed = try await ListNotesTool(store: store).run(context)

        XCTAssertEqual(saved, .observation("Saved note: Buy milk"))
        XCTAssertEqual(listed, .observation("Saved notes:\n1. Buy milk"))
    }

    func testCurrentTimeIsFormattedInTheGivenZone() async throws {
        let tool = CurrentTimeTool(timeZone: TimeZone(identifier: "UTC")!) { Date(timeIntervalSince1970: 0) }
        let context = ToolContext(userMessage: "", history: [], calls: [], textGenerator: FakeGenerator(""))

        let outcome = try await tool.run(context)

        XCTAssertEqual(outcome, .observation("It is Thursday, January 1, 1970 at 12:00 AM GMT."))
    }

    func testDuplicateToolNamesAreRejected() {
        XCTAssertThrowsError(try ChatAgent(
            decider: ScriptedDecider([]),
            textGenerator: FakeGenerator(""),
            tools: [ObservingTool(name: "generate_text", observation: "")]
        )) { error in
            XCTAssertEqual(error as? ChatError, .duplicateToolName("generate_text"))
        }
    }
}
