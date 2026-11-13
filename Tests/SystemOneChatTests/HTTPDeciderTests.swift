import SystemOne
import XCTest

@testable import SystemOneChat

final class HTTPDeciderTests: XCTestCase {
    func testRequestBodyKeepsOptionOrderAndEscapesText() throws {
        let body = try HTTPDecider.requestBody(
            state: .text("say \"hi\""),
            options: [
                ChoiceOption(key: "save_note", description: "To remember something"),
                ChoiceOption(key: "generate_text"),
            ],
            instructions: "What is the user asking for?"
        )

        XCTAssertEqual(
            String(decoding: body, as: UTF8.self),
            #"{"state":"say \"hi\"","questions":{"route":{"type":"choice","#
                + #""instructions":"What is the user asking for?","#
                + #""criteria":{"save_note":"To remember something","generate_text":null}}}}"#
        )
    }

    func testFieldStateIsAnOrderedObject() throws {
        let body = try HTTPDecider.requestBody(
            state: .fields([StateField(key: "b", value: "2"), StateField(key: "a", value: "1")]),
            options: [ChoiceOption(key: "x")],
            instructions: "?"
        )

        XCTAssertTrue(String(decoding: body, as: UTF8.self).hasPrefix(#"{"state":{"b":"2","a":"1"},"#))
    }

    func testMediaStateIsNull() throws {
        let body = try HTTPDecider.requestBody(state: .media, options: [ChoiceOption(key: "x")], instructions: "?")

        XCTAssertTrue(String(decoding: body, as: UTF8.self).hasPrefix(#"{"state":null,"#))
    }

    func testAnswerProbabilitiesFollowTheOfferedOrder() throws {
        let response = Data("""
            {"model":"d1-omni-600M","answers":{"route":{"type":"choice","choice":"b",
            "probabilities":{"b":0.75,"a":0.25},"confidence":0.75}},
            "usage":{"input_tokens":12,"output_tokens":0}}
            """.utf8)

        let answer = try HTTPDecider.decodeAnswer(
            response, options: [ChoiceOption(key: "a"), ChoiceOption(key: "b")]
        )

        XCTAssertEqual(answer, ChoiceAnswer(
            choice: "b",
            probabilities: [
                ProbabilityEntry(key: "a", probability: 0.25),
                ProbabilityEntry(key: "b", probability: 0.75),
            ],
            confidence: 0.75
        ))
    }

    func testAMissingAnswerIsAnError() {
        let response = Data(#"{"answers":{}}"#.utf8)

        XCTAssertThrowsError(try HTTPDecider.decodeAnswer(response, options: [])) { error in
            XCTAssertEqual(error as? ChatError, .routerUnavailable("The response has no 'route' answer."))
        }
    }
}
