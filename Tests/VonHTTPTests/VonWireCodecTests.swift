import Foundation
import JSON
import VonSwift
import XCTest

@testable import VonHTTP

final class VonWireCodecTests: XCTestCase {
    private let codec = VonWireCodec()

    func testGoldenMetadataFreeRequestPreservesSemanticOrder() throws {
        let json =
            #"{"state":{"title":"Ship","facts":{"count":2,"ready":true},"items":["a",null,1.0]},"questions":{"route":{"instructions":" Pick route ","criteria":{"fast":null,"safe":"Low risk"}},"valid":{"type":"noul","instructions":"Is valid?","criteria":{"ignored":"x","true":"Definitely"}},"rating":{"type":"score","instructions":"Rate","criteria":[" low ",{"what":"high","examples":["a","b"],"ignored":42}]}},"ignored":true}"#
        let request = try codec.decodeRequest(Data(json.utf8))

        XCTAssertEqual(request.model, "von-latest")
        XCTAssertEqual(
            request.state,
            .fields([
                .init(key: "title", value: "Ship"),
                .init(key: "facts", value: "{'count': 2, 'ready': True}"),
                .init(key: "items", value: "['a', None, 1.0]"),
            ]))
        XCTAssertEqual(request.questions.map(\.id), ["route", "valid", "rating"])
        guard case .choice(let choice) = request.questions[0].question else {
            return XCTFail("expected choice")
        }
        XCTAssertEqual(
            choice.criteria,
            [
                .init(key: "fast", description: nil),
                .init(key: "safe", description: "Low risk"),
            ])
        guard case .noul(let noul) = request.questions[1].question else {
            return XCTFail("expected noul")
        }
        XCTAssertEqual(noul.criteria, NoulCriteria(true: "Definitely", false: nil))
        guard case .score(let score) = request.questions[2].question else {
            return XCTFail("expected score")
        }
        XCTAssertEqual(
            score.criteria, [.text(" low "), .described(what: "high", examples: ["a", "b"])])
    }

    func testStateUsesPythonStringConventions() throws {
        let json =
            #"{"state":[1,"it's ok",true,null,{"x":"line\nnext","emoji":"😀"}],"questions":{}}"#
        let request = try codec.decodeRequest(Data(json.utf8))
        XCTAssertEqual(
            request.state, .text(#"[1, "it's ok", True, None, {'x': 'line\nnext', 'emoji': '😀'}]"#))
    }

    func testMissingTypeDefaultsToChoiceAndExtraFieldsAreIgnored() throws {
        let json =
            #"{"model":"jev-latest","state":"x","questions":{"q":{"instructions":"Choose","criteria":{"b":"B","a":"A"},"future":{"nested":true}}},"future":1}"#
        let request = try codec.decodeRequest(Data(json.utf8))
        XCTAssertEqual(request.model, "jev-latest")
        guard case .choice(let choice) = request.questions[0].question else {
            return XCTFail("expected choice")
        }
        XCTAssertEqual(choice.criteria.map(\.key), ["b", "a"])
    }

    func testEmptyChoiceAndScoreCriteriaAreRejectedAtWireBoundary() {
        assertRequestError(
            #"{"state":"x","questions":{"q":{"type":"choice","instructions":"Choose","criteria":{}}}}"#
        )
        assertRequestError(
            #"{"state":"x","questions":{"q":{"type":"score","instructions":"Rate","criteria":[]}}}"#
        )
    }

    func testNoulDefaultsAndRecognizedCriteria() throws {
        let absent = try codec.decodeRequest(
            Data(#"{"state":null,"questions":{"q":{"type":"noul","instructions":"Ready?"}}}"#.utf8))
        guard case .noul(let first) = absent.questions[0].question else {
            return XCTFail("expected noul")
        }
        XCTAssertNil(first.criteria)

        let explicit = try codec.decodeRequest(
            Data(
                #"{"state":null,"questions":{"q":{"type":"noul","instructions":"Ready?","criteria":{"other":"ignored","false":"Nope"}}}}"#
                    .utf8))
        guard case .noul(let second) = explicit.questions[0].question else {
            return XCTFail("expected noul")
        }
        XCTAssertEqual(second.criteria, NoulCriteria(true: nil, false: "Nope"))
    }

    func testScoreDictionaryValuesFollowPythonFormattingAndJoinBehavior() throws {
        let json =
            #"{"state":"x","questions":{"q":{"type":"score","instructions":"Rate","criteria":[{"what":null,"examples":false},{"what":2,"examples":0},{"what":[true,"x"],"examples":"ab"},{"what":{"a":1},"examples":{"first":99,"second":null}}]}}}"#
        let request = try codec.decodeRequest(Data(json.utf8))
        guard case .score(let score) = request.questions[0].question else {
            return XCTFail("expected score")
        }
        XCTAssertEqual(
            score.criteria,
            [
                .described(what: "None", examples: []),
                .described(what: "2", examples: []),
                .described(what: "[True, 'x']", examples: ["a", "b"]),
                .described(what: "{'a': 1}", examples: ["first", "second"]),
            ])
    }

    func testScoreExamplesExpansionIsCappedBeforeAllocation() {
        let longExamples = String(repeating: "a", count: 65_536)
        let json =
            #"{"state":"x","questions":{"q":{"type":"score","instructions":"Rate","criteria":[{"what":"level","examples":"\#(longExamples)"}]}}}"#
        XCTAssertThrowsError(try codec.decodeRequest(Data(json.utf8))) {
            XCTAssertEqual(
                $0 as? VonWireCodecError,
                .safetyLimitExceeded(name: "score examples per level", limit: 256)
            )
        }
    }

    func testScoreExamplesArrayHasIndependentSemanticCap() {
        let values = Array(repeating: #""x""#, count: 257).joined(separator: ",")
        let json =
            #"{"state":"x","questions":{"q":{"type":"score","instructions":"Rate","criteria":[{"examples":[\#(values)]}]}}}"#
        let permissiveJSON = OrderedJSONLimits(maximumContainerEntries: 300)
        XCTAssertThrowsError(
            try VonWireCodec(limits: permissiveJSON).decodeRequest(Data(json.utf8))
        ) {
            XCTAssertEqual(
                $0 as? VonWireCodecError,
                .safetyLimitExceeded(name: "score examples per level", limit: 256)
            )
        }
    }

    func testSemanticBudgetLimitsQuestionsForwardsMarkersTextAndPackedLength() {
        let tooManyQuestions = (0..<17).map {
            #""q\#($0)":{"type":"noul","instructions":"x","criteria":{"true":"yes"}}"#
        }.joined(separator: ",")
        assertSafetyLimit(
            #"{"state":"x","questions":{\#(tooManyQuestions)}}"#, name: "questions", limit: 16)

        let defaultNouls = (0..<13).map { #""q\#($0)":{"type":"noul","instructions":"x"}"# }.joined(
            separator: ",")
        assertSafetyLimit(
            #"{"state":"x","questions":{\#(defaultNouls)}}"#, name: "planned forwards", limit: 24)

        let options = (0..<65).map { #""o\#($0)":null"# }.joined(separator: ",")
        assertSafetyLimit(
            #"{"state":"x","questions":{"q":{"instructions":"x","criteria":{\#(options)}}}}"#,
            name: "options per question", limit: 64)

        let longInstruction = String(repeating: "i", count: 2_049)
        assertSafetyLimit(
            #"{"state":"x","questions":{"q":{"instructions":"\#(longInstruction)","criteria":{"a":null}}}}"#,
            name: "instruction UTF-8 bytes per question", limit: 2_048)

        let longState = String(repeating: "s", count: 8_180)
        let relaxedState = VonWireSafetyLimits(maximumStateUTF8Bytes: 10_000)
        XCTAssertThrowsError(
            try VonWireCodec(safetyLimits: relaxedState).decodeRequest(
                Data(
                    #"{"state":"\#(longState)","questions":{"q":{"instructions":"x","criteria":{"a":null}}}}"#
                        .utf8))
        ) {
            XCTAssertEqual(
                $0 as? VonWireCodecError,
                .safetyLimitExceeded(name: "packed UTF-8 bytes per forward", limit: 8_190))
        }
    }

    func testSemanticBudgetRejectsLiteralMaskInjection() {
        assertRequestError(
            #"{"state":"[MASK]","questions":{"q":{"instructions":"x","criteria":{"a":null}}}}"#)
    }

    func testWorstCaseEscapedResponseExpansionIsRejectedBeforeInference() {
        let escapedControlKey = String(repeating: "\\u0001", count: 65_000)
        let json =
            #"{"state":"x","questions":{"\#(escapedControlKey)":{"instructions":"Choose","criteria":{"\#(escapedControlKey)":"short"}}}}"#
        XCTAssertLessThan(Data(json.utf8).count, OrderedJSONLimits.httpDefault.maximumBodyBytes)
        XCTAssertThrowsError(try codec.decodeRequest(Data(json.utf8))) {
            XCTAssertEqual(
                $0 as? VonWireCodecError,
                .safetyLimitExceeded(
                    name: "worst-case serialized response bytes",
                    limit: OrderedJSONLimits.httpDefault.maximumBodyBytes
                )
            )
        }
    }

    func testStrictlyRejectsUnsupportedTypedValuesAndDuplicateKeys() {
        assertRequestError(
            #"{"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":1}}}}"#
        )
        assertRequestError(
            #"{"state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":[{"examples":[1]}]}}}"#
        )
        assertRequestError(
            #"{"state":"x","questions":{"q":{"type":"noul","instructions":"x","criteria":{"other":null}}}}"#
        )
        XCTAssertThrowsError(
            try codec.decodeRequest(Data(#"{"state":"x","state":"y","questions":{}}"#.utf8))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .duplicateKey("state"))
        }
    }

    func testGoldenResponseHasPythonFieldOrderAndNoSwiftOrderMetadata() throws {
        let response = SystemOneResponse(
            model: "von-1.1.0",
            answers: [
                .init(
                    id: "pick",
                    answer: .choice(
                        .init(
                            choice: "b",
                            probabilities: [
                                .init(key: "b", probability: 0.75),
                                .init(key: "a", probability: 0.25),
                            ],
                            confidence: 0.5
                        ))),
                .init(id: "yes", answer: .noul(.init(noul: 0.8))),
                .init(
                    id: "rate",
                    answer: .score(
                        .init(
                            score: 1.25,
                            confidence: 0.2,
                            legend: [
                                .init(key: "0", description: "low"),
                                .init(key: "1", description: "high"),
                            ],
                            probabilities: [
                                .init(key: "0", probability: 0.25),
                                .init(key: "1", probability: 0.75),
                            ]
                        ))),
            ],
            usage: Usage(inputTokens: 12, outputTokens: 3)
        )
        let encoded = String(decoding: try codec.encodeResponse(response), as: UTF8.self)
        XCTAssertEqual(
            encoded,
            #"{"model":"von-1.1.0","answers":{"pick":{"type":"choice","choice":"b","probabilities":{"b":0.75,"a":0.25},"confidence":0.5},"yes":{"type":"noul","noul":0.8},"rate":{"type":"score","score":1.25,"confidence":0.2,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.25,"1":0.75}}},"usage":{"input_tokens":12,"output_tokens":3}}"#
        )
        XCTAssertFalse(encoded.contains("_order"))
    }

    func testResponseUsesConfiguredCodecLimits() {
        let limited = VonWireCodec(limits: OrderedJSONLimits(maximumBodyBytes: 10))
        let response = SystemOneResponse(model: "von", answers: [], usage: .init())
        XCTAssertThrowsError(try limited.encodeResponse(response)) {
            XCTAssertEqual($0 as? OrderedJSONError, .bodyTooLarge)
        }
    }

    func testResponseRejectsNonFiniteAndDuplicateSemanticKeys() {
        let nonFinite = SystemOneResponse(
            model: "von",
            answers: [.init(id: "q", answer: .noul(.init(noul: .infinity)))],
            usage: .init()
        )
        XCTAssertThrowsError(try codec.encodeResponse(nonFinite))

        let duplicate = SystemOneResponse(
            model: "von",
            answers: [
                .init(id: "q", answer: .noul(.init(noul: 0.5))),
                .init(id: "q", answer: .noul(.init(noul: 0.4))),
            ],
            usage: .init()
        )
        XCTAssertThrowsError(try codec.encodeResponse(duplicate))
    }

    private func assertRequestError(_ json: String) {
        XCTAssertThrowsError(try codec.decodeRequest(Data(json.utf8))) {
            XCTAssertTrue($0 is VonWireCodecError, "unexpected error: \($0)")
        }
    }

    private func assertSafetyLimit(_ json: String, name: String, limit: Int) {
        XCTAssertThrowsError(try codec.decodeRequest(Data(json.utf8))) {
            XCTAssertEqual($0 as? VonWireCodecError, .safetyLimitExceeded(name: name, limit: limit))
        }
    }
}
