import Foundation
import XCTest
@testable import VonSwift

final class VonTypesTests: XCTestCase {
    func testRequestUsesPythonWireKeysAndDiscriminators() throws {
        let request = SystemOneRequest(
            state: .fields([
                VonStateField(key: "region", value: "Tokyo"),
                VonStateField(key: "status", value: "degraded"),
            ]),
            questions: [
                QuestionEntry(
                    id: "route",
                    question: .choice(Choice(
                        instructions: "Route it",
                        criteria: [
                            ChoiceOption(key: "ops", description: nil),
                            ChoiceOption(key: "billing", description: "Payments"),
                        ]
                    ))
                ),
                QuestionEntry(
                    id: "urgent",
                    question: .noul(Noul(
                        instructions: "Urgent?",
                        criteria: NoulCriteria(true: "yes", false: "no")
                    ))
                ),
                QuestionEntry(
                    id: "severity",
                    question: .score(Score(
                        instructions: "Rate",
                        criteria: [.text("low"), ScoreLevel(what: "high", examples: ["outage"])]
                    ))
                ),
            ]
        )

        let data = try JSONEncoder().encode(request)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["model"] as? String, "von-latest")
        XCTAssertEqual(object["state"] as? [String: String], ["region": "Tokyo", "status": "degraded"])
        XCTAssertEqual(object["state_order"] as? [String], ["region", "status"])
        XCTAssertEqual(object["question_order"] as? [String], ["route", "urgent", "severity"])
        let questions = try XCTUnwrap(object["questions"] as? [String: [String: Any]])
        XCTAssertEqual(questions["route"]?["type"] as? String, "choice")
        XCTAssertEqual(questions["urgent"]?["type"] as? String, "noul")
        XCTAssertEqual(questions["severity"]?["type"] as? String, "score")
        XCTAssertTrue((questions["route"]?["criteria"] as? [String: Any])?["ops"] is NSNull)
        XCTAssertEqual(
            (questions["urgent"]?["criteria"] as? [String: String])?["true"],
            "yes"
        )
        XCTAssertEqual(questions["route"]?["criteria_order"] as? [String], ["ops", "billing"])
        XCTAssertEqual(try JSONDecoder().decode(SystemOneRequest.self, from: data), request)
    }

    func testResponseUsesPythonWireKeysAndRoundTrips() throws {
        let response = SystemOneResponse(
            model: "von-1.1.0",
            answers: [
                AnswerEntry(id: "decision", answer: .choice(ChoiceAnswer(
                    choice: "a",
                    probabilities: [
                        ProbabilityEntry(key: "a", probability: 0.75),
                        ProbabilityEntry(key: "b", probability: 0.25),
                    ],
                    confidence: 0.5
                ))),
                AnswerEntry(id: "judgment", answer: .noul(NoulAnswer(noul: 0.9))),
                AnswerEntry(id: "rating", answer: .score(ScoreAnswer(
                    score: 1.2,
                    confidence: 0.1,
                    legend: [LegendEntry(key: "0", description: "low")],
                    probabilities: [ProbabilityEntry(key: "0", probability: 1)]
                ))),
            ],
            usage: Usage(inputTokens: 12, outputTokens: 3)
        )
        let data = try JSONEncoder().encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Int])
        XCTAssertEqual(usage["input_tokens"], 12)
        XCTAssertEqual(usage["output_tokens"], 3)
        let answers = try XCTUnwrap(object["answers"] as? [String: [String: Any]])
        XCTAssertEqual(answers["decision"]?["type"] as? String, "choice")
        XCTAssertEqual(answers["judgment"]?["type"] as? String, "noul")
        XCTAssertEqual(answers["rating"]?["type"] as? String, "score")

        let decoded = try JSONDecoder().decode(SystemOneResponse.self, from: data)
        XCTAssertEqual(decoded, response)
    }

    func testKeyedEncodersRejectDuplicateKeysBeforeWritingContainers() {
        assertDuplicateEncoding(
            Choice(
                instructions: "choose",
                criteria: [ChoiceOption(key: "dup"), ChoiceOption(key: "dup")]
            ),
            field: "Choice.criteria",
            key: "dup"
        )
        assertDuplicateEncoding(
            SystemOneRequest(
                state: .fields([
                    VonStateField(key: "dup", value: "first"),
                    VonStateField(key: "dup", value: "second"),
                ]),
                questions: []
            ),
            field: "SystemOneRequest.state",
            key: "dup"
        )
        assertDuplicateEncoding(
            SystemOneRequest(
                state: .text("state"),
                questions: [
                    QuestionEntry(id: "dup", question: .noul(Noul(instructions: "first"))),
                    QuestionEntry(id: "dup", question: .noul(Noul(instructions: "second"))),
                ]
            ),
            field: "SystemOneRequest.questions",
            key: "dup"
        )
        assertDuplicateEncoding(
            ChoiceAnswer(
                choice: "dup",
                probabilities: [
                    ProbabilityEntry(key: "dup", probability: 0.4),
                    ProbabilityEntry(key: "dup", probability: 0.6),
                ],
                confidence: 0.2
            ),
            field: "ChoiceAnswer.probabilities",
            key: "dup"
        )
        assertDuplicateEncoding(
            ScoreAnswer(
                score: 0,
                confidence: 0,
                legend: [
                    LegendEntry(key: "dup", description: "first"),
                    LegendEntry(key: "dup", description: "second"),
                ],
                probabilities: [ProbabilityEntry(key: "0", probability: 1)]
            ),
            field: "ScoreAnswer.legend",
            key: "dup"
        )
        assertDuplicateEncoding(
            ScoreAnswer(
                score: 0,
                confidence: 0,
                legend: [LegendEntry(key: "0", description: "only")],
                probabilities: [
                    ProbabilityEntry(key: "dup", probability: 0.4),
                    ProbabilityEntry(key: "dup", probability: 0.6),
                ]
            ),
            field: "ScoreAnswer.probabilities",
            key: "dup"
        )
        assertDuplicateEncoding(
            SystemOneResponse(
                model: VonEngine.modelID,
                answers: [
                    AnswerEntry(id: "dup", answer: .noul(NoulAnswer(noul: 0.4))),
                    AnswerEntry(id: "dup", answer: .noul(NoulAnswer(noul: 0.6))),
                ],
                usage: Usage()
            ),
            field: "SystemOneResponse.answers",
            key: "dup"
        )
    }

    func testUnknownQuestionTypeIsRejected() {
        let data = Data(#"{"model":"von-latest","state":"x","questions":{"q":{"type":"other","instructions":"x","criteria":{}}},"question_order":["q"]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(SystemOneRequest.self, from: data)) { error in
            guard case DecodingError.dataCorrupted(let context) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(context.debugDescription.contains("Unknown Von type 'other'"))
        }
    }

    private func assertDuplicateEncoding<T: Encodable>(
        _ value: T,
        field: String,
        key: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try JSONEncoder().encode(value), file: file, line: line) { error in
            guard case EncodingError.invalidValue(_, let context) = error else {
                return XCTFail("Unexpected error: \(error)", file: file, line: line)
            }
            XCTAssertEqual(
                context.debugDescription,
                "\(field) contains duplicate key '\(key)'.",
                file: file,
                line: line
            )
        }
    }
}
