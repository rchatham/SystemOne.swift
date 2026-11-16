import XCTest
@testable import SystemOne

final class SystemOneTests: XCTestCase {
    func testVersion() {
        XCTAssertFalse(SystemOneLibrary.version.isEmpty)
    }

    func testEngineErrorsThatBlameTheRequestCarryTheirReason() {
        let rejections: [SystemOneEngineError] = [
            .emptyChoiceOptions, .emptyScoreLevels, .duplicateStateFieldKey("a"), .duplicateChoiceKey("b"),
            .duplicateQuestionID("q"), .unsupportedMedia(model: "jev"),
            .invalidQuestion(id: "urgency", reason: "a JEV score has exactly 6 levels, 0 to 5"),
        ]
        for error in rejections {
            XCTAssertEqual(error.rejectionReason, error.errorDescription, "\(error)")
        }
        XCTAssertEqual(
            (rejections[6] as any RequestRejection).rejectionReason,
            "Question 'urgency' is invalid: a JEV score has exactly 6 levels, 0 to 5")

        // A provider that breaks its contract is the server's fault, not the request's.
        let faults: [SystemOneEngineError] = [
            .unexpectedMaskCount(expected: 2, actual: 1), .unexpectedLogitCount(expected: 2, actual: 1),
            .unexpectedBatchOffsets(expected: [0, 2], actual: [0]),
        ]
        for error in faults {
            XCTAssertNil(error.rejectionReason, "\(error)")
        }
    }
}
