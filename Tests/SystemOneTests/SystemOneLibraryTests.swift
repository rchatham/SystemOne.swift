import XCTest
@testable import SystemOne

final class SystemOneTests: XCTestCase {
    func testVersion() {
        XCTAssertFalse(SystemOneLibrary.version.isEmpty)
    }
}