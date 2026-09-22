import XCTest
@testable import VonSwift

final class VonSwiftTests: XCTestCase {
    func testVersion() {
        XCTAssertFalse(VonSwift.version.isEmpty)
    }
}