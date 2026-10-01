import Foundation
import XCTest

@testable import VonHTTP

final class OrderedJSONTests: XCTestCase {
    func testParserAndSerializerPreserveObjectOrderAndUnicode() throws {
        let data = Data(#"{"z":"é","a":"\uD83D\uDE00","nested":{"b":1,"a":2}}"#.utf8)
        let value = try OrderedJSONParser().parse(data)
        guard case .object(let members) = value else { return XCTFail("expected object") }
        XCTAssertEqual(members.map(\.key), ["z", "a", "nested"])
        XCTAssertEqual(members[1].value, .string("😀"))
        guard case .object(let nested) = members[2].value else {
            return XCTFail("expected nested object")
        }
        XCTAssertEqual(nested.map(\.key), ["b", "a"])
        XCTAssertEqual(
            String(decoding: try OrderedJSONSerializer().serialize(value), as: UTF8.self),
            #"{"z":"é","a":"😀","nested":{"b":1,"a":2}}"#)
    }

    func testRejectsDuplicateKeysInvalidUTF8EscapesAndSurrogates() {
        assertError(#"{"a":1,"a":2}"#, equals: .duplicateKey("a"))
        assertError(#"{"a":1,"\u0061":2}"#, equals: .duplicateKey("a"))
        assertError(#""\q""#, equals: .invalidEscape)
        assertError(#""\uD800""#, equals: .invalidSurrogate)
        assertError(#""\uDC00""#, equals: .invalidSurrogate)
        XCTAssertThrowsError(try OrderedJSONParser().parse(Data([0x22, 0xFF, 0x22]))) {
            XCTAssertEqual($0 as? OrderedJSONError, .invalidUTF8)
        }
    }

    func testRejectsMalformedAndNonFiniteNumbers() throws {
        assertError("01", equals: .invalidNumber)
        assertError("1.", equals: .invalidNumber)
        assertError("1e400", equals: .nonFiniteNumber)
        for source in ["-0", "5e-324", "1.7976931348623157e308", "123456789012345678901234567890"] {
            let parsed = try OrderedJSONParser().parse(Data(source.utf8))
            XCTAssertEqual(
                String(decoding: try OrderedJSONSerializer().serialize(parsed), as: UTF8.self),
                source)
        }
    }

    func testSerializerEscapesAllRequiredCharacters() throws {
        let encoded = try OrderedJSONSerializer().serialize(
            .string("\"\\\u{08}\u{0C}\n\r\t\u{01}é"))
        XCTAssertEqual(String(decoding: encoded, as: UTF8.self), #""\"\\\b\f\n\r\t\u0001é""#)
    }

    func testBodyDepthNodeStringAndContainerLimits() throws {
        XCTAssertThrowsError(
            try OrderedJSONParser(limits: limits(body: 2)).parse(Data("null".utf8))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .bodyTooLarge)
        }
        XCTAssertThrowsError(
            try OrderedJSONParser(limits: limits(depth: 2)).parse(Data("[[0]]".utf8))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .depthLimitExceeded)
        }
        XCTAssertThrowsError(
            try OrderedJSONParser(limits: limits(nodes: 2)).parse(Data("[0,1]".utf8))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .nodeLimitExceeded)
        }
        XCTAssertThrowsError(
            try OrderedJSONParser(limits: limits(string: 3)).parse(Data(#""😀""#.utf8))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .stringLimitExceeded)
        }
        XCTAssertThrowsError(
            try OrderedJSONParser(limits: limits(container: 1)).parse(Data("[0,1]".utf8))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .containerLimitExceeded)
        }
    }

    func testSerializerRejectsDuplicateKeys() {
        let value = OrderedJSON.object([
            .init(key: "x", value: .null),
            .init(key: "x", value: .bool(true)),
        ])
        XCTAssertThrowsError(try OrderedJSONSerializer().serialize(value)) {
            XCTAssertEqual($0 as? OrderedJSONError, .duplicateKey("x"))
        }
    }

    func testSerializerEnforcesAllStructuralLimits() {
        XCTAssertThrowsError(try OrderedJSONSerializer(limits: limits(body: 3)).serialize(.null)) {
            XCTAssertEqual($0 as? OrderedJSONError, .bodyTooLarge)
        }
        XCTAssertThrowsError(
            try OrderedJSONSerializer(limits: limits(depth: 1)).serialize(.array([.null]))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .depthLimitExceeded)
        }
        XCTAssertThrowsError(
            try OrderedJSONSerializer(limits: limits(nodes: 1)).serialize(.array([.null]))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .nodeLimitExceeded)
        }
        XCTAssertThrowsError(
            try OrderedJSONSerializer(limits: limits(string: 1)).serialize(.string("é"))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .stringLimitExceeded)
        }
        XCTAssertThrowsError(
            try OrderedJSONSerializer(limits: limits(container: 1)).serialize(
                .array([.null, .null]))
        ) {
            XCTAssertEqual($0 as? OrderedJSONError, .containerLimitExceeded)
        }
    }

    private func assertError(_ source: String, equals expected: OrderedJSONError) {
        XCTAssertThrowsError(try OrderedJSONParser().parse(Data(source.utf8))) {
            XCTAssertEqual($0 as? OrderedJSONError, expected)
        }
    }

    private func limits(
        body: Int = 1_048_576,
        depth: Int = 32,
        nodes: Int = 10_000,
        string: Int = 65_536,
        container: Int = 256
    ) -> OrderedJSONLimits {
        OrderedJSONLimits(
            maximumBodyBytes: body,
            maximumDepth: depth,
            maximumNodes: nodes,
            maximumStringBytes: string,
            maximumContainerEntries: container
        )
    }
}
