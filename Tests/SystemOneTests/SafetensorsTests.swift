import Darwin
import Foundation
import XCTest
@testable import SystemOne

final class SafetensorsTests: XCTestCase {
    func testLoadsScalarZeroDimensionMetadataPaddingAndScopedBytes() throws {
        let header = """
        {
          "__metadata__":{"source":"unit","escaped\\u005fkey":"value"},
          "scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},
          "zero":{"dtype":"F32","shape":[3,0,9],"data_offsets":[4,4]},
          "matrix":{"dtype":"F32","shape":[1,2],"data_offsets":[4,12]}
        }
        """ + "    "
        let url = try makeFile(header: Array(header.utf8), payload: Array(0..<12))
        let file = try SafetensorsFile(contentsOf: url)

        XCTAssertEqual(file.metadata, ["source": "unit", "escaped_key": "value"])
        XCTAssertEqual(file.descriptors.count, 3)
        XCTAssertEqual(file.payloadByteCount, 12)
        XCTAssertEqual(file.dataStart, 8 + header.utf8.count)

        let scalar = try XCTUnwrap(file.tensor(named: "scalar"))
        XCTAssertEqual(scalar.descriptor.shape, [])
        XCTAssertEqual(scalar.descriptor.elementCount, 1)
        XCTAssertEqual(scalar.descriptor.byteCount, 4)
        XCTAssertEqual(scalar.descriptor.payloadRange, 0..<4)
        XCTAssertEqual(scalar.withUnsafeBytes { Array($0) }, [0, 1, 2, 3])

        let zero = try XCTUnwrap(file.tensor(named: "zero"))
        XCTAssertEqual(zero.descriptor.elementCount, 0)
        XCTAssertEqual(zero.descriptor.byteCount, 0)
        XCTAssertEqual(zero.withUnsafeBytes { $0.count }, 0)

        XCTAssertEqual(
            try XCTUnwrap(file.tensor(named: "matrix")).withUnsafeBytes { Array($0) },
            Array(4..<12)
        )
        XCTAssertEqual(file.withUnsafePayloadBytes { Array($0) }, Array(0..<12))
        XCTAssertEqual(file.withUnsafeBytes { $0.count }, file.fileByteCount)
    }

    func testFloat16BytesRoundToNearestEven() {
        let values: [Float] = [1, -2.5, 0.1, 2_049, 2_051, 65_504, 70_000, 6e-8, 1e-8]
        let half = values.withUnsafeBytes { SafetensorsTensorView.float16Bytes($0) }
        let bits = half.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        // 2049 and 2051 are ties: they round to the even neighbours 2048 and 2052.
        XCTAssertEqual(bits, [0x3C00, 0xC100, 0x2E66, 0x6800, 0x6802, 0x7BFF, 0x7C00, 0x0001, 0x0000])
        XCTAssertEqual([Float]().withUnsafeBytes { SafetensorsTensorView.float16Bytes($0) }.count, 0)
    }

    func testTensorViewRetainsMappingAfterFileAndPathAreReleased() throws {
        let url = try makeFile(
            header: Array(singleTensorHeader(shape: "[2]", offsets: "[0,8]").utf8),
            payload: [9, 8, 7, 6, 5, 4, 3, 2]
        )
        var file: SafetensorsFile? = try SafetensorsFile(contentsOf: url)
        let view = try XCTUnwrap(file?.tensor(named: "tensor"))
        file = nil
        try FileManager.default.removeItem(at: url)

        XCTAssertEqual(view.withUnsafeBytes { Array($0) }, [9, 8, 7, 6, 5, 4, 3, 2])
    }

    func testLoadsReadOnlyRegularFile() throws {
        let url = try makeFile(
            header: Array(singleTensorHeader(shape: "[1]", offsets: "[0,4]").utf8),
            payload: [1, 2, 3, 4]
        )
        XCTAssertEqual(chmod(url.path, S_IRUSR | S_IRGRP | S_IROTH), 0)
        defer { _ = chmod(url.path, S_IRUSR | S_IWUSR) }

        let file = try SafetensorsFile(contentsOf: url)
        XCTAssertEqual(file.tensor(named: "tensor")?.withUnsafeBytes { Array($0) }, [1, 2, 3, 4])
    }

    func testRejectsPathAndHeaderFramingFailures() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        assertThrows({ try SafetensorsFile(contentsOf: missing) }) {
            if case .openFailed = $0 { return true }
            return false
        }

        let directory = try makeTemporaryDirectory()
        assertThrows({ try SafetensorsFile(contentsOf: directory) }) {
            if case .notRegularFile = $0 { return true }
            return false
        }

        let empty = try makeRawFile([])
        assertThrows({ try SafetensorsFile(contentsOf: empty) }) {
            $0 == .fileTooSmall(actual: 0)
        }

        let short = try makeRawFile([0, 1, 2, 3])
        assertThrows({ try SafetensorsFile(contentsOf: short) }) {
            $0 == .fileTooSmall(actual: 4)
        }

        let huge = try makeFile(header: [], payload: [], headerLengthOverride: UInt64.max)
        assertThrows({ try SafetensorsFile(contentsOf: huge) }) {
            $0 == .headerTooLarge(actual: UInt64.max, maximum: 100_000_000)
        }

        let beyondFile = try makeFile(header: [], payload: [], headerLengthOverride: 1)
        assertThrows({ try SafetensorsFile(contentsOf: beyondFile) }) {
            $0 == .headerLengthOutOfRange(1)
        }
    }

    func testRejectsOversizedSparseHeaderBeforeParsing() throws {
        let oversizedHeaderLength: UInt64 = 100_000_001
        let url = try makeSparseHeaderFile(headerLength: oversizedHeaderLength)
        var status = stat()
        XCTAssertEqual(lstat(url.path, &status), 0)
        XCTAssertEqual(UInt64(status.st_size), 8 + oversizedHeaderLength)
        XCTAssertLessThan(Int64(status.st_blocks) * 512, 1_000_000)

        assertThrows({ try SafetensorsFile(contentsOf: url) }) {
            $0 == .headerTooLarge(actual: oversizedHeaderLength, maximum: 100_000_000)
        }
    }

    func testHeaderLengthIsStrictLittleEndian() throws {
        let header = Array("{}".utf8)
        let valid = try makeFile(header: header)
        XCTAssertEqual(try SafetensorsFile(contentsOf: valid).headerByteCount, 2)

        var bigEndianPrefix = [UInt8](repeating: 0, count: 8)
        bigEndianPrefix[7] = 2
        let invalid = try makeRawFile(bigEndianPrefix + header)
        assertThrows({ try SafetensorsFile(contentsOf: invalid) }) {
            $0 == .headerTooLarge(
                actual: 144_115_188_075_855_872,
                maximum: 100_000_000
            )
        }
    }

    func testRejectsLeadingWhitespaceAndNonSpaceTrailingPadding() throws {
        let jsonWhitespace: [UInt8] = [0x20, 0x09, 0x0A, 0x0D]
        for byte in jsonWhitespace {
            let url = try makeFile(header: [byte] + Array("{}".utf8))
            assertThrows({ try SafetensorsFile(contentsOf: url) }, "leading byte \(byte)") {
                if case .invalidJSON = $0 { return true }
                return false
            }
        }

        let nonSpacePadding: [UInt8] = [0x09, 0x0A, 0x0D]
        for byte in nonSpacePadding {
            let url = try makeFile(header: Array("{}".utf8) + [byte])
            assertThrows({ try SafetensorsFile(contentsOf: url) }, "trailing byte \(byte)") {
                if case .invalidJSON = $0 { return true }
                return false
            }
        }
    }

    func testRejectsMalformedJSONRootUTF8AndPadding() throws {
        let cases: [[UInt8]] = [
            Array("[]".utf8),
            Array("{}x".utf8),
            Array("{\"x\":\"\\q\"}".utf8),
            Array("{\"x\":\"\\uD800\"}".utf8),
            Array("{\"x\":\"\\uDC00\"}".utf8),
            Array("{\"x\":\"\\uD800\\u0041\"}".utf8),
            [0x7B, 0x22, 0xFF, 0x22, 0x3A, 0x7B, 0x7D, 0x7D],
        ]

        for (index, header) in cases.enumerated() {
            let url = try makeFile(header: header)
            XCTAssertThrowsError(try SafetensorsFile(contentsOf: url), "case \(index)")
        }

        let root = try makeFile(header: Array("[]".utf8))
        assertThrows({ try SafetensorsFile(contentsOf: root) }) { $0 == .rootMustBeObject }
    }

    func testRejectsExcessiveJSONNestingWithoutRecursingUnboundedly() throws {
        let nesting = 140
        let header = String(repeating: "[", count: nesting)
            + "null"
            + String(repeating: "]", count: nesting)
        let url = try makeFile(header: Array(header.utf8))

        assertThrows({ try SafetensorsFile(contentsOf: url) }) {
            if case .invalidJSON(let reason) = $0 {
                return reason.contains("nesting exceeds")
            }
            return false
        }
    }

    func testRejectsDuplicateDecodedKeysAtEveryObjectLevel() throws {
        let headers = [
            "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"ten\\u0073or\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            "{\"tensor\":{\"dtype\":\"F32\",\"d\\u0074ype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            "{\"__metadata__\":{\"source\":\"a\",\"sou\\u0072ce\":\"b\"}}",
            "{\"tensor\":{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}",
        ]

        for (index, header) in headers.enumerated() {
            let url = try makeFile(header: Array(header.utf8), payload: [0, 0, 0, 0])
            assertThrows({ try SafetensorsFile(contentsOf: url) }, "case \(index)") {
                if case .duplicateJSONKey = $0 { return true }
                return false
            }
        }
    }

    func testRejectsInvalidDescriptorFieldsMetadataAndDType() throws {
        let headers = [
            "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1]}}",
            "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"extra\":0}}",
            "{\"tensor\":42}",
            "{\"tensor\":{\"dtype\":1,\"shape\":[1],\"data_offsets\":[0,4]}}",
            "{\"tensor\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            "{\"__metadata__\":null}",
            "{\"__metadata__\":{\"number\":1}}",
        ]

        for (index, header) in headers.enumerated() {
            let url = try makeFile(header: Array(header.utf8), payload: [0, 0, 0, 0])
            XCTAssertThrowsError(try SafetensorsFile(contentsOf: url), "case \(index)")
        }
    }

    func testRejectsInvalidShapeNumbersAndCheckedProducts() throws {
        let invalidShapes = [
            "[-1]",
            "[1.0]",
            "[1e0]",
            "[true]",
            "[9223372036854775808]",
            "[4611686018427387904,2]",
            "[2305843009213693952]",
        ]

        for shape in invalidShapes {
            let url = try makeFile(
                header: Array(singleTensorHeader(shape: shape, offsets: "[0,0]").utf8)
            )
            assertThrows({ try SafetensorsFile(contentsOf: url) }, shape) {
                if case .invalidDescriptor = $0 { return true }
                return false
            }
        }
    }

    func testRejectsMalformedOffsetsAndByteLength() throws {
        let cases = [
            ("[1]", "[0]"),
            ("[1]", "[0,4,4]"),
            ("[1]", "[-1,3]"),
            ("[1]", "[4,0]"),
            ("[1]", "[0,8]"),
            ("[1]", "[0,1.0]"),
            ("[1]", "[0,true]"),
            ("[1]", "[0,9223372036854775808]"),
        ]

        for (shape, offsets) in cases {
            let url = try makeFile(
                header: Array(singleTensorHeader(shape: shape, offsets: offsets).utf8),
                payload: [0, 0, 0, 0, 0, 0, 0, 0]
            )
            assertThrows({ try SafetensorsFile(contentsOf: url) }, offsets) {
                if case .invalidDescriptor = $0 { return true }
                return false
            }
        }
    }

    func testRejectsPayloadGapsOverlapsBoundsAndTrailingBytes() throws {
        let gap = """
        {"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
         "b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}
        """
        let overlap = """
        {"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
         "b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
        """
        let interiorZero = """
        {"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
         "zero":{"dtype":"F32","shape":[0],"data_offsets":[2,2]},
         "b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
        """
        let cases = [
            (gap, 12),
            (overlap, 8),
            (interiorZero, 8),
            (singleTensorHeader(shape: "[1]", offsets: "[0,4]"), 8),
            (singleTensorHeader(shape: "[2]", offsets: "[0,8]"), 4),
        ]

        for (index, item) in cases.enumerated() {
            let url = try makeFile(
                header: Array(item.0.utf8),
                payload: [UInt8](repeating: 0, count: item.1)
            )
            assertThrows({ try SafetensorsFile(contentsOf: url) }, "case \(index)") {
                if case .invalidLayout = $0 { return true }
                return false
            }
        }
    }

    func testAllowsMultipleZeroSizedTensorsAtCoveredBoundaries() throws {
        let header = """
        {"before":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
         "data":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
         "afterA":{"dtype":"F32","shape":[2,0],"data_offsets":[4,4]},
         "afterB":{"dtype":"F32","shape":[0],"data_offsets":[4,4]}}
        """
        let file = try SafetensorsFile(
            contentsOf: makeFile(header: Array(header.utf8), payload: [0, 1, 2, 3])
        )
        XCTAssertEqual(file.descriptors.count, 4)
        XCTAssertEqual(file.tensor(named: "before")?.descriptor.byteCount, 0)
        XCTAssertEqual(file.tensor(named: "afterB")?.withUnsafeBytes { $0.count }, 0)
    }
}

private extension SafetensorsTests {
    func singleTensorHeader(shape: String, offsets: String) -> String {
        "{\"tensor\":{\"dtype\":\"F32\",\"shape\":\(shape),\"data_offsets\":\(offsets)}}"
    }

    func makeFile(
        header: [UInt8],
        payload: [UInt8] = [],
        headerLengthOverride: UInt64? = nil
    ) throws -> URL {
        let length = headerLengthOverride ?? UInt64(header.count)
        var bytes: [UInt8] = []
        bytes.reserveCapacity(8 + header.count + payload.count)
        for index in 0..<8 {
            bytes.append(UInt8(truncatingIfNeeded: length >> UInt64(index * 8)))
        }
        bytes.append(contentsOf: header)
        bytes.append(contentsOf: payload)
        return try makeRawFile(bytes)
    }

    func makeRawFile(_ bytes: [UInt8]) throws -> URL {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("weights.safetensors")
        try Data(bytes).write(to: url)
        return url
    }

    func makeSparseHeaderFile(headerLength: UInt64) throws -> URL {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("oversized.safetensors")
        var prefix = [UInt8](repeating: 0, count: 8)
        for index in 0..<8 {
            prefix[index] = UInt8(truncatingIfNeeded: headerLength >> UInt64(index * 8))
        }
        try Data(prefix + [0x7B]).write(to: url)

        let descriptor = Darwin.open(url.path, O_RDWR)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { Darwin.close(descriptor) }
        guard ftruncate(descriptor, off_t(8 + headerLength)) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return url
    }

    func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SystemOneTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func assertThrows<Result>(
        _ expression: () throws -> Result,
        _ message: String = "",
        matching predicate: (SafetensorsError) -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try expression(), message, file: file, line: line) { error in
            guard let error = error as? SafetensorsError else {
                XCTFail("Unexpected error type: \(error)", file: file, line: line)
                return
            }
            XCTAssertTrue(predicate(error), "Unexpected error: \(error)", file: file, line: line)
        }
    }
}
