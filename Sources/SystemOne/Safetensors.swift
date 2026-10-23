import Darwin
import Foundation

/// The tensor element types the loader accepts. Models compute in float32;
/// `I64` appears only for bookkeeping tensors such as BatchNorm counters.
public enum SafetensorsDataType: String, Sendable {
    case float32 = "F32"
    case int64 = "I64"

    /// Bytes per element.
    public var byteWidth: Int {
        switch self {
        case .float32: return 4
        case .int64: return 8
        }
    }
}

/// A validated tensor entry from a safetensors header.
public struct SafetensorsTensorDescriptor: Equatable, Sendable {
    public let name: String
    public let dataType: SafetensorsDataType
    public let shape: [Int]
    public let elementCount: Int
    public let payloadRange: Range<Int>
    public let byteCount: Int

    public init(
        name: String,
        dataType: SafetensorsDataType,
        shape: [Int],
        elementCount: Int,
        payloadRange: Range<Int>,
        byteCount: Int
    ) {
        self.name = name
        self.dataType = dataType
        self.shape = shape
        self.elementCount = elementCount
        self.payloadRange = payloadRange
        self.byteCount = byteCount
    }
}

/// Errors raised while opening or validating a safetensors file.
public enum SafetensorsError: Error, Equatable, Sendable {
    case openFailed(path: String, code: Int32)
    case statFailed(path: String, code: Int32)
    case notRegularFile(String)
    case invalidFileSize(String)
    case mapFailed(path: String, code: Int32)
    case closeFailed(path: String, code: Int32)
    case fileTooSmall(actual: Int)
    case headerTooLarge(actual: UInt64, maximum: UInt64)
    case headerLengthOutOfRange(UInt64)
    case invalidJSON(String)
    case duplicateJSONKey(String)
    case rootMustBeObject
    case invalidMetadata(String)
    case invalidDescriptor(name: String, reason: String)
    case invalidLayout(String)
}

extension SafetensorsError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .openFailed(let path, let code):
            return "Could not open safetensors file \(path): errno \(code)."
        case .statFailed(let path, let code):
            return "Could not stat safetensors file \(path): errno \(code)."
        case .notRegularFile(let path):
            return "Safetensors path is not a regular file: \(path)."
        case .invalidFileSize(let path):
            return "Safetensors file size is invalid or not representable: \(path)."
        case .mapFailed(let path, let code):
            return "Could not mmap safetensors file \(path): errno \(code)."
        case .closeFailed(let path, let code):
            return "Could not close safetensors file \(path) after mmap: errno \(code)."
        case .fileTooSmall(let actual):
            return "Safetensors file has \(actual) bytes; at least 8 are required."
        case .headerTooLarge(let actual, let maximum):
            return "Safetensors header has \(actual) bytes; maximum is \(maximum)."
        case .headerLengthOutOfRange(let length):
            return "Safetensors header length \(length) is outside the mapped file."
        case .invalidJSON(let reason):
            return "Safetensors header is not valid JSON: \(reason)."
        case .duplicateJSONKey(let key):
            return "Safetensors header contains duplicate decoded key \(key)."
        case .rootMustBeObject:
            return "Safetensors header root must be an object."
        case .invalidMetadata(let reason):
            return "Safetensors metadata is invalid: \(reason)."
        case .invalidDescriptor(let name, let reason):
            return "Safetensors descriptor \(name) is invalid: \(reason)."
        case .invalidLayout(let reason):
            return "Safetensors payload layout is invalid: \(reason)."
        }
    }
}

/// A zero-copy tensor byte view that retains its read-only file mapping.
public struct SafetensorsTensorView: @unchecked Sendable {
    public let descriptor: SafetensorsTensorDescriptor

    private let owner: SafetensorsMappedFile
    private let absoluteRange: Range<Int>

    fileprivate init(
        descriptor: SafetensorsTensorDescriptor,
        owner: SafetensorsMappedFile,
        dataStart: Int
    ) {
        self.descriptor = descriptor
        self.owner = owner
        self.absoluteRange = (dataStart + descriptor.payloadRange.lowerBound)..<(
            dataStart + descriptor.payloadRange.upperBound
        )
    }

    /// Provides scoped access to the tensor's mapped bytes without copying them.
    public func withUnsafeBytes<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        try owner.withUnsafeBytes(in: absoluteRange, body)
    }
}

/// A read-only, memory-mapped safetensors file with strict format and layout validation.
///
/// This type does not validate a model-specific schema or artifact identity.
/// The mapped inode must not be truncated or mutated while this object or one
/// of its tensor views is alive.
public final class SafetensorsFile: @unchecked Sendable {
    private static let maximumHeaderByteCount: UInt64 = 100_000_000

    public let metadata: [String: String]?
    public let descriptors: [String: SafetensorsTensorDescriptor]
    public let fileByteCount: Int
    public let headerByteCount: Int
    public let dataStart: Int
    public let payloadByteCount: Int

    private let owner: SafetensorsMappedFile

    /// Opens and validates a local safetensors file's generic format and layout
    /// without copying its payload.
    public convenience init(contentsOf url: URL) throws {
        try self.init(contentsOf: url, validatingMappedBytes: { _ in })
    }

    /// Opens a local file, verifies its complete raw mapping, then parses that
    /// same mapping as safetensors. The validator runs before any header reads.
    internal init(
        contentsOf url: URL,
        validatingFileByteCount validateFileByteCount: (Int) throws -> Void = { _ in },
        validatingMappedBytes validateMappedBytes: (UnsafeRawBufferPointer) throws -> Void
    ) throws {
        let owner = try SafetensorsMappedFile(
            url: url,
            validatingFileByteCount: validateFileByteCount
        )
        try owner.withUnsafeBytes(in: 0..<owner.byteCount, validateMappedBytes)

        guard owner.byteCount >= 8 else {
            throw SafetensorsError.fileTooSmall(actual: owner.byteCount)
        }

        let headerLength = owner.withUnsafeBytes(in: 0..<8) { bytes -> UInt64 in
            var value: UInt64 = 0
            for index in 0..<8 {
                value |= UInt64(bytes[index]) << UInt64(index * 8)
            }
            return value
        }
        guard headerLength <= Self.maximumHeaderByteCount else {
            throw SafetensorsError.headerTooLarge(
                actual: headerLength,
                maximum: Self.maximumHeaderByteCount
            )
        }
        guard let headerByteCount = Int(exactly: headerLength) else {
            throw SafetensorsError.headerLengthOutOfRange(headerLength)
        }
        let (dataStart, additionOverflow) = 8.addingReportingOverflow(headerByteCount)
        guard !additionOverflow, dataStart <= owner.byteCount else {
            throw SafetensorsError.headerLengthOutOfRange(headerLength)
        }

        let header = try owner.withUnsafeBytes(in: 8..<dataStart) { bytes in
            try SafetensorsHeader.parse(bytes: bytes)
        }
        let payloadByteCount = owner.byteCount - dataStart
        try Self.validateLayout(
            descriptors: header.descriptors,
            payloadByteCount: payloadByteCount,
            dataStart: dataStart,
            fileByteCount: owner.byteCount
        )

        self.owner = owner
        self.metadata = header.metadata
        self.descriptors = header.descriptors
        self.fileByteCount = owner.byteCount
        self.headerByteCount = headerByteCount
        self.dataStart = dataStart
        self.payloadByteCount = payloadByteCount
    }

    /// Returns a retained zero-copy view for a named tensor.
    public func tensor(named name: String) -> SafetensorsTensorView? {
        guard let descriptor = descriptors[name] else { return nil }
        return SafetensorsTensorView(descriptor: descriptor, owner: owner, dataStart: dataStart)
    }

    /// Provides scoped access to the complete mapped file without copying it.
    public func withUnsafeBytes<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        try owner.withUnsafeBytes(in: 0..<fileByteCount, body)
    }

    /// Provides scoped access to the mapped payload without copying it.
    public func withUnsafePayloadBytes<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        try owner.withUnsafeBytes(in: dataStart..<fileByteCount, body)
    }

    private static func validateLayout(
        descriptors: [String: SafetensorsTensorDescriptor],
        payloadByteCount: Int,
        dataStart: Int,
        fileByteCount: Int
    ) throws {
        let ordered = descriptors.values.sorted {
            if $0.payloadRange.lowerBound == $1.payloadRange.lowerBound {
                return $0.payloadRange.upperBound < $1.payloadRange.upperBound
            }
            return $0.payloadRange.lowerBound < $1.payloadRange.lowerBound
        }

        var cursor = 0
        for descriptor in ordered {
            let range = descriptor.payloadRange
            guard range.lowerBound == cursor else {
                let kind = range.lowerBound < cursor ? "overlap" : "gap"
                throw SafetensorsError.invalidLayout(
                    "\(kind) before \(descriptor.name): expected offset \(cursor), got \(range.lowerBound)"
                )
            }
            guard range.upperBound <= payloadByteCount else {
                throw SafetensorsError.invalidLayout(
                    "\(descriptor.name) ends past payload byte \(payloadByteCount)"
                )
            }
            let (absoluteEnd, overflow) = dataStart.addingReportingOverflow(range.upperBound)
            guard !overflow, absoluteEnd <= fileByteCount else {
                throw SafetensorsError.invalidLayout(
                    "\(descriptor.name) has an out-of-bounds absolute range"
                )
            }
            cursor = range.upperBound
        }

        guard cursor == payloadByteCount else {
            throw SafetensorsError.invalidLayout(
                "trailing payload bytes: tensors end at \(cursor), payload ends at \(payloadByteCount)"
            )
        }
    }
}

private final class SafetensorsMappedFile: @unchecked Sendable {
    let byteCount: Int

    private let baseAddress: UnsafeMutableRawPointer

    init(
        url: URL,
        validatingFileByteCount validateFileByteCount: (Int) throws -> Void
    ) throws {
        let path = url.path
        let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw SafetensorsError.openFailed(path: path, code: errno)
        }

        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0 else {
            let code = errno
            Darwin.close(descriptor)
            throw SafetensorsError.statFailed(path: path, code: code)
        }
        guard (status.st_mode & S_IFMT) == S_IFREG else {
            Darwin.close(descriptor)
            throw SafetensorsError.notRegularFile(path)
        }
        guard status.st_size >= 0, let byteCount = Int(exactly: status.st_size) else {
            Darwin.close(descriptor)
            throw SafetensorsError.invalidFileSize(path)
        }
        do {
            try validateFileByteCount(byteCount)
        } catch {
            Darwin.close(descriptor)
            throw error
        }
        guard byteCount > 0 else {
            Darwin.close(descriptor)
            throw SafetensorsError.fileTooSmall(actual: byteCount)
        }

        let mapping = mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, descriptor, 0)
        guard mapping != MAP_FAILED, let baseAddress = mapping else {
            let code = errno
            Darwin.close(descriptor)
            throw SafetensorsError.mapFailed(path: path, code: code)
        }
        guard Darwin.close(descriptor) == 0 else {
            let code = errno
            munmap(baseAddress, byteCount)
            throw SafetensorsError.closeFailed(path: path, code: code)
        }

        self.baseAddress = baseAddress
        self.byteCount = byteCount
    }

    deinit {
        let unmapResult = munmap(baseAddress, byteCount)
        assert(unmapResult == 0, "munmap failed with errno \(errno)")
    }

    func withUnsafeBytes<Result>(
        in range: Range<Int>,
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        let bytes = UnsafeRawBufferPointer(
            start: baseAddress.advanced(by: range.lowerBound),
            count: range.count
        )
        return try withExtendedLifetime(self) {
            try body(bytes)
        }
    }
}

private struct SafetensorsHeader {
    let metadata: [String: String]?
    let descriptors: [String: SafetensorsTensorDescriptor]

    static func parse(bytes: UnsafeRawBufferPointer) throws -> SafetensorsHeader {
        let root = try StrictJSONParser(bytes: bytes).parseDocument()
        guard case .object(let entries) = root else {
            throw SafetensorsError.rootMustBeObject
        }

        var metadata: [String: String]?
        var descriptors: [String: SafetensorsTensorDescriptor] = [:]
        for (name, value) in entries {
            if name == "__metadata__" {
                metadata = try parseMetadata(value)
            } else {
                descriptors[name] = try parseDescriptor(name: name, value: value)
            }
        }
        return SafetensorsHeader(metadata: metadata, descriptors: descriptors)
    }

    private static func parseMetadata(_ value: JSONValue) throws -> [String: String] {
        guard case .object(let entries) = value else {
            throw SafetensorsError.invalidMetadata("__metadata__ must be an object")
        }
        var metadata: [String: String] = [:]
        for (key, value) in entries {
            guard case .string(let string) = value else {
                throw SafetensorsError.invalidMetadata("value for \(key) must be a string")
            }
            metadata[key] = string
        }
        return metadata
    }

    private static func parseDescriptor(
        name: String,
        value: JSONValue
    ) throws -> SafetensorsTensorDescriptor {
        guard case .object(let fields) = value else {
            throw SafetensorsError.invalidDescriptor(name: name, reason: "entry must be an object")
        }
        let expectedFields: Set<String> = ["dtype", "shape", "data_offsets"]
        let actualFields = Set(fields.keys)
        guard actualFields == expectedFields else {
            let missing = expectedFields.subtracting(actualFields).sorted()
            let unknown = actualFields.subtracting(expectedFields).sorted()
            throw SafetensorsError.invalidDescriptor(
                name: name,
                reason: "expected exactly dtype, shape, data_offsets; missing \(missing), unknown \(unknown)"
            )
        }

        guard case .string(let dataTypeName)? = fields["dtype"] else {
            throw SafetensorsError.invalidDescriptor(name: name, reason: "dtype must be a string")
        }
        guard let dataType = SafetensorsDataType(rawValue: dataTypeName) else {
            throw SafetensorsError.invalidDescriptor(
                name: name,
                reason: "unsupported dtype \(dataTypeName); expected F32 or I64"
            )
        }

        guard case .array(let shapeValues)? = fields["shape"] else {
            throw SafetensorsError.invalidDescriptor(name: name, reason: "shape must be an array")
        }
        let shape = try shapeValues.enumerated().map { index, value in
            try nonnegativeInteger(value, name: name, field: "shape[\(index)]")
        }

        var elementCount = 1
        for dimension in shape {
            let (product, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw SafetensorsError.invalidDescriptor(name: name, reason: "shape product overflows Int")
            }
            elementCount = product
        }
        let (byteCount, byteOverflow) = elementCount.multipliedReportingOverflow(by: dataType.byteWidth)
        guard !byteOverflow else {
            throw SafetensorsError.invalidDescriptor(name: name, reason: "\(dataTypeName) byte count overflows Int")
        }

        guard case .array(let offsetValues)? = fields["data_offsets"], offsetValues.count == 2 else {
            throw SafetensorsError.invalidDescriptor(
                name: name,
                reason: "data_offsets must contain exactly two integers"
            )
        }
        let start = try nonnegativeInteger(offsetValues[0], name: name, field: "data_offsets[0]")
        let end = try nonnegativeInteger(offsetValues[1], name: name, field: "data_offsets[1]")
        guard start <= end else {
            throw SafetensorsError.invalidDescriptor(name: name, reason: "data_offsets are not ordered")
        }
        let (rangeByteCount, subtractionOverflow) = end.subtractingReportingOverflow(start)
        guard !subtractionOverflow, rangeByteCount == byteCount else {
            throw SafetensorsError.invalidDescriptor(
                name: name,
                reason: "offset length \(rangeByteCount) does not match shape byte count \(byteCount)"
            )
        }

        return SafetensorsTensorDescriptor(
            name: name,
            dataType: dataType,
            shape: shape,
            elementCount: elementCount,
            payloadRange: start..<end,
            byteCount: byteCount
        )
    }

    private static func nonnegativeInteger(
        _ value: JSONValue,
        name: String,
        field: String
    ) throws -> Int {
        guard case .number(let token) = value,
              !token.isEmpty,
              token.utf8.allSatisfy({ $0 >= 0x30 && $0 <= 0x39 }),
              let unsigned = UInt64(token),
              let integer = Int(exactly: unsigned)
        else {
            throw SafetensorsError.invalidDescriptor(
                name: name,
                reason: "\(field) must be a nonnegative representable integer"
            )
        }
        return integer
    }
}

private indirect enum JSONValue {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case number(String)
    case bool(Bool)
    case null
}

private final class StrictJSONParser {
    private static let maximumNestingDepth = 128

    private let bytes: UnsafeRawBufferPointer
    private var index = 0

    init(bytes: UnsafeRawBufferPointer) {
        self.bytes = bytes
    }

    func parseDocument() throws -> JSONValue {
        let value = try parseValue(depth: 0)
        while index < bytes.count, bytes[index] == 0x20 {
            index += 1
        }
        guard index == bytes.count else {
            throw invalid("non-space padding follows the root value at byte \(index)")
        }
        return value
    }

    private func parseValue(depth: Int) throws -> JSONValue {
        guard depth <= Self.maximumNestingDepth else {
            throw invalid("JSON nesting exceeds \(Self.maximumNestingDepth) levels")
        }
        guard index < bytes.count else { throw invalid("unexpected end of input") }
        switch bytes[index] {
        case 0x7B:
            return try parseObject(depth: depth)
        case 0x5B:
            return try parseArray(depth: depth)
        case 0x22:
            return .string(try parseString())
        case 0x74:
            try consumeLiteral("true")
            return .bool(true)
        case 0x66:
            try consumeLiteral("false")
            return .bool(false)
        case 0x6E:
            try consumeLiteral("null")
            return .null
        case 0x2D, 0x30...0x39:
            return .number(try parseNumber())
        default:
            throw invalid("unexpected byte at offset \(index)")
        }
    }

    private func parseObject(depth: Int) throws -> JSONValue {
        index += 1
        skipWhitespace()
        var result: [String: JSONValue] = [:]
        if consumeIf(0x7D) { return .object(result) }

        while true {
            guard index < bytes.count, bytes[index] == 0x22 else {
                throw invalid("object key must be a string at byte \(index)")
            }
            let key = try parseString()
            guard result[key] == nil else {
                throw SafetensorsError.duplicateJSONKey(key)
            }
            skipWhitespace()
            try consume(0x3A, message: "expected ':' after object key")
            skipWhitespace()
            result[key] = try parseValue(depth: depth + 1)
            skipWhitespace()
            if consumeIf(0x7D) { break }
            try consume(0x2C, message: "expected ',' or '}' in object")
            skipWhitespace()
        }
        return .object(result)
    }

    private func parseArray(depth: Int) throws -> JSONValue {
        index += 1
        skipWhitespace()
        var result: [JSONValue] = []
        if consumeIf(0x5D) { return .array(result) }

        while true {
            result.append(try parseValue(depth: depth + 1))
            skipWhitespace()
            if consumeIf(0x5D) { break }
            try consume(0x2C, message: "expected ',' or ']' in array")
            skipWhitespace()
        }
        return .array(result)
    }

    private func parseString() throws -> String {
        try consume(0x22, message: "expected string")
        var result = ""
        var segmentStart = index

        while index < bytes.count {
            let byte = bytes[index]
            if byte == 0x22 {
                try appendUTF8(from: segmentStart, to: index, into: &result)
                index += 1
                return result
            }
            if byte == 0x5C {
                try appendUTF8(from: segmentStart, to: index, into: &result)
                index += 1
                guard index < bytes.count else { throw invalid("unterminated string escape") }
                let escape = bytes[index]
                index += 1
                switch escape {
                case 0x22: result.append("\"")
                case 0x5C: result.append("\\")
                case 0x2F: result.append("/")
                case 0x62: result.append("\u{0008}")
                case 0x66: result.append("\u{000C}")
                case 0x6E: result.append("\n")
                case 0x72: result.append("\r")
                case 0x74: result.append("\t")
                case 0x75:
                    try appendUnicodeEscape(into: &result)
                default:
                    throw invalid("invalid string escape at byte \(index - 1)")
                }
                segmentStart = index
            } else {
                guard byte >= 0x20 else {
                    throw invalid("unescaped control byte in string at byte \(index)")
                }
                index += 1
            }
        }
        throw invalid("unterminated string")
    }

    private func appendUTF8(from start: Int, to end: Int, into result: inout String) throws {
        guard start < end else { return }
        let slice = UnsafeRawBufferPointer(rebasing: bytes[start..<end])
        guard let string = String(bytes: slice, encoding: .utf8) else {
            throw invalid("invalid UTF-8 in string at byte \(start)")
        }
        result.append(string)
    }

    private func appendUnicodeEscape(into result: inout String) throws {
        let first = try parseHexQuad()
        let scalarValue: UInt32
        if (0xD800...0xDBFF).contains(first) {
            guard index + 2 <= bytes.count,
                  bytes[index] == 0x5C,
                  bytes[index + 1] == 0x75
            else {
                throw invalid("high surrogate is not followed by a low surrogate")
            }
            index += 2
            let second = try parseHexQuad()
            guard (0xDC00...0xDFFF).contains(second) else {
                throw invalid("high surrogate is not followed by a low surrogate")
            }
            scalarValue = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
        } else {
            guard !(0xDC00...0xDFFF).contains(first) else {
                throw invalid("unpaired low surrogate")
            }
            scalarValue = first
        }
        guard let scalar = Unicode.Scalar(scalarValue) else {
            throw invalid("invalid Unicode scalar")
        }
        result.unicodeScalars.append(scalar)
    }

    private func parseHexQuad() throws -> UInt32 {
        guard index + 4 <= bytes.count else { throw invalid("incomplete Unicode escape") }
        var value: UInt32 = 0
        for _ in 0..<4 {
            let byte = bytes[index]
            index += 1
            let digit: UInt32
            switch byte {
            case 0x30...0x39: digit = UInt32(byte - 0x30)
            case 0x41...0x46: digit = UInt32(byte - 0x41 + 10)
            case 0x61...0x66: digit = UInt32(byte - 0x61 + 10)
            default: throw invalid("non-hexadecimal Unicode escape")
            }
            value = (value << 4) | digit
        }
        return value
    }

    private func parseNumber() throws -> String {
        let start = index
        if consumeIf(0x2D), index == bytes.count {
            throw invalid("incomplete number")
        }

        if consumeIf(0x30) {
            if index < bytes.count, (0x30...0x39).contains(bytes[index]) {
                throw invalid("number has a leading zero")
            }
        } else {
            guard index < bytes.count, (0x31...0x39).contains(bytes[index]) else {
                throw invalid("invalid number at byte \(start)")
            }
            repeat { index += 1 } while index < bytes.count && (0x30...0x39).contains(bytes[index])
        }

        if consumeIf(0x2E) {
            guard index < bytes.count, (0x30...0x39).contains(bytes[index]) else {
                throw invalid("fraction has no digits")
            }
            repeat { index += 1 } while index < bytes.count && (0x30...0x39).contains(bytes[index])
        }

        if index < bytes.count, bytes[index] == 0x65 || bytes[index] == 0x45 {
            index += 1
            if index < bytes.count, bytes[index] == 0x2B || bytes[index] == 0x2D {
                index += 1
            }
            guard index < bytes.count, (0x30...0x39).contains(bytes[index]) else {
                throw invalid("exponent has no digits")
            }
            repeat { index += 1 } while index < bytes.count && (0x30...0x39).contains(bytes[index])
        }

        let tokenBytes = UnsafeRawBufferPointer(rebasing: bytes[start..<index])
        return String(decoding: tokenBytes, as: UTF8.self)
    }

    private func consumeLiteral(_ literal: StaticString) throws {
        let expected = Array("\(literal)".utf8)
        guard index + expected.count <= bytes.count else {
            throw invalid("incomplete literal")
        }
        for byte in expected {
            guard bytes[index] == byte else { throw invalid("invalid literal") }
            index += 1
        }
    }

    private func consume(_ byte: UInt8, message: String) throws {
        guard consumeIf(byte) else { throw invalid(message + " at byte \(index)") }
    }

    private func consumeIf(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }

    private func skipWhitespace() {
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x09, 0x0A, 0x0D:
                index += 1
            default:
                return
            }
        }
    }

    private func invalid(_ reason: String) -> SafetensorsError {
        .invalidJSON(reason)
    }
}
