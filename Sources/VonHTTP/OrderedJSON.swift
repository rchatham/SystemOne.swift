import Foundation

/// An order-preserving JSON value used by the HTTP wire codec.
public enum OrderedJSON: Equatable, Sendable {
    case null
    case bool(Bool)
    case number(OrderedJSONNumber)
    case string(String)
    case array([OrderedJSON])
    case object([OrderedJSONMember])
}

public struct OrderedJSONMember: Equatable, Sendable {
    public let key: String
    public let value: OrderedJSON

    public init(key: String, value: OrderedJSON) {
        self.key = key
        self.value = value
    }
}

/// A syntactically valid, finite JSON number. Its source spelling is retained.
public struct OrderedJSONNumber: Equatable, Sendable {
    public let rawValue: String

    public init(rawValue: String) throws {
        guard OrderedJSONParser.isValidNumber(rawValue) else {
            throw OrderedJSONError.invalidNumber
        }
        if rawValue.contains(".") || rawValue.contains("e") || rawValue.contains("E") {
            guard let value = Double(rawValue), value.isFinite else {
                throw OrderedJSONError.nonFiniteNumber
            }
        }
        self.rawValue = rawValue
    }

    public init(_ value: Double) throws {
        guard value.isFinite else { throw OrderedJSONError.nonFiniteNumber }
        self.rawValue = String(value)
    }

    public init(_ value: Int) {
        self.rawValue = String(value)
    }
}

public struct OrderedJSONLimits: Equatable, Sendable {
    public var maximumBodyBytes: Int
    public var maximumDepth: Int
    public var maximumNodes: Int
    public var maximumStringBytes: Int
    public var maximumContainerEntries: Int

    public init(
        maximumBodyBytes: Int = 1_048_576,
        maximumDepth: Int = 32,
        maximumNodes: Int = 10_000,
        maximumStringBytes: Int = 65_536,
        maximumContainerEntries: Int = 256
    ) {
        self.maximumBodyBytes = maximumBodyBytes
        self.maximumDepth = maximumDepth
        self.maximumNodes = maximumNodes
        self.maximumStringBytes = maximumStringBytes
        self.maximumContainerEntries = maximumContainerEntries
    }

    public static let httpDefault = OrderedJSONLimits()
}

public enum OrderedJSONError: Error, Equatable, Sendable {
    case bodyTooLarge
    case invalidUTF8
    case unexpectedEnd
    case unexpectedToken
    case trailingData
    case invalidEscape
    case invalidSurrogate
    case invalidNumber
    case nonFiniteNumber
    case duplicateKey(String)
    case depthLimitExceeded
    case nodeLimitExceeded
    case stringLimitExceeded
    case containerLimitExceeded
}

public struct OrderedJSONParser: Sendable {
    public let limits: OrderedJSONLimits

    public init(limits: OrderedJSONLimits = .httpDefault) {
        self.limits = limits
    }

    public func parse(_ data: Data) throws -> OrderedJSON {
        guard data.count <= limits.maximumBodyBytes else { throw OrderedJSONError.bodyTooLarge }
        guard let text = String(data: data, encoding: .utf8) else {
            throw OrderedJSONError.invalidUTF8
        }
        var implementation = ParserImplementation(text: text, limits: limits)
        return try implementation.parse()
    }

    fileprivate static func isValidNumber(_ text: String) -> Bool {
        var parser = NumberSyntax(bytes: Array(text.utf8))
        return parser.consumeNumber() && parser.index == parser.bytes.count
    }
}

public struct OrderedJSONSerializer: Sendable {
    public let limits: OrderedJSONLimits

    public init(limits: OrderedJSONLimits = .httpDefault) {
        self.limits = limits
    }

    public func serialize(_ value: OrderedJSON) throws -> Data {
        var output = ""
        var byteCount = 0
        var nodeCount = 0
        try append(value, depth: 1, nodeCount: &nodeCount, byteCount: &byteCount, to: &output)
        guard let data = output.data(using: .utf8) else { throw OrderedJSONError.invalidUTF8 }
        return data
    }

    private func append(
        _ value: OrderedJSON,
        depth: Int,
        nodeCount: inout Int,
        byteCount: inout Int,
        to output: inout String
    ) throws {
        guard depth <= limits.maximumDepth else { throw OrderedJSONError.depthLimitExceeded }
        nodeCount += 1
        guard nodeCount <= limits.maximumNodes else { throw OrderedJSONError.nodeLimitExceeded }
        switch value {
        case .null:
            try appendText("null", byteCount: &byteCount, to: &output)
        case .bool(let value):
            try appendText(value ? "true" : "false", byteCount: &byteCount, to: &output)
        case .number(let number):
            guard OrderedJSONParser.isValidNumber(number.rawValue) else {
                throw OrderedJSONError.invalidNumber
            }
            if number.rawValue.contains(".") || number.rawValue.contains("e")
                || number.rawValue.contains("E")
            {
                guard let value = Double(number.rawValue), value.isFinite else {
                    throw OrderedJSONError.nonFiniteNumber
                }
            }
            try appendText(number.rawValue, byteCount: &byteCount, to: &output)
        case .string(let string):
            try appendEscaped(string, byteCount: &byteCount, to: &output)
        case .array(let values):
            guard values.count <= limits.maximumContainerEntries else {
                throw OrderedJSONError.containerLimitExceeded
            }
            try appendText("[", byteCount: &byteCount, to: &output)
            for (index, item) in values.enumerated() {
                if index > 0 { try appendText(",", byteCount: &byteCount, to: &output) }
                try append(
                    item, depth: depth + 1, nodeCount: &nodeCount, byteCount: &byteCount,
                    to: &output)
            }
            try appendText("]", byteCount: &byteCount, to: &output)
        case .object(let members):
            guard members.count <= limits.maximumContainerEntries else {
                throw OrderedJSONError.containerLimitExceeded
            }
            var keys = Set<String>()
            try appendText("{", byteCount: &byteCount, to: &output)
            for (index, member) in members.enumerated() {
                guard keys.insert(member.key).inserted else {
                    throw OrderedJSONError.duplicateKey(member.key)
                }
                if index > 0 { try appendText(",", byteCount: &byteCount, to: &output) }
                try appendEscaped(member.key, byteCount: &byteCount, to: &output)
                try appendText(":", byteCount: &byteCount, to: &output)
                try append(
                    member.value, depth: depth + 1, nodeCount: &nodeCount, byteCount: &byteCount,
                    to: &output)
            }
            try appendText("}", byteCount: &byteCount, to: &output)
        }
    }

    private func appendEscaped(
        _ string: String,
        byteCount: inout Int,
        to output: inout String
    ) throws {
        guard string.utf8.count <= limits.maximumStringBytes else {
            throw OrderedJSONError.stringLimitExceeded
        }
        var escaped = "\""
        for scalar in string.unicodeScalars {
            switch scalar.value {
            case 0x22: escaped += "\\\""
            case 0x5C: escaped += "\\\\"
            case 0x08: escaped += "\\b"
            case 0x0C: escaped += "\\f"
            case 0x0A: escaped += "\\n"
            case 0x0D: escaped += "\\r"
            case 0x09: escaped += "\\t"
            case 0x00...0x1F:
                escaped += String(format: "\\u%04x", scalar.value)
            default:
                escaped.unicodeScalars.append(scalar)
            }
        }
        escaped += "\""
        try appendText(escaped, byteCount: &byteCount, to: &output)
    }

    private func appendText(
        _ text: String,
        byteCount: inout Int,
        to output: inout String
    ) throws {
        let (nextCount, overflow) = byteCount.addingReportingOverflow(text.utf8.count)
        guard !overflow, nextCount <= limits.maximumBodyBytes else {
            throw OrderedJSONError.bodyTooLarge
        }
        byteCount = nextCount
        output += text
    }
}

private struct ParserImplementation {
    let scalars: [Unicode.Scalar]
    let limits: OrderedJSONLimits
    var index = 0
    var nodes = 0

    init(text: String, limits: OrderedJSONLimits) {
        self.scalars = Array(text.unicodeScalars)
        self.limits = limits
    }

    mutating func parse() throws -> OrderedJSON {
        skipWhitespace()
        let value = try parseValue(depth: 1)
        skipWhitespace()
        guard index == scalars.count else { throw OrderedJSONError.trailingData }
        return value
    }

    mutating func parseValue(depth: Int) throws -> OrderedJSON {
        guard depth <= limits.maximumDepth else { throw OrderedJSONError.depthLimitExceeded }
        nodes += 1
        guard nodes <= limits.maximumNodes else { throw OrderedJSONError.nodeLimitExceeded }
        guard let scalar = current else { throw OrderedJSONError.unexpectedEnd }
        switch scalar.value {
        case 0x6E:
            try consume("null")
            return .null
        case 0x74:
            try consume("true")
            return .bool(true)
        case 0x66:
            try consume("false")
            return .bool(false)
        case 0x22: return .string(try parseString())
        case 0x5B: return try parseArray(depth: depth)
        case 0x7B: return try parseObject(depth: depth)
        case 0x2D, 0x30...0x39: return .number(try parseNumber())
        default: throw OrderedJSONError.unexpectedToken
        }
    }

    mutating func parseArray(depth: Int) throws -> OrderedJSON {
        index += 1
        skipWhitespace()
        var values: [OrderedJSON] = []
        if consumeIf(0x5D) { return .array(values) }
        while true {
            guard values.count < limits.maximumContainerEntries else {
                throw OrderedJSONError.containerLimitExceeded
            }
            values.append(try parseValue(depth: depth + 1))
            skipWhitespace()
            if consumeIf(0x5D) { return .array(values) }
            guard consumeIf(0x2C) else { throw OrderedJSONError.unexpectedToken }
            skipWhitespace()
        }
    }

    mutating func parseObject(depth: Int) throws -> OrderedJSON {
        index += 1
        skipWhitespace()
        var members: [OrderedJSONMember] = []
        var keys = Set<String>()
        if consumeIf(0x7D) { return .object(members) }
        while true {
            guard members.count < limits.maximumContainerEntries else {
                throw OrderedJSONError.containerLimitExceeded
            }
            guard current?.value == 0x22 else { throw OrderedJSONError.unexpectedToken }
            let key = try parseString()
            guard keys.insert(key).inserted else { throw OrderedJSONError.duplicateKey(key) }
            skipWhitespace()
            guard consumeIf(0x3A) else { throw OrderedJSONError.unexpectedToken }
            skipWhitespace()
            members.append(OrderedJSONMember(key: key, value: try parseValue(depth: depth + 1)))
            skipWhitespace()
            if consumeIf(0x7D) { return .object(members) }
            guard consumeIf(0x2C) else { throw OrderedJSONError.unexpectedToken }
            skipWhitespace()
        }
    }

    mutating func parseString() throws -> String {
        index += 1
        var result = String.UnicodeScalarView()
        var decodedUTF8Bytes = 0
        while let scalar = current {
            index += 1
            switch scalar.value {
            case 0x22:
                return String(result)
            case 0x00...0x1F:
                throw OrderedJSONError.unexpectedToken
            case 0x5C:
                try appendDecoded(try parseEscape(), bytes: &decodedUTF8Bytes, to: &result)
            default:
                try appendDecoded(scalar, bytes: &decodedUTF8Bytes, to: &result)
            }
        }
        throw OrderedJSONError.unexpectedEnd
    }

    mutating func parseEscape() throws -> Unicode.Scalar {
        guard let escaped = current else { throw OrderedJSONError.unexpectedEnd }
        index += 1
        switch escaped.value {
        case 0x22, 0x2F, 0x5C: return escaped
        case 0x62: return Unicode.Scalar(0x08)!
        case 0x66: return Unicode.Scalar(0x0C)!
        case 0x6E: return Unicode.Scalar(0x0A)!
        case 0x72: return Unicode.Scalar(0x0D)!
        case 0x74: return Unicode.Scalar(0x09)!
        case 0x75:
            let first = try parseHexQuad()
            if (0xD800...0xDBFF).contains(first) {
                guard consumeIf(0x5C), consumeIf(0x75) else {
                    throw OrderedJSONError.invalidSurrogate
                }
                let second = try parseHexQuad()
                guard (0xDC00...0xDFFF).contains(second) else {
                    throw OrderedJSONError.invalidSurrogate
                }
                let value = 0x10000 + ((first - 0xD800) << 10) + second - 0xDC00
                guard let scalar = Unicode.Scalar(value) else {
                    throw OrderedJSONError.invalidSurrogate
                }
                return scalar
            }
            guard !(0xDC00...0xDFFF).contains(first), let scalar = Unicode.Scalar(first) else {
                throw OrderedJSONError.invalidSurrogate
            }
            return scalar
        default: throw OrderedJSONError.invalidEscape
        }
    }

    func appendDecoded(
        _ scalar: Unicode.Scalar,
        bytes: inout Int,
        to result: inout String.UnicodeScalarView
    ) throws {
        let width: Int
        switch scalar.value {
        case 0...0x7F: width = 1
        case 0x80...0x7FF: width = 2
        case 0x800...0xFFFF: width = 3
        default: width = 4
        }
        let (next, overflow) = bytes.addingReportingOverflow(width)
        guard !overflow, next <= limits.maximumStringBytes else {
            throw OrderedJSONError.stringLimitExceeded
        }
        bytes = next
        result.append(scalar)
    }

    mutating func parseHexQuad() throws -> UInt32 {
        var value: UInt32 = 0
        for _ in 0..<4 {
            guard let scalar = current else { throw OrderedJSONError.unexpectedEnd }
            index += 1
            let digit: UInt32
            switch scalar.value {
            case 0x30...0x39: digit = scalar.value - 0x30
            case 0x41...0x46: digit = scalar.value - 0x41 + 10
            case 0x61...0x66: digit = scalar.value - 0x61 + 10
            default: throw OrderedJSONError.invalidEscape
            }
            value = value * 16 + digit
        }
        return value
    }

    mutating func parseNumber() throws -> OrderedJSONNumber {
        let start = index
        var bytes: [UInt8] = []
        while let scalar = current, scalar.isASCII,
            scalar.value == 0x2D || scalar.value == 0x2B || scalar.value == 0x2E
                || scalar.value == 0x45 || scalar.value == 0x65
                || (0x30...0x39).contains(scalar.value)
        {
            bytes.append(UInt8(scalar.value))
            index += 1
        }
        let raw = String(decoding: bytes, as: UTF8.self)
        guard index > start, OrderedJSONParser.isValidNumber(raw) else {
            throw OrderedJSONError.invalidNumber
        }
        return try OrderedJSONNumber(rawValue: raw)
    }

    mutating func consume(_ literal: String) throws {
        for expected in literal.unicodeScalars {
            guard current == expected else { throw OrderedJSONError.unexpectedToken }
            index += 1
        }
    }

    mutating func skipWhitespace() {
        while let value = current?.value,
            value == 0x20 || value == 0x09 || value == 0x0A || value == 0x0D
        {
            index += 1
        }
    }

    mutating func consumeIf(_ value: UInt32) -> Bool {
        guard current?.value == value else { return false }
        index += 1
        return true
    }

    var current: Unicode.Scalar? { index < scalars.count ? scalars[index] : nil }
}

private struct NumberSyntax {
    let bytes: [UInt8]
    var index = 0

    mutating func consumeNumber() -> Bool {
        _ = consume(0x2D)
        if consume(0x30) {
            if isDigit(current) { return false }
        } else {
            guard consumeDigits(requireOne: true) else { return false }
        }
        if consume(0x2E), !consumeDigits(requireOne: true) { return false }
        if consume(0x65) || consume(0x45) {
            _ = consume(0x2B) || consume(0x2D)
            guard consumeDigits(requireOne: true) else { return false }
        }
        return true
    }

    mutating func consumeDigits(requireOne: Bool) -> Bool {
        let start = index
        while isDigit(current) { index += 1 }
        return !requireOne || index > start
    }

    mutating func consume(_ byte: UInt8) -> Bool {
        guard current == byte else { return false }
        index += 1
        return true
    }

    func isDigit(_ byte: UInt8?) -> Bool {
        guard let byte else { return false }
        return (0x30...0x39).contains(byte)
    }

    var current: UInt8? { index < bytes.count ? bytes[index] : nil }
}
