import Accelerate
import CryptoKit
import Foundation
import MLX

/// One tensor a model expects: its exact name, shape, and element type.
public struct WeightSpecification: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dataType: SafetensorsDataType

    public init(name: String, shape: [Int], dataType: SafetensorsDataType = .float32) {
        self.name = name
        self.shape = shape
        self.dataType = dataType
    }
}

/// A released artifact identified by its exact size and whole-file SHA-256.
public struct PinnedArtifact: Equatable, Sendable {
    public let byteCount: Int
    public let sha256: String

    public init(byteCount: Int, sha256: String) {
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

/// Errors raised when a checkpoint does not match a model's schema or pin.
public enum WeightsError: Error, Equatable, Sendable {
    case artifactSizeMismatch(expected: Int, actual: Int)
    case artifactSHA256Mismatch(expected: String, actual: String)
    case tensorSetMismatch(missing: [String], unexpected: [String])
    case dataTypeMismatch(name: String, expected: SafetensorsDataType, actual: SafetensorsDataType)
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
    case float16Overflow(name: String, maximumMagnitude: Float)
}

extension WeightsError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .artifactSizeMismatch(let expected, let actual):
            return "Weights artifact has \(actual) bytes; expected exactly \(expected)."
        case .artifactSHA256Mismatch(let expected, let actual):
            return "Weights artifact has SHA256 \(actual); expected \(expected)."
        case .tensorSetMismatch(let missing, let unexpected):
            return "Tensor names do not match the model schema; missing \(missing), unexpected \(unexpected)."
        case .dataTypeMismatch(let name, let expected, let actual):
            return "Tensor \(name) has dtype \(actual.rawValue); expected \(expected.rawValue)."
        case .shapeMismatch(let name, let expected, let actual):
            return "Tensor \(name) has shape \(actual); expected \(expected)."
        case .float16Overflow(let name, let maximumMagnitude):
            return "Tensor \(name) holds \(maximumMagnitude), beyond float16's range (65504); load it at float32."
        }
    }
}

extension SafetensorsFile {
    private static let hashChunkByteCount = 16 * 1_024 * 1_024

    /// Checks the file size before mapping, then hashes the complete mapping
    /// before parsing it, so a pinned artifact is verified byte for byte.
    public convenience init(contentsOf url: URL, pinnedTo pin: PinnedArtifact) throws {
        try self.init(
            contentsOf: url,
            validatingFileByteCount: { byteCount in
                guard byteCount == pin.byteCount else {
                    throw WeightsError.artifactSizeMismatch(expected: pin.byteCount, actual: byteCount)
                }
            },
            validatingMappedBytes: { bytes in
                var hasher = SHA256()
                var offset = 0
                while offset < bytes.count {
                    let end = min(offset + Self.hashChunkByteCount, bytes.count)
                    hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: bytes[offset..<end]))
                    offset = end
                }
                let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
                guard actual == pin.sha256 else {
                    throw WeightsError.artifactSHA256Mismatch(expected: pin.sha256, actual: actual)
                }
            }
        )
    }
}

extension TensorStore {
    /// Requires exactly the schema's tensor names, with matching dtypes and shapes.
    public func validate(schema: [WeightSpecification]) throws {
        let expectedNames = Set(schema.map(\.name))
        let actualNames = Set(descriptors.keys)
        let missing = expectedNames.subtracting(actualNames).sorted()
        let unexpected = actualNames.subtracting(expectedNames).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw WeightsError.tensorSetMismatch(missing: missing, unexpected: unexpected)
        }
        for specification in schema {
            let descriptor = descriptors[specification.name]!
            guard descriptor.dataType == specification.dataType else {
                throw WeightsError.dataTypeMismatch(
                    name: specification.name,
                    expected: specification.dataType,
                    actual: descriptor.dataType
                )
            }
            guard descriptor.shape == specification.shape else {
                throw WeightsError.shapeMismatch(
                    name: specification.name,
                    expected: specification.shape,
                    actual: descriptor.shape
                )
            }
        }
    }

    /// Returns a named tensor or reports it missing.
    public func requiredTensor(_ name: String) throws -> SafetensorsTensorView {
        guard let view = tensor(named: name) else {
            throw WeightsError.tensorSetMismatch(missing: [name], unexpected: [])
        }
        return view
    }
}

extension SafetensorsTensorView {
    /// Copies this tensor into an MLX-owned array of its stored dtype, so no
    /// MLX array ever retains a pointer into the read-only safetensors mapping.
    package func mlxArray() -> MLXArray {
        withUnsafeBytes { bytes in
            let data = bytes.baseAddress.map { Data(bytes: $0, count: bytes.count) } ?? Data()
            return MLXArray(data, descriptor.shape, dtype: descriptor.dataType.mlxDataType)
        }
    }

    /// Copies this tensor into MLX and casts it to `dataType` when that differs
    /// from the stored dtype.
    package func mlxArray(_ dataType: DType) -> MLXArray {
        let array = mlxArray()
        return array.dtype == dataType ? array : array.asType(dataType)
    }

    /// The values of a floating-point tensor as float32, decoded on the CPU.
    package func floatValues() -> [Float] {
        switch descriptor.dataType {
        case .float32:
            return withFloats { Array($0) }
        case .float16:
            return withHalfWords { $0.map(Self.float(fromFloat16:)) }
        case .bfloat16:
            return withHalfWords { $0.map(Self.float(fromBFloat16:)) }
        default:
            preconditionFailure("\(descriptor.dataType.rawValue) tensors have no float values")
        }
    }

    /// The smallest magnitude float16 rounds to infinity: values below it round
    /// to at most 65504.
    package static let float16OverflowMagnitude: Float = 65_520

    /// Copies this float32 tensor into an MLX-owned float16 array, converted on
    /// the CPU straight into a page-aligned buffer the array owns, so Metal wraps
    /// it without a second copy and MLX never holds a float32 copy.
    ///
    /// Also returns the source's largest magnitude, read in the same pass:
    /// values of `float16OverflowMagnitude` or more become infinity.
    /// float16 and bfloat16 sources are accepted too: float16 is copied as is,
    /// and bfloat16 is widened to float32 and rounded like a float32 source.
    package func float16Array() -> (array: MLXArray, maximumMagnitude: Float) {
        switch descriptor.dataType {
        case .float32:
            break
        case .float16:
            return (mlxArray(), maximumMagnitude())
        case .bfloat16:
            let values = floatValues()
            let half = Self.float16Bytes(values)
            let maximum = values.withUnsafeBufferPointer(Self.maximumMagnitude)
            return (MLXArray(half, descriptor.shape, dtype: .float16), maximum)
        default:
            preconditionFailure("Only floating-point tensors convert to float16")
        }
        let page = Int(getpagesize())
        var buffer: UnsafeMutableRawPointer?
        let capacity = max(page, (descriptor.elementCount * 2 + page - 1) / page * page)
        precondition(posix_memalign(&buffer, page, capacity) == 0, "float16 buffer allocation failed")
        let maximum = withFloats { values in
            Self.convertToFloat16(values, into: buffer!)
            return Self.maximumMagnitude(values)
        }
        return (MLXArray(rawPointer: buffer!, descriptor.shape, dtype: .float16) { free(buffer) }, maximum)
    }

    /// The largest absolute value in this floating-point tensor.
    package func maximumMagnitude() -> Float {
        switch descriptor.dataType {
        case .float32:
            return withFloats(Self.maximumMagnitude)
        case .float16, .bfloat16:
            return floatValues().withUnsafeBufferPointer(Self.maximumMagnitude)
        default:
            preconditionFailure("Only floating-point tensors have a float magnitude")
        }
    }

    /// The float32 value of an IEEE float16 bit pattern (exact).
    package static func float(fromFloat16 bits: UInt16) -> Float {
        let sign: UInt32 = UInt32(bits & 0x8000) << 16
        let exponent = UInt32((bits >> 10) & 0x1F)
        let mantissa = UInt32(bits & 0x03FF)
        switch exponent {
        case 0:
            // Zero or subnormal: mantissa × 2^-24.
            let magnitude = Float(mantissa) * Float(bitPattern: 0x3380_0000)
            return sign == 0 ? magnitude : -magnitude
        case 0x1F:
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        default:
            return Float(bitPattern: sign | ((exponent + 112) << 23) | (mantissa << 13))
        }
    }

    /// The float32 value of a bfloat16 bit pattern (exact).
    package static func float(fromBFloat16 bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// IEEE float16 bytes for float32 values, rounding to nearest even.
    package static func float16Bytes(_ values: [Float]) -> Data {
        var half = Data(count: values.count * 2)
        half.withUnsafeMutableBytes { output in
            values.withUnsafeBufferPointer { if let base = output.baseAddress { convertToFloat16($0, into: base) } }
        }
        return half
    }

    /// The mapped values as floats, copied first when the payload is not
    /// 4-byte aligned (safetensors does not require an aligned header length).
    private func withFloats<Result>(_ body: (UnsafeBufferPointer<Float>) -> Result) -> Result {
        withUnsafeBytes { bytes in
            let count = bytes.count / MemoryLayout<Float>.size
            guard let base = bytes.baseAddress, count > 0 else { return body(UnsafeBufferPointer(start: nil, count: 0)) }
            if Int(bitPattern: base) % MemoryLayout<Float>.alignment == 0 {
                return body(UnsafeBufferPointer(start: base.assumingMemoryBound(to: Float.self), count: count))
            }
            let aligned = [Float](unsafeUninitializedCapacity: count) { storage, initialized in
                memcpy(storage.baseAddress!, base, count * MemoryLayout<Float>.size)
                initialized = count
            }
            return aligned.withUnsafeBufferPointer(body)
        }
    }

    /// The mapped 16-bit words of a float16 or bfloat16 tensor, copied out so
    /// misaligned payloads read correctly.
    private func withHalfWords<Result>(_ body: ([UInt16]) -> Result) -> Result {
        withUnsafeBytes { bytes in
            let count = bytes.count / 2
            let words = [UInt16](unsafeUninitializedCapacity: count) { storage, initialized in
                if count > 0 { memcpy(storage.baseAddress!, bytes.baseAddress!, count * 2) }
                initialized = count
            }
            return body(words.map { UInt16(littleEndian: $0) })
        }
    }

    private static func maximumMagnitude(_ values: UnsafeBufferPointer<Float>) -> Float {
        guard let base = values.baseAddress, !values.isEmpty else { return 0 }
        var maximum: Float = 0
        vDSP_maxmgv(base, 1, &maximum, vDSP_Length(values.count))
        return maximum
    }

    /// Converts float32 values into IEEE float16 at `destination`, rounding to nearest even.
    private static func convertToFloat16(_ values: UnsafeBufferPointer<Float>, into destination: UnsafeMutableRawPointer) {
        guard let base = values.baseAddress, !values.isEmpty else { return }
        var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: base), height: 1,
                                   width: vImagePixelCount(values.count), rowBytes: values.count * 4)
        var output = vImage_Buffer(data: destination, height: 1, width: vImagePixelCount(values.count),
                                   rowBytes: values.count * 2)
        let status = vImageConvert_PlanarFtoPlanar16F(&source, &output, vImage_Flags(kvImageNoFlags))
        precondition(status == kvImageNoError, "float16 conversion failed: \(status)")
    }
}

extension SafetensorsDataType {
    /// The MLX dtype that holds this element type unchanged.
    public var mlxDataType: DType {
        switch self {
        case .float64: return .float64
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int64: return .int64
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        case .uint64: return .uint64
        case .uint32: return .uint32
        case .uint16: return .uint16
        case .uint8: return .uint8
        case .bool: return .bool
        }
    }
}
