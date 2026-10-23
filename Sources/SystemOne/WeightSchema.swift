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

    /// Requires every float32 tensor to fit float16, so a half-precision load can
    /// never turn a weight into infinity. Values below 65520 round to 65504.
    public func validateFitsFloat16() throws {
        for (name, descriptor) in descriptors.sorted(by: { $0.key < $1.key }) where descriptor.dataType == .float32 {
            let maximum = tensor(named: name)!.maximumMagnitude()
            if maximum >= 65_520 {
                throw WeightsError.float16Overflow(name: name, maximumMagnitude: maximum)
            }
        }
    }

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
    /// Copies this float32 tensor into an MLX-owned array, so no MLX array ever
    /// retains a pointer into the read-only safetensors mapping.
    package func mlxArray() -> MLXArray {
        precondition(descriptor.dataType == .float32, "Only float32 tensors are copied into MLX")
        return withUnsafeBytes { bytes in
            MLXArray(Data(bytes: bytes.baseAddress!, count: bytes.count), descriptor.shape, dtype: .float32)
        }
    }

    /// Copies this float32 tensor into an MLX-owned array of `dataType`.
    ///
    /// float16 is converted on the CPU, straight into a page-aligned buffer the
    /// array takes ownership of, so Metal wraps it without a second copy and a
    /// half-precision load never holds a float32 copy in MLX memory. Call
    /// `SafetensorsFile.validateFitsFloat16()` first: out-of-range values become
    /// infinity here.
    package func mlxArray(as dataType: DType) -> MLXArray {
        switch dataType {
        case .float32:
            return mlxArray()
        case .float16:
            precondition(descriptor.dataType == .float32, "Only float32 tensors are copied into MLX")
            let byteCount = descriptor.elementCount * 2
            let page = Int(getpagesize())
            var buffer: UnsafeMutableRawPointer?
            precondition(posix_memalign(&buffer, page, max(page, (byteCount + page - 1) / page * page)) == 0,
                         "float16 buffer allocation failed")
            withUnsafeBytes { Self.convertToFloat16($0, into: buffer!) }
            return MLXArray(rawPointer: buffer!, descriptor.shape, dtype: .float16) { free(buffer) }
        default:
            preconditionFailure("Unsupported weight dtype \(dataType)")
        }
    }

    /// The largest absolute value in this float32 tensor.
    package func maximumMagnitude() -> Float {
        precondition(descriptor.dataType == .float32, "Only float32 tensors have a float magnitude")
        return withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: Float.self)
            guard let base = values.baseAddress, !values.isEmpty else { return 0 }
            var maximum: Float = 0
            vDSP_maxmgv(base, 1, &maximum, vDSP_Length(values.count))
            return maximum
        }
    }

    /// IEEE float16 bytes for float32 bytes, rounding to nearest even.
    package static func float16Bytes(_ float32: UnsafeRawBufferPointer) -> Data {
        var half = Data(count: float32.count / 2)
        half.withUnsafeMutableBytes { if let base = $0.baseAddress { convertToFloat16(float32, into: base) } }
        return half
    }

    /// Converts float32 values into IEEE float16 at `destination`, rounding to nearest even.
    static func convertToFloat16(_ float32: UnsafeRawBufferPointer, into destination: UnsafeMutableRawPointer) {
        let count = float32.count / MemoryLayout<Float>.size
        guard count > 0 else { return }
        var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: float32.baseAddress!),
                                   height: 1, width: vImagePixelCount(count), rowBytes: float32.count)
        var output = vImage_Buffer(data: destination, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        let status = vImageConvert_PlanarFtoPlanar16F(&source, &output, vImage_Flags(kvImageNoFlags))
        precondition(status == kvImageNoError, "float16 conversion failed: \(status)")
    }
}
