import Foundation
import MLX
import MLXNN

/// The group size, bit width, and mode of MLX-quantized weights.
public struct QuantizationParameters: Equatable, Sendable {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int, bits: Int, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }
}

/// Errors raised while reading MLX-quantized weights.
public enum QuantizationError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case unsupportedMode(String)
    case inconsistentShapes(name: String, reason: String)
    case parametersMismatch(name: String, configured: QuantizationParameters, stored: QuantizationParameters)
}

extension QuantizationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason):
            return "The quantization configuration is invalid: \(reason)."
        case .unsupportedMode(let mode):
            return "Quantization mode \(mode) is not supported."
        case .inconsistentShapes(let name, let reason):
            return "Quantized weight \(name) has inconsistent tensors: \(reason)."
        case .parametersMismatch(let name, let configured, let stored):
            return "Quantized weight \(name) is stored as \(stored.bits)-bit groups of \(stored.groupSize); "
                + "the configuration says \(configured.bits)-bit groups of \(configured.groupSize)."
        }
    }
}

/// The quantization an MLX export declares in `config.json`, as mlx-lm and
/// mlx-vlm write it:
///
/// ```json
/// "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
///                  "model.layers.0.mlp.down_proj": {"group_size": 64, "bits": 8},
///                  "lm_head": false}
/// ```
///
/// Module paths override the defaults; `false` keeps a module unquantized.
public struct MLXQuantizationConfiguration: Equatable, Sendable {
    public let defaults: QuantizationParameters
    /// Per-module parameters; `nil` marks a module left unquantized.
    public let overrides: [String: QuantizationParameters?]

    public init(defaults: QuantizationParameters, overrides: [String: QuantizationParameters?] = [:]) {
        self.defaults = defaults
        self.overrides = overrides
    }

    /// Parses the `quantization` object (or an MLX-format `quantization_config`).
    public init(json: [String: Any]) throws {
        defaults = try Self.parameters(json, path: "quantization")
        var overrides: [String: QuantizationParameters?] = [:]
        for (key, value) in json where !["group_size", "bits", "mode"].contains(key) {
            if let flag = value as? Bool, !flag {
                overrides[key] = .some(nil)
            } else if let object = value as? [String: Any] {
                overrides[key] = try Self.parameters(object, path: key)
            } else {
                throw QuantizationError.invalidConfiguration("\(key) must be an object or false")
            }
        }
        self.overrides = overrides
    }

    /// Reads a model folder's `config.json`; `nil` when it declares no MLX quantization.
    public static func load(modelFolder: URL) throws -> MLXQuantizationConfiguration? {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw QuantizationError.invalidConfiguration("config.json is not an object")
        }
        if let quantization = root["quantization"] as? [String: Any] {
            return try MLXQuantizationConfiguration(json: quantization)
        }
        if let legacy = root["quantization_config"] as? [String: Any], legacy["bits"] != nil,
           legacy["quant_method"] == nil {
            return try MLXQuantizationConfiguration(json: legacy)
        }
        return nil
    }

    /// The parameters for one module path, or `nil` when it stays unquantized.
    public func parameters(for modulePath: String) -> QuantizationParameters? {
        if let override = overrides[modulePath] { return override }
        return defaults
    }

    private static func parameters(_ json: [String: Any], path: String) throws -> QuantizationParameters {
        guard let groupSize = json["group_size"] as? Int, let bits = json["bits"] as? Int else {
            throw QuantizationError.invalidConfiguration("\(path) needs integer group_size and bits")
        }
        let modeName = json["mode"] as? String ?? "affine"
        guard let mode = QuantizationMode(rawValue: modeName) else {
            throw QuantizationError.unsupportedMode(modeName)
        }
        return QuantizationParameters(groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// MLX's affine quantization, in pure Swift, for checks that run without MLX.
///
/// Each row of a `[rows, columns]` matrix is cut into groups of `groupSize`
/// columns. Every value is stored as `q = round((w - bias) / scale)` in
/// `bits` bits, and the integers of a row form one little-endian bit stream
/// packed into `UInt32` words: element `j` starts at bit `j × bits`.
/// Dequantization is `q × scale + bias` with the group's scale and bias.
public enum AffineQuantization {
    public static let supportedBits: Set<Int> = [2, 3, 4, 5, 6, 8]
    public static let supportedGroupSizes: Set<Int> = [32, 64, 128]

    /// Packed `UInt32` words per row.
    public static func packedWidth(columns: Int, bits: Int) -> Int {
        columns * bits / 32
    }

    /// The `count` integers of one row's packed bit stream.
    public static func unpack(_ words: ArraySlice<UInt32>, bits: Int, count: Int) -> [UInt32] {
        precondition(supportedBits.contains(bits), "unsupported bit width \(bits)")
        let mask: UInt64 = (1 << UInt64(bits)) - 1
        let base = words.startIndex
        return (0..<count).map { index in
            let offset = index * bits
            let word = offset / 32
            let shift = UInt64(offset % 32)
            var value = UInt64(words[base + word]) >> shift
            if shift + UInt64(bits) > 32 {
                value |= UInt64(words[base + word + 1]) << (32 - shift)
            }
            return UInt32(value & mask)
        }
    }

    /// Dequantizes a row-major `[rows, columns]` matrix.
    public static func dequantize(
        packed: [UInt32], scales: [Float], biases: [Float],
        rows: Int, columns: Int, groupSize: Int, bits: Int
    ) -> [Float] {
        let width = packedWidth(columns: columns, bits: bits)
        let groups = columns / groupSize
        precondition(packed.count == rows * width, "packed has \(packed.count) words; expected \(rows * width)")
        precondition(scales.count == rows * groups && biases.count == rows * groups, "one scale and bias per group")
        var output: [Float] = []
        output.reserveCapacity(rows * columns)
        for row in 0..<rows {
            let integers = unpack(packed[(row * width)..<((row + 1) * width)], bits: bits, count: columns)
            for column in 0..<columns {
                let group = row * groups + column / groupSize
                output.append(Float(integers[column]) * scales[group] + biases[group])
            }
        }
        return output
    }
}

/// An MLX-quantized weight loaded from a checkpoint.
package struct QuantizedWeight {
    package let weight: MLXArray
    package let scales: MLXArray
    package let biases: MLXArray?
    package let parameters: QuantizationParameters

    /// The dense `[out, in]` weight in `dataType`.
    package func dequantized(_ dataType: DType) -> MLXArray {
        MLX.dequantized(
            weight, scales: scales, biases: biases,
            groupSize: parameters.groupSize, bits: parameters.bits, mode: parameters.mode,
            dtype: dataType
        )
    }
}

extension TensorStore {
    /// Whether MLX stored `prefix.weight` quantized: `prefix.scales` exists.
    public func isQuantized(_ prefix: String) -> Bool {
        descriptors["\(prefix).scales"] != nil
    }

    /// The group size and bits of a quantized `prefix.weight`, read from its
    /// packed and scale shapes given the unpacked input width, as mlx-lm does.
    /// `configured` is checked against them and supplies a non-affine mode.
    public func quantizationParameters(
        of prefix: String, inputWidth: Int, configured: QuantizationParameters? = nil
    ) throws -> QuantizationParameters {
        let name = "\(prefix).weight"
        let weight = try requiredTensor(name).descriptor
        let scales = try requiredTensor("\(prefix).scales").descriptor
        let biases = descriptors["\(prefix).biases"]
        func inconsistent(_ reason: String) -> QuantizationError {
            .inconsistentShapes(name: name, reason: reason)
        }
        guard weight.dataType == .uint32 else { throw inconsistent("packed weight is \(weight.dataType.rawValue), not U32") }
        guard let packed = weight.shape.last, let groups = scales.shape.last, inputWidth > 0, groups > 0,
              Array(weight.shape.dropLast()) == Array(scales.shape.dropLast())
        else {
            throw inconsistent("weight \(weight.shape) and scales \(scales.shape) do not share leading dimensions")
        }
        guard (packed * 32) % inputWidth == 0, inputWidth % groups == 0 else {
            throw inconsistent("\(packed) packed words and \(groups) groups do not divide \(inputWidth) inputs")
        }
        let bits = packed * 32 / inputWidth
        let groupSize = inputWidth / groups
        let mode: QuantizationMode
        if let biases {
            guard biases.shape == scales.shape else { throw inconsistent("biases \(biases.shape) differ from scales") }
            mode = .affine
        } else {
            guard let configured, configured.mode != .affine else {
                throw inconsistent("affine quantization needs biases")
            }
            mode = configured.mode
        }
        let stored = QuantizationParameters(groupSize: groupSize, bits: bits, mode: mode)
        if mode == .affine {
            guard AffineQuantization.supportedBits.contains(bits) else { throw inconsistent("\(bits) bits per value") }
            guard AffineQuantization.supportedGroupSizes.contains(groupSize) else {
                throw inconsistent("groups of \(groupSize) values")
            }
        }
        if let configured, configured != stored {
            throw QuantizationError.parametersMismatch(name: name, configured: configured, stored: stored)
        }
        return stored
    }

    /// The packed weight, scales, and biases of `prefix`; scales and biases in `dataType`.
    package func quantizedWeight(
        _ prefix: String, inputWidth: Int, dataType: DType, configured: QuantizationParameters? = nil
    ) throws -> QuantizedWeight {
        let parameters = try quantizationParameters(of: prefix, inputWidth: inputWidth, configured: configured)
        let scales = try requiredTensor("\(prefix).scales")
        // mxfp/nvfp scales are integer exponents and keep their stored type.
        let scaleType = scales.descriptor.dataType.isFloatingPoint ? dataType : scales.descriptor.dataType.mlxDataType
        return QuantizedWeight(
            weight: try requiredTensor("\(prefix).weight").mlxArray(),
            scales: scales.mlxArray(scaleType),
            biases: tensor(named: "\(prefix).biases").map { $0.mlxArray(dataType) },
            parameters: parameters
        )
    }

    /// `prefix.weight` as a dense array in `dataType`, dequantized when MLX stored
    /// it quantized (`inputWidth` is the unpacked last dimension).
    package func denseWeight(_ prefix: String, inputWidth: Int, dataType: DType) throws -> MLXArray {
        guard isQuantized(prefix) else { return try requiredTensor("\(prefix).weight").mlxArray(dataType) }
        return try quantizedWeight(prefix, inputWidth: inputWidth, dataType: dataType).dequantized(dataType)
    }

    /// A linear layer over `prefix.weight` (`[out, in]`) and, with `bias`,
    /// `prefix.bias`: a `QuantizedLinear` when MLX stored the weight quantized.
    package func linear(
        _ prefix: String, inputWidth: Int, bias: Bool, dataType: DType,
        configured: QuantizationParameters? = nil
    ) throws -> Linear {
        let biasArray = try bias ? requiredTensor("\(prefix).bias").mlxArray(dataType) : nil
        guard isQuantized(prefix) else {
            return Linear(weight: try requiredTensor("\(prefix).weight").mlxArray(dataType), bias: biasArray)
        }
        let quantized = try quantizedWeight(prefix, inputWidth: inputWidth, dataType: dataType, configured: configured)
        return QuantizedLinear(
            weight: quantized.weight, bias: biasArray, scales: quantized.scales, biases: quantized.biases,
            groupSize: quantized.parameters.groupSize, bits: quantized.parameters.bits,
            mode: quantized.parameters.mode
        )
    }

    /// An embedding table over `prefix.weight` (`[count, width]`), quantized
    /// when MLX stored it quantized.
    package func embedding(
        _ prefix: String, width: Int, dataType: DType, configured: QuantizationParameters? = nil
    ) throws -> Embedding {
        guard isQuantized(prefix) else {
            return Embedding(weight: try requiredTensor("\(prefix).weight").mlxArray(dataType))
        }
        let quantized = try quantizedWeight(prefix, inputWidth: width, dataType: dataType, configured: configured)
        return PackedQuantizedEmbedding(quantized)
    }
}

/// An embedding table read from MLX's packed quantized storage.
///
/// `MLXNN.QuantizedEmbedding` can only quantize a dense table; this holds the
/// packed words, scales, and biases an export already contains.
package final class PackedQuantizedEmbedding: Embedding {
    package let scales: MLXArray
    package let biases: MLXArray?
    package let parameters: QuantizationParameters

    package init(_ quantized: QuantizedWeight) {
        scales = quantized.scales
        biases = quantized.biases
        parameters = quantized.parameters
        super.init(weight: quantized.weight)
    }

    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.flattened()
        let rows = dequantized(
            weight[flat], scales: scales[flat], biases: biases.map { $0[flat] },
            groupSize: parameters.groupSize, bits: parameters.bits, mode: parameters.mode
        )
        return rows.reshaped(shape + [-1])
    }

    package override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: parameters.groupSize, bits: parameters.bits, mode: parameters.mode
        )
    }
}

extension WeightSpecification {
    /// The tensors MLX stores for a quantized `[output, input]` weight under
    /// `prefix`: packed `U32` words, scales, and (affine) biases.
    public static func quantized(
        _ prefix: String, output: Int, input: Int, parameters: QuantizationParameters,
        scaleType: SafetensorsDataType
    ) -> [WeightSpecification] {
        let groups = input / parameters.groupSize
        var specifications = [
            WeightSpecification(
                name: "\(prefix).weight",
                shape: [output, AffineQuantization.packedWidth(columns: input, bits: parameters.bits)],
                dataType: .uint32
            ),
            WeightSpecification(name: "\(prefix).scales", shape: [output, groups], dataType: scaleType),
        ]
        if parameters.mode == .affine {
            specifications.append(WeightSpecification(name: "\(prefix).biases", shape: [output, groups], dataType: scaleType))
        }
        return specifications
    }
}

extension Array where Element == WeightSpecification {
    /// Rewrites a float32 schema as an MLX export stores it: floating-point
    /// tensors in `floatType`, and every 2-D `.weight` whose module `quantization`
    /// returns parameters for packed as MLX packs it.
    public func mlxExport(
        floatType: SafetensorsDataType,
        quantization: (_ modulePath: String) -> QuantizationParameters?
    ) -> [WeightSpecification] {
        flatMap { specification -> [WeightSpecification] in
            let dataType = specification.dataType.isFloatingPoint ? floatType : specification.dataType
            if specification.name.hasSuffix(".weight"), specification.shape.count == 2 {
                let prefix = String(specification.name.dropLast(".weight".count))
                if let parameters = quantization(prefix) {
                    return WeightSpecification.quantized(
                        prefix, output: specification.shape[0], input: specification.shape[1],
                        parameters: parameters, scaleType: floatType
                    )
                }
            }
            return [WeightSpecification(name: specification.name, shape: specification.shape, dataType: dataType)]
        }
    }
}
