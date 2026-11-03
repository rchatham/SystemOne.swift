import Foundation
import SystemOne

/// One tensor to write into a test safetensors file.
struct TestTensor {
    let name: String
    let dataType: SafetensorsDataType
    let shape: [Int]
    let bytes: [UInt8]

    static func float32(_ name: String, _ shape: [Int], _ values: [Float]) -> TestTensor {
        TestTensor(name: name, dataType: .float32, shape: shape, bytes: values.withUnsafeBytes { Array($0) })
    }

    static func uint32(_ name: String, _ shape: [Int], _ values: [UInt32]) -> TestTensor {
        TestTensor(name: name, dataType: .uint32, shape: shape, bytes: values.withUnsafeBytes { Array($0) })
    }

    /// Raw 16-bit patterns stored as `F16` or `BF16`.
    static func halfBits(_ name: String, _ dataType: SafetensorsDataType, _ shape: [Int], _ bits: [UInt16]) -> TestTensor {
        TestTensor(name: name, dataType: dataType, shape: shape, bytes: bits.withUnsafeBytes { Array($0) })
    }
}

/// Writes gap-free safetensors files the way `safetensors` and MLX do.
enum SafetensorsWriter {
    static func data(_ tensors: [TestTensor]) -> Data {
        var entries: [String] = []
        var payload: [UInt8] = []
        for tensor in tensors.sorted(by: { $0.name < $1.name }) {
            let start = payload.count
            payload.append(contentsOf: tensor.bytes)
            let shape = tensor.shape.map(String.init).joined(separator: ",")
            entries.append(
                #""\#(tensor.name)":{"dtype":"\#(tensor.dataType.rawValue)","shape":[\#(shape)],"data_offsets":[\#(start),\#(payload.count)]}"#
            )
        }
        let header = Array("{\(entries.joined(separator: ","))}".utf8)
        var bytes: [UInt8] = []
        let length = UInt64(header.count)
        for index in 0..<8 { bytes.append(UInt8(truncatingIfNeeded: length >> UInt64(index * 8))) }
        bytes.append(contentsOf: header)
        bytes.append(contentsOf: payload)
        return Data(bytes)
    }

    static func write(_ tensors: [TestTensor], to url: URL) throws {
        try data(tensors).write(to: url)
    }
}

extension URL {
    /// A fresh directory under the temporary folder.
    static func temporaryTestDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SystemOneTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }
}
