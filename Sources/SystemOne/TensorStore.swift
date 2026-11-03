import Foundation

/// Named tensors a model loads from: one safetensors file, or a checkpoint
/// split across several.
public protocol TensorStore: AnyObject, Sendable {
    /// Every tensor's validated descriptor, by name.
    var descriptors: [String: SafetensorsTensorDescriptor] { get }

    /// A retained zero-copy view of a named tensor.
    func tensor(named name: String) -> SafetensorsTensorView?
}

extension SafetensorsFile: TensorStore {}

/// Errors raised while assembling a checkpoint from its files.
public enum CheckpointError: Error, Equatable, Sendable {
    case missingWeights(folder: String)
    case invalidIndex(String)
    case invalidShardName(String)
    case tensorMissingFromShard(name: String, shard: String)
    case tensorNotInIndex(name: String, shard: String)
    case duplicateTensor(name: String)
    case missingPin(shard: String)
    case unusedPin(shard: String)
}

extension CheckpointError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingWeights(let folder):
            return "\(folder) has neither model.safetensors nor model.safetensors.index.json."
        case .invalidIndex(let reason):
            return "model.safetensors.index.json is invalid: \(reason)."
        case .invalidShardName(let name):
            return "Shard name \(name) is not a plain .safetensors file name."
        case .tensorMissingFromShard(let name, let shard):
            return "The index places \(name) in \(shard), which does not contain it."
        case .tensorNotInIndex(let name, let shard):
            return "\(shard) contains \(name), which the index does not list there."
        case .duplicateTensor(let name):
            return "Tensor \(name) appears in more than one shard."
        case .missingPin(let shard):
            return "No pinned artifact is given for shard \(shard)."
        case .unusedPin(let shard):
            return "A pin names \(shard), which the checkpoint does not use."
        }
    }
}

/// A checkpoint split across safetensors shards, as Hugging Face and MLX
/// exports write it: `model.safetensors.index.json` maps every tensor to the
/// shard that holds it.
///
/// The index must agree exactly with the shards: every listed tensor is in
/// its shard, every shard tensor is listed against that shard, and no tensor
/// appears twice.
public final class SafetensorsCheckpoint: TensorStore, @unchecked Sendable {
    public static let indexFileName = "model.safetensors.index.json"
    public static let singleFileName = "model.safetensors"

    /// The open shards, by file name.
    public let shards: [String: SafetensorsFile]
    public let descriptors: [String: SafetensorsTensorDescriptor]
    private let owners: [String: SafetensorsFile]

    /// Opens a model folder's `model.safetensors.index.json` and its shards,
    /// or its single `model.safetensors` when there is no index.
    ///
    /// With `pins`, every file opened must have a pin (size and SHA-256,
    /// verified before parsing) and every pin must name an opened file.
    public convenience init(modelFolder: URL, pins: [String: PinnedArtifact]? = nil) throws {
        let fileManager = FileManager.default
        let indexURL = modelFolder.appendingPathComponent(Self.indexFileName)
        let names: [String]
        let weightMap: [String: String]?
        if fileManager.fileExists(atPath: indexURL.path) {
            let map = try Self.weightMap(fromIndex: Data(contentsOf: indexURL))
            names = Set(map.values).sorted()
            weightMap = map
        } else if fileManager.fileExists(atPath: modelFolder.appendingPathComponent(Self.singleFileName).path) {
            names = [Self.singleFileName]
            weightMap = nil
        } else {
            throw CheckpointError.missingWeights(folder: modelFolder.path)
        }
        if let pins {
            for name in names where pins[name] == nil { throw CheckpointError.missingPin(shard: name) }
            for name in pins.keys.sorted() where !names.contains(name) { throw CheckpointError.unusedPin(shard: name) }
        }
        var shards: [String: SafetensorsFile] = [:]
        for name in names {
            let url = modelFolder.appendingPathComponent(name)
            if let pin = pins?[name] {
                shards[name] = try SafetensorsFile(contentsOf: url, pinnedTo: pin)
            } else {
                shards[name] = try SafetensorsFile(contentsOf: url)
            }
        }
        try self.init(shards: shards, weightMap: weightMap)
    }

    /// Assembles open shards. A `nil` weight map requires a single shard.
    public init(shards: [String: SafetensorsFile], weightMap: [String: String]?) throws {
        var descriptors: [String: SafetensorsTensorDescriptor] = [:]
        var owners: [String: SafetensorsFile] = [:]
        for (shardName, file) in shards.sorted(by: { $0.key < $1.key }) {
            try Self.validateShardName(shardName)
            for (name, descriptor) in file.descriptors {
                if let weightMap, weightMap[name] != shardName {
                    throw CheckpointError.tensorNotInIndex(name: name, shard: shardName)
                }
                guard descriptors[name] == nil else { throw CheckpointError.duplicateTensor(name: name) }
                descriptors[name] = descriptor
                owners[name] = file
            }
        }
        if let weightMap {
            for (name, shardName) in weightMap.sorted(by: { $0.key < $1.key }) {
                guard shards[shardName] != nil else {
                    throw CheckpointError.invalidIndex("\(name) names shard \(shardName), which is not open")
                }
                guard owners[name] != nil else {
                    throw CheckpointError.tensorMissingFromShard(name: name, shard: shardName)
                }
            }
        } else if shards.count != 1 {
            throw CheckpointError.invalidIndex("several shards need a weight map")
        }
        self.shards = shards
        self.descriptors = descriptors
        self.owners = owners
    }

    public func tensor(named name: String) -> SafetensorsTensorView? {
        owners[name]?.tensor(named: name)
    }

    /// Parses an index file's `weight_map`, validating every shard name.
    static func weightMap(fromIndex data: Data) throws -> [String: String] {
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw CheckpointError.invalidIndex("not JSON")
        }
        guard let object = root as? [String: Any], let map = object["weight_map"] as? [String: Any] else {
            throw CheckpointError.invalidIndex("missing weight_map object")
        }
        var weightMap: [String: String] = [:]
        for (name, value) in map {
            guard let shard = value as? String else {
                throw CheckpointError.invalidIndex("weight_map value for \(name) is not a string")
            }
            try validateShardName(shard)
            weightMap[name] = shard
        }
        guard !weightMap.isEmpty else { throw CheckpointError.invalidIndex("weight_map is empty") }
        return weightMap
    }

    /// A shard is a plain file in the model folder: no directories, no traversal.
    static func validateShardName(_ name: String) throws {
        guard name.hasSuffix(".safetensors"), !name.hasPrefix("."), !name.contains("/"),
              !name.contains("\\"), !name.contains("\0")
        else {
            throw CheckpointError.invalidShardName(name)
        }
    }
}
