import Foundation

/// A checkpoint split across several safetensors files by a Hugging Face
/// `model.safetensors.index.json`.
///
/// Every shard is opened with the same strict validation as a single file,
/// and each shard's own header must hold exactly the tensors the index assigns
/// it, so the index cannot point a name at an unexpected file. When pins are
/// given, every shard the index names must be pinned, and nothing else.
public final class ShardedSafetensors: @unchecked Sendable {
    /// The opened shards, by file name.
    public let shards: [String: SafetensorsFile]
    /// The file name holding each tensor.
    public let weightMap: [String: String]
    /// Every tensor's descriptor across all shards.
    public let descriptors: [String: SafetensorsTensorDescriptor]

    /// Opens the index at `indexURL` and every shard it names, from the same
    /// folder. Pass `pins` keyed by shard file name to verify each shard byte
    /// for byte before parsing it.
    public init(indexURL: URL, pins: [String: PinnedArtifact]? = nil) throws {
        let weightMap = try Self.weightMap(at: indexURL)
        let shardNames = Set(weightMap.values)
        if let pins {
            let missing = shardNames.subtracting(pins.keys).sorted()
            let unexpected = Set(pins.keys).subtracting(shardNames).sorted()
            guard missing.isEmpty, unexpected.isEmpty else {
                throw WeightsError.shardPinsMismatch(missing: missing, unexpected: unexpected)
            }
        }

        let folder = indexURL.deletingLastPathComponent()
        var shards: [String: SafetensorsFile] = [:]
        var descriptors: [String: SafetensorsTensorDescriptor] = [:]
        for name in shardNames.sorted() {
            let url = folder.appendingPathComponent(name)
            let file = try pins.map { try SafetensorsFile(contentsOf: url, pinnedTo: $0[name]!) }
                ?? SafetensorsFile(contentsOf: url)
            let assigned = Set(weightMap.filter { $0.value == name }.keys)
            let held = Set(file.descriptors.keys)
            guard assigned == held else {
                throw WeightsError.shardContentsMismatch(
                    shard: name,
                    missing: assigned.subtracting(held).sorted(),
                    unexpected: held.subtracting(assigned).sorted()
                )
            }
            descriptors.merge(file.descriptors) { first, _ in first }
            shards[name] = file
        }
        self.shards = shards
        self.weightMap = weightMap
        self.descriptors = descriptors
    }

    /// Requires exactly the schema's tensor names across all shards, with
    /// matching dtypes and shapes.
    public func validate(schema: [WeightSpecification]) throws {
        try SafetensorsFile.validate(descriptors, schema: schema)
    }

    /// Returns a named tensor from whichever shard holds it, or reports it missing.
    public func requiredTensor(_ name: String) throws -> SafetensorsTensorView {
        guard let shard = weightMap[name], let view = shards[shard]?.tensor(named: name) else {
            throw WeightsError.tensorSetMismatch(missing: [name], unexpected: [])
        }
        return view
    }

    /// The index's `weight_map`, with shard names restricted to plain file
    /// names in the index's own folder.
    private static func weightMap(at url: URL) throws -> [String: String] {
        let data = try Data(contentsOf: url)
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WeightsError.invalidShardIndex("root must be a JSON object")
        }
        guard let entries = root["weight_map"] as? [String: Any], !entries.isEmpty else {
            throw WeightsError.invalidShardIndex("weight_map must be a non-empty object")
        }
        var map: [String: String] = [:]
        for (tensor, value) in entries {
            guard let file = value as? String else {
                throw WeightsError.invalidShardIndex("shard for \(tensor) must be a string")
            }
            guard !file.isEmpty, !file.contains("/"), file != ".", file != ".." else {
                throw WeightsError.invalidShardIndex("shard \(file) must be a file name in the index's folder")
            }
            map[tensor] = file
        }
        return map
    }
}
