import Foundation
import MLX
import MLXNN
import SystemOne

/// Errors raised while reading or merging a LoRA adapter.
public enum LoRAAdapterError: Error, Equatable, Sendable {
    case missingFile(String)
    case unsupported(String)
    case quantizedTarget(String)
    case unusedTensors([String])
}

extension LoRAAdapterError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            return "The LoRA adapter is missing \(path)."
        case .unsupported(let reason):
            return "The LoRA adapter is not supported: \(reason)."
        case .quantizedTarget(let module):
            return "The LoRA adapter targets \(module), which is stored quantized and cannot be merged."
        case .unusedTensors(let names):
            return "The LoRA adapter has tensors no loaded linear layer uses: \(names)."
        }
    }
}

/// A peft LoRA adapter folder (`adapter_config.json`, `adapter_model.safetensors`)
/// merged into linear weights as they load, exactly as peft's
/// `merge_and_unload` does onto the stored base:
/// `W ← cast(f32(W) + (alpha / r) · B @ A)` in `W`'s stored type.
///
/// Plain LoRA only: DoRA, rsLoRA, `fan_in_fan_out`, LoRA biases and per-module
/// rank or alpha patterns are rejected. A module is targeted when its last
/// path component is one of `target_modules`, as peft matches a list.
public final class LoRAAdapter: @unchecked Sendable {
    public let rank: Int
    public let alpha: Double
    public let targetModules: Set<String>
    private let store: any TensorStore
    private var used: Set<String> = []

    /// peft's `lora_alpha / r`.
    public var scale: Float { Float(alpha / Double(rank)) }

    /// Opens an adapter folder, optionally pinning `adapter_model.safetensors`.
    public convenience init(folder: URL, pin: PinnedArtifact? = nil) throws {
        let configURL = folder.appendingPathComponent("adapter_config.json")
        let weightsURL = folder.appendingPathComponent("adapter_model.safetensors")
        for url in [configURL, weightsURL] where !FileManager.default.fileExists(atPath: url.path) {
            throw LoRAAdapterError.missingFile(url.path)
        }
        guard let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] else {
            throw LoRAAdapterError.unsupported("adapter_config.json is not a JSON object")
        }
        let store = try pin.map { try SafetensorsFile(contentsOf: weightsURL, pinnedTo: $0) }
            ?? SafetensorsFile(contentsOf: weightsURL)
        try self.init(config: config, store: store)
    }

    /// An adapter from its parsed `adapter_config.json` and its tensors.
    public init(config: [String: Any], store: any TensorStore) throws {
        func flag(_ key: String) -> Bool { (config[key] as? Bool) ?? false }
        func isEmpty(_ key: String) -> Bool { (config[key] as? [String: Any])?.isEmpty ?? (config[key] == nil || config[key] is NSNull) }
        guard (config["peft_type"] as? String) == "LORA" else { throw LoRAAdapterError.unsupported("peft_type must be LORA") }
        guard let rank = (config["r"] as? NSNumber)?.intValue, rank > 0 else {
            throw LoRAAdapterError.unsupported("r must be a positive integer")
        }
        guard let alpha = (config["lora_alpha"] as? NSNumber)?.doubleValue, alpha.isFinite else {
            throw LoRAAdapterError.unsupported("lora_alpha must be a number")
        }
        guard let targets = config["target_modules"] as? [String], !targets.isEmpty else {
            throw LoRAAdapterError.unsupported("target_modules must be a list of module names")
        }
        guard !flag("use_dora"), !flag("use_rslora"), !flag("fan_in_fan_out") else {
            throw LoRAAdapterError.unsupported("DoRA, rsLoRA and fan_in_fan_out are not supported")
        }
        guard (config["bias"] as? String ?? "none") == "none" else {
            throw LoRAAdapterError.unsupported("LoRA biases are not supported")
        }
        guard isEmpty("rank_pattern"), isEmpty("alpha_pattern") else {
            throw LoRAAdapterError.unsupported("per-module rank or alpha patterns are not supported")
        }
        self.rank = rank
        self.alpha = alpha
        targetModules = Set(targets)
        self.store = store
    }

    /// Whether the adapter changes the linear layer at `module` (a weight
    /// prefix such as `model.layers.0.mlp.gate_proj`).
    public func targets(_ module: String) -> Bool {
        targetModules.contains(String(module.split(separator: ".").last ?? ""))
    }

    /// `module.weight` from `base` with the adapter merged, in `dataType`.
    /// Untargeted modules load unchanged.
    func linear(
        _ module: String, base: any TensorStore, inputWidth: Int, outputWidth: Int, bias: Bool, dataType: DType
    ) throws -> Linear {
        guard targets(module) else {
            return try base.linear(module, inputWidth: inputWidth, bias: bias, dataType: dataType)
        }
        guard !base.isQuantized(module) else { throw LoRAAdapterError.quantizedTarget(module) }
        guard !bias else { throw LoRAAdapterError.unsupported("\(module) has a bias") }
        let weight = try base.requiredTensor("\(module).weight")
        let prefix = "base_model.model.\(module)"
        let a = try store.requiredTensor("\(prefix).lora_A.weight")
        let b = try store.requiredTensor("\(prefix).lora_B.weight")
        guard a.descriptor.shape == [rank, inputWidth] else {
            throw WeightsError.shapeMismatch(name: "\(prefix).lora_A.weight", expected: [rank, inputWidth], actual: a.descriptor.shape)
        }
        guard b.descriptor.shape == [outputWidth, rank] else {
            throw WeightsError.shapeMismatch(name: "\(prefix).lora_B.weight", expected: [outputWidth, rank], actual: b.descriptor.shape)
        }
        used.formUnion(["\(prefix).lora_A.weight", "\(prefix).lora_B.weight"])
        let stored = weight.descriptor.dataType.mlxDataType
        let merged = (weight.mlxArray(.float32) + matmul(b.mlxArray(.float32), a.mlxArray(.float32)) * scale)
            .asType(stored).asType(dataType)
        // MLX `eval` materialises the lazy array (it runs no code), one layer at
        // a time, so float32 intermediates never accumulate.
        eval(merged)
        return Linear(weight: merged)
    }

    /// Fails unless every adapter tensor was merged into some layer, so a
    /// renamed or mismatched checkpoint cannot silently drop the adapter.
    func requireAllUsed() throws {
        let unused = Set(store.descriptors.keys).subtracting(used).sorted()
        guard unused.isEmpty else { throw LoRAAdapterError.unusedTensors(unused) }
    }
}
