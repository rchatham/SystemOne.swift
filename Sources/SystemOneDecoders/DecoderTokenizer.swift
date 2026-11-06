import Foundation
import Hub
import SystemOne
import Tokenizers

/// The tokenizer a decoder decision model reads its prompt with.
public protocol DecoderTokenizer: Sendable {
    /// Ids of `text` as rendered: special-token text such as `<|im_start|>`
    /// becomes its token, and nothing (no BOS) is added.
    func encode(_ text: String) -> [Int]
}

/// A tokenizer that can also turn ids back into text, for backends that cut
/// over-long states at token boundaries.
public protocol ReversibleDecoderTokenizer: DecoderTokenizer {
    /// The text of `ids`, special tokens included.
    func decode(_ ids: [Int]) -> String
}

public enum DecoderTokenizerError: Error, Equatable, Sendable {
    case missingFile(String)
    case invalidJSON(String)
}

extension DecoderTokenizerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingFile(let path): return "The tokenizer file \(path) is missing."
        case .invalidJSON(let path): return "The tokenizer file \(path) is not a JSON object."
        }
    }
}

/// A local tokenizer folder (`tokenizer.json`, `tokenizer_config.json`),
/// loaded with swift-transformers. Nothing is downloaded.
public struct HubDecoderTokenizer: ReversibleDecoderTokenizer {
    private let tokenizer: any Tokenizer

    public init(folder: URL) throws {
        func object(_ name: String) throws -> [NSString: Any] {
            let url = folder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw DecoderTokenizerError.missingFile(url.path)
            }
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [NSString: Any] else {
                throw DecoderTokenizerError.invalidJSON(url.path)
            }
            return object
        }
        var configuration = try object("tokenizer_config.json")
        // Transformers 5 names fast tokenizers `TokenizersBackend`; swift-transformers
        // knows the same class as `PreTrainedTokenizerFast`.
        if (configuration["tokenizer_class"] as? String) == "TokenizersBackend" {
            configuration["tokenizer_class"] = "PreTrainedTokenizerFast"
        }
        tokenizer = try AutoTokenizer.from(
            tokenizerConfig: Config(configuration),
            tokenizerData: Config(try object("tokenizer.json"))
        )
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }
}

/// Qwen's ChatML chat template (Qwen3.5, `chat_template.jinja`) for a system
/// and a user message with the generation prompt and thinking disabled.
public enum QwenChat {
    /// `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
    /// each message trimmed as the template's `|trim` does.
    public static func prompt(system: String?, user: String) -> String {
        var text = ""
        if let system {
            text += "<|im_start|>system\n" + PythonSemantics.pythonStrip(system) + "<|im_end|>\n"
        }
        text += "<|im_start|>user\n" + PythonSemantics.pythonStrip(user) + "<|im_end|>\n"
        return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}
