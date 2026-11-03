import Foundation
import Hub
import Tokenizers

/// The tokenizer an encoder decision model reads its prompt with.
public protocol EncoderTokenizer: Sendable {
    /// Ids of `text`, with no special tokens added.
    func encode(_ text: String) -> [Int]
    /// The id of one token, such as `[MASK]`.
    func id(of token: String) -> Int?
}

public enum EncoderTokenizerError: Error, Equatable, Sendable {
    case missingFile(String)
    case invalidJSON(String)
    case missingToken(String)
}

extension EncoderTokenizerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            return "The tokenizer folder is missing \(path)."
        case .invalidJSON(let path):
            return "\(path) is not a JSON object."
        case .missingToken(let token):
            return "The tokenizer has no id for \(token)."
        }
    }
}

/// A local tokenizer folder (`tokenizer.json` and `tokenizer_config.json`),
/// loaded with swift-transformers. Nothing is downloaded.
public struct HubEncoderTokenizer: EncoderTokenizer {
    private let tokenizer: any Tokenizer
    /// Special token strings by `tokenizer_config.json` key (`mask_token`, `cls_token`, ...).
    public let specialTokens: [String: String]

    public init(folder: URL) throws {
        func object(_ name: String) throws -> [NSString: Any] {
            let url = folder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw EncoderTokenizerError.missingFile(url.path)
            }
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [NSString: Any] else {
                throw EncoderTokenizerError.invalidJSON(url.path)
            }
            return object
        }
        var configuration = try object("tokenizer_config.json")
        var special: [String: String] = [:]
        for key in ["mask_token", "cls_token", "sep_token", "pad_token", "unk_token", "bos_token", "eos_token"] {
            switch configuration[key as NSString] {
            case let token as String: special[key] = token
            case let token as [String: Any]: if let content = token["content"] as? String { special[key] = content }
            default: break
            }
        }
        // Transformers 5 names fast tokenizers `TokenizersBackend`; swift-transformers
        // knows the same class as `PreTrainedTokenizerFast`.
        if (configuration["tokenizer_class"] as? String) == "TokenizersBackend" {
            configuration["tokenizer_class"] = "PreTrainedTokenizerFast"
        }
        tokenizer = try AutoTokenizer.from(
            tokenizerConfig: Config(configuration),
            tokenizerData: Config(try object("tokenizer.json"))
        )
        specialTokens = special
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func id(of token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}

/// The ids and strings of the special tokens encoder prompts are built from.
public struct EncoderSpecialTokens: Equatable, Sendable {
    public let cls: Int
    public let sep: Int
    public let mask: Int
    public let pad: Int
    /// The mask token's text, which callers' text must not contain.
    public let maskText: String

    public init(cls: Int, sep: Int, mask: Int, pad: Int, maskText: String) {
        self.cls = cls
        self.sep = sep
        self.mask = mask
        self.pad = pad
        self.maskText = maskText
    }

    /// Resolves `[CLS]`-style tokens by name, preferring the folder's own names.
    public init(tokenizer: EncoderTokenizer, names: [String: String] = [:]) throws {
        func id(_ key: String, _ fallback: String) throws -> (Int, String) {
            let token = names[key] ?? fallback
            guard let id = tokenizer.id(of: token) else { throw EncoderTokenizerError.missingToken(token) }
            return (id, token)
        }
        let mask = try id("mask_token", "[MASK]")
        self.init(
            cls: try id("cls_token", "[CLS]").0, sep: try id("sep_token", "[SEP]").0,
            mask: mask.0, pad: try id("pad_token", "[PAD]").0, maskText: mask.1
        )
    }
}
