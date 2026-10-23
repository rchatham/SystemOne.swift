import Foundation
import Hub
import Tokenizers

/// Errors raised while loading a local d1 tokenizer snapshot.
public enum D1TokenizerError: Error, Equatable, Sendable {
    case missingFile(String)
    case invalidJSON(String)
    case missingSpecialToken(String)
}

extension D1TokenizerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "The local d1 model folder is missing \(name)."
        case .invalidJSON(let name):
            return "\(name) is not a JSON object."
        case .missingSpecialToken(let token):
            return "The d1 tokenizer has no id for \(token)."
        }
    }
}

/// The LFM2 byte-level BPE tokenizer shipped with d1 checkpoints, loaded only
/// from an explicit local folder.
public struct D1Tokenizer: Sendable {
    /// Delimiters from the tokenizer's reserved block, as `prompt.py` names them.
    public struct SpecialTokens: Equatable, Sendable {
        public let bos: Int
        public let pad: Int
        public let mask: Int
        public let state: Int
        public let question: Int
        public let option: Int
        public let optionEnd: Int
        public let decide: Int
    }

    public let special: SpecialTokens
    private let tokenizer: any Tokenizer

    public init(modelFolder: URL) async throws {
        func object(_ name: String) throws -> [NSString: Any] {
            let url = modelFolder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw D1TokenizerError.missingFile(name)
            }
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [NSString: Any] else {
                throw D1TokenizerError.invalidJSON(name)
            }
            return object
        }
        var tokenizerConfig = try object("tokenizer_config.json")
        // Transformers 5 names this class `TokenizersBackend`; it is a plain fast
        // BPE tokenizer, which swift-transformers knows by its classic name.
        tokenizerConfig["tokenizer_class"] = "PreTrainedTokenizerFast"
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: Config(tokenizerConfig),
            tokenizerData: Config(try object("tokenizer.json"))
        )

        func id(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw D1TokenizerError.missingSpecialToken(token)
            }
            return id
        }
        special = SpecialTokens(
            bos: try id("<|startoftext|>"),
            pad: try id("<|pad|>"),
            mask: try id("<|mask|>"),
            state: try id("<|reserved_7|>"),
            question: try id("<|reserved_8|>"),
            option: try id("<|reserved_9|>"),
            optionEnd: try id("<|reserved_10|>"),
            decide: try id("<|reserved_11|>")
        )
        self.tokenizer = tokenizer
    }

    /// Token ids of caller text, without special tokens. Callers escape first.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }
}
