import Foundation
import Hub
import Tokenizers

/// Text ↔ token ids for `bare-v1` prompts. The prompt is tokenised as one
/// string without special tokens, as the reference does.
public protocol JevTextCoding: Sendable {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
}

/// The Qwen3.5 byte-level BPE tokenizer shipped with JEV checkpoints, loaded
/// only from an explicit local folder.
public struct JevTokenizer: JevTextCoding {
    private let tokenizer: any Tokenizer

    public init(modelFolder: URL) async throws {
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            guard FileManager.default.fileExists(atPath: modelFolder.appendingPathComponent(name).path) else {
                throw JevConfigurationError.missingFile(name)
            }
        }
        tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }
}
