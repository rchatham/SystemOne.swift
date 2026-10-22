import Foundation
import SystemOne
import Tokenizers

/// Tokenized input consumed by the Von option-marker model.
public struct VonPackedInput: Equatable, Sendable {
    public let text: String
    public let inputIDs: [Int]
    public let attentionMask: [Int]
    public let maskPositions: [Int]

    public init(
        text: String,
        inputIDs: [Int],
        attentionMask: [Int],
        maskPositions: [Int]
    ) {
        self.text = text
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
        self.maskPositions = maskPositions
    }
}

/// Errors raised while loading a local Von tokenizer snapshot.
public enum VonTokenizerError: Error, Equatable, Sendable {
    case missingFile(String)
    case unexpectedSpecialToken(token: String, expectedID: Int, actualID: Int?)
}

extension VonTokenizerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "The local tokenizer folder is missing \(name)."
        case .unexpectedSpecialToken(let token, let expectedID, let actualID):
            let actual = actualID.map(String.init) ?? "nil"
            return "Tokenizer special token \(token) has id \(actual); expected \(expectedID)."
        }
    }
}

/// A local-only wrapper around the tokenizer shipped with `wfzyx/von`.
///
/// Loading requires a folder containing `tokenizer.json` and
/// `tokenizer_config.json`. This type never resolves a model name or downloads
/// tokenizer files from the Hugging Face Hub.
public struct VonTokenizer: Sendable {
    public static let classificationTokenID: Int = 50_281
    public static let separatorTokenID: Int = 50_282
    public static let paddingTokenID: Int = 50_283
    public static let maskTokenID: Int = 50_284

    private static let requiredFiles: [String] = [
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    private static let specialTokenIDs: [(token: String, id: Int)] = [
        ("[CLS]", classificationTokenID),
        ("[SEP]", separatorTokenID),
        ("[PAD]", paddingTokenID),
        ("[MASK]", maskTokenID),
    ]

    private let tokenizer: any Tokenizer

    /// Loads a tokenizer from local files without a network fallback.
    public init(modelFolder: URL) async throws {
        for file in Self.requiredFiles {
            let fileURL = modelFolder.appendingPathComponent(file, isDirectory: false)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                throw VonTokenizerError.missingFile(file)
            }
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        for specialToken in Self.specialTokenIDs {
            let actualID = tokenizer.convertTokenToId(specialToken.token)
            guard actualID == specialToken.id else {
                throw VonTokenizerError.unexpectedSpecialToken(
                    token: specialToken.token,
                    expectedID: specialToken.id,
                    actualID: actualID
                )
            }
        }
        self.tokenizer = tokenizer
    }

    /// Encodes text using the pinned Von tokenizer configuration.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    /// Returns the calibration token count used by Python von.
    ///
    /// Python clamps the special-token-free count to one, so an empty state has
    /// a count of one even though encoding it directly produces no token IDs.
    public func stateTokenCount(_ state: String) -> Int {
        max(encode(state, addSpecialTokens: false).count, 1)
    }

    /// Reproduces Python von's option-marker sequence packing.
    public static func packSequence(
        state: String,
        question: String,
        options: [String]
    ) -> String {
        let prefix: String
        if question.isEmpty {
            prefix = PythonSemantics.pythonStrip(state)
        } else {
            prefix = PythonSemantics.pythonStrip("\(question) \(state)")
        }

        let packedOptions = options
            .map { "[MASK] \(PythonSemantics.pythonStrip($0))" }
            .joined(separator: " ")
        return "\(prefix) [SEP] \(packedOptions)"
    }

    /// Packs and tokenizes one model input, including mask marker positions.
    public func packedInput(
        state: String,
        question: String,
        options: [String]
    ) -> VonPackedInput {
        let text = Self.packSequence(state: state, question: question, options: options)
        let inputIDs = encode(text)
        let maskPositions = inputIDs.indices.filter { inputIDs[$0] == Self.maskTokenID }
        return VonPackedInput(
            text: text,
            inputIDs: inputIDs,
            attentionMask: [Int](repeating: 1, count: inputIDs.count),
            maskPositions: maskPositions
        )
    }

}
