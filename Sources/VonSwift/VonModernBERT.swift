import Foundation
import MLX
import MLXNN

/// Errors raised before a ModernBERT forward is constructed.
public enum VonModernBERTError: Error, Equatable, Sendable {
    case emptyBatch
    case emptySequence
    case raggedInput(batchIndex: Int, expected: Int, actual: Int)
    case sequenceTooLong(maximum: Int, actual: Int)
    case invalidTokenID(batchIndex: Int, sequenceIndex: Int, value: Int)
    case attentionBatchMismatch(expected: Int, actual: Int)
    case attentionSequenceMismatch(batchIndex: Int, expected: Int, actual: Int)
    case invalidAttentionMaskValue(batchIndex: Int, sequenceIndex: Int, value: Int)
    case noUnmaskedTokens(batchIndex: Int)
}

extension VonModernBERTError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyBatch:
            return "ModernBERT input must contain at least one batch row."
        case .emptySequence:
            return "ModernBERT input sequences must contain at least one token."
        case .raggedInput(let batchIndex, let expected, let actual):
            return "ModernBERT input row \(batchIndex) has length \(actual); expected \(expected)."
        case .sequenceTooLong(let maximum, let actual):
            return "ModernBERT input has \(actual) tokens; the maximum is \(maximum)."
        case .invalidTokenID(let batchIndex, let sequenceIndex, let value):
            return "ModernBERT token [\(batchIndex), \(sequenceIndex)] is \(value), outside the vocabulary."
        case .attentionBatchMismatch(let expected, let actual):
            return "ModernBERT attention mask has \(actual) rows; expected \(expected)."
        case .attentionSequenceMismatch(let batchIndex, let expected, let actual):
            return "ModernBERT attention mask row \(batchIndex) has length \(actual); expected \(expected)."
        case .invalidAttentionMaskValue(let batchIndex, let sequenceIndex, let value):
            return "ModernBERT attention mask [\(batchIndex), \(sequenceIndex)] is \(value); expected 0 or 1."
        case .noUnmaskedTokens(let batchIndex):
            return "ModernBERT attention mask row \(batchIndex) has no unmasked keys."
        }
    }
}

/// One ModernBERT backbone forward.
public struct VonModernBERTOutput {
    /// Final, normalized hidden states with shape `[batch, sequence, 1024]`.
    public let lastHiddenState: MLXArray

    /// Optional parity boundaries: embeddings, layers 0...27, then final norm.
    /// This is empty unless `captureHiddenStates` was requested.
    public let hiddenStates: [MLXArray]
}

/// The production ModernBERT-large encoder used by Von's option-marker model.
///
/// All checkpoint tensors are copied while this model is initialized. The MLX
/// arrays therefore own their storage and never refer back to the safetensors
/// mmap exposed by `SafetensorsTensorView`.
public final class VonModernBERT {
    public static let vocabularySize = 50_368
    public static let hiddenSize = 1_024
    public static let layerCount = 28
    public static let attentionHeadCount = 16
    public static let attentionHeadSize = 64
    public static let intermediateSize = 2_624
    public static let maximumSequenceLength = 8_192
    public static let slidingAttentionRadius = 64

    private static let normEpsilon: Float = 1e-5
    private static let attentionScale: Float = 0.125
    private static let fullAttentionRoPEBase: Float = 160_000
    private static let slidingAttentionRoPEBase: Float = 10_000

    private let tokenEmbedding: Embedding
    private let embeddingNorm: MLXArray
    private let layers: [EncoderLayer]
    private let finalNorm: MLXArray

    /// Copies the encoder portion of a validated Von checkpoint into MLX-owned arrays.
    public convenience init(weights: VonWeights) {
        self.init(encoderWeights: weights.encoder)
    }

    /// Copies validated encoder tensors into MLX-owned arrays.
    public init(encoderWeights: VonEncoderWeights) {
        tokenEmbedding = Embedding(
            weight: Self.copyToMLX(encoderWeights.embeddings.tokenEmbedding)
        )
        embeddingNorm = Self.copyToMLX(encoderWeights.embeddings.norm)
        layers = encoderWeights.layers.map { weights in
            EncoderLayer(
                attentionNorm: weights.attentionNorm.map(Self.copyToMLX),
                queryKeyValue: Linear(weight: Self.copyToMLX(weights.attention.queryKeyValue)),
                attentionOutput: Linear(weight: Self.copyToMLX(weights.attention.output)),
                mlpNorm: Self.copyToMLX(weights.mlp.norm),
                mlpInput: Linear(weight: Self.copyToMLX(weights.mlp.input)),
                mlpOutput: Linear(weight: Self.copyToMLX(weights.mlp.output))
            )
        }
        finalNorm = Self.copyToMLX(encoderWeights.finalNorm)
    }

    /// Runs one unpadded sequence and returns a batched `[1, length, 1024]` result.
    public func forward(
        inputIDs: [Int],
        attentionMask: [Int]? = nil,
        captureHiddenStates: Bool = false
    ) throws -> VonModernBERTOutput {
        try forward(
            inputIDs: [inputIDs],
            attentionMask: attentionMask.map { [$0] },
            captureHiddenStates: captureHiddenStates
        )
    }

    /// Runs a rectangular batch through the exact 28-layer production encoder.
    public func forward(
        inputIDs: [[Int]],
        attentionMask: [[Int]]? = nil,
        captureHiddenStates: Bool = false
    ) throws -> VonModernBERTOutput {
        let validatedMask = try Self.validateInputs(
            inputIDs: inputIDs,
            attentionMask: attentionMask
        )
        let batchSize = inputIDs.count
        let sequenceLength = inputIDs[0].count
        let ids = MLXArray(inputIDs.flatMap { $0 }, [batchSize, sequenceLength])

        let fullMask = Self.makeAdditiveAttentionMask(
            validatedMask,
            type: .full
        )
        let slidingMask = Self.makeAdditiveAttentionMask(
            validatedMask,
            type: .sliding
        )

        var hidden = MLXFast.layerNorm(
            tokenEmbedding(ids),
            weight: embeddingNorm,
            eps: Self.normEpsilon
        )
        var captured: [MLXArray] = captureHiddenStates ? [hidden] : []
        captured.reserveCapacity(captureHiddenStates ? Self.layerCount + 2 : 0)

        for (index, layer) in layers.enumerated() {
            let attentionInput: MLXArray
            if let attentionNorm = layer.attentionNorm {
                attentionInput = MLXFast.layerNorm(
                    hidden,
                    weight: attentionNorm,
                    eps: Self.normEpsilon
                )
            } else {
                // ModernBERT layer zero intentionally has no attention norm.
                attentionInput = hidden
            }

            let qkv = layer.queryKeyValue(attentionInput)
            var (queries, keys, values) = Self.reshapeQueryKeyValue(qkv)
            let isFullAttention = Self.isFullAttention(layerIndex: index)
            let ropeBase = isFullAttention
                ? Self.fullAttentionRoPEBase
                : Self.slidingAttentionRoPEBase
            queries = MLXFast.RoPE(
                queries,
                dimensions: Self.attentionHeadSize,
                traditional: false,
                base: ropeBase,
                scale: 1,
                offset: 0
            )
            keys = MLXFast.RoPE(
                keys,
                dimensions: Self.attentionHeadSize,
                traditional: false,
                base: ropeBase,
                scale: 1,
                offset: 0
            )

            var attentionOutput = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: Self.attentionScale,
                mask: isFullAttention ? fullMask : slidingMask
            )
            attentionOutput = attentionOutput
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, Self.hiddenSize)
            hidden = hidden + layer.attentionOutput(attentionOutput)

            let normalizedMLPInput = MLXFast.layerNorm(
                hidden,
                weight: layer.mlpNorm,
                eps: Self.normEpsilon
            )
            let projectedMLPInput = layer.mlpInput(normalizedMLPInput)
            let (input, gate) = projectedMLPInput.split(axis: -1)
            hidden = hidden + layer.mlpOutput(gelu(input) * gate)

            if captureHiddenStates {
                captured.append(hidden)
            }
        }

        hidden = MLXFast.layerNorm(hidden, weight: finalNorm, eps: Self.normEpsilon)
        if captureHiddenStates {
            captured.append(hidden)
        }
        return VonModernBERTOutput(lastHiddenState: hidden, hiddenStates: captured)
    }
}

extension VonModernBERT {
    enum AttentionType {
        case full
        case sliding
    }

    static func isFullAttention(layerIndex: Int) -> Bool {
        layerIndex % 3 == 0
    }

    static func queryKeyValueShape(batchSize: Int, sequenceLength: Int) -> [Int] {
        [batchSize, sequenceLength, 3, attentionHeadCount, attentionHeadSize]
    }

    static func attentionProjectionShape(batchSize: Int, sequenceLength: Int) -> [Int] {
        [batchSize, attentionHeadCount, sequenceLength, attentionHeadSize]
    }

    /// Reshapes `[B, L, 3072]` through `[B, L, 3, 16, 64]` and returns
    /// Q/K/V in the `[B, 16, L, 64]` layout consumed by scaled SDPA.
    static func reshapeQueryKeyValue(
        _ queryKeyValue: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        precondition(
            queryKeyValue.ndim == 3 && queryKeyValue.dim(-1) == 3 * hiddenSize,
            "ModernBERT QKV projection must have shape [B, L, 3072]"
        )
        let batchSize = queryKeyValue.dim(0)
        let sequenceLength = queryKeyValue.dim(1)
        let pieces = queryKeyValue
            .reshaped(queryKeyValueShape(batchSize: batchSize, sequenceLength: sequenceLength))
            .split(parts: 3, axis: 2)
        precondition(pieces.count == 3)

        func attentionLayout(_ piece: MLXArray) -> MLXArray {
            piece.squeezed(axis: 2).transposed(0, 2, 1, 3)
        }
        return (
            attentionLayout(pieces[0]),
            attentionLayout(pieces[1]),
            attentionLayout(pieces[2])
        )
    }

    /// Builds a float32 additive mask. Padding masks keys only; sliding
    /// attention additionally permits exactly `abs(query-key) <= 64`.
    static func makeAdditiveAttentionMask(
        _ attentionMask: [[Int]],
        type: AttentionType
    ) -> MLXArray {
        let mask = additiveAttentionMaskValues(attentionMask, type: type)
        return MLXArray(mask.values, mask.shape)
    }

    /// Pure Swift mask construction used by offline tests. The runtime wrapper
    /// above only transfers this already-validated float32 mask into MLX.
    static func additiveAttentionMaskValues(
        _ attentionMask: [[Int]],
        type: AttentionType
    ) -> (values: [Float], shape: [Int]) {
        let batchSize = attentionMask.count
        let sequenceLength = attentionMask[0].count
        let blocked = -Float.greatestFiniteMagnitude
        var values: [Float] = []

        switch type {
        case .full:
            values.reserveCapacity(batchSize * sequenceLength)
            for row in attentionMask {
                values.append(contentsOf: row.map { $0 == 1 ? 0 : blocked })
            }
            return (values, [batchSize, 1, 1, sequenceLength])

        case .sliding:
            values.reserveCapacity(batchSize * sequenceLength * sequenceLength)
            for row in attentionMask {
                for query in 0..<sequenceLength {
                    for key in 0..<sequenceLength {
                        let isVisible = row[key] == 1
                            && abs(query - key) <= slidingAttentionRadius
                        values.append(isVisible ? 0 : blocked)
                    }
                }
            }
            return (values, [batchSize, 1, sequenceLength, sequenceLength])
        }
    }

    @discardableResult
    static func validateInputs(
        inputIDs: [[Int]],
        attentionMask: [[Int]]?
    ) throws -> [[Int]] {
        guard !inputIDs.isEmpty else { throw VonModernBERTError.emptyBatch }
        let sequenceLength = inputIDs[0].count
        guard sequenceLength > 0 else { throw VonModernBERTError.emptySequence }
        guard sequenceLength <= maximumSequenceLength else {
            throw VonModernBERTError.sequenceTooLong(
                maximum: maximumSequenceLength,
                actual: sequenceLength
            )
        }

        for (batchIndex, row) in inputIDs.enumerated() {
            guard row.count == sequenceLength else {
                throw VonModernBERTError.raggedInput(
                    batchIndex: batchIndex,
                    expected: sequenceLength,
                    actual: row.count
                )
            }
            for (sequenceIndex, tokenID) in row.enumerated() {
                guard (0..<vocabularySize).contains(tokenID) else {
                    throw VonModernBERTError.invalidTokenID(
                        batchIndex: batchIndex,
                        sequenceIndex: sequenceIndex,
                        value: tokenID
                    )
                }
            }
        }

        let mask = attentionMask
            ?? [[Int]](
                repeating: [Int](repeating: 1, count: sequenceLength),
                count: inputIDs.count
            )
        guard mask.count == inputIDs.count else {
            throw VonModernBERTError.attentionBatchMismatch(
                expected: inputIDs.count,
                actual: mask.count
            )
        }
        for (batchIndex, row) in mask.enumerated() {
            guard row.count == sequenceLength else {
                throw VonModernBERTError.attentionSequenceMismatch(
                    batchIndex: batchIndex,
                    expected: sequenceLength,
                    actual: row.count
                )
            }
            for (sequenceIndex, value) in row.enumerated() where value != 0 && value != 1 {
                throw VonModernBERTError.invalidAttentionMaskValue(
                    batchIndex: batchIndex,
                    sequenceIndex: sequenceIndex,
                    value: value
                )
            }
            guard row.contains(1) else {
                throw VonModernBERTError.noUnmaskedTokens(batchIndex: batchIndex)
            }
        }
        return mask
    }

    private static func copyToMLX(_ tensor: SafetensorsTensorView) -> MLXArray {
        tensor.withUnsafeBytes { bytes in
            // Make ownership explicit rather than letting an MLX array retain a
            // pointer into the read-only safetensors mmap.
            let copiedBytes = Data(bytes: bytes.baseAddress!, count: bytes.count)
            return MLXArray(copiedBytes, tensor.descriptor.shape, dtype: .float32)
        }
    }
}

private extension VonModernBERT {
    final class EncoderLayer {
        let attentionNorm: MLXArray?
        let queryKeyValue: Linear
        let attentionOutput: Linear
        let mlpNorm: MLXArray
        let mlpInput: Linear
        let mlpOutput: Linear

        init(
            attentionNorm: MLXArray?,
            queryKeyValue: Linear,
            attentionOutput: Linear,
            mlpNorm: MLXArray,
            mlpInput: Linear,
            mlpOutput: Linear
        ) {
            self.attentionNorm = attentionNorm
            self.queryKeyValue = queryKeyValue
            self.attentionOutput = attentionOutput
            self.mlpNorm = mlpNorm
            self.mlpInput = mlpInput
            self.mlpOutput = mlpOutput
        }
    }
}
