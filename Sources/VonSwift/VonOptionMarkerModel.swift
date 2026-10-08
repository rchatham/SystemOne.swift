import Foundation
import MLX
import MLXNN

/// Errors raised while selecting option-marker representations.
public enum VonOptionMarkerModelError: Error, Equatable, Sendable {
    case emptyMaskPositions
    case invalidMaskPosition(index: Int, value: Int, sequenceLength: Int)
    case markerTokenMismatch(position: Int, tokenID: Int)
    case emptyBatch
    case maskPositionsBatchMismatch(expected: Int, actual: Int)
    case invalidInputRow(batchIndex: Int, error: VonModernBERTError)
    case emptyMaskPositionsInBatch(batchIndex: Int)
    case invalidMaskPositionInBatch(
        batchIndex: Int,
        index: Int,
        value: Int,
        sequenceLength: Int
    )
    case markerTokenMismatchInBatch(batchIndex: Int, position: Int, tokenID: Int)
}

extension VonOptionMarkerModelError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyMaskPositions:
            return "Option-marker inference requires at least one mask position."
        case .invalidMaskPosition(let index, let value, let sequenceLength):
            return "Mask position \(index) is \(value), outside a sequence of length \(sequenceLength)."
        case .markerTokenMismatch(let position, let tokenID):
            return "Token \(tokenID) at marker position \(position) is not Von's [MASK] token."
        case .emptyBatch:
            return "Option-marker batch inference requires at least one input row."
        case .maskPositionsBatchMismatch(let expected, let actual):
            return "Option-marker batch has \(actual) mask-position rows; expected \(expected)."
        case .invalidInputRow(let batchIndex, let error):
            return "Option-marker batch row \(batchIndex) is invalid: \(error.localizedDescription)"
        case .emptyMaskPositionsInBatch(let batchIndex):
            return "Option-marker batch row \(batchIndex) requires at least one mask position."
        case .invalidMaskPositionInBatch(
            let batchIndex,
            let index,
            let value,
            let sequenceLength
        ):
            return "Mask position \(index) in batch row \(batchIndex) is \(value), outside a sequence of length \(sequenceLength)."
        case .markerTokenMismatchInBatch(let batchIndex, let position, let tokenID):
            return "Token \(tokenID) at marker position \(position) in batch row \(batchIndex) is not Von's [MASK] token."
        }
    }
}

/// Batched option logits in row-major marker order.
public struct VonOptionMarkerBatchOutput {
    /// One lazy rank-one MLX array containing every row's logits.
    public let logits: MLXArray

    /// Prefix offsets into `logits`; row `i` occupies
    /// `rowOffsets[i]..<rowOffsets[i + 1]`.
    public let rowOffsets: [Int]
}

/// The float32 option scorer shipped with the production Von checkpoint.
///
/// The checkpoint tensors are copied into MLX-owned arrays during
/// initialization. Production inference intentionally omits the scorer's
/// training-only dropout.
public final class VonOptionMarkerScorer {
    public static let inputSize = 1_024
    public static let hiddenSize = 512

    private static let normEpsilon: Float = 1e-5

    private let inputNormWeight: MLXArray
    private let inputNormBias: MLXArray
    private let dense: Linear
    private let normWeight: MLXArray
    private let normBias: MLXArray
    private let outputProjection: Linear

    /// Copies all eight validated scorer tensors into MLX-owned float32 arrays.
    public init(weights: VonScorerWeights) {
        inputNormWeight = Self.copyToMLX(weights.inputNormWeight)
        inputNormBias = Self.copyToMLX(weights.inputNormBias)
        dense = Linear(
            weight: Self.copyToMLX(weights.denseWeight),
            bias: Self.copyToMLX(weights.denseBias)
        )
        normWeight = Self.copyToMLX(weights.normWeight)
        normBias = Self.copyToMLX(weights.normBias)
        outputProjection = Linear(
            weight: Self.copyToMLX(weights.outputProjectionWeight),
            bias: Self.copyToMLX(weights.outputProjectionBias)
        )
    }

    /// Scores option representations whose final dimension is 1024.
    ///
    /// The result has the same leading dimensions with the final singleton
    /// projection dimension removed.
    public func forward(_ representations: MLXArray) -> MLXArray {
        precondition(
            representations.ndim > 0 && representations.dim(-1) == Self.inputSize,
            "Von option representations must have a final dimension of 1024"
        )
        var hidden = MLXFast.layerNorm(
            representations,
            weight: inputNormWeight,
            bias: inputNormBias,
            eps: Self.normEpsilon
        )
        hidden = gelu(dense(hidden))
        hidden = MLXFast.layerNorm(
            hidden,
            weight: normWeight,
            bias: normBias,
            eps: Self.normEpsilon
        )
        return outputProjection(hidden).squeezed(axis: -1)
    }

    private static func copyToMLX(_ tensor: SafetensorsTensorView) -> MLXArray {
        tensor.withUnsafeBytes { bytes in
            let copiedBytes = Data(bytes: bytes.baseAddress!, count: bytes.count)
            return MLXArray(copiedBytes, tensor.descriptor.shape, dtype: .float32)
        }
    }
}

/// The production ModernBERT backbone composed with its option-marker scorer.
public final class VonOptionMarkerModel {
    public let backbone: VonModernBERT
    public let scorer: VonOptionMarkerScorer

    /// Copies both encoder and scorer weights out of the validated mapping.
    public init(weights: VonWeights) {
        backbone = VonModernBERT(weights: weights)
        scorer = VonOptionMarkerScorer(weights: weights.scorer)
    }

    /// Runs one packed sequence and returns one raw float32 logit per marker.
    public func forward(
        inputIDs: [Int],
        attentionMask: [Int]? = nil,
        maskPositions: [Int]
    ) throws -> MLXArray {
        try Self.validateMaskPositions(maskPositions, inputIDs: inputIDs)
        let output = try backbone.forward(
            inputIDs: inputIDs,
            attentionMask: attentionMask
        )
        let representations = output.lastHiddenState
            .take(MLXArray(maskPositions), axis: 1)
            .squeezed(axis: 0)
        return scorer.forward(representations)
    }

    /// Runs tokenized requests as one right-padded backbone/scorer batch.
    public func forward(_ inputs: [VonPackedInput]) throws -> VonOptionMarkerBatchOutput {
        try forward(
            inputIDs: inputs.map(\.inputIDs),
            attentionMask: inputs.map(\.attentionMask),
            maskPositions: inputs.map(\.maskPositions)
        )
    }

    /// Runs ragged rows as one right-padded backbone/scorer batch.
    ///
    /// Padding is internal and never changes marker positions. The returned
    /// logits remain lazy and are grouped by `rowOffsets`.
    public func forward(
        inputIDs: [[Int]],
        attentionMask: [[Int]]? = nil,
        maskPositions: [[Int]]
    ) throws -> VonOptionMarkerBatchOutput {
        let batch = try Self.prepareBatch(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            maskPositions: maskPositions
        )
        let output = try backbone.forward(
            inputIDs: batch.inputIDs,
            attentionMask: batch.attentionMask
        )
        let flattenedHidden = output.lastHiddenState.reshaped(
            batch.inputIDs.count * batch.maximumSequenceLength,
            VonModernBERT.hiddenSize
        )
        let representations = flattenedHidden.take(
            MLXArray(batch.flattenedMaskPositions),
            axis: 0
        )
        return VonOptionMarkerBatchOutput(
            logits: scorer.forward(representations),
            rowOffsets: batch.rowOffsets
        )
    }
}

extension VonOptionMarkerModel {
    struct PreparedBatch {
        let inputIDs: [[Int]]
        let attentionMask: [[Int]]
        let flattenedMaskPositions: [Int]
        let rowOffsets: [Int]
        let maximumSequenceLength: Int
    }

    static func prepareBatch(
        inputIDs: [[Int]],
        attentionMask: [[Int]]?,
        maskPositions: [[Int]]
    ) throws -> PreparedBatch {
        guard !inputIDs.isEmpty else {
            throw VonOptionMarkerModelError.emptyBatch
        }
        if let attentionMask, attentionMask.count != inputIDs.count {
            throw VonModernBERTError.attentionBatchMismatch(
                expected: inputIDs.count,
                actual: attentionMask.count
            )
        }
        guard maskPositions.count == inputIDs.count else {
            throw VonOptionMarkerModelError.maskPositionsBatchMismatch(
                expected: inputIDs.count,
                actual: maskPositions.count
            )
        }

        var validatedMasks: [[Int]] = []
        validatedMasks.reserveCapacity(inputIDs.count)
        var rowOffsets = [0]
        rowOffsets.reserveCapacity(inputIDs.count + 1)

        for batchIndex in inputIDs.indices {
            let rowMask = attentionMask.map { $0[batchIndex] }
            do {
                let validated = try VonModernBERT.validateInputs(
                    inputIDs: [inputIDs[batchIndex]],
                    attentionMask: rowMask.map { [$0] }
                )
                validatedMasks.append(validated[0])
            } catch let error as VonModernBERTError {
                throw VonOptionMarkerModelError.invalidInputRow(
                    batchIndex: batchIndex,
                    error: remap(error, toBatchIndex: batchIndex)
                )
            }

            do {
                try validateMaskPositions(
                    maskPositions[batchIndex],
                    inputIDs: inputIDs[batchIndex]
                )
            } catch VonOptionMarkerModelError.emptyMaskPositions {
                throw VonOptionMarkerModelError.emptyMaskPositionsInBatch(
                    batchIndex: batchIndex
                )
            } catch VonOptionMarkerModelError.invalidMaskPosition(
                let index,
                let value,
                let sequenceLength
            ) {
                throw VonOptionMarkerModelError.invalidMaskPositionInBatch(
                    batchIndex: batchIndex,
                    index: index,
                    value: value,
                    sequenceLength: sequenceLength
                )
            } catch VonOptionMarkerModelError.markerTokenMismatch(let position, let tokenID) {
                throw VonOptionMarkerModelError.markerTokenMismatchInBatch(
                    batchIndex: batchIndex,
                    position: position,
                    tokenID: tokenID
                )
            }
            rowOffsets.append(rowOffsets[rowOffsets.count - 1] + maskPositions[batchIndex].count)
        }

        let maximumSequenceLength = inputIDs.map(\.count).max()!
        var paddedIDs: [[Int]] = []
        var paddedMasks: [[Int]] = []
        var flattenedMaskPositions: [Int] = []
        paddedIDs.reserveCapacity(inputIDs.count)
        paddedMasks.reserveCapacity(inputIDs.count)
        flattenedMaskPositions.reserveCapacity(rowOffsets.last!)

        for batchIndex in inputIDs.indices {
            let paddingCount = maximumSequenceLength - inputIDs[batchIndex].count
            paddedIDs.append(
                inputIDs[batchIndex]
                    + [Int](repeating: VonTokenizer.paddingTokenID, count: paddingCount)
            )
            paddedMasks.append(
                validatedMasks[batchIndex] + [Int](repeating: 0, count: paddingCount)
            )
            flattenedMaskPositions.append(
                contentsOf: maskPositions[batchIndex].map {
                    batchIndex * maximumSequenceLength + $0
                }
            )
        }

        return PreparedBatch(
            inputIDs: paddedIDs,
            attentionMask: paddedMasks,
            flattenedMaskPositions: flattenedMaskPositions,
            rowOffsets: rowOffsets,
            maximumSequenceLength: maximumSequenceLength
        )
    }

    static func remap(
        _ error: VonModernBERTError,
        toBatchIndex batchIndex: Int
    ) -> VonModernBERTError {
        switch error {
        case .invalidTokenID(_, let sequenceIndex, let value):
            return .invalidTokenID(
                batchIndex: batchIndex,
                sequenceIndex: sequenceIndex,
                value: value
            )
        case .attentionSequenceMismatch(_, let expected, let actual):
            return .attentionSequenceMismatch(
                batchIndex: batchIndex,
                expected: expected,
                actual: actual
            )
        case .invalidAttentionMaskValue(_, let sequenceIndex, let value):
            return .invalidAttentionMaskValue(
                batchIndex: batchIndex,
                sequenceIndex: sequenceIndex,
                value: value
            )
        case .noUnmaskedTokens:
            return .noUnmaskedTokens(batchIndex: batchIndex)
        default:
            return error
        }
    }

    static func validateMaskPositions(_ positions: [Int], inputIDs: [Int]) throws {
        guard !positions.isEmpty else {
            throw VonOptionMarkerModelError.emptyMaskPositions
        }
        for (index, position) in positions.enumerated() {
            guard inputIDs.indices.contains(position) else {
                throw VonOptionMarkerModelError.invalidMaskPosition(
                    index: index,
                    value: position,
                    sequenceLength: inputIDs.count
                )
            }
            let tokenID = inputIDs[position]
            guard tokenID == VonTokenizer.maskTokenID else {
                throw VonOptionMarkerModelError.markerTokenMismatch(
                    position: position,
                    tokenID: tokenID
                )
            }
        }
    }
}
