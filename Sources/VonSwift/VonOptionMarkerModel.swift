import Foundation
import MLX
import MLXNN

/// Errors raised while selecting option-marker representations.
public enum VonOptionMarkerModelError: Error, Equatable, Sendable {
    case emptyMaskPositions
    case invalidMaskPosition(index: Int, value: Int, sequenceLength: Int)
    case markerTokenMismatch(position: Int, tokenID: Int)
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
        }
    }
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
}

extension VonOptionMarkerModel {
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
