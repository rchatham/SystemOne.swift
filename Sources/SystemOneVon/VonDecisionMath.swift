import Foundation
import SystemOne

/// Errors raised by the pure decision and calibration layer.
public enum VonDecisionMathError: Error, Equatable, Sendable {
    case emptyLogits
    case nonFiniteLogit(index: Int, value: Float)
    case invalidStateTokenCount(Int)
    case invalidTemperature(Double)
    case noulRequiresTwoLogits(actual: Int)
}

extension VonDecisionMathError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyLogits:
            return "Decision math requires at least one logit."
        case .nonFiniteLogit(let index, let value):
            return "Logit \(index) is not finite: \(value)."
        case .invalidStateTokenCount(let count):
            return "State token count must be at least one; got \(count)."
        case .invalidTemperature(let temperature):
            return "Temperature must be finite; got \(temperature)."
        case .noulRequiresTwoLogits(let actual):
            return "Noul requires exactly two logits; got \(actual)."
        }
    }
}

/// Unrounded intermediate values from Von's pinned calibration map.
public struct VonCalibrationResult: Equatable, Sendable {
    public let inputLogits: [Float]
    public let unscaledProbabilities: [Float]
    public let entropyNumerator: Float
    public let entropyNormalizer: Double
    public let normalizedEntropy: Double
    public let stateTokenCount: Int
    public let rawTemperature: Double
    public let effectiveTemperature: Double
    public let scaledLogits: [Float]
    public let probabilities: [Float]
}

/// Low-level result of Choice postprocessing.
public struct VonChoiceMathResult: Equatable, Sendable {
    public let selectedIndex: Int
    public let calibration: VonCalibrationResult
    public let roundedProbabilities: [Double]
    public let confidence: Double
}

/// Low-level result of Noul postprocessing.
public struct VonNoulMathResult: Equatable, Sendable {
    public let calibration: VonCalibrationResult
    public let debiasApplied: Bool
    public let probabilityTrue: Double
}

/// Low-level result of Score postprocessing.
public struct VonScoreMathResult: Equatable, Sendable {
    public let calibration: VonCalibrationResult
    public let roundedProbabilities: [Double]
    public let score: Double
    public let confidence: Double
}

/// Pure float32-compatible calibration and decision postprocessing.
///
/// Coefficients are embedded from `marker_calibration.json` in the pinned
/// `wfzyx/von` revision `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd`
/// (artifact SHA-256
/// `c3b5345c89be9ee802a5587dcda4e530e69a9bb5965dd839b877b7e8cfff048f`).
/// Entropy is deliberately reduced in float32, matching Torch, before the
/// fitted feature map is evaluated in double precision.
public enum VonDecisionMath {
    public static let calibrationBias = 0.2056
    public static let calibrationEntropyCoefficient = -3.255
    public static let calibrationLogTokensCoefficient = 20.2391
    public static let calibrationOptionCountCoefficient = -5.2263
    public static let calibrationLowerBound = 0.3
    public static let calibrationUpperBound = 12.0
    public static let defaultNoulDebiasFactor: Float = 0.7

    /// Applies the pinned input-conditioned temperature map and calibrated softmax.
    public static func calibrate(
        logits: [Float],
        stateTokenCount: Int,
        temperatureOverride: Double? = nil
    ) throws -> VonCalibrationResult {
        try validate(logits)
        guard stateTokenCount >= 1 else {
            throw VonDecisionMathError.invalidStateTokenCount(stateTokenCount)
        }
        if let temperatureOverride, !temperatureOverride.isFinite {
            throw VonDecisionMathError.invalidTemperature(temperatureOverride)
        }

        let unscaledProbabilities = softmaxValidated(logits)
        let optionCount = logits.count
        let entropyNumerator: Float
        let entropyNormalizer: Double
        let normalizedEntropy: Double
        if optionCount > 1 {
            var entropySum: Float = 0
            for probability in unscaledProbabilities {
                entropySum += probability * log(max(probability, Float(1e-12)))
            }
            entropyNumerator = -entropySum
            entropyNormalizer = log(Double(optionCount))
            normalizedEntropy = Double(entropyNumerator) / entropyNormalizer
        } else {
            entropyNumerator = 0
            entropyNormalizer = 1
            normalizedEntropy = 0
        }

        // Preserve Python's feature insertion and summation order. The Torch
        // entropy scalar has already crossed into double precision here.
        var rawTemperature = 0.0
        rawTemperature += calibrationBias
        rawTemperature += calibrationEntropyCoefficient * normalizedEntropy
        rawTemperature += calibrationLogTokensCoefficient
            * (log10(Double(stateTokenCount)) / 4.0)
        rawTemperature += calibrationOptionCountCoefficient
            * (Double(optionCount) / 8.0)

        let mappedTemperature = min(
            calibrationUpperBound,
            max(calibrationLowerBound, rawTemperature)
        )
        let effectiveTemperature = temperatureOverride ?? mappedTemperature
        let denominator = Float(max(effectiveTemperature, 1e-4))
        let scaledLogits = logits.map { $0 / denominator }
        let probabilities = softmaxValidated(scaledLogits)

        return VonCalibrationResult(
            inputLogits: logits,
            unscaledProbabilities: unscaledProbabilities,
            entropyNumerator: entropyNumerator,
            entropyNormalizer: entropyNormalizer,
            normalizedEntropy: normalizedEntropy,
            stateTokenCount: stateTokenCount,
            rawTemperature: rawTemperature,
            effectiveTemperature: effectiveTemperature,
            scaledLogits: scaledLogits,
            probabilities: probabilities
        )
    }

    /// Returns Torch-style stable float32 softmax probabilities.
    public static func softmax(_ logits: [Float]) throws -> [Float] {
        try validate(logits)
        return softmaxValidated(logits)
    }

    /// Applies Choice calibration, first-maximum selection, and production rounding.
    public static func choice(
        logits: [Float],
        stateTokenCount: Int,
        temperatureOverride: Double? = nil
    ) throws -> VonChoiceMathResult {
        let calibration = try calibrate(
            logits: logits,
            stateTokenCount: stateTokenCount,
            temperatureOverride: temperatureOverride
        )
        let selectedIndex = firstMaximumIndex(logits)
        let probabilities = calibration.probabilities.map(Double.init)
        return VonChoiceMathResult(
            selectedIndex: selectedIndex,
            calibration: calibration,
            roundedProbabilities: probabilities.map { rounded($0, places: 4) },
            confidence: rounded(confidence(probabilities), places: 3)
        )
    }

    /// Applies optional default-Noul null-state debiasing and calibration.
    ///
    /// Supply `nullLogits` only when neither criterion was explicit. The 0.7
    /// bias multiplication and subtraction are performed in float32, as in
    /// production Torch.
    public static func noul(
        logits: [Float],
        nullLogits: [Float]? = nil,
        stateTokenCount: Int,
        temperatureOverride: Double? = nil
    ) throws -> VonNoulMathResult {
        guard logits.count == 2 else {
            throw VonDecisionMathError.noulRequiresTwoLogits(actual: logits.count)
        }
        try validate(logits)

        let inputLogits: [Float]
        if let nullLogits {
            guard nullLogits.count == 2 else {
                throw VonDecisionMathError.noulRequiresTwoLogits(actual: nullLogits.count)
            }
            try validate(nullLogits)
            let nullBias = nullLogits[0] - nullLogits[1]
            let adjustment = defaultNoulDebiasFactor * nullBias
            inputLogits = [logits[0] - adjustment, logits[1]]
        } else {
            inputLogits = logits
        }

        let calibration = try calibrate(
            logits: inputLogits,
            stateTokenCount: stateTokenCount,
            temperatureOverride: temperatureOverride
        )
        let clamped = min(1.0, max(0.0, Double(calibration.probabilities[0])))
        return VonNoulMathResult(
            calibration: calibration,
            debiasApplied: nullLogits != nil,
            probabilityTrue: rounded(clamped, places: 4)
        )
    }

    /// Applies Score calibration, expected-value scoring, and production rounding.
    public static func score(
        logits: [Float],
        stateTokenCount: Int,
        temperatureOverride: Double? = nil
    ) throws -> VonScoreMathResult {
        let calibration = try calibrate(
            logits: logits,
            stateTokenCount: stateTokenCount,
            temperatureOverride: temperatureOverride
        )
        let probabilities = calibration.probabilities.map(Double.init)
        var expectedValue = 0.0
        for (index, probability) in probabilities.enumerated() {
            expectedValue += Double(index) * probability
        }
        return VonScoreMathResult(
            calibration: calibration,
            roundedProbabilities: probabilities.map { rounded($0, places: 4) },
            score: rounded(expectedValue, places: 2),
            confidence: rounded(confidence(probabilities), places: 3)
        )
    }
}

extension VonDecisionMath {
    static func firstMaximumIndex(_ values: [Float]) -> Int {
        var bestIndex = 0
        for index in values.indices.dropFirst() where values[index] > values[bestIndex] {
            bestIndex = index
        }
        return bestIndex
    }

    static func rounded(_ value: Double, places: Int) -> Double {
        precondition(places >= 0 && places <= 9)
        let factor = pow(10.0, Double(places))
        return (value * factor).rounded(.toNearestOrEven) / factor
    }

    private static func validate(_ logits: [Float]) throws {
        guard !logits.isEmpty else { throw VonDecisionMathError.emptyLogits }
        for (index, value) in logits.enumerated() where !value.isFinite {
            throw VonDecisionMathError.nonFiniteLogit(index: index, value: value)
        }
    }

    private static func softmaxValidated(_ logits: [Float]) -> [Float] {
        let maximum = logits.max()!
        let exponentials = logits.map { exp($0 - maximum) }
        var total: Float = 0
        for exponential in exponentials {
            total += exponential
        }
        // Torch's CPU softmax computes one float32 reciprocal and multiplies
        // every lane. Dividing each lane separately differs by an ULP near
        // some serialization-rounding boundaries.
        let reciprocal: Float = 1 / total
        return exponentials.map { $0 * reciprocal }
    }

    private static func confidence(_ probabilities: [Double]) -> Double {
        let sorted = probabilities.sorted(by: >)
        let runnerUp = sorted.count > 1 ? sorted[1] : 0
        return min(1.0, max(0.0, sorted[0] - runnerUp))
    }
}
