import Foundation
import MLX
import SystemOne

extension VonPackedInput: PlannedRow {
    public var sequenceLength: Int { inputIDs.count }
    public var logitCount: Int { maskPositions.count }
}

/// The von option-marker model as a System One backend.
///
/// Planning reproduces the pinned Python backend exactly: one packed sequence
/// per question, plus a context-free null pass for default Noul debiasing, with
/// von's input-conditioned temperature map and rounded answers.
public final class VonBackend: DecisionBackend {
    public static let modelID = "von-1.1.0"
    public static let defaultModelAlias = "von-latest"

    public var modelID: String { Self.modelID }

    private let tokenizer: VonTokenizer
    private let model: VonOptionMarkerModel?
    private let rowLogitsProvider: ((VonPackedInput) async throws -> [Float])?
    private let batchLogitsProvider: (([VonPackedInput]) async throws -> BatchLogitsOutput)?

    /// Loads the local tokenizer and verifies the exact pinned production weights.
    public convenience init(tokenizerFolder: URL, verifiedWeightsURL: URL) async throws {
        let tokenizer = try await VonTokenizer(modelFolder: tokenizerFolder)
        let weights = try VonWeights.verifiedProduction(contentsOf: verifiedWeightsURL)
        self.init(tokenizer: tokenizer, model: VonOptionMarkerModel(weights: weights))
    }

    init(tokenizer: VonTokenizer, model: VonOptionMarkerModel) {
        self.tokenizer = tokenizer
        self.model = model
        rowLogitsProvider = nil
        batchLogitsProvider = nil
    }

    /// Offline test seam: one provider call per row, in plan order.
    init(
        tokenizer: VonTokenizer,
        logitsProvider: @escaping @Sendable (VonPackedInput) async throws -> [Float]
    ) {
        self.tokenizer = tokenizer
        model = nil
        rowLogitsProvider = logitsProvider
        batchLogitsProvider = nil
    }

    /// Offline batched test seam: one provider call per microbatch.
    init(
        tokenizer: VonTokenizer,
        batchLogitsProvider: @escaping @Sendable ([VonPackedInput]) async throws -> BatchLogitsOutput
    ) {
        self.tokenizer = tokenizer
        model = nil
        rowLogitsProvider = nil
        self.batchLogitsProvider = batchLogitsProvider
    }

    public func plan(
        _ request: SystemOneRequest,
        media: SystemOneMedia?
    ) throws -> DecisionPlan<VonPackedInput> {
        guard media == nil else {
            throw SystemOneEngineError.unsupportedMedia(model: modelID)
        }
        let stateText = VonPrompt.formatState(request.state)
        let stateTokenCount = tokenizer.stateTokenCount(stateText)
        var rows: [VonPackedInput] = []
        var decisions: [PlannedDecision] = []
        var instructionCharacters = 0

        for entry in request.questions {
            let options: [String]
            let instructions: String
            let needsNullRow: Bool
            let resolve: ([[Float]]) throws -> Answer
            switch entry.question {
            case .choice(let question):
                let keys = question.criteria.map(\.key)
                options = VonPrompt.renderChoiceOptions(question.criteria)
                instructions = question.instructions
                needsNullRow = false
                resolve = { logits in
                    let result = try VonDecisionMath.choice(logits: logits[0], stateTokenCount: stateTokenCount)
                    return .choice(ChoiceAnswer(
                        choice: keys[result.selectedIndex],
                        probabilities: zip(keys, result.roundedProbabilities).map {
                            ProbabilityEntry(key: $0.0, probability: $0.1)
                        },
                        confidence: result.confidence
                    ))
                }
            case .noul(let question):
                let rendered = VonPrompt.renderNoulCriteria(question.criteria)
                options = rendered.descriptions
                instructions = question.instructions
                needsNullRow = !rendered.hasExplicitCriteria
                resolve = { logits in
                    let result = try VonDecisionMath.noul(
                        logits: logits[0],
                        nullLogits: logits.count > 1 ? logits[1] : nil,
                        stateTokenCount: stateTokenCount
                    )
                    return .noul(NoulAnswer(noul: result.probabilityTrue))
                }
            case .score(let question):
                let descriptions = VonPrompt.renderScoreLevels(question.criteria)
                let keys = descriptions.indices.map(String.init)
                options = descriptions
                instructions = question.instructions
                needsNullRow = false
                resolve = { logits in
                    let result = try VonDecisionMath.score(logits: logits[0], stateTokenCount: stateTokenCount)
                    return .score(ScoreAnswer(
                        score: result.score,
                        confidence: result.confidence,
                        legend: zip(keys, descriptions).map { LegendEntry(key: $0.0, description: $0.1) },
                        probabilities: zip(keys, result.roundedProbabilities).map {
                            ProbabilityEntry(key: $0.0, probability: $0.1)
                        }
                    ))
                }
            }

            var rowIndices = [rows.count]
            rows.append(try packedRow(state: stateText, question: instructions, options: options))
            if needsNullRow {
                rowIndices.append(rows.count)
                rows.append(try packedRow(state: "", question: instructions, options: options))
            }
            decisions.append(PlannedDecision(id: entry.id, rowIndices: rowIndices, resolve: resolve))
            instructionCharacters += PythonSemantics.characterCount(instructions)
        }

        // Python von reports an approximate character-based count, not tokens.
        let stateCharacters = PythonSemantics.characterCount(stateText)
        let usage = Usage(
            inputTokens: max(1, stateCharacters / 4) + max(1, instructionCharacters / 4),
            outputTokens: decisions.count
        )
        return DecisionPlan(rows: rows, decisions: decisions, usage: usage)
    }

    public func microbatches(for rows: [VonPackedInput], strategy: BatchingStrategy) -> [[Int]] {
        // The per-row seam reproduces the original sequential plan order.
        if rowLogitsProvider != nil {
            return rows.indices.map { [$0] }
        }
        return SystemOneEngine.microbatchIndices(
            sequenceLengths: rows.map(\.sequenceLength),
            maxAmplification: strategy.maxAmplification
        )
    }

    public func logits(for rows: [VonPackedInput]) async throws -> BatchLogitsOutput {
        if let rowLogitsProvider {
            var logits: [Float] = []
            var offsets = [0]
            for row in rows {
                let rowLogits = try await rowLogitsProvider(row)
                guard rowLogits.count == row.logitCount else {
                    throw SystemOneEngineError.unexpectedLogitCount(
                        expected: row.logitCount,
                        actual: rowLogits.count
                    )
                }
                logits += rowLogits
                offsets.append(logits.count)
            }
            return BatchLogitsOutput(logits: logits, rowOffsets: offsets)
        }
        if let batchLogitsProvider {
            return try await batchLogitsProvider(rows)
        }
        guard let model else { preconditionFailure("VonBackend has no logits provider") }
        let output = try model.forward(rows)
        eval(output.logits)
        return BatchLogitsOutput(logits: output.logits.asArray(Float.self), rowOffsets: output.rowOffsets)
    }

    private func packedRow(state: String, question: String, options: [String]) throws -> VonPackedInput {
        let packed = tokenizer.packedInput(state: state, question: question, options: options)
        guard packed.maskPositions.count == options.count else {
            throw SystemOneEngineError.unexpectedMaskCount(
                expected: options.count,
                actual: packed.maskPositions.count
            )
        }
        return packed
    }
}

extension SystemOneEngine {
    /// Loads von from an explicit local tokenizer folder and the verified
    /// production safetensors produced by `tools/convert_option_marker.py`.
    public static func von(tokenizerFolder: URL, verifiedWeightsURL: URL) async throws -> SystemOneEngine {
        SystemOneEngine(backend: try await VonBackend(
            tokenizerFolder: tokenizerFolder,
            verifiedWeightsURL: verifiedWeightsURL
        ))
    }

    /// Offline test seam: one provider call per row, in plan order.
    init(
        tokenizer: VonTokenizer,
        logitsProvider: @escaping @Sendable (VonPackedInput) async throws -> [Float]
    ) {
        self.init(backend: VonBackend(tokenizer: tokenizer, logitsProvider: logitsProvider))
    }

    /// Offline batched test seam: one provider call per microbatch.
    init(
        tokenizer: VonTokenizer,
        batchLogitsProvider: @escaping @Sendable ([VonPackedInput]) async throws -> BatchLogitsOutput
    ) {
        self.init(backend: VonBackend(tokenizer: tokenizer, batchLogitsProvider: batchLogitsProvider))
    }
}
