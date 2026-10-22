import Foundation

/// One planned forward pass. The engine only needs its size to group rows into
/// microbatches and its logit count to split batched results.
public protocol PlannedRow {
    /// Positions the model reads for this row (media prefix plus tokens).
    var sequenceLength: Int { get }

    /// Number of option logits this row produces.
    var logitCount: Int { get }
}

/// One question's planned rows and the model-specific postprocessing that turns
/// their raw logits into a public answer.
public struct PlannedDecision {
    public let id: String

    /// Indices into `DecisionPlan.rows`, in the order `resolve` receives them.
    public let rowIndices: [Int]

    /// Converts this question's row logits, ordered like `rowIndices`, into an answer.
    public let resolve: ([[Float]]) throws -> Answer

    public init(
        id: String,
        rowIndices: [Int],
        resolve: @escaping ([[Float]]) throws -> Answer
    ) {
        self.id = id
        self.rowIndices = rowIndices
        self.resolve = resolve
    }
}

/// Every forward pass a request needs, planned before any inference runs.
public struct DecisionPlan<Row: PlannedRow> {
    public let rows: [Row]
    public let decisions: [PlannedDecision]
    public let usage: Usage

    public init(rows: [Row], decisions: [PlannedDecision], usage: Usage) {
        self.rows = rows
        self.decisions = decisions
        self.usage = usage
    }
}

/// A System One model: it plans requests into rows, runs microbatches of rows,
/// and resolves each question's logits into an answer.
///
/// `SystemOneEngine` owns orchestration that every model shares (validation,
/// planning before inference, microbatch fan-out, cancellation, and answer
/// assembly). Conformers own prompt packing, the network, and calibration.
/// A backend is only ever used from the engine's actor, so it need not be
/// `Sendable`.
public protocol DecisionBackend: AnyObject {
    associatedtype Row: PlannedRow

    /// The identifier reported in `SystemOneResponse.model`.
    var modelID: String { get }

    /// Validates model-specific constraints and plans every row up front, so
    /// invalid input fails before any forward pass runs.
    func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<Row>

    /// Groups row indices into microbatches. Defaults to
    /// `SystemOneEngine.microbatchIndices` over `sequenceLength`.
    func microbatches(for rows: [Row], strategy: BatchingStrategy) -> [[Int]]

    /// Runs one microbatch and returns its row-major logits.
    func logits(for rows: [Row]) async throws -> BatchLogitsOutput
}

extension DecisionBackend {
    public func microbatches(for rows: [Row], strategy: BatchingStrategy) -> [[Int]] {
        SystemOneEngine.microbatchIndices(
            sequenceLengths: rows.map(\.sequenceLength),
            maxAmplification: strategy.maxAmplification
        )
    }
}

/// Row-major logits for one microbatch. Row `i` occupies
/// `logits[rowOffsets[i]..<rowOffsets[i + 1]]`.
public struct BatchLogitsOutput: Equatable, Sendable {
    public let logits: [Float]
    public let rowOffsets: [Int]

    public init(logits: [Float], rowOffsets: [Int]) {
        self.logits = logits
        self.rowOffsets = rowOffsets
    }
}
