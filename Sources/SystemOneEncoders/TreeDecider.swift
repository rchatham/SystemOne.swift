import Foundation
import MLX
import MLXNN
import SystemOne

/// How tasksource-decider-nano (`tasksource/tasksource-decider-nano`,
/// `decider.py`, layout "shared") renders requests.
///
/// Each option is read as the virtual pair `(state, "Question: <q>\nOption: <o>")`,
/// but a row writes the state once: `[CLS] state [SEP]`, then each question's
/// `"Question: <q>\nOption:"` once, then each `" <option> [SEP]"`. Tree
/// attention keeps the pairs apart: the state sees only the state, a question
/// sees the state and itself, an option sees the state, its question, and itself.
/// Positions follow the virtual pair.
public enum TreeDeciderPrompt {
    /// The state, or `"(empty)"` for an empty text or no fields; fields as compact JSON.
    public static func stateText(_ state: SystemOneState) -> String {
        switch state {
        case .text(let text) where text.isEmpty: return "(empty)"
        case .fields(let fields) where fields.isEmpty: return "(empty)"
        default: return EncoderText.compactJSON(state)
        }
    }

    /// `format_option`: placeholder keys (`option_…`, `a`, `(b)`, ...) give way to
    /// the description; otherwise `key: description`, or the key alone when the
    /// description is missing or repeats it.
    public static func formatOption(key: String, description: String?) -> String {
        let description = PythonSemantics.pythonStrip(description ?? "")
        let key = PythonSemantics.pythonStrip(key)
        guard !description.isEmpty else { return key }
        let lower = key.lowercased()
        let placeholder = ["option_", "k_", "choice_", "item_"].contains { lower.hasPrefix($0) }
            || (key.count == 1 && key.first!.isLetter)
            || (key.hasPrefix("(") && key.hasSuffix(")") && key.count <= 4)
        if placeholder { return description }
        if description.lowercased() == lower { return key }
        return "\(key): \(description)"
    }

    /// Option texts for choice and noul (`[false, true]`); score is not supported.
    public static func options(for question: Question) throws -> [String] {
        switch question {
        case .choice(let choice):
            return choice.criteria.map { formatOption(key: $0.key, description: $0.description) }
        case .noul(let noul):
            return [
                formatOption(key: "false", description: noul.criteria?.falseDescription ?? "False"),
                formatOption(key: "true", description: noul.criteria?.trueDescription ?? "True"),
            ]
        case .score:
            throw TreeDeciderError.unsupportedScore
        }
    }
}

public enum TreeDeciderError: Error, Equatable, Sendable {
    case unsupportedScore
    case unsupportedLayout(String)
}

extension TreeDeciderError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedScore: return "tasksource-decider-nano answers choice and noul questions only."
        case .unsupportedLayout(let layout): return "decider_config.json asks for \(layout); only shared/opt/cached is supported."
        }
    }
}

/// One shared-layout row: the state, then question blocks with their options.
public struct TreeRow: PlannedRow, Equatable, Sendable {
    public let ids: [Int]
    public let positions: [Int]
    /// Per token: 0 state, 1 question block, 2 option block.
    public let kinds: [Int]
    /// Per token: the question block's index in this row, or -1.
    public let questions: [Int]
    /// Per token: the option's index in this row, or -1.
    public let options: [Int]
    public let optionCount: Int

    public var sequenceLength: Int { ids.count }
    public var logitCount: Int { optionCount }

    /// `decider.py`'s attention: every token sees the state; question and option
    /// tokens see their question block; option tokens see their own option; every
    /// token sees itself.
    public var visibility: [[Bool]] {
        ids.indices.map { query in
            ids.indices.map { key in
                if query == key { return true }
                switch kinds[key] {
                case 0: return true
                case 1: return kinds[query] >= 1 && questions[query] == questions[key]
                default: return kinds[query] == 2 && options[query] == options[key]
                }
            }
        }
    }
}

/// tasksource-decider-nano's network: an Ettin encoder and a `Linear(d, 1)`
/// over each option block's mean state.
final class TreeDeciderModel {
    let encoder: ModernBERTEncoder
    let scorer: Linear
    let pad: Int

    init(store: any TensorStore, encoder configuration: ModernBERTConfiguration, pad: Int, dataType: DType = .float32) throws {
        let d = configuration.hiddenSize
        encoder = try ModernBERTEncoder(store: store, configuration: configuration, prefix: "backbone.", dataType: dataType)
        for (name, shape) in [("scorer.weight", [1, d]), ("scorer.bias", [1])] {
            let descriptor = try store.requiredTensor(name).descriptor
            guard descriptor.shape == shape else { throw WeightsError.shapeMismatch(name: name, expected: shape, actual: descriptor.shape) }
        }
        scorer = try store.linear("scorer", inputWidth: d, bias: true, dataType: dataType)
        self.pad = pad
    }

    func logits(_ rows: [TreeRow]) throws -> [[Float]] {
        let attention = EncoderAttention(
            lengths: rows.map(\.ids.count), positions: rows.map(\.positions), visible: rows.map(\.visibility)
        )
        let hidden = try encoder(paddedIDs(rows.map(\.ids), pad: pad), attention: attention)
        var pooled: [MLXArray] = []
        for (index, row) in rows.enumerated() {
            for option in 0..<row.optionCount {
                let tokens = row.options.indices.filter { row.options[$0] == option }.map(Int32.init)
                pooled.append(hidden[index].take(MLXArray(tokens), axis: 0).mean(axis: 0))
            }
        }
        let scores = scorer(stacked(pooled)).squeezed(axis: -1).asType(.float32).asArray(Float.self)
        var offset = 0
        return rows.map { row in
            defer { offset += row.optionCount }
            return Array(scores[offset..<(offset + row.optionCount)])
        }
    }
}

/// tasksource-decider-nano as a System One backend: choice and noul only, the
/// plain softmax over option scores (no calibration). A choice's confidence is
/// its top probability; a noul is `P(true)`.
public final class TreeDeciderBackend: DecisionBackend {
    public let modelID: String
    let tokenizer: EncoderTokenizer
    let tokens: EncoderSpecialTokens
    let maxLength: Int
    /// Tokens a row may grow past its state: `max(2P, P + rowExtra)`.
    let rowExtra = 2_048
    private let model: TreeDeciderModel?

    /// Loads a local folder: `config.json`, `decider_config.json`,
    /// `tokenizer.json`, and `model.safetensors`.
    public convenience init(modelFolder: URL, modelID: String = "tasksource-decider-nano", dataType: DType = .float32) throws {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("decider_config.json"))
        let decider = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let layout = [decider["layout"] as? String ?? "pair", decider["pool"] as? String ?? "opt", decider["attn"] as? String ?? "cached"]
        guard layout == ["shared", "opt", "cached"] else { throw TreeDeciderError.unsupportedLayout(layout.joined(separator: "/")) }
        let configuration = try ModernBERTConfiguration(modelFolder: modelFolder)
        let tokenizer = try HubEncoderTokenizer(folder: modelFolder)
        let tokens = try EncoderSpecialTokens(tokenizer: tokenizer, names: tokenizer.specialTokens)
        self.init(
            modelID: modelID, tokenizer: tokenizer, tokens: tokens,
            maxLength: decider["inference_max_length"] as? Int ?? 8_192,
            model: try TreeDeciderModel(
                store: SafetensorsFile(contentsOf: modelFolder.appendingPathComponent("model.safetensors")),
                encoder: configuration, pad: tokens.pad, dataType: dataType
            )
        )
    }

    init(modelID: String, tokenizer: EncoderTokenizer, tokens: EncoderSpecialTokens, maxLength: Int, model: TreeDeciderModel?) {
        self.modelID = modelID
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.maxLength = maxLength
        self.model = model
    }

    /// The reference's packing: rows of at most `max(2P, P + 2048)` tokens, the
    /// state repeated in each, question blocks repeated when their options spill
    /// into a new row. Returns rows and, per question, where each option landed.
    func pack(state: SystemOneState, questions: [(id: String, instructions: String, options: [String])]) throws
        -> (rows: [TreeRow], places: [[(row: Int, option: Int)]])
    {
        let prefix = [tokens.cls] + tokenizer.encode(TreeDeciderPrompt.stateText(state)) + [tokens.sep]
        let cap = max(2 * prefix.count, prefix.count + rowExtra)
        typealias Block = (question: [Int], options: [(index: Int, ids: [Int])])
        var rows: [[Block]] = []
        var sizes: [Int] = []
        for (number, question) in questions.enumerated() {
            let block = tokenizer.encode("Question: \(question.instructions)\nOption:")
            let optionIDs = question.options.map { tokenizer.encode(" " + $0) + [tokens.sep] }
            let longest = prefix.count + block.count + (optionIDs.map(\.count).max() ?? 0)
            guard longest <= maxLength else {
                throw SystemOneEngineError.invalidQuestion(
                    id: question.id, reason: "a (state, question, option) pair needs \(longest) tokens > \(maxLength)"
                )
            }
            if rows.isEmpty || number == 0 || sizes[sizes.count - 1] + block.count + optionIDs[0].count > cap {
                rows.append([])
                sizes.append(prefix.count)
            }
            var taken = 0
            while taken < optionIDs.count {
                var take: [(index: Int, ids: [Int])] = []
                var size = block.count
                while taken + take.count < optionIDs.count,
                      take.isEmpty || sizes[sizes.count - 1] + size + optionIDs[taken + take.count].count <= cap
                {
                    take.append((taken + take.count, optionIDs[taken + take.count]))
                    size += take[take.count - 1].ids.count
                }
                if !rows[rows.count - 1].isEmpty, sizes[sizes.count - 1] + size > cap {
                    rows.append([])
                    sizes.append(prefix.count)
                    continue
                }
                rows[rows.count - 1].append((block, take))
                sizes[sizes.count - 1] += size
                taken += take.count
            }
        }

        var built: [TreeRow] = []
        for blocks in rows {
            var ids = prefix, positions = Array(prefix.indices)
            var kinds = Array(repeating: 0, count: prefix.count)
            var questionIDs = Array(repeating: -1, count: prefix.count), optionIDs = questionIDs
            var count = 0
            for (blockIndex, block) in blocks.enumerated() {
                ids += block.question
                positions += (0..<block.question.count).map { prefix.count + $0 }
                kinds += Array(repeating: 1, count: block.question.count)
                questionIDs += Array(repeating: blockIndex, count: block.question.count)
                optionIDs += Array(repeating: -1, count: block.question.count)
                for option in block.options {
                    ids += option.ids
                    positions += (0..<option.ids.count).map { prefix.count + block.question.count + $0 }
                    kinds += Array(repeating: 2, count: option.ids.count)
                    questionIDs += Array(repeating: blockIndex, count: option.ids.count)
                    optionIDs += Array(repeating: count, count: option.ids.count)
                    count += 1
                }
            }
            built.append(TreeRow(ids: ids, positions: positions, kinds: kinds, questions: questionIDs, options: optionIDs, optionCount: count))
        }
        // Rows fill question by question and option by option, so each question's
        // options are the next ones in row order.
        var cursor = (row: 0, option: 0)
        let places = questions.map { question -> [(row: Int, option: Int)] in
            question.options.map { _ in
                while cursor.option >= built[cursor.row].optionCount { cursor = (cursor.row + 1, 0) }
                defer { cursor.option += 1 }
                return cursor
            }
        }
        return (built, places)
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<TreeRow> {
        guard media == nil else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        var questions: [(id: String, instructions: String, options: [String])] = []
        for entry in request.questions {
            do {
                let options = try TreeDeciderPrompt.options(for: entry.question)
                if case .choice = entry.question, options.isEmpty {
                    throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: "choice needs criteria")
                }
                questions.append((entry.id, entry.question.instructions, options))
            } catch let error as TreeDeciderError {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: error.localizedDescription)
            }
        }
        let (rows, places) = try pack(state: request.state, questions: questions)
        var decisions: [PlannedDecision] = []
        for (entry, place) in zip(request.questions, places) {
            let rowIndices = place.map { $0.row }.reduce(into: [Int]()) { if !$0.contains($1) { $0.append($1) } }
            let question = entry.question
            decisions.append(PlannedDecision(id: entry.id, rowIndices: rowIndices) { logits in
                let z = place.map { logits[rowIndices.firstIndex(of: $0.row)!][$0.option] }
                let p = EncoderAnswers.softmax(z)
                switch question {
                case .noul:
                    return .noul(NoulAnswer(noul: min(max(p[1], 0), 1)))
                case .choice(let choice):
                    return EncoderAnswers.choice(keys: choice.criteria.map(\.key), probabilities: p, confidence: p.max() ?? 0)
                case .score:
                    preconditionFailure("score questions are rejected while planning")
                }
            })
        }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: rows.reduce(0) { $0 + $1.ids.count }))
    }

    public func logits(for rows: [TreeRow]) async throws -> BatchLogitsOutput {
        guard let model else { preconditionFailure("TreeDeciderBackend has no model") }
        return BatchLogitsOutput(rows: try model.logits(rows))
    }
}

extension SystemOneEngine {
    /// tasksource-decider-nano from a local folder.
    public static func treeDecider(modelFolder: URL, dataType: DType = .float32) throws -> SystemOneEngine {
        SystemOneEngine(backend: try TreeDeciderBackend(modelFolder: modelFolder, dataType: dataType))
    }
}
