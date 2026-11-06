import Foundation
import MLX
import SystemOne

/// The three question kinds of JEV's `bare-v1` template.
public enum JEVKind: String, CaseIterable, Sendable {
    case noul, choice, score
}

/// Errors raised while loading or prompting a JEV model.
public enum JEVError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case invalidHead(String)
    /// Even with an empty state the prompt exceeds `maxLength` tokens.
    case promptTooLong(id: String, tokens: Int, maximum: Int)
}

extension JEVError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason): return "The JEV configuration is invalid: \(reason)."
        case .invalidHead(let reason): return "The JEV head is invalid: \(reason)."
        case .promptTooLong(let id, let tokens, let maximum):
            return "Question \(id) needs \(tokens) tokens without its state; the limit is \(maximum)."
        }
    }
}

/// A JEV folder's `judge_config.json` (head slots, template, adapter folder)
/// and `calibration.json` (one temperature per kind).
public struct JEVConfiguration: Equatable, Sendable {
    public static let templateVersion = "bare-v1"

    public let slotCount: Int
    /// The head slots each kind reads, in option order.
    public let slots: [JEVKind: Range<Int>]
    public let temperatures: [JEVKind: Double]
    /// The peft adapter's folder inside the model folder.
    public let adapterSubfolder: String

    public init(modelFolder: URL) throws {
        func object(_ name: String) throws -> [String: Any] {
            let url = modelFolder.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url),
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw JEVError.invalidConfiguration("\(name) is missing or not a JSON object") }
            return object
        }
        try self.init(judge: object("judge_config.json"), calibration: object("calibration.json"))
    }

    public init(judge: [String: Any], calibration: [String: Any]) throws {
        guard let slotsObject = judge["slots"] as? [String: Any],
              let count = (slotsObject["num_slots"] as? NSNumber)?.intValue, count > 0,
              let ranges = slotsObject["ranges"] as? [String: [Int]]
        else { throw JEVError.invalidConfiguration("judge_config.json needs slots.num_slots and slots.ranges") }
        guard (slotsObject["template_version"] as? String) == Self.templateVersion else {
            throw JEVError.invalidConfiguration("only the \(Self.templateVersion) template is supported")
        }
        guard let temperatures = calibration["per_kind"] as? [String: NSNumber] else {
            throw JEVError.invalidConfiguration("calibration.json needs per_kind temperatures")
        }
        let minimum: [JEVKind: Int] = [.noul: 2, .score: JEVPrompt.scoreLevels, .choice: 2]
        var slots: [JEVKind: Range<Int>] = [:]
        var perKind: [JEVKind: Double] = [:]
        for kind in JEVKind.allCases {
            guard let range = ranges[kind.rawValue], range.count == 2, 0 <= range[0], range[0] < range[1],
                  range[1] <= count, range[1] - range[0] >= minimum[kind]!
            else {
                throw JEVError.invalidConfiguration("slots.ranges.\(kind.rawValue) must hold at least \(minimum[kind]!) slots within num_slots")
            }
            guard let temperature = temperatures[kind.rawValue]?.doubleValue, temperature.isFinite, temperature > 0 else {
                throw JEVError.invalidConfiguration("calibration.json needs a positive \(kind.rawValue) temperature")
            }
            slots[kind] = range[0]..<range[1]
            perKind[kind] = temperature
        }
        let subfolder = judge["adapter_subfolder"] as? String ?? "adapter"
        guard !subfolder.isEmpty, !subfolder.hasPrefix("/"), !subfolder.split(separator: "/").contains("..") else {
            throw JEVError.invalidConfiguration("adapter_subfolder must name a folder inside the model folder")
        }
        slotCount = count
        self.slots = slots
        self.temperatures = perKind
        adapterSubfolder = subfolder
    }
}

/// The files a JEV checkpoint is verified against before parsing.
public struct JEVArtifacts: Equatable, Sendable {
    /// Backbone shards, by file name.
    public let shards: [String: PinnedArtifact]
    /// The adapter's `adapter_model.safetensors`.
    public let adapter: PinnedArtifact
    /// `head.safetensors`.
    public let head: PinnedArtifact

    public init(shards: [String: PinnedArtifact], adapter: PinnedArtifact, head: PinnedArtifact) {
        self.shards = shards
        self.adapter = adapter
        self.head = head
    }

    /// `autotrust/JEV-9B` at revision `7ac669b8fb476ba926491e3d1c39f9330d68dbed`.
    public static let jev9B = JEVArtifacts(
        shards: [
            "model-00001-of-00005.safetensors": PinnedArtifact(
                byteCount: 4_234_572_344, sha256: "fdce203bd6767c95690c4695699e9e796fa53caeb06c7f6d43a014f7c513b32c"),
            "model-00002-of-00005.safetensors": PinnedArtifact(
                byteCount: 4_199_125_880, sha256: "64003b88d6e6871f2315e39356188082063ff879994434d5ce8c209b3c736a01"),
            "model-00003-of-00005.safetensors": PinnedArtifact(
                byteCount: 4_215_396_752, sha256: "ef7cf4227fb3f391bb2bcd5d79a3c10cd90d823dbf964f41a0aa34591377ae82"),
            "model-00004-of-00005.safetensors": PinnedArtifact(
                byteCount: 3_224_330_536, sha256: "2265f2fcf900572fa0b7c645a9a64f662d4123daeeeda23e917411e907d66272"),
            "model-00005-of-00005.safetensors": PinnedArtifact(
                byteCount: 2_034_237_568, sha256: "dd63614f1dc80dce2d83be3f3e69af1f8a0ebc8add9b8e0abf3910f563e44344"),
        ],
        adapter: PinnedArtifact(
            byteCount: 160_486_456, sha256: "796386ace560ec5624bbfe83f5c505121e5c9f9654200a847f032cb333cd1f71"),
        head: PinnedArtifact(
            byteCount: 393_464, sha256: "d960a6f5989402c6d08d74d8d435a21a031cc7878f3c4ac274a126c900f627e7")
    )
}

/// JEV's `bare-v1` prompt, as the autotrust/JEV-9B model card renders it:
///
///     [kind] choice
///     [state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
///     [question] Supplier response for this scenario.
///     [options]
///     A) issue_warning
///     B) renegotiate
///     [decision]:
///
/// The string is tokenised whole, with no chat template or special tokens.
/// A noul lists `false`, `true` and a score `0` … `5`, so what their sides or
/// levels mean travels in the `[question]` line (``foldedQuestion(_:)``).
public enum JEVPrompt {
    /// Labels of the choice slots.
    static let choiceLabels = "ABCDEFGHIJKLMNOP".map(String.init)
    public static let maximumChoiceOptions = 16
    public static let scoreLevels = 6

    /// The whole prompt for one question.
    public static func render(kind: JEVKind, state: String, question: String, options: [String]) -> String {
        let lines = kind == .choice
            ? options.enumerated().map { "\(choiceLabels[$0.offset])) \($0.element)" }
            : options
        return "[kind] \(kind.rawValue)\n[state] \(state)\n[question] \(question)\n[options]\n"
            + lines.joined(separator: "\n") + "\n[decision]:"
    }

    /// A question's kind, answer keys, and option lines. A choice option
    /// reads `key: description`, or its key alone; a score's keys are `"0"` … `"5"`.
    public static func options(for question: Question) -> (kind: JEVKind, keys: [String], texts: [String]) {
        switch question {
        case .noul:
            return (.noul, ["false", "true"], ["false", "true"])
        case .score:
            let levels = (0..<scoreLevels).map(String.init)
            return (.score, levels, levels)
        case .choice(let choice):
            let texts = choice.criteria.map { option -> String in
                let description = option.description.flatMap { $0.isEmpty ? nil : $0 }
                return description.map { "\(option.key): \($0)" } ?? option.key
            }
            return (.choice, choice.criteria.map(\.key), texts)
        }
    }

    /// Why the head cannot read `question`, or `nil`: a choice takes 1 to 16
    /// options and a score exactly six levels.
    public static func validate(_ question: Question, choiceSlots: Int = maximumChoiceOptions) -> String? {
        switch question {
        case .choice(let choice):
            let maximum = min(maximumChoiceOptions, choiceSlots)
            guard (1...maximum).contains(choice.criteria.count) else {
                return "a JEV choice takes 1 to \(maximum) options"
            }
        case .score(let score):
            guard score.criteria.count == scoreLevels else {
                return "a JEV score has exactly \(scoreLevels) levels, 0 to 5"
            }
        case .noul:
            break
        }
        return nil
    }

    /// The `[question]` line. JEV's noul and score options are fixed, so
    /// what a noul's `true`/`false` or a score's six levels mean must travel in
    /// the question itself. A choice keeps its instructions as they are.
    ///
    /// The result must stay on one line: `\n[options]` follows it.
    public static func foldedQuestion(_ question: Question) -> String {
        switch question {
        case .choice(let choice):
            return choice.instructions
        case .noul(let noul):
            // TODO(human): fold `noul.criteria?.trueDescription` and
            // `falseDescription` into the instructions.
            return noul.instructions
        case .score(let score):
            // TODO(human): fold the six levels (`score.criteria`, `.text` or
            // `.described(what:examples:)`) into the instructions, so the model
            // knows what 0 and 5 mean.
            return score.instructions
        }
    }
}

/// One `bare-v1` prompt, read through the head slots of its kind.
public struct JEVRow: PlannedRow, Equatable, Sendable {
    public let ids: [Int]
    /// The head slots this row's options occupy.
    public let slots: Range<Int>

    public var sequenceLength: Int { ids.count }
    public var logitCount: Int { slots.count }
}

/// autotrust/JEV-9B (and its packaging) as a System One backend: a
/// Qwen3.5 trunk with its peft LoRA merged at load, read at each prompt's
/// last token by a 24-slot head (`head.safetensors`: `proj.weight`
/// `[slots, hidden]` and `proj.bias` `[slots]`, float32).
///
/// One causal row per question, right-padded. A kind's active slots, divided
/// by its `calibration.json` temperature, give the softmax answer. The
/// reference's vLLM path reads the log-probabilities of 24 verbalizer tokens;
/// the log-softmax normaliser cancels within a kind, so the head alone gives
/// the same distribution and the language-model head is never loaded.
public final class JEVBackend: DecisionBackend {
    public let modelID: String
    public let configuration: JEVConfiguration
    /// The longest prompt in tokens. A longer state keeps the first 60 % and
    /// last 40 % of its tokens, as JEV's own server truncates.
    public let maxLength: Int
    let tokenizer: any ReversibleDecoderTokenizer
    private let trunk: Qwen35Trunk?
    private let headWeight: MLXArray?
    private let headBias: MLXArray?

    /// Loads a JEV folder: `config.json`, the sharded backbone, the adapter,
    /// `head.safetensors`, `judge_config.json`, `calibration.json`, and the
    /// tokenizer. Pass `artifacts: nil` to skip the byte-for-byte pins.
    public convenience init(
        modelFolder: URL, modelID: String? = nil, artifacts: JEVArtifacts? = .jev9B,
        dataType: DType = .bfloat16, maxLength: Int = 4_096
    ) throws {
        let configuration = try JEVConfiguration(modelFolder: modelFolder)
        let adapter = try LoRAAdapter(
            folder: modelFolder.appendingPathComponent(configuration.adapterSubfolder), pin: artifacts?.adapter)
        let trunk = try Qwen35Trunk(
            store: SafetensorsCheckpoint(modelFolder: modelFolder, pins: artifacts?.shards),
            configuration: Qwen35Configuration(modelFolder: modelFolder),
            dataType: dataType, adapter: adapter, loadsReadout: false
        )
        let headURL = modelFolder.appendingPathComponent("head.safetensors")
        let head = try artifacts.map { try SafetensorsFile(contentsOf: headURL, pinnedTo: $0.head) }
            ?? SafetensorsFile(contentsOf: headURL)
        try self.init(
            modelID: modelID ?? modelFolder.lastPathComponent, configuration: configuration,
            tokenizer: HubDecoderTokenizer(folder: modelFolder), maxLength: maxLength, trunk: trunk, head: head
        )
    }

    init(
        modelID: String, configuration: JEVConfiguration, tokenizer: any ReversibleDecoderTokenizer,
        maxLength: Int, trunk: Qwen35Trunk?, head: (any TensorStore)?
    ) throws {
        self.modelID = modelID
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.trunk = trunk
        if let trunk, let head {
            let slots = configuration.slotCount, hidden = trunk.configuration.hiddenSize
            let weight = try head.requiredTensor("proj.weight").descriptor.shape
            let bias = try head.requiredTensor("proj.bias").descriptor.shape
            guard weight == [slots, hidden], bias == [slots] else {
                throw JEVError.invalidHead("proj.weight is \(weight) and proj.bias \(bias); expected [\(slots), \(hidden)] and [\(slots)]")
            }
            headWeight = try head.requiredTensor("proj.weight").mlxArray(.float32)
            headBias = try head.requiredTensor("proj.bias").mlxArray(.float32)
        } else {
            headWeight = nil
            headBias = nil
        }
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<JEVRow> {
        guard media == nil else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        let state = D3Prompt.describe(request.state)
        var rows: [JEVRow] = []
        var decisions: [PlannedDecision] = []
        for entry in request.questions {
            let question = entry.question
            if let reason = JEVPrompt.validate(question, choiceSlots: configuration.slots[.choice]!.count) {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: reason)
            }
            let (kind, keys, texts) = JEVPrompt.options(for: question)
            if keys.count == 1 {
                // A softmax over one option is 1 whatever the model reads.
                decisions.append(PlannedDecision(id: entry.id, rowIndices: []) { _ in
                    DecoderAnswers.answer(question, keys: keys, probabilities: [1])
                })
                continue
            }
            let ids = try encode(
                id: entry.id, kind: kind, state: state, question: JEVPrompt.foldedQuestion(question), options: texts)
            let temperature = configuration.temperatures[kind]!
            decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                DecoderAnswers.answer(
                    question, keys: keys, probabilities: DecoderAnswers.probabilities(logits[0], temperature: temperature))
            })
            let start = configuration.slots[kind]!.lowerBound
            rows.append(JEVRow(ids: ids, slots: start ..< start + keys.count))
        }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: rows.reduce(0) { $0 + $1.ids.count }))
    }

    public func logits(for rows: [JEVRow]) async throws -> BatchLogitsOutput {
        let perRow = try slotLogits(for: rows)
        var offsets = [0]
        for row in perRow { offsets.append(offsets[offsets.count - 1] + row.count) }
        return BatchLogitsOutput(logits: perRow.flatMap { $0 }, rowOffsets: offsets)
    }

    /// Each row's head logits over its slots: `W · h + b` on the float32
    /// final-norm state at the row's last token.
    func slotLogits(for rows: [JEVRow]) throws -> [[Float]] {
        let scores = try headLogits(rows.map(\.ids))
        return rows.enumerated().map { index, row in
            let start = index * configuration.slotCount
            return Array(scores[(start + row.slots.lowerBound) ..< (start + row.slots.upperBound)])
        }
    }

    /// All head slots for each row, row-major `[rows × slots]`.
    func headLogits(_ rows: [[Int]]) throws -> [Float] {
        guard let trunk, let headWeight, let headBias else { preconditionFailure("JEVBackend has no model") }
        let width = rows.map(\.count).max() ?? 0
        let lengths = rows.map(\.count)
        let hidden = try trunk(
            rows.map { $0 + Array(repeating: 0, count: width - $0.count) },
            attention: DecoderAttention(lengths: lengths, plan: .causal)
        )
        let last = MLXArray(lengths.enumerated().map { Int32($0.offset * width + $0.element - 1) })
        let states = hidden.reshaped(-1, hidden.dim(-1)).take(last, axis: 0).asType(.float32)
        return (matmul(states, headWeight.transposed()) + headBias).asArray(Float.self)
    }

    /// One row's final-norm states `[length, hidden]` in float32, for parity probes.
    func trunkStates(_ ids: [Int]) throws -> MLXArray {
        guard let trunk else { preconditionFailure("JEVBackend has no model") }
        return try trunk([ids], attention: DecoderAttention(lengths: [ids.count], plan: .causal))[0].asType(.float32)
    }

    /// Token ids of one prompt, cutting an over-long state to its head and tail.
    func encode(id: String, kind: JEVKind, state: String, question: String, options: [String]) throws -> [Int] {
        let ids = tokenizer.encode(JEVPrompt.render(kind: kind, state: state, question: question, options: options))
        guard ids.count > maxLength else { return ids }
        let empty = tokenizer.encode(JEVPrompt.render(kind: kind, state: "", question: question, options: options))
        guard empty.count <= maxLength else {
            throw JEVError.promptTooLong(id: id, tokens: empty.count, maximum: maxLength)
        }
        let stateIDs = tokenizer.encode(state)
        var budget = min(maxLength - empty.count, stateIDs.count - 1)
        while budget > 0 {
            let head = budget * 3 / 5
            let kept = tokenizer.decode(Array(stateIDs[..<head]))
                + tokenizer.decode(Array(stateIDs[(stateIDs.count - (budget - head))...]))
            let ids = tokenizer.encode(JEVPrompt.render(kind: kind, state: kept, question: question, options: options))
            if ids.count <= maxLength { return ids }
            // Re-tokenising across the cut can merge differently: shrink by the overshoot.
            budget -= ids.count - maxLength
        }
        return empty
    }
}

extension SystemOneEngine {
    /// A JEV model from a local folder, verifying the pinned autotrust/JEV-9B
    /// files unless other artifacts (or `nil`) are given.
    public static func jev(
        modelFolder: URL, artifacts: JEVArtifacts? = .jev9B, dataType: DType = .bfloat16
    ) throws -> SystemOneEngine {
        SystemOneEngine(backend: try JEVBackend(modelFolder: modelFolder, artifacts: artifacts, dataType: dataType))
    }
}
