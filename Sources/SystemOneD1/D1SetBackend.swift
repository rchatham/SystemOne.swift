import Foundation
import MLX
import SystemOne
import SystemOneDecoders

/// d1-2B's settings: the `d1_*` keys of `config.json`, the tokenizer's BOS and
/// special tokens, and `calibration.json`.
public struct D1SetConfiguration: Equatable, Sendable {
    /// Questions packed after one state in a row (`d1_set_size`).
    public let setSize: Int
    /// The option-code scheme past 26 options (`d1_codes`: 1 `00`…, 2 `AA`…).
    public let codeScheme: Int
    /// The BOS token's text, written at the head of every row.
    public let bos: String
    /// Special tokens' texts, escaped in caller text (`plain`).
    public let specialTokens: [String]
    /// The temperature for buckets `temperatures` does not name.
    public let defaultTemperature: Double
    /// Per bucket: `noul`, `score:k`, `choice:small|mid|large|inventory`.
    public let temperatures: [String: Double]

    public init(
        setSize: Int = 4, codeScheme: Int = 1, bos: String = "", specialTokens: [String] = [],
        defaultTemperature: Double = 1, temperatures: [String: Double] = [:]
    ) {
        self.setSize = setSize
        self.codeScheme = codeScheme
        self.bos = bos
        self.specialTokens = specialTokens
        self.defaultTemperature = defaultTemperature
        self.temperatures = temperatures
    }

    /// Reads `config.json`, `tokenizer_config.json`, `tokenizer.json`, and
    /// (when present) `calibration.json`, with mlx-vlm's defaults for missing keys.
    public init(modelFolder: URL) throws {
        func object(_ name: String) throws -> [String: Any]? {
            let url = modelFolder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw D1SetError.invalidConfiguration("\(name) is not a JSON object")
            }
            return object
        }
        let config = try object("config.json") ?? [:]
        let attention = config["d1_attention"] as? String ?? "causal"
        guard attention == "set" else {
            throw D1SetError.invalidConfiguration("d1_attention \(attention); this backend reads `set` checkpoints")
        }
        let setSize = config["d1_set_size"] as? Int ?? 4
        guard (2...128).contains(setSize) else { throw D1SetError.invalidConfiguration("d1_set_size \(setSize)") }
        let tokenizerConfig = try object("tokenizer_config.json") ?? [:]
        func text(_ value: Any?) -> String? {
            (value as? String) ?? ((value as? [String: Any])?["content"] as? String)
        }
        // transformers' `all_special_tokens` and added tokens marked special.
        var specials = Set(["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]
            .compactMap { text(tokenizerConfig[$0]) })
        specials.formUnion((tokenizerConfig["additional_special_tokens"] as? [Any] ?? []).compactMap(text))
        for case let token as [String: Any] in (tokenizerConfig["added_tokens_decoder"] as? [String: Any] ?? [:]).values
        where token["special"] as? Bool == true {
            if let content = token["content"] as? String { specials.insert(content) }
        }
        for case let token as [String: Any] in try object("tokenizer.json")?["added_tokens"] as? [Any] ?? []
        where token["special"] as? Bool == true {
            if let content = token["content"] as? String { specials.insert(content) }
        }
        let calibration = try object("calibration.json") ?? [:]
        self.init(
            setSize: setSize,
            codeScheme: config["d1_codes"] as? Int ?? 1,
            bos: text(tokenizerConfig["bos_token"]) ?? "",
            specialTokens: specials.sorted(),
            defaultTemperature: (calibration["default"] as? NSNumber)?.doubleValue ?? 1,
            temperatures: (calibration["temperatures"] as? [String: Any] ?? [:]).compactMapValues { ($0 as? NSNumber)?.doubleValue }
        )
    }

    /// The calibration bucket of a question with `options` options.
    static func bucket(_ question: Question, options: Int) -> String {
        switch question {
        case .noul: return "noul"
        case .score: return "score:\(options)"
        case .choice:
            switch options {
            case ...3: return "choice:small"
            case ...8: return "choice:mid"
            case ...26: return "choice:large"
            default: return "choice:inventory"
            }
        }
    }

    func temperature(_ question: Question, options: Int) -> Double {
        temperatures[Self.bucket(question, options: options)] ?? defaultTemperature
    }
}

public enum D1SetError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
}

extension D1SetError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason): return "The d1-2B config is invalid: \(reason)."
        }
    }
}

/// d1-2B's prompt (mlx-vlm `d1_lfm3_vl.prompt`, from the code LiquidAI ships
/// with d1 checkpoints): d1-3B's ChatML state and question blocks, caller text
/// with special tokens broken by U+200B, and each question closed by
/// `<think></think>`, whose last token is the answer slot.
public enum D1SetPrompt {
    static let breakCharacter: Unicode.Scalar = "\u{200B}"
    static let lead = "<think></think>"
    public static let header = "QUESTION:\n"

    /// `plain`: every special token's text in `text` gets U+200B after its
    /// first character (longest name first, again until none is left), so it
    /// reads the same and no longer encodes to the token.
    public static func plain(_ text: String, specialTokens: [String]) -> String {
        let names = specialTokens.map { Array($0.unicodeScalars) }
            .filter { $0.count > 1 && !$0.contains(breakCharacter) }
            .sorted { $0.count > $1.count }
        guard !names.isEmpty else { return text }
        var scalars = Array(text.unicodeScalars)
        while true {
            var out: [Unicode.Scalar] = []
            var index = 0
            var changed = false
            while index < scalars.count {
                if let name = names.first(where: { index + $0.count <= scalars.count && Array(scalars[index ..< index + $0.count]) == $0 }) {
                    out.append(name[0])
                    out.append(breakCharacter)
                    out.append(contentsOf: name[1...])
                    index += name.count
                    changed = true
                } else {
                    out.append(scalars[index])
                    index += 1
                }
            }
            scalars = out
            if !changed { break }
        }
        var result = String.UnicodeScalarView()
        result.append(contentsOf: scalars)
        return String(result)
    }

    /// The row's head: BOS, the user turn, the state, and the first `QUESTION:`.
    public static func prefix(state: SystemOneState, configuration: D1SetConfiguration) -> String {
        configuration.bos + "<|im_start|>user\n"
            + plain(D1ChatPrompt.stateBlock(state), specialTokens: configuration.specialTokens) + "\n" + header
    }

    /// The question and the assistant header up to the answer slot.
    public static func suffix(question: Question, codes: [String], configuration: D1SetConfiguration) -> String {
        plain(D1ChatPrompt.questionBlock(question, codes: codes), specialTokens: configuration.specialTokens)
            + "<|im_end|>\n<|im_start|>assistant\n" + lead
    }

    /// `check_question` for noul, choice, and score.
    static func validate(_ question: Question) -> String? {
        guard !PythonSemantics.pythonStrip(question.instructions).isEmpty else {
            return "\"instructions\" is the question as text"
        }
        if case .score(let score) = question {
            guard (2...10).contains(score.criteria.count), score.criteria.allSatisfy({ if case .text = $0 { return true }; return false }) else {
                return "a score's \"criteria\" lists 2 to 10 levels, lowest first"
            }
        }
        return nil
    }

    /// `api.answer` at its defaults: a noul's P(yes); a choice's first best
    /// option; a score's median level (the first whose cumulative probability
    /// reaches one half). Confidence is the best option's probability.
    static func answer(_ question: Question, probabilities p: [Double]) -> Answer {
        var best = 0
        for index in p.indices.dropFirst() where p[index] > p[best] { best = index }
        switch question {
        case .noul:
            return .noul(NoulAnswer(noul: p[0]))
        case .choice(let choice):
            return .choice(ChoiceAnswer(
                choice: choice.criteria[best].key,
                probabilities: zip(choice.criteria, p).map { ProbabilityEntry(key: $0.key, probability: $1) },
                confidence: p[best]
            ))
        case .score(let score):
            let total = p.reduce(0, +)
            var mass = 0.0
            var level = p.count - 1
            for (index, value) in p.enumerated() {
                mass += value / (total == 0 ? 1 : total)
                if mass >= 0.5 { level = index; break }
            }
            return .score(ScoreAnswer(
                score: Double(level),
                confidence: p[best],
                legend: score.criteria.enumerated().map { LegendEntry(key: String($0.offset), description: D1ChatPrompt.pythonString($0.element)) },
                probabilities: p.enumerated().map { ProbabilityEntry(key: String($0.offset), probability: $0.element) }
            ))
        }
    }
}

/// One packed row: a state and up to `d1_set_size` questions after it.
public struct D1SetRow: PlannedRow, Equatable, Sendable {
    public let ids: [Int]
    /// Tokens of the state (BOS, user turn, state, first `QUESTION:`).
    public let stateLength: Int
    /// Per token: its question (1…), or -1 on the state.
    public let questions: [Int]
    /// Each question's answer slot.
    public let ends: [Int]
    /// Each question's token groups, one per option.
    public let groups: [[[Int]]]

    public var sequenceLength: Int { ids.count }
    public var logitCount: Int { groups.reduce(0) { $0 + $1.count } }

    /// Set attention on full-attention layers: the state sees the whole row,
    /// every token sees the state, and each question sees itself.
    var visibility: [[Bool]] {
        ids.indices.map { query in
            ids.indices.map { key in query < stateLength || key < stateLength || questions[query] == questions[key] }
        }
    }
}

/// d1-2B (`LiquidAI/d1-2B`, LFM3) as a System One backend for noul, choice,
/// and score questions on text states, under its `set` attention: up to
/// `d1_set_size` questions share one row after the state, read at their own
/// slots through the tied head, with `calibration.json`'s temperature.
/// Pictures, audio, and the typed questions (multiple choice, extraction,
/// estimates, boxes) are not supported yet.
public final class D1SetBackend: DecisionBackend {
    public let modelID: String
    public let configuration: D1SetConfiguration
    let tokenizer: DecoderTokenizer
    private let trunk: LFM3Trunk?

    public convenience init(modelFolder: URL, modelID: String = "d1-2B", dataType: DType = .bfloat16) throws {
        self.init(
            modelID: modelID,
            configuration: try D1SetConfiguration(modelFolder: modelFolder),
            tokenizer: try HubDecoderTokenizer(folder: modelFolder),
            trunk: try LFM3Trunk(modelFolder: modelFolder, dataType: dataType)
        )
    }

    init(modelID: String, configuration: D1SetConfiguration, tokenizer: DecoderTokenizer, trunk: LFM3Trunk?) {
        self.modelID = modelID
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.trunk = trunk
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<D1SetRow> {
        guard media == nil else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        var suffixes: [String] = []
        var readouts: [[[Int]]] = []
        for entry in request.questions {
            if let reason = D1SetPrompt.validate(entry.question) {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: reason)
            }
            do {
                let readout = try D1ChatPrompt.readout(entry.question, tokenizer: tokenizer, scheme: configuration.codeScheme)
                readouts.append(readout.groups)
                suffixes.append(D1SetPrompt.suffix(question: entry.question, codes: readout.codes, configuration: configuration))
            } catch let error as D1ChatError {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: error.localizedDescription)
            }
        }
        let prefix = D1SetPrompt.prefix(state: request.state, configuration: configuration)
        var rows: [D1SetRow] = []
        var decisions: [PlannedDecision] = []
        for start in stride(from: 0, to: suffixes.count, by: configuration.setSize) {
            let members = Array(start ..< min(start + configuration.setSize, suffixes.count))
            // `pack_set`: the first question is encoded with the state; the suffix
            // opens on a token boundary, so its own tokens are the row's last ones.
            var ids = tokenizer.encode(prefix + suffixes[members[0]])
            let stateLength = ids.count - tokenizer.encode(suffixes[members[0]]).count
            var questions = Array(repeating: -1, count: stateLength) + Array(repeating: 1, count: ids.count - stateLength)
            var ends = [ids.count - 1]
            for (offset, member) in members.enumerated().dropFirst() {
                let own = tokenizer.encode(D1SetPrompt.header + suffixes[member])
                ids += own
                questions += Array(repeating: offset + 1, count: own.count)
                ends.append(ids.count - 1)
            }
            var logitOffset = 0
            for member in members {
                let entry = request.questions[member]
                let count = readouts[member].count
                let temperature = configuration.temperature(entry.question, options: count)
                let range = logitOffset ..< logitOffset + count
                decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                    let scores = logits[0][range].map { temperature == 1 ? Double($0) : Double($0) / temperature }
                    let top = scores.max() ?? 0
                    let weights = scores.map { Foundation.exp($0 - top) }
                    let total = weights.reduce(0, +)
                    return D1SetPrompt.answer(entry.question, probabilities: weights.map { $0 / total })
                })
                logitOffset += count
            }
            rows.append(D1SetRow(
                ids: ids, stateLength: stateLength, questions: questions, ends: ends,
                groups: members.map { readouts[$0] }
            ))
        }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: rows.reduce(0) { $0 + $1.ids.count }))
    }

    public func logits(for rows: [D1SetRow]) async throws -> BatchLogitsOutput {
        let perRow = try optionLogits(for: rows)
        var offsets = [0]
        for row in perRow { offsets.append(offsets[offsets.count - 1] + row.count) }
        return BatchLogitsOutput(logits: perRow.flatMap { $0 }, rowOffsets: offsets)
    }

    /// Each row's option scores, question by question: the best logit among
    /// each option's token forms at the question's slot.
    func optionLogits(for rows: [D1SetRow]) throws -> [[Float]] {
        guard let trunk else { preconditionFailure("D1SetBackend has no model") }
        let lengths = rows.map(\.ids.count)
        let width = lengths.max() ?? 0
        let hidden = try trunk(
            rows.map { $0.ids + Array(repeating: 0, count: width - $0.ids.count) },
            attention: DecoderAttention(lengths: lengths, plan: .visible(rows.map(\.visibility)))
        )
        let slots = rows.enumerated().flatMap { index, row in row.ends.map { Int32(index * width + $0) } }
        let states = hidden.reshaped(-1, hidden.dim(-1)).take(MLXArray(slots), axis: 0).asType(.float32)
        let tokens = Array(Set(rows.flatMap { $0.groups.joined().joined() })).sorted()
        let column = Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($1, $0) })
        let scores = matmul(states, try trunk.readoutRows(tokens).asType(.float32).transposed()).asArray(Float.self)
        var slot = 0
        return rows.map { row in
            row.groups.flatMap { question -> [Float] in
                defer { slot += 1 }
                let base = slot * tokens.count
                return question.map { group in group.map { scores[base + column[$0]!] }.max() ?? -.infinity }
            }
        }
    }
}

extension SystemOneEngine {
    /// d1-2B from a local folder.
    public static func d1Set(modelFolder: URL, dataType: DType = .bfloat16) throws -> SystemOneEngine {
        SystemOneEngine(backend: try D1SetBackend(modelFolder: modelFolder, dataType: dataType))
    }
}
