import Foundation
import MLX
import SystemOne
import SystemOneDecoders

/// d1-3B's prompt and readout (`LiquidAI/d1-3B`, `prompt.py`, `api.py`, and
/// `runner.py`, the published defaults: no system turn, `json_only` states,
/// `desc` options, no lead, no calibration).
///
/// A state is written once, `<|im_start|>user\n{images}{state}\n\n\nQUESTION:\n`,
/// and each question follows it with its options under single-token codes, up
/// to the assistant header. The answer is read off the next-token logits: each
/// option scores its best token form, softmaxed over the options. Pictures are
/// capped at one megapixel (Pillow's bicubic) and written as the chat
/// template's `<image>` per picture, which LFM2-VL's processor expands.
public enum D1ChatPrompt {
    static let yesForms = ["yes", "Yes", "YES"]
    static let noForms = ["no", "No", "NO"]
    /// Codes tried, in order, when an option's own code is not one free token.
    static let fallbackPool: [String] = {
        let upper: [String] = (65...90).map { String(UnicodeScalar(UInt8($0))) }
        let digits: [String] = (0..<100).map { String(format: "%02d", $0) }
        let lower: [String] = (97...122).map { String(UnicodeScalar(UInt8($0))) }
        let numbered: [String] = (0..<200).map { "#\($0)" }
        let pairs: [String] = upper.flatMap { first in upper.map { first + $0 } }
        return upper + digits + lower + numbered + pairs
    }()

    /// Pixels a picture may have before the processor (`VISION_MAX_PIXELS`).
    public static let maximumPixels = 1024 * 1024

    /// The text before the questions: the user turn's head, the pictures'
    /// markup (`images`), and the state.
    public static func prefix(state: SystemOneState, images: String = "") -> String {
        // No state (`state=None`): the question follows the pictures directly.
        guard state != .media else { return "<|im_start|>user\n" + images }
        return "<|im_start|>user\n" + images + stateBlock(state) + "\nQUESTION:\n"
    }

    /// `cap_pixels`: a picture over `limit` pixels downscaled with Pillow's
    /// bicubic filter to `int(side · sqrt(limit / pixels))` per side.
    public static func capPixels(_ image: RGBImage, limit: Int = D1ChatPrompt.maximumPixels) -> RGBImage {
        let pixels = image.width * image.height
        guard pixels > limit else { return image }
        let scale = (Double(limit) / Double(pixels)).squareRoot()
        let width = max(1, Int(Double(image.width) * scale))
        let height = max(1, Int(Double(image.height) * scale))
        return RGBImage(width: width, height: height, pixels: AntialiasedResize.resize(
            image.pixels, width: image.width, height: image.height, channels: 3,
            toWidth: width, toHeight: height, filter: .cubic, kernel: .pillow
        ))
    }

    /// A text state as is; fields as `json.dumps(…, indent=2, ensure_ascii=False)`;
    /// no state writes no block.
    static func stateBlock(_ state: SystemOneState) -> String {
        switch state {
        case .media:
            return ""
        case .text(let text):
            return text + "\n\n"
        case .fields(let fields):
            guard !fields.isEmpty else { return "{}\n\n" }
            let lines = fields.map { "  \(PythonSemantics.jsonString($0.key)): \(PythonSemantics.jsonString($0.value))" }
            return "{\n" + lines.joined(separator: ",\n") + "\n}\n\n"
        }
    }

    /// The question and the assistant header, up to the answer slot. `codes`
    /// are the choice's option codes.
    public static func suffix(question: Question, codes: [String]) -> String {
        questionBlock(question, codes: codes) + "<|im_end|>\n<|im_start|>assistant\n"
    }

    static func questionBlock(_ question: Question, codes: [String]) -> String {
        switch question {
        case .choice(let choice):
            let lines = zip(codes, choice.criteria).map { code, option -> String in
                let text = option.description.flatMap { $0.isEmpty ? nil : $0 } ?? option.key.replacingOccurrences(of: "_", with: " ")
                return "\(code) \(text)"
            }
            return "\(choice.instructions)\n\nOptions:\n\(lines.joined(separator: "\n"))\n\nReply with the option code only."
        case .noul(let noul):
            var extra = ""
            if let criteria = noul.criteria, criteria.trueDescription != nil || criteria.falseDescription != nil {
                // Python's f-string prints a missing description as `None`.
                extra = "\nYes: \(criteria.trueDescription ?? "None")\nNo: \(criteria.falseDescription ?? "None")"
            }
            return "\(noul.instructions)\(extra)\n\nReply with yes or no only."
        case .score(let score):
            let legend = score.criteria.enumerated().map { "\($0.offset) \(pythonString($0.element))" }
            return "\(score.instructions)\n\n\(legend.joined(separator: "\n"))\n\nReply with a single digit 0-\(score.criteria.count - 1) only."
        }
    }

    /// Native letters when every label is one letter, else `A`…`Z`, else the
    /// checkpoint's `d1_codes` scheme: `00`… (1, d1-3B) or `AA`, `AB`… (2, d1-2B).
    static func optionCodes(_ labels: [String], scheme: Int = 1) -> [String] {
        let stripped = labels.map(PythonSemantics.pythonStrip)
        if !stripped.isEmpty, stripped.allSatisfy({ $0.unicodeScalars.count == 1 && $0.first!.isLetter }) {
            return stripped
        }
        let letters = (65...90).map { String(UnicodeScalar(UInt8($0))) }
        if labels.count <= 26 { return Array(letters.prefix(labels.count)) }
        if scheme == 2 { return Array((letters + letters.flatMap { a in letters.map { a + $0 } }).prefix(labels.count)) }
        return labels.indices.map { String(format: "%02d", $0) }
    }

    /// A distinct single-token code and its token for every label.
    static func aliases(_ labels: [String], tokenizer: DecoderTokenizer, scheme: Int = 1) throws -> [(code: String, token: Int)] {
        var used = Set<Int>()
        var out: [(code: String, token: Int)] = []
        func take(_ raw: String) -> Bool {
            let ids = tokenizer.encode(raw)
            guard ids.count == 1, !used.contains(ids[0]) else { return false }
            out.append((raw, ids[0]))
            used.insert(ids[0])
            return true
        }
        for code in optionCodes(labels, scheme: scheme) where !take(code) {
            guard fallbackPool.contains(where: take) else { throw D1ChatError.noAlias(options: labels.count) }
        }
        return out
    }

    /// The single-token encodings among `texts`, without repeats.
    static func singleTokens(_ texts: [String], tokenizer: DecoderTokenizer) -> [Int] {
        var out: [Int] = []
        for text in texts {
            let ids = tokenizer.encode(text)
            if ids.count == 1, !out.contains(ids[0]) { out.append(ids[0]) }
        }
        return out
    }

    /// The token groups scored for each option (max-pooled), and the choice codes.
    public static func readout(_ question: Question, tokenizer: DecoderTokenizer, scheme: Int = 1) throws -> (groups: [[Int]], codes: [String]) {
        switch question {
        case .noul:
            let yes = singleTokens(yesForms, tokenizer: tokenizer), no = singleTokens(noForms, tokenizer: tokenizer)
            guard !yes.isEmpty, !no.isEmpty else { throw D1ChatError.noYesNoTokens }
            return ([yes, no], [])
        case .score(let score):
            let groups = score.criteria.indices.map { singleTokens([String($0)], tokenizer: tokenizer) }
            guard groups.allSatisfy({ !$0.isEmpty }) else { throw D1ChatError.noDigitTokens(levels: score.criteria.count) }
            return (groups, [])
        case .choice(let choice):
            let codes = try aliases(choice.criteria.map(\.key), tokenizer: tokenizer, scheme: scheme)
            let groups = codes.map { code in
                [code.token] + singleTokens([" \(code.code)"], tokenizer: tokenizer).filter { $0 != code.token }
            }
            return (groups, codes.map(\.code))
        }
    }

    /// Python's `str()` of a score level: the text, or the dict as Python prints it.
    static func pythonString(_ level: ScoreLevel) -> String {
        switch level {
        case .text(let text):
            return text
        case .described(let what, let examples):
            return "{'what': \(pythonRepr(what)), 'examples': [\(examples.map(pythonRepr).joined(separator: ", "))]}"
        }
    }

    /// CPython `repr(str)` for printable text.
    static func pythonRepr(_ text: String) -> String {
        let quote: Character = text.contains("'") && !text.contains("\"") ? "\"" : "'"
        var out = String(quote)
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where Character(scalar) == quote: out += "\\\(quote)"
            case _ where scalar.value < 0x20 || scalar.value == 0x7F: out += String(format: "\\x%02x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + String(quote)
    }

    /// `api.answer`: a noul's P(yes); a choice's first best option; a score's
    /// expected level. Confidence is the best option's probability.
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
            return .score(ScoreAnswer(
                score: p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element },
                confidence: p[best],
                legend: score.criteria.enumerated().map { LegendEntry(key: String($0.offset), description: pythonString($0.element)) },
                probabilities: p.enumerated().map { ProbabilityEntry(key: String($0.offset), probability: $0.element) }
            ))
        }
    }
}

public enum D1ChatError: Error, Equatable, Sendable {
    case noAlias(options: Int)
    case noYesNoTokens
    case noDigitTokens(levels: Int)
    /// The prompt's image tokens do not match its pictures' features.
    case imagePlaceholders(expected: Int, found: Int)
    /// The prompt's audio tokens do not match its clip's features.
    case audioPlaceholders(expected: Int, found: Int)
}

extension D1ChatError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .noAlias(let options): return "No single-token code is left for \(options) options."
        case .noYesNoTokens: return "The tokenizer has no single-token yes or no."
        case .noDigitTokens(let levels): return "A score with \(levels) levels needs single-token digits."
        case .imagePlaceholders(let expected, let found):
            return "The prompt holds \(found) image tokens where its pictures fill \(expected); text cannot contain the image token."
        case .audioPlaceholders(let expected, let found):
            return "The prompt holds \(found) audio tokens where its clip fills \(expected); text cannot contain the audio token."
        }
    }
}

/// One question's tokens after its request's shared prefix, and the token
/// groups that score its options.
public struct D1ChatRow: PlannedRow, Equatable, Sendable {
    /// The request's state tokens (with BOS), read once; empty for a lone question.
    public let prefix: [Int]
    public let ids: [Int]
    public let groups: [[Int]]
    /// The capped pictures whose features fill the image tokens: those of
    /// `prefix`, or of `ids` when there is no prefix.
    public let images: [RGBImage]

    public var sequenceLength: Int { ids.count }
    public var logitCount: Int { groups.count }
}

/// d1-3B (`LiquidAI/d1-3B`, LFM2.5-VL-3B) as a System One backend: the causal
/// LFM2 trunk read at the answer slot through its tied embeddings.
///
/// A lone question is one pass over its whole prompt. Several questions share
/// their state: it is prefilled once and every question continues it, which
/// is the reference's tree (`hybrid.py`) evaluated as prefix and branches.
/// Pictures (`SystemOneMedia.images`) go through LFM2-VL's processor, SigLIP2
/// tower, and projector, and their features fill the prompt's `<image>`
/// tokens; audio is not supported.
public final class D1ChatBackend: DecisionBackend {
    public let modelID: String
    let tokenizer: DecoderTokenizer
    let bos: Int?
    /// The image processor's settings; `nil` for a text-only checkpoint.
    let visionConfiguration: LFM2VLVisionConfiguration?
    private let trunk: LFM2Trunk?
    private let vision: LFM2VLVision?
    private let lock = NSLock()
    private var prefixCache: (ids: [Int], images: [RGBImage], cache: DecoderCache)?

    /// Loads a d1-3B folder: `config.json`, the weights, the tokenizer, and,
    /// when the config has a `vision_config`, the image path and `processor_config.json`.
    public convenience init(modelFolder: URL, modelID: String = "d1-3B", dataType: DType = .bfloat16) throws {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
        let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let bos = config["bos_token_id"] as? Int ?? (config["text_config"] as? [String: Any])?["bos_token_id"] as? Int
        let store = try SafetensorsCheckpoint(modelFolder: modelFolder)
        let visionConfiguration = try config["vision_config"] == nil ? nil : LFM2VLVisionConfiguration(modelFolder: modelFolder)
        let vision = try visionConfiguration.map { try LFM2VLVision(store: store, configuration: $0, dataType: dataType) }
        self.init(
            modelID: modelID, tokenizer: try HubDecoderTokenizer(folder: modelFolder), bos: bos,
            trunk: try LFM2Trunk(store: store, configuration: LFM2Configuration(modelFolder: modelFolder), dataType: dataType),
            visionConfiguration: visionConfiguration, vision: vision
        )
    }

    init(
        modelID: String, tokenizer: DecoderTokenizer, bos: Int?, trunk: LFM2Trunk?,
        visionConfiguration: LFM2VLVisionConfiguration? = nil, vision: LFM2VLVision? = nil
    ) {
        self.modelID = modelID
        self.tokenizer = tokenizer
        self.bos = bos
        self.trunk = trunk
        self.visionConfiguration = vision?.configuration ?? visionConfiguration
        self.vision = vision
    }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<D1ChatRow> {
        var images: [RGBImage] = []
        var markup = ""
        var placeholders = 0
        if let media {
            guard case .images(let pictures) = media, let configuration = visionConfiguration else {
                throw SystemOneEngineError.unsupportedMedia(model: modelID)
            }
            images = pictures.map { D1ChatPrompt.capPixels($0) }
            for image in images {
                let layout = LFM2VLVision.layout(width: image.width, height: image.height, configuration: configuration)
                markup += LFM2VLVision.markup(layout, configuration: configuration)
                placeholders += LFM2VLVision.featureCount(layout, configuration: configuration)
            }
        }
        let head = bos.map { [$0] } ?? []
        let prefixText = D1ChatPrompt.prefix(state: request.state, images: markup)
        let lone = request.questions.count == 1
        let prefix = lone ? [] : head + tokenizer.encode(prefixText)
        var rows: [D1ChatRow] = []
        var decisions: [PlannedDecision] = []
        for entry in request.questions {
            let question = entry.question
            if case .score(let score) = question, !(2...10).contains(score.criteria.count) {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: "a score needs 2 to 10 levels")
            }
            let readout: (groups: [[Int]], codes: [String])
            do {
                readout = try D1ChatPrompt.readout(question, tokenizer: tokenizer)
            } catch let error as D1ChatError {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: error.localizedDescription)
            }
            let suffix = D1ChatPrompt.suffix(question: question, codes: readout.codes)
            let ids = lone ? head + tokenizer.encode(prefixText + suffix) : tokenizer.encode(suffix)
            decisions.append(PlannedDecision(id: entry.id, rowIndices: [rows.count]) { logits in
                let top = logits[0].max() ?? 0
                let weights = logits[0].map { Foundation.exp(Double($0 - top)) }
                let total = weights.reduce(0, +)
                return D1ChatPrompt.answer(question, probabilities: weights.map { $0 / total })
            })
            rows.append(D1ChatRow(prefix: prefix, ids: ids, groups: readout.groups, images: images))
        }
        // The prompt's image tokens must be exactly those the pictures fill.
        if !images.isEmpty, let token = visionConfiguration?.imageTokenID, let carrier = lone ? rows.first?.ids : prefix {
            let found = carrier.filter { $0 == token }.count
            let stray = lone ? 0 : rows.reduce(0) { total, row in total + row.ids.filter { $0 == token }.count }
            guard found == placeholders, stray == 0 else {
                throw D1ChatError.imagePlaceholders(expected: placeholders, found: found + stray)
            }
        }
        let tokens = prefix.count + rows.reduce(0) { $0 + $1.ids.count }
        return DecisionPlan(rows: rows, decisions: decisions, usage: Usage(inputTokens: tokens))
    }

    public func logits(for rows: [D1ChatRow]) async throws -> BatchLogitsOutput {
        let perRow = try optionLogits(for: rows)
        var offsets = [0]
        for row in perRow { offsets.append(offsets[offsets.count - 1] + row.count) }
        return BatchLogitsOutput(logits: perRow.flatMap { $0 }, rowOffsets: offsets)
    }

    /// Each row's option scores: the best logit among each group's tokens at
    /// the row's last token.
    func optionLogits(for rows: [D1ChatRow]) throws -> [[Float]] {
        guard let trunk else { preconditionFailure("D1ChatBackend has no model") }
        var result = [[Float]](repeating: [], count: rows.count)
        // Rows sharing a prefix and pictures run together, in first-seen order.
        var groups: [[Int]] = []
        for (index, row) in rows.enumerated() {
            if let group = groups.firstIndex(where: { rows[$0[0]].prefix == row.prefix && rows[$0[0]].images == row.images }) {
                groups[group].append(index)
            } else {
                groups.append([index])
            }
        }
        for members in groups {
            let first = rows[members[0]]
            let cache = try first.prefix.isEmpty ? nil : prefixState(first.prefix, images: first.images, trunk: trunk)
            // Without a prefix each row holds its own pictures' tokens.
            var rowFeatures: DecoderFeatures?
            if first.prefix.isEmpty, !first.images.isEmpty {
                let value = try imageFeatures(first.images)
                rowFeatures = DecoderFeatures(token: imageToken, rows: Array(repeating: value, count: members.count))
            }
            let lengths = members.map { rows[$0].ids.count }
            let width = lengths.max() ?? 0
            let hidden = try trunk(
                members.map { rows[$0].ids + Array(repeating: 0, count: width - rows[$0].ids.count) },
                attention: DecoderAttention(lengths: lengths), prefix: cache, features: rowFeatures
            )
            let last = MLXArray(lengths.enumerated().map { Int32($0.offset * width + $0.element - 1) })
            let states = hidden.reshaped(-1, hidden.dim(-1)).take(last, axis: 0).asType(.float32)
            let tokens = Array(Set(members.flatMap { rows[$0].groups.joined() })).sorted()
            let column = Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($1, $0) })
            let scores = matmul(states, try trunk.readoutRows(tokens).asType(.float32).transposed()).asArray(Float.self)
            for (position, index) in members.enumerated() {
                let base = position * tokens.count
                result[index] = rows[index].groups.map { group in
                    group.map { scores[base + column[$0]!] }.max() ?? -.infinity
                }
            }
        }
        return result
    }

    /// The image token id the pictures' features replace.
    private var imageToken: Int { visionConfiguration?.imageTokenID ?? -1 }

    /// `[tokens, hidden]` features of `images`, one after another.
    private func imageFeatures(_ images: [RGBImage]) throws -> MLXArray {
        guard let vision else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
        let features = images.map { vision($0) }
        return features.count == 1 ? features[0] : concatenated(features, axis: 0)
    }

    /// The prefilled state of `prefix` and its pictures, kept for the next
    /// microbatch of the same request.
    private func prefixState(_ prefix: [Int], images: [RGBImage], trunk: LFM2Trunk) throws -> DecoderCache {
        lock.lock()
        defer { lock.unlock() }
        if let cached = prefixCache, cached.ids == prefix, cached.images == images { return cached.cache }
        let cache = DecoderCache()
        let placed = try images.isEmpty ? nil : DecoderFeatures(token: imageToken, rows: [imageFeatures(images)])
        try trunk.prefill(prefix, cache: cache, features: placed)
        prefixCache = (prefix, images, cache)
        return cache
    }
}

extension SystemOneEngine {
    /// d1-3B from a local folder.
    public static func d1Chat(modelFolder: URL, dataType: DType = .bfloat16) throws -> SystemOneEngine {
        SystemOneEngine(backend: try D1ChatBackend(modelFolder: modelFolder, dataType: dataType))
    }
}
