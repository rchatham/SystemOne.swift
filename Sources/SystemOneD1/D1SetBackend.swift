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
    /// Pixels a picture may have before the processor (`d1_vision_max_pixels`).
    public let visionMaxPixels: Int

    public init(
        setSize: Int = 4, codeScheme: Int = 1, bos: String = "", specialTokens: [String] = [],
        defaultTemperature: Double = 1, temperatures: [String: Double] = [:], visionMaxPixels: Int = 1024 * 1024
    ) {
        self.setSize = setSize
        self.codeScheme = codeScheme
        self.bos = bos
        self.specialTokens = specialTokens
        self.defaultTemperature = defaultTemperature
        self.temperatures = temperatures
        self.visionMaxPixels = visionMaxPixels
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
            temperatures: (calibration["temperatures"] as? [String: Any] ?? [:]).compactMapValues { ($0 as? NSNumber)?.doubleValue },
            visionMaxPixels: config["d1_vision_max_pixels"] as? Int ?? 1024 * 1024
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
    /// A spoken question's place in its suffix until its clip's markup replaces it (`media.SPOKEN`).
    static let spokenPlaceholder = "\u{E000}"

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

    /// The row's head: BOS, the user turn, the pictures' markup (`images`, not
    /// escaped), the state, and the first `QUESTION:`.
    public static func prefix(state: SystemOneState, configuration: D1SetConfiguration, images: String = "") -> String {
        configuration.bos + "<|im_start|>user\n" + images
            + plain(D1ChatPrompt.stateBlock(state), specialTokens: configuration.specialTokens) + "\n" + header
    }

    /// The question and the assistant header up to the answer slot. A spoken
    /// question's `spoken` markup (not escaped) sits where its instructions would.
    public static func suffix(
        question: Question, codes: [String], configuration: D1SetConfiguration, spoken: String? = nil
    ) -> String {
        guard let spoken else {
            return plain(D1ChatPrompt.questionBlock(question, codes: codes), specialTokens: configuration.specialTokens)
                + "<|im_end|>\n<|im_start|>assistant\n" + lead
        }
        return suffix(question: question.withInstructions(spokenPlaceholder), codes: codes, configuration: configuration)
            .replacingOccurrences(of: spokenPlaceholder, with: spoken)
    }

    /// `check_question` for noul, choice, and score. A spoken question's clip
    /// takes the place of its instructions, which stay empty.
    static func validate(_ question: Question, spoken: Bool = false) -> String? {
        if spoken {
            guard question.instructions.isEmpty else {
                return "a spoken question's clip is its instructions; leave \"instructions\" empty"
            }
        } else {
            guard !PythonSemantics.pythonStrip(question.instructions).isEmpty else {
                return "\"instructions\" is the question as text"
            }
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
    /// The capped pictures whose features fill the state's `<|image|>` tokens.
    public let images: [RGBImage]
    /// The clip whose features fill the state's `<|audio|>` tokens.
    public let audio: AudioClip?
    /// The clips of the row's spoken questions, in question order; their
    /// features fill the `<|audio|>` tokens after the state's.
    public let voices: [AudioClip]

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
/// and score questions, under its `set` attention: up to `d1_set_size`
/// questions share one row after the state, read at their own slots through
/// the tied head, with `calibration.json`'s temperature.
///
/// Pictures (`SystemOneMedia.images`) are capped at `d1_vision_max_pixels`
/// (Pillow's bicubic), written before the state as
/// `<|vision_start|><|image|>…<|vision_end|>`, and encoded once by LFM3-VL's
/// tower; their features fill the state's `<|image|>` tokens in every row.
///
/// With an `audio_encoder_config` (d1-Omni-2B, `LiquidAI/d1-Omni-2B`), a clip
/// (`SystemOneMedia.audio`) is resampled to 16 kHz, written before the state
/// as `<|vision_start|><|audio|>…<|vision_end|>`, and encoded once by
/// LFM3-Omni's audio encoder; its features fill the `<|audio|>` tokens.
/// A spoken question (`QuestionEntry.spoken`) has its clip's markup where its
/// instructions would be, and its features fill those `<|audio|>` tokens.
/// Video, a spoken state with no text, and the typed questions (multiple
/// choice, extraction, estimates, boxes) are not supported yet.
public final class D1SetBackend: DecisionBackend {
    public let modelID: String
    public let configuration: D1SetConfiguration
    let tokenizer: DecoderTokenizer
    /// The image processor's settings; `nil` for a text-only checkpoint.
    let visionConfiguration: LFM3VLVisionConfiguration?
    /// The audio encoder's and feature extractor's settings; `nil` without speech.
    let audioConfiguration: LFM3AudioConfiguration?
    private let trunk: LFM3Trunk?
    private let vision: LFM3VLVision?
    private let audio: LFM3AudioEncoder?
    private let lock = NSLock()
    /// The last media encoded, kept for the next microbatch of the same request.
    private var featureCache: (media: Media, features: MLXArray)?

    /// A row's media, as encoded together.
    private enum Media: Equatable {
        case images([RGBImage])
        case audio(AudioClip)
    }

    public convenience init(modelFolder: URL, modelID: String = "d1-2B", dataType: DType = .bfloat16) throws {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
        let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let store = try SafetensorsCheckpoint(modelFolder: modelFolder)
        let visionConfiguration = try config["vision_config"] == nil ? nil : LFM3VLVisionConfiguration(modelFolder: modelFolder)
        let vision = try visionConfiguration.map { try LFM3VLVision(store: store, configuration: $0, dataType: dataType) }
        let audioConfiguration = try config["audio_encoder_config"] == nil ? nil : LFM3AudioConfiguration(modelFolder: modelFolder)
        let audio = try audioConfiguration.map { try LFM3AudioEncoder(store: store, configuration: $0, dataType: dataType) }
        self.init(
            modelID: modelID,
            configuration: try D1SetConfiguration(modelFolder: modelFolder),
            tokenizer: try HubDecoderTokenizer(folder: modelFolder),
            trunk: try LFM3Trunk(store: store, configuration: LFM3Configuration(modelFolder: modelFolder), dataType: dataType),
            visionConfiguration: visionConfiguration, vision: vision,
            audioConfiguration: audioConfiguration, audio: audio
        )
    }

    init(
        modelID: String, configuration: D1SetConfiguration, tokenizer: DecoderTokenizer, trunk: LFM3Trunk?,
        visionConfiguration: LFM3VLVisionConfiguration? = nil, vision: LFM3VLVision? = nil,
        audioConfiguration: LFM3AudioConfiguration? = nil, audio: LFM3AudioEncoder? = nil
    ) {
        self.modelID = modelID
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.trunk = trunk
        self.visionConfiguration = vision?.configuration ?? visionConfiguration
        self.vision = vision
        self.audioConfiguration = audio?.configuration ?? audioConfiguration
        self.audio = audio
    }

    /// A checkpoint with an audio encoder (d1-Omni-2B) hears spoken questions.
    public var hearsSpokenQuestions: Bool { audioConfiguration != nil }

    public func plan(_ request: SystemOneRequest, media: SystemOneMedia?) throws -> DecisionPlan<D1SetRow> {
        var images: [RGBImage] = []
        var clip: AudioClip?
        var markup = ""
        var placeholders = 0
        switch media {
        case nil:
            break
        case .images(let pictures)?:
            guard let settings = visionConfiguration else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
            images = pictures.map { D1ChatPrompt.capPixels($0, limit: configuration.visionMaxPixels) }
            for image in images {
                let size = try LFM3VLVision.smartResize(height: image.height, width: image.width, configuration: settings)
                let count = LFM3VLVision.featureCount(height: size.height, width: size.width, configuration: settings)
                markup += LFM3VLVision.markup(featureCount: count)
                placeholders += count
            }
        case .audio(let value)?:
            guard let settings = audioConfiguration else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
            placeholders = try LFM3AudioEncoder.featureCount(of: value, configuration: settings)
            markup = LFM3AudioEncoder.markup(featureCount: placeholders)
            clip = value
        }
        let stateAudio = placeholders
        var suffixes: [String] = []
        var readouts: [[[Int]]] = []
        // Per question: its clip's `<|audio|>` count, 0 when written.
        var voiceCounts: [Int] = []
        for entry in request.questions {
            if let reason = D1SetPrompt.validate(entry.question, spoken: entry.spoken != nil) {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: reason)
            }
            var voice: String?
            var voiceCount = 0
            if let spoken = entry.spoken {
                guard let settings = audioConfiguration else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
                voiceCount = try LFM3AudioEncoder.featureCount(of: spoken, configuration: settings)
                voice = LFM3AudioEncoder.markup(featureCount: voiceCount)
            }
            voiceCounts.append(voiceCount)
            do {
                let readout = try D1ChatPrompt.readout(entry.question, tokenizer: tokenizer, scheme: configuration.codeScheme)
                readouts.append(readout.groups)
                suffixes.append(D1SetPrompt.suffix(
                    question: entry.question, codes: readout.codes, configuration: configuration, spoken: voice
                ))
            } catch let error as D1ChatError {
                throw SystemOneEngineError.invalidQuestion(id: entry.id, reason: error.localizedDescription)
            }
        }
        let prefix = D1SetPrompt.prefix(state: request.state, configuration: configuration, images: markup)
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
            // Every row repeats the state, and with it each picture's or the clip's tokens.
            if let token = visionConfiguration?.imageTokenID, !images.isEmpty {
                let found = ids.filter { $0 == token }.count
                guard found == placeholders else {
                    throw D1ChatError.imagePlaceholders(expected: placeholders, found: found)
                }
            }
            // The state's clip, then each spoken question's, in row order.
            let heard = (clip == nil ? 0 : stateAudio) + members.reduce(0) { $0 + voiceCounts[$1] }
            if let token = audioConfiguration?.audioTokenID, heard > 0 {
                let found = ids.filter { $0 == token }.count
                guard found == heard else {
                    throw D1ChatError.audioPlaceholders(expected: heard, found: found)
                }
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
                groups: members.map { readouts[$0] }, images: images, audio: clip,
                voices: members.compactMap { request.questions[$0].spoken }
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
            attention: DecoderAttention(lengths: lengths, plan: .visible(rows.map(\.visibility))),
            features: try mediaFeatures(rows)
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

    /// Each row's media features, every picture set or clip encoded once;
    /// `nil` when no row has media.
    private func mediaFeatures(_ rows: [D1SetRow]) throws -> DecoderFeatures? {
        guard rows.contains(where: { !$0.images.isEmpty || $0.audio != nil || !$0.voices.isEmpty }) else { return nil }
        let imageToken = visionConfiguration?.imageTokenID
        let audioToken = audioConfiguration?.audioTokenID
        var encoded: [(media: Media, features: MLXArray)] = []
        func features(_ media: Media, cached: Bool) throws -> MLXArray {
            if let known = encoded.first(where: { $0.media == media }) { return known.features }
            let value = try encode(media, cached: cached)
            encoded.append((media: media, features: value))
            return value
        }
        let placed = try rows.map { row -> [Int: MLXArray] in
            var filled: [Int: MLXArray] = [:]
            if !row.images.isEmpty {
                guard let imageToken else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
                filled[imageToken] = try features(.images(row.images), cached: true)
            }
            // The state's clip, then each spoken question's: the order their `<|audio|>` tokens run.
            var heard = try row.audio.map { [try features(.audio($0), cached: true)] } ?? []
            heard += try row.voices.map { try features(.audio($0), cached: false) }
            if !heard.isEmpty {
                guard let audioToken else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
                filled[audioToken] = heard.count == 1 ? heard[0] : concatenated(heard, axis: 0)
            }
            return filled
        }
        return DecoderFeatures(tokens: [imageToken, audioToken].compactMap { $0 }, rows: placed)
    }

    /// `media`'s features (pictures one after another). A request's state
    /// media (`cached`) is kept for its next microbatch; a spoken question's
    /// clip sits in one row only.
    private func encode(_ media: Media, cached: Bool) throws -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let known = featureCache, known.media == media { return known.features }
        let value: MLXArray
        switch media {
        case .images(let images):
            guard let vision else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
            let parts = try images.map { try vision($0) }
            value = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
        case .audio(let clip):
            guard let audio else { throw SystemOneEngineError.unsupportedMedia(model: modelID) }
            value = try audio(clip)
        }
        if cached { featureCache = (media: media, features: value) }
        return value
    }
}

extension Question {
    /// The same question with other instructions.
    func withInstructions(_ instructions: String) -> Question {
        switch self {
        case .choice(let choice): return .choice(Choice(instructions: instructions, criteria: choice.criteria))
        case .noul(let noul): return .noul(Noul(instructions: instructions, criteria: noul.criteria))
        case .score(let score): return .score(Score(instructions: instructions, criteria: score.criteria))
        }
    }
}

extension SystemOneEngine {
    /// d1-2B (or d1-Omni-2B, which adds audio) from a local folder.
    public static func d1Set(modelFolder: URL, dataType: DType = .bfloat16) throws -> SystemOneEngine {
        SystemOneEngine(backend: try D1SetBackend(modelFolder: modelFolder, dataType: dataType))
    }
}
