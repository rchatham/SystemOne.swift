import Foundation

/// The text one question renders to before tokenization.
public struct PromptSketch: Equatable, Sendable {
    public let instructions: String
    /// Option texts in the model's order.
    public let options: [String]
    /// The packed text of each forward pass the question plans.
    public let forwards: [String]

    public init(instructions: String, options: [String], forwards: [String]) {
        self.instructions = instructions
        self.options = options
        self.forwards = forwards
    }
}

/// What a trust boundary needs to know about a model without loading it:
/// how it renders prompts, the literal marker callers must not inject, its
/// hard per-forward size, and how it describes itself to clients.
public protocol ModelProfile: Sendable {
    var modelID: String { get }

    /// The `model` assumed when a request omits one.
    var defaultModelAlias: String { get }

    /// Literal text that marks options in packed prompts. `nil` when the
    /// model escapes caller text, so injection is impossible.
    var markerText: String? { get }

    /// Upper bound on one packed forward's UTF-8 bytes, from the model's context.
    var maximumForwardUTF8Bytes: Int { get }

    /// The most options one question may list: the model's own limit (64 for
    /// von and d1-omni, 255 for most decoders and for the TypeSafe and Decision
    /// Index schemas).
    var maximumOptionsPerQuestion: Int { get }

    /// The model's description and release date in `GET /v1/models`.
    var modelDescription: String { get }
    var releaseDate: String? { get }

    func renderState(_ state: SystemOneState) -> String
    func sketch(_ question: Question, renderedState: String) -> PromptSketch

    /// Body of `GET /health`.
    var healthJSON: String { get }

    /// Body of `GET /v1/models`.
    var modelsJSON: String { get }
}

extension ModelProfile {
    public var defaultModelAlias: String { modelID }

    public var healthJSON: String {
        #"{"status":"ok","service":"system-one","engine":\#(PythonSemantics.jsonString(modelID))}"#
    }

    public var maximumOptionsPerQuestion: Int { 64 }
    public var modelDescription: String { "System One decision model" }
    public var releaseDate: String? { nil }

    /// Both shapes clients read: TypeSafe's `models` (name, description,
    /// release date) and OpenAI's `data`.
    public var modelsJSON: String {
        let id = PythonSemantics.jsonString(modelID)
        let description = PythonSemantics.jsonString(modelDescription)
        let date = releaseDate.map(PythonSemantics.jsonString) ?? "null"
        return #"{"models":[{"name":\#(id),"description":\#(description),"release_date":\#(date)}],"#
            + #""object":"list","data":[{"id":\#(id),"object":"model","owned_by":"system-one"}]}"#
    }
}

extension Question {
    /// The instructions shared by every question type.
    public var instructions: String {
        switch self {
        case .choice(let choice): return choice.instructions
        case .noul(let noul): return noul.instructions
        case .score(let score): return score.instructions
        }
    }
}

/// The profile of a decision backend that renders its own prompt (d3, d1-3B,
/// the encoder backends): the wire limits come from the model's option cap and
/// context, and the budget sketch is the state, instructions, and option texts
/// as plain text. Such models tokenize caller text as written, so there is no
/// marker to guard.
public struct DecisionModelProfile: ModelProfile {
    public let modelID: String
    public let maximumForwardUTF8Bytes: Int
    public let maximumOptionsPerQuestion: Int
    public let modelDescription: String
    public let releaseDate: String?

    public init(
        modelID: String, maximumOptionsPerQuestion: Int = 255, maximumForwardUTF8Bytes: Int = 1 << 20,
        modelDescription: String = "System One decision model", releaseDate: String? = nil
    ) {
        self.modelID = modelID
        self.maximumOptionsPerQuestion = maximumOptionsPerQuestion
        self.maximumForwardUTF8Bytes = maximumForwardUTF8Bytes
        self.modelDescription = modelDescription
        self.releaseDate = releaseDate
    }

    public var markerText: String? { nil }

    /// A text state as is; fields as a JSON object.
    public func renderState(_ state: SystemOneState) -> String {
        switch state {
        case .text(let text):
            return text
        case .fields(let fields):
            return "{" + fields.map { "\(PythonSemantics.jsonString($0.key)): \(PythonSemantics.jsonString($0.value))" }
                .joined(separator: ", ") + "}"
        }
    }

    public func sketch(_ question: Question, renderedState: String) -> PromptSketch {
        let options: [String]
        switch question {
        case .choice(let choice):
            options = choice.criteria.map { option in option.description.map { "\(option.key): \($0)" } ?? option.key }
        case .noul(let noul):
            options = ["false: \(noul.criteria?.falseDescription ?? "")", "true: \(noul.criteria?.trueDescription ?? "")"]
        case .score(let score):
            options = score.criteria.map { level in
                switch level {
                case .text(let text): return text
                case .described(let what, let examples): return ([what] + examples).joined(separator: " ")
                }
            }
        }
        let packed = ([renderedState, question.instructions] + options).joined(separator: "\n")
        return PromptSketch(instructions: question.instructions, options: options, forwards: [packed])
    }
}
