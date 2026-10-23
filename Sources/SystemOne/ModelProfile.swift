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

    public var modelsJSON: String {
        let id = PythonSemantics.jsonString(modelID)
        return #"{"object":"list","data":[{"id":\#(id),"object":"model","owned_by":"system-one"}]}"#
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
