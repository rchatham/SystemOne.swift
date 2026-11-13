import Foundation
import SystemOne

/// Routes through a `system-one serve` endpoint instead of an in-process engine.
///
/// The iOS Simulator lacks the Metal features MLX needs, so a simulator build
/// sends its routing questions to a server running on the Mac, which the
/// simulator reaches at `127.0.0.1`.
public struct HTTPDecider: ActionDecider {
    public let endpoint: URL
    private let apiKey: String?
    private let session: URLSession

    /// - Parameters:
    ///   - endpoint: The server's `/v1/systemone` URL.
    ///   - apiKey: Sent as a bearer token when the server sets `SYSTEMONE_API_KEY`.
    public init(endpoint: URL, apiKey: String? = nil, session: URLSession = .shared) {
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.session = session
    }

    public func chooseAction(
        state: SystemOneState,
        options: [ChoiceOption],
        instructions: String
    ) async throws -> ChoiceAnswer {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try Self.requestBody(state: state, options: options, instructions: instructions)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError {
            throw ChatError.routerUnavailable("\(endpoint.absoluteString): \(error.localizedDescription)")
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            let body = String(decoding: data.prefix(500), as: UTF8.self)
            throw ChatError.routerUnavailable("\(endpoint.absoluteString) answered \(status): \(body)")
        }
        return try Self.decodeAnswer(data, options: options)
    }

    static let questionID = "route"

    /// The server reads JSON objects in order (option order is part of the
    /// question), and `JSONSerialization` does not keep key order, so the body
    /// is assembled from individually encoded strings.
    static func requestBody(
        state: SystemOneState,
        options: [ChoiceOption],
        instructions: String
    ) throws -> Data {
        let encoder = JSONEncoder()
        func json(_ value: String) throws -> String {
            String(decoding: try encoder.encode(value), as: UTF8.self)
        }
        let stateJSON: String
        switch state {
        case .text(let text):
            stateJSON = try json(text)
        case .fields(let fields):
            stateJSON = "{\(try fields.map { "\(try json($0.key)):\(try json($0.value))" }.joined(separator: ","))}"
        case .media:
            // HTTP requests carry no media, and the server reads a null state as "None".
            stateJSON = "null"
        }
        let criteria = try options.map { option in
            "\(try json(option.key)):\(try option.description.map(json) ?? "null")"
        }.joined(separator: ",")
        let question = "{\"type\":\"choice\",\"instructions\":\(try json(instructions)),\"criteria\":{\(criteria)}}"
        return Data("{\"state\":\(stateJSON),\"questions\":{\"\(questionID)\":\(question)}}".utf8)
    }

    private struct Response: Decodable {
        struct Answer: Decodable {
            let choice: String
            let probabilities: [String: Double]
            let confidence: Double
        }

        let answers: [String: Answer]
    }

    /// Reads the choice answer back, restoring the offered order of the probabilities.
    static func decodeAnswer(_ data: Data, options: [ChoiceOption]) throws -> ChoiceAnswer {
        guard let answer = try JSONDecoder().decode(Response.self, from: data).answers[questionID] else {
            throw ChatError.routerUnavailable("The response has no '\(questionID)' answer.")
        }
        return ChoiceAnswer(
            choice: answer.choice,
            probabilities: options.compactMap { option in
                answer.probabilities[option.key].map { ProbabilityEntry(key: option.key, probability: $0) }
            },
            confidence: answer.confidence
        )
    }
}
