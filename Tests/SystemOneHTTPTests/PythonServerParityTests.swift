import Foundation
import SystemOne
import XCTest

@testable import SystemOneHTTP

final class PythonServerParityTests: XCTestCase {
    private struct HTTPResult {
        let status: Int
        let body: Data
    }

    func testPinnedPythonServerParityWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["VON_HTTP_PARITY"] == "1" else {
            throw XCTSkip("Set VON_HTTP_PARITY=1 to run live Python/Swift HTTP parity.")
        }
        let pythonBaseURL = try requiredLoopbackURL(environment["VON_PYTHON_SERVER_URL"])
        XCTAssertNotEqual(
            pythonBaseURL.port, 8_000, "HTTP parity must not use the workspace-assigned port 8000")
        let weightsURL = try requiredRegularFile(
            environment["VON_PRODUCTION_WEIGHTS_PATH"], name: "VON_PRODUCTION_WEIGHTS_PATH")
        let tokenizerURL = try tokenizerResourceURL(environment["VON_TOKENIZER_PATH"])
        guard let apiKey = environment["VON_API_KEY"], !apiKey.isEmpty else {
            throw XCTSkip(
                "Set a non-empty VON_API_KEY on both servers to cover authentication parity.")
        }

        let pythonHealth = try await request(pythonBaseURL.appendingPathComponent("health"))
        XCTAssertEqual(pythonHealth.status, 200, "The external pinned Python server is not healthy")

        let engine = try await SystemOneEngine.von(tokenizerFolder: tokenizerURL, verifiedWeightsURL: weightsURL)
        let swiftServer = SystemOneHTTPServer(
            configuration: .init(
                host: "127.0.0.1",
                port: 0,
                apiKey: apiKey,
                cors: .disabled,
                requestDeadline: .seconds(120)
            )
        ) { request in
            try await engine.systemOne(request)
        }
        try await swiftServer.start()
        let swiftPort = try XCTUnwrap(swiftServer.port)
        XCTAssertNotEqual(swiftPort, 8_000)
        let swiftBaseURL = try XCTUnwrap(URL(string: "http://127.0.0.1:\(swiftPort)/"))

        do {
            for route in ["", "health", "v1/models"] {
                let python = try await request(pythonBaseURL.appendingPathComponent(route))
                let swift = try await request(swiftBaseURL.appendingPathComponent(route))
                XCTAssertEqual(swift.status, python.status, "GET /\(route) status")
                XCTAssertEqual(swift.body, python.body, "GET /\(route) raw JSON")
            }

            let successRequests = [
                #"{"model":"von-latest","state":"The service is degraded in Tokyo.","questions":{"route":{"type":"choice","instructions":"Choose the best response.","criteria":{"restart":"Restart the service","observe":"Continue monitoring","escalate":null}},"default_noul":{"type":"noul","instructions":"Is immediate action required?"},"explicit_noul":{"type":"noul","instructions":"Is the incident contained?","criteria":{"true":"The incident is fully contained.","false":"The incident remains active."}},"severity":{"type":"score","instructions":"Rate incident severity.","criteria":["low",{"what":"medium","examples":["partial degradation"]},{"what":"high","examples":["outage","data loss"]}]}}}"#,
                #"{"state":{"region":"Tokyo","status":"degraded","attempts":2},"questions":{"severity":{"type":"score","instructions":"Rate incident severity.","criteria":[{"what":"high","examples":["outage"]},"medium","low"]},"route":{"instructions":"Choose the best response.","criteria":{"escalate":"Escalate now","observe":null,"restart":"Restart safely"}},"contained":{"type":"noul","instructions":"Is the incident contained?","criteria":{"false":"Still active","true":"Fully contained"}}}}"#,
            ]
            let authorization = ["Authorization": "Bearer \(apiKey)"]
            for (index, rawJSON) in successRequests.enumerated() {
                let body = Data(rawJSON.utf8)
                let python = try await request(
                    pythonBaseURL.appendingPathComponent("v1/systemone"),
                    method: "POST",
                    body: body,
                    headers: authorization,
                    timeout: 180
                )
                let swift = try await request(
                    swiftBaseURL.appendingPathComponent("v1/systemone"),
                    method: "POST",
                    body: body,
                    headers: authorization,
                    timeout: 180
                )
                XCTAssertEqual(
                    python.status, 200,
                    "Python success case \(index): \(String(decoding: python.body, as: UTF8.self))")
                XCTAssertEqual(swift.status, python.status, "success case \(index) status")
                XCTAssertEqual(swift.body, python.body, "success case \(index) ordered raw JSON")
            }

            let validBody = Data(successRequests[0].utf8)
            let statusCases:
                [(
                    name: String, method: String, route: String, body: Data?,
                    headers: [String: String]
                )] = [
                    ("missing auth", "POST", "v1/systemone", validBody, [:]),
                    (
                        "invalid auth", "POST", "v1/systemone", validBody,
                        ["Authorization": "Bearer wrong"]
                    ),
                    ("invalid JSON", "POST", "v1/systemone", Data("not json".utf8), authorization),
                    ("missing route", "GET", "missing", nil, [:]),
                    ("wrong method", "GET", "v1/systemone", nil, [:]),
                ]
            for item in statusCases {
                let python = try await request(
                    pythonBaseURL.appendingPathComponent(item.route),
                    method: item.method,
                    body: item.body,
                    headers: item.headers
                )
                let swift = try await request(
                    swiftBaseURL.appendingPathComponent(item.route),
                    method: item.method,
                    body: item.body,
                    headers: item.headers
                )
                XCTAssertEqual(swift.status, python.status, "\(item.name) status-category parity")
            }

            try await swiftServer.shutdown()
        } catch {
            try? await swiftServer.shutdown()
            throw error
        }
    }

    private func request(
        _ url: URL,
        method: String = "GET",
        body: Data? = nil,
        headers: [String: String] = [:],
        timeout: TimeInterval = 10
    ) async throws -> HTTPResult {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = timeout
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        return HTTPResult(status: http.statusCode, body: data)
    }

    private func requiredLoopbackURL(_ value: String?) throws -> URL {
        guard let value, let url = URL(string: value), url.scheme == "http",
            let host = url.host, ["127.0.0.1", "::1", "localhost"].contains(host),
            url.port != nil
        else {
            throw XCTSkip(
                "VON_PYTHON_SERVER_URL must be an explicit loopback http URL with a port.")
        }
        return url
    }

    private func requiredRegularFile(_ value: String?, name: String) throws -> URL {
        guard let value else {
            throw XCTSkip("Set \(name) to the local production safetensors file.")
        }
        let url = URL(fileURLWithPath: value).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            !isDirectory.boolValue
        else {
            throw XCTSkip("\(name) is not a local file: \(url.path)")
        }
        return url
    }

    private func tokenizerResourceURL(_ override: String?) throws -> URL {
        if let override {
            let url = URL(fileURLWithPath: override).standardizedFileURL
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            else {
                throw XCTSkip("VON_TOKENIZER_PATH is not a local folder: \(url.path)")
            }
            return url
        }
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let resource = repository.appendingPathComponent("Tests/SystemOneVonTests/Resources/Tokenizer")
        guard
            FileManager.default.fileExists(
                atPath: resource.appendingPathComponent("tokenizer.json").path)
        else {
            throw XCTSkip("Tokenizer test resources are missing at \(resource.path)")
        }
        return resource
    }
}
