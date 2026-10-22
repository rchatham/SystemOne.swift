import Foundation
import NIOCore
import NIOPosix
import VonSwift
import XCTest

@testable import VonHTTP

final class VonHTTPServerTests: XCTestCase {
    private let validRequest =
        #"{"state":"ship","questions":{"route":{"instructions":"Choose","criteria":{"fast":null,"safe":"Low risk"}}}}"#

    func testHealthAndModelsMatchPythonPayloads() async throws {
        try await withServer { baseURL in
            let root = try await self.request(baseURL.appendingPathComponent(""))
            let health = try await self.request(baseURL.appendingPathComponent("health"))
            XCTAssertEqual(root.status, 200)
            XCTAssertEqual(root.body, health.body)
            XCTAssertEqual(
                String(decoding: root.body, as: UTF8.self),
                #"{"status":"ok","service":"von-decision-server","version":"1.1.0","engine":"von-1.1","homage":"John von Neumann & Ludwig von Mises"}"#
            )

            let models = try await self.request(baseURL.appendingPathComponent("v1/models"))
            XCTAssertEqual(models.status, 200)
            XCTAssertEqual(
                String(decoding: models.body, as: UTF8.self),
                #"{"models":[{"name":"von-latest","description":"Current Von System One decision model","release_date":"2026-09-21"},{"name":"von-1.1.0","description":"Von 1.1 stable release","release_date":"2026-09-21"},{"name":"jev-latest","description":"TypeSafe Jev compatibility alias","release_date":"2026-09-21"}],"object":"list","data":[{"id":"von-latest","object":"model","owned_by":"von"},{"id":"von-1.1.0","object":"model","owned_by":"von"},{"id":"jev-latest","object":"model","owned_by":"von"}]}"#
            )
        }
    }

    func testSystemOneUsesOrderedWireCodec() async throws {
        try await withServer(handler: { request in
            XCTAssertEqual(request.questions.map(\.id), ["route"])
            return SystemOneResponse(
                model: "von-1.1.0",
                answers: [
                    .init(
                        id: "route",
                        answer: .choice(
                            .init(
                                choice: "safe",
                                probabilities: [
                                    .init(key: "fast", probability: 0.25),
                                    .init(key: "safe", probability: 0.75),
                                ],
                                confidence: 0.5
                            )))
                ],
                usage: .init(inputTokens: 9, outputTokens: 1)
            )
        }) { baseURL in
            let response = try await self.request(
                baseURL.appendingPathComponent("v1/systemone"),
                method: "POST",
                body: Data(self.validRequest.utf8)
            )
            XCTAssertEqual(response.status, 200)
            XCTAssertEqual(
                String(decoding: response.body, as: UTF8.self),
                #"{"model":"von-1.1.0","answers":{"route":{"type":"choice","choice":"safe","probabilities":{"fast":0.25,"safe":0.75},"confidence":0.5}},"usage":{"input_tokens":9,"output_tokens":1}}"#
            )
        }
    }

    func testAuthenticationPrecedesValidationAndUsesGenericFailures() async throws {
        let invocations = TestCounter()
        let configuration = VonHTTPServerConfiguration(apiKey: "secret")
        try await withServer(
            configuration: configuration,
            handler: { request in
                await invocations.increment()
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            let invalid = try await self.request(endpoint, method: "POST", body: Data("{}".utf8))
            XCTAssertEqual(invalid.status, 401)
            XCTAssertEqual(
                String(decoding: invalid.body, as: UTF8.self), #"{"detail":"Unauthorized"}"#)

            let wrongPrefix = try await self.request(
                endpoint, method: "POST", body: Data(self.validRequest.utf8),
                headers: ["Authorization": "bearer secret"])
            XCTAssertEqual(wrongPrefix.status, 401)
            XCTAssertEqual(
                String(decoding: wrongPrefix.body, as: UTF8.self), #"{"detail":"Unauthorized"}"#)

            let wrongKey = try await self.request(
                endpoint, method: "POST", body: Data(self.validRequest.utf8),
                headers: ["Authorization": "Bearer wrong"])
            XCTAssertEqual(wrongKey.status, 401)
            XCTAssertEqual(
                String(decoding: wrongKey.body, as: UTF8.self), #"{"detail":"Unauthorized"}"#)

            let accepted = try await self.request(
                endpoint, method: "POST", body: Data(self.validRequest.utf8),
                headers: ["Authorization": "Bearer  secret \t"])
            XCTAssertEqual(accepted.status, 200)
            let invocationCount = await invocations.value()
            XCTAssertEqual(invocationCount, 1)
        }
    }

    func testDuplicateAuthorizationIsRejectedBeforeBodyAndHandler() async throws {
        let invocations = TestCounter()
        let configuration = VonHTTPServerConfiguration(apiKey: "secret")
        try await withServer(
            configuration: configuration,
            handler: { request in
                await invocations.increment()
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let duplicateLines =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nAuthorization: Bearer secret\r\nContent-Length: 999999\r\n\r\n"
            let lineResponse = try await self.rawExchange(
                port: baseURL.port!, request: duplicateLines, expectedResponses: 1)
            XCTAssertTrue(lineResponse.contains("HTTP/1.1 401 Unauthorized"))
            XCTAssertTrue(lineResponse.contains(#"{"detail":"Unauthorized"}"#))

            let commaValue =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret, Bearer secret\r\nContent-Length: 999999\r\n\r\n"
            let commaResponse = try await self.rawExchange(
                port: baseURL.port!, request: commaValue, expectedResponses: 1)
            XCTAssertTrue(commaResponse.contains("HTTP/1.1 401 Unauthorized"))
            let invocationCount = await invocations.value()
            XCTAssertEqual(invocationCount, 0)
        }
    }

    func testExpectContinueWaitsForValidatedAdmissionAndUnsupportedExpectationsFail() async throws {
        let invocations = TestCounter()
        let configuration = VonHTTPServerConfiguration(apiKey: "secret")
        try await withServer(
            configuration: configuration,
            handler: { request in
                await invocations.increment()
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let body = self.validRequest
            let acceptedHead =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: 100-continue\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n"
            let accepted = try await self.rawExpectContinueExchange(
                port: baseURL.port!,
                head: acceptedHead,
                body: body
            )
            XCTAssertTrue(accepted.contains("HTTP/1.1 100 Continue"), accepted)
            XCTAssertTrue(accepted.contains("HTTP/1.1 200 OK"), accepted)

            let missingAuth =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nExpect: 100-continue\r\nContent-Length: \(body.utf8.count)\r\n\r\n"
            let unauthorized = try await self.rawExchange(
                port: baseURL.port!, request: missingAuth, expectedResponses: 1
            )
            XCTAssertTrue(unauthorized.contains("HTTP/1.1 401 Unauthorized"), unauthorized)
            XCTAssertFalse(unauthorized.contains("100 Continue"), unauthorized)

            let oversized =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: 100-continue\r\nContent-Length: 1048577\r\n\r\n"
            let tooLarge = try await self.rawExchange(
                port: baseURL.port!, request: oversized, expectedResponses: 1
            )
            XCTAssertTrue(tooLarge.contains("HTTP/1.1 413 Payload Too Large"), tooLarge)
            XCTAssertFalse(tooLarge.contains("100 Continue"), tooLarge)

            let unsupported =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: 100-continue, custom\r\nContent-Length: \(body.utf8.count)\r\n\r\n"
            let rejected = try await self.rawExchange(
                port: baseURL.port!, request: unsupported, expectedResponses: 1
            )
            XCTAssertTrue(rejected.contains("HTTP/1.1 417 Expectation Failed"), rejected)
            XCTAssertFalse(rejected.contains("100 Continue"), rejected)

            let http10 =
                "POST /v1/systemone HTTP/1.0\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: 100-continue\r\nContent-Length: \(body.utf8.count)\r\n\r\n"
            let oldProtocol = try await self.rawExchange(
                port: baseURL.port!, request: http10 + body, expectedResponses: 1
            )
            XCTAssertTrue(oldProtocol.contains("200 OK"), oldProtocol)
            XCTAssertFalse(oldProtocol.contains("100 Continue"), oldProtocol)

            for bodyless in [
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: 100-continue\r\nContent-Length: 0\r\n\r\n",
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: 100-continue\r\n\r\n",
            ] {
                let response = try await self.rawExchange(
                    port: baseURL.port!, request: bodyless, expectedResponses: 1
                )
                XCTAssertTrue(response.contains("422 Unprocessable Entity"), response)
                XCTAssertFalse(response.contains("100 Continue"), response)
            }

            let invocationCount = await invocations.value()
            XCTAssertEqual(invocationCount, 2)
        }
    }

    func testUnsupportedExpectationReturns417EvenWhenAdmissionIsFull() async throws {
        let entered = TestSignal()
        let release = TestSignal()
        let configuration = VonHTTPServerConfiguration(
            apiKey: "secret",
            maximumActiveRequests: 1,
            maximumQueuedRequests: 0
        )
        try await withServer(
            configuration: configuration,
            handler: { request in
                await entered.signal()
                await release.wait()
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            let first = Task {
                try await self.request(
                    endpoint,
                    method: "POST",
                    body: Data(self.validRequest.utf8),
                    headers: ["Authorization": "Bearer secret"]
                ).status
            }
            await entered.wait()

            let unsupported =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nExpect: custom\r\nContent-Length: \(self.validRequest.utf8.count)\r\n\r\n"
            let response = try await self.rawExchange(
                port: baseURL.port!, request: unsupported, expectedResponses: 1
            )
            XCTAssertTrue(response.contains("HTTP/1.1 417 Expectation Failed"), response)
            await release.signal()
            let firstStatus = try await first.value
            XCTAssertEqual(firstStatus, 200)
        }
    }

    func testMethodUnknownInvalidAndOversizedStatuses() async throws {
        try await withServer { baseURL in
            let wrongMethod = try await self.request(
                baseURL.appendingPathComponent("health"), method: "POST", body: Data())
            XCTAssertEqual(wrongMethod.status, 405)
            XCTAssertEqual(self.header("Allow", in: wrongMethod.headers), "GET")
            XCTAssertEqual(
                String(decoding: wrongMethod.body, as: UTF8.self),
                #"{"detail":"Method Not Allowed"}"#)

            let wrongSystemOneMethod = try await self.request(
                baseURL.appendingPathComponent("v1/systemone"))
            XCTAssertEqual(wrongSystemOneMethod.status, 405)
            XCTAssertEqual(self.header("Allow", in: wrongSystemOneMethod.headers), "POST")

            let missing = try await self.request(baseURL.appendingPathComponent("missing"))
            XCTAssertEqual(missing.status, 404)
            XCTAssertEqual(
                String(decoding: missing.body, as: UTF8.self), #"{"detail":"Not Found"}"#)

            let invalid = try await self.request(
                baseURL.appendingPathComponent("v1/systemone"), method: "POST",
                body: Data("not json".utf8))
            XCTAssertEqual(invalid.status, 422)

            let declaredOversized =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1048577\r\n\r\n"
            let oversized = try await self.rawExchange(
                port: baseURL.port!, request: declaredOversized, expectedResponses: 1)
            XCTAssertTrue(oversized.contains("HTTP/1.1 413 Payload Too Large"))
            XCTAssertTrue(
                oversized.contains("Connection: close") || oversized.contains("connection: close"))
            XCTAssertTrue(oversized.contains(#"{"detail":"Payload Too Large"}"#))
        }
    }

    func testEmptyCriteriaAndWorstCaseResponseExpansionReturn422BeforeHandler() async throws {
        let invocations = TestCounter()
        try await withServer(handler: { request in
            await invocations.increment()
            return SystemOneResponse(model: request.model, answers: [], usage: .init())
        }) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            for body in [
                #"{"state":"x","questions":{"q":{"type":"choice","instructions":"Choose","criteria":{}}}}"#,
                #"{"state":"x","questions":{"q":{"type":"score","instructions":"Rate","criteria":[]}}}"#,
            ] {
                let response = try await self.request(
                    endpoint, method: "POST", body: Data(body.utf8))
                XCTAssertEqual(response.status, 422)
            }

            let escapedControlKey = String(repeating: "\\u0001", count: 65_000)
            let adversarial =
                #"{"state":"x","questions":{"\#(escapedControlKey)":{"instructions":"Choose","criteria":{"\#(escapedControlKey)":"short"}}}}"#
            XCTAssertLessThan(Data(adversarial.utf8).count, 1_048_576)
            let rejected = try await self.request(
                endpoint, method: "POST", body: Data(adversarial.utf8))
            XCTAssertEqual(rejected.status, 422)
            let invocationCount = await invocations.value()
            XCTAssertEqual(invocationCount, 0)
        }
    }

    func testHandlerErrorsAreSanitized() async throws {
        struct SecretError: Error {}
        try await withServer(handler: { _ in throw SecretError() }) { baseURL in
            let response = try await self.request(
                baseURL.appendingPathComponent("v1/systemone"),
                method: "POST",
                body: Data(self.validRequest.utf8)
            )
            XCTAssertEqual(response.status, 500)
            XCTAssertEqual(
                String(decoding: response.body, as: UTF8.self),
                #"{"detail":"Internal Server Error"}"#)
        }
    }

    func testCORSRequiresExplicitAllowlistAndHandlesPreflight() async throws {
        let configuration = VonHTTPServerConfiguration(
            cors: .init(allowedOrigins: ["https://allowed.example", "*"]))
        try await withServer(configuration: configuration) { baseURL in
            let allowed = try await self.request(
                baseURL.appendingPathComponent("health"),
                headers: ["Origin": "https://allowed.example"])
            XCTAssertEqual(
                self.header("Access-Control-Allow-Origin", in: allowed.headers),
                "https://allowed.example")
            XCTAssertEqual(
                self.header("Access-Control-Allow-Credentials", in: allowed.headers), "true")

            let denied = try await self.request(
                baseURL.appendingPathComponent("health"),
                headers: ["Origin": "https://denied.example"])
            XCTAssertNil(self.header("Access-Control-Allow-Origin", in: denied.headers))

            let preflight = try await self.request(
                baseURL.appendingPathComponent("v1/systemone"),
                method: "OPTIONS",
                headers: [
                    "Origin": "https://allowed.example", "Access-Control-Request-Method": "POST",
                ]
            )
            XCTAssertEqual(preflight.status, 200)
            XCTAssertEqual(
                self.header("Access-Control-Allow-Origin", in: preflight.headers),
                "https://allowed.example"
            )
            XCTAssertEqual(
                self.header("Access-Control-Allow-Credentials", in: preflight.headers), "true")
            XCTAssertFalse(
                (self.header("Access-Control-Allow-Methods", in: preflight.headers) ?? "").contains(
                    "*"))
        }
    }

    func testChunkedPipelinedRequestsStayOrderedAndKeepAlive() async throws {
        let handler: VonHTTPSystemOneHandler = { request in
            if request.state == .text("first") {
                try await Task.sleep(nanoseconds: 80_000_000)
            }
            let model: String
            if case .text(let state) = request.state { model = state } else { model = "unexpected" }
            return SystemOneResponse(model: model, answers: [], usage: .init())
        }
        try await withServer(handler: handler) { baseURL in
            let first = #"{"state":"first","questions":{}}"#
            let second = #"{"state":"second","questions":{}}"#
            let chunks = [String(first.prefix(9)), String(first.dropFirst(9))]
            let chunkedBody =
                chunks.map {
                    String($0.utf8.count, radix: 16) + "\r\n" + $0 + "\r\n"
                }.joined() + "0\r\n\r\n"
            let firstHead =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
            let secondRequest =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: \(second.utf8.count)\r\nConnection: close\r\n\r\n\(second)"
            let response = try await self.rawExchange(
                port: baseURL.port!,
                request: firstHead + chunkedBody + secondRequest,
                expectedResponses: 2
            )
            XCTAssertEqual(response.components(separatedBy: "HTTP/1.1 200 OK").count - 1, 2)
            let firstRange = try XCTUnwrap(response.range(of: #""model":"first""#))
            let secondRange = try XCTUnwrap(response.range(of: #""model":"second""#))
            XCTAssertLessThan(firstRange.lowerBound, secondRange.lowerBound)
        }
    }

    func testObservedChunkedOverflowReturnsImmediate413AndCloses() async throws {
        let configuration = VonHTTPServerConfiguration(maximumBodyBytes: 16)
        try await withServer(configuration: configuration) { baseURL in
            let firstHead =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
            let chunk = "20\r\n" + String(repeating: "x", count: 32) + "\r\n"
            let response = try await self.rawExchange(
                port: baseURL.port!, request: firstHead + chunk, expectedResponses: 1)
            XCTAssertTrue(response.contains("HTTP/1.1 413 Payload Too Large"))
            XCTAssertTrue(response.contains(#"{"detail":"Payload Too Large"}"#))
            XCTAssertTrue(
                response.contains("Connection: close") || response.contains("connection: close"))
        }
    }

    func testConcurrentStartShutdownPublicationRaceIsBounded() async throws {
        for _ in 0..<20 {
            let server = VonHTTPServer { request in
                SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
            let startTask = Task { try await server.start() }
            let shutdownTask = Task { try await server.shutdown() }
            do {
                try await startTask.value
            } catch {
                let serverError = error as? VonHTTPServerError
                XCTAssertTrue(serverError == .alreadyShutdown || serverError == .alreadyStarted)
            }
            try await shutdownTask.value
            try await server.shutdown()
        }
    }

    func testShutdownRemainsIdempotentAfterBindFailure() async throws {
        let first = VonHTTPServer { request in
            SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        try await first.start()
        let occupiedPort = try XCTUnwrap(first.port)
        let second = VonHTTPServer(configuration: .init(port: occupiedPort)) { request in
            SystemOneResponse(model: request.model, answers: [], usage: .init())
        }

        do {
            try await second.start()
            XCTFail("Expected binding an occupied port to fail")
        } catch {
            try await second.shutdown()
            try await second.shutdown()
        }
        try await first.shutdown()
    }

    func testGracefulShutdownClosesStalledPartialUploadPromptly() async throws {
        let server = VonHTTPServer { request in
            SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        try await server.start()
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let channel = try await ClientBootstrap(group: clientGroup)
            .connect(host: "127.0.0.1", port: try XCTUnwrap(server.port))
            .get()
        var buffer = channel.allocator.buffer(capacity: 128)
        buffer.writeString(
            "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{")
        try await channel.writeAndFlush(buffer).get()
        try await Task.sleep(nanoseconds: 20_000_000)

        let clock = ContinuousClock()
        let start = clock.now
        try await server.shutdown()
        XCTAssertLessThan(start.duration(to: clock.now), .seconds(1))
        try await channel.closeFuture.get()
        try await clientGroup.shutdownGracefully()
    }

    func testShutdownWaitsForWorkAfterClientDisconnects() async throws {
        let entered = TestSignal()
        let release = TestSignal()
        let shutdownFinished = TestFlag()
        let server = VonHTTPServer { request in
            await entered.signal()
            await release.wait()
            return SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        try await server.start()
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let channel = try await ClientBootstrap(group: clientGroup)
            .connect(host: "127.0.0.1", port: try XCTUnwrap(server.port))
            .get()
        let body = #"{"state":"disconnect","questions":{}}"#
        var buffer = channel.allocator.buffer(capacity: body.utf8.count + 128)
        buffer.writeString(
            "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
        )
        try await channel.writeAndFlush(buffer).get()
        await entered.wait()
        try await channel.close().get()

        let shutdownTask = Task {
            try await server.shutdown()
            await shutdownFinished.set()
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        let finishedBeforeRelease = await shutdownFinished.value()
        XCTAssertFalse(finishedBeforeRelease)
        await release.signal()
        try await shutdownTask.value
        let finishedAfterRelease = await shutdownFinished.value()
        XCTAssertTrue(finishedAfterRelease)
        try await clientGroup.shutdownGracefully()
    }

    func testShutdownCancelsAnInFlightCancellationAwareRequest() async throws {
        let entered = TestSignal()
        let handler: VonHTTPSystemOneHandler = { request in
            await entered.signal()
            try await Task.sleep(nanoseconds: 60_000_000)
            return SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        let server = VonHTTPServer(handler: handler)
        try await server.start()
        let port = try XCTUnwrap(server.port)
        let endpoint = URL(string: "http://127.0.0.1:\(port)/v1/systemone")!
        let responseTask = Task { () throws -> Int in
            try await self.request(endpoint, method: "POST", body: Data(self.validRequest.utf8))
                .status
        }
        await entered.wait()
        async let firstShutdown: Void = server.shutdown()
        async let secondShutdown: Void = server.shutdown()
        let status = try await responseTask.value
        XCTAssertEqual(status, 504)
        try await firstShutdown
        try await secondShutdown
    }

    func testQueueOverflowReturnsSanitized500AndConfigurationClampsCaps() async throws {
        let configuration = VonHTTPServerConfiguration(
            maximumActiveRequests: 1, maximumQueuedRequests: 1)
        XCTAssertEqual(
            VonHTTPServerConfiguration(maximumActiveRequests: 99, maximumQueuedRequests: 99)
                .maximumActiveRequests, 2)
        XCTAssertEqual(
            VonHTTPServerConfiguration(maximumActiveRequests: 99, maximumQueuedRequests: 99)
                .maximumQueuedRequests, 8)
        let entered = TestSignal()
        let release = TestSignal()
        let handler: VonHTTPSystemOneHandler = { request in
            await entered.signal()
            await release.wait()
            return SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        try await withServer(configuration: configuration, handler: handler) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            let first = Task { () throws -> (Int, String) in
                let response = try await self.request(
                    endpoint, method: "POST", body: Data(self.validRequest.utf8))
                return (response.status, String(decoding: response.body, as: UTF8.self))
            }
            await entered.wait()
            let mailbox = TestResponseMailbox()
            let second = Task { () throws -> (Int, String) in
                let response = try await self.request(
                    endpoint, method: "POST", body: Data(self.validRequest.utf8))
                let result = (response.status, String(decoding: response.body, as: UTF8.self))
                await mailbox.submit(.init(status: result.0, body: result.1))
                return result
            }
            let third = Task { () throws -> (Int, String) in
                let response = try await self.request(
                    endpoint, method: "POST", body: Data(self.validRequest.utf8))
                let result = (response.status, String(decoding: response.body, as: UTF8.self))
                await mailbox.submit(.init(status: result.0, body: result.1))
                return result
            }
            let overflow = await mailbox.next()
            XCTAssertEqual(overflow.status, 503)
            XCTAssertEqual(overflow.body, #"{"detail":"Service Unavailable"}"#)
            await release.signal()
            let results = try await [first.value, second.value, third.value]
            XCTAssertEqual(results.map(\.0).sorted(), [200, 200, 503])
        }
    }

    func testAbsoluteHeaderAndBodyDeadlinesRejectTrickledPartialRequests() async throws {
        let configuration = VonHTTPServerConfiguration(
            headerDeadline: .milliseconds(70),
            bodyDeadline: .milliseconds(70),
            idleKeepAliveTimeout: .milliseconds(200)
        )
        try await withServer(configuration: configuration) { baseURL in
            let headerResponse = try await self.rawTrickledExchange(
                port: baseURL.port!,
                writes: ["GET /health HTTP/1.1\r\n", "Host: local", "host"],
                intervalNanoseconds: 20_000_000
            )
            XCTAssertTrue(headerResponse.contains("HTTP/1.1 408 Request Timeout"))

            let bodyResponse = try await self.rawTrickledExchange(
                port: baseURL.port!,
                writes: [
                    "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{",
                    "\"state\":",
                    "\"slow\"",
                ],
                intervalNanoseconds: 20_000_000
            )
            XCTAssertTrue(bodyResponse.contains("HTTP/1.1 408 Request Timeout"))
        }
    }

    func testRawMetadataBudgetsRejectManyHeadersAndOversizedChunkedTrailers() async throws {
        let configuration = VonHTTPServerConfiguration(
            maximumHeaderBytes: 4_096,
            maximumHeaderCount: 4,
            maximumTrailerBytes: 256
        )
        try await withServer(configuration: configuration) { baseURL in
            let fields = (0..<5).map { "X-Field-\($0): value\r\n" }.joined()
            let manyHeaders = "GET /health HTTP/1.1\r\nHost: localhost\r\n\(fields)\r\n"
            let headerResponse = try await self.rawExchange(
                port: baseURL.port!, request: manyHeaders, expectedResponses: 1
            )
            XCTAssertTrue(headerResponse.contains("HTTP/1.1 431"), headerResponse)
            XCTAssertTrue(headerResponse.contains(#"{"detail":"Request Header Fields Too Large"}"#))

            let body = #"{"state":"trailers","questions":{}}"#
            let oversizedTrailer =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nTrailer: X-Large\r\n\r\n"
                + String(body.utf8.count, radix: 16) + "\r\n" + body + "\r\n0\r\n"
                + "X-Large: " + String(repeating: "x", count: 300) + "\r\n\r\n"
            let trailerResponse = try await self.rawExchange(
                port: baseURL.port!, request: oversizedTrailer, expectedResponses: 1
            )
            XCTAssertTrue(trailerResponse.contains("HTTP/1.1 431"), trailerResponse)
            XCTAssertTrue(
                trailerResponse.contains(#"{"detail":"Request Header Fields Too Large"}"#))
        }
    }

    func testValidatedPipelinedResponseSurvivesLaterRawHeaderFailureInSameWrite() async throws {
        let configuration = VonHTTPServerConfiguration(maximumHeaderCount: 4)
        try await withServer(configuration: configuration) { baseURL in
            let valid = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
            let fields = (0..<5).map { "X-Field-\($0): value\r\n" }.joined()
            let oversized = "GET /health HTTP/1.1\r\nHost: localhost\r\n\(fields)\r\n"
            let response = try await self.rawExchange(
                port: baseURL.port!, request: valid + oversized, expectedResponses: 2
            )
            let ok = try XCTUnwrap(response.range(of: "HTTP/1.1 200 OK"))
            let rejected = try XCTUnwrap(response.range(of: "HTTP/1.1 431"))
            XCTAssertLessThan(ok.lowerBound, rejected.lowerBound)
            XCTAssertTrue(response.contains(#"{"status":"ok""#))
        }
    }

    func testRawFramingHandlesSplitChunkExtensionsAndTrailers() async throws {
        try await withServer { baseURL in
            let body = #"{"state":"split","questions":{}}"#
            let size = String(body.utf8.count, radix: 16)
            let midpoint = body.index(body.startIndex, offsetBy: body.count / 2)
            let firstBody = String(body[..<midpoint])
            let secondBody = String(body[midpoint...])
            let response = try await self.rawTrickledExchange(
                port: baseURL.port!,
                writes: [
                    "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nTrailer: X-Trace\r\n\r\n\(size);trace",
                    "=yes\r\n\(firstBody)",
                    "\(secondBody)\r\n0\r\nX-Tra",
                    "ce: yes\r\n\r\nGET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                ],
                intervalNanoseconds: 5_000_000,
                expectedResponses: 2
            )
            XCTAssertEqual(response.components(separatedBy: "HTTP/1.1 200 OK").count - 1, 2)
            XCTAssertTrue(response.contains(#""model":"von-latest""#))
        }
    }

    func testAmbiguousTransferEncodingAndContentLengthFailClosed() async throws {
        try await withServer { baseURL in
            let transferAndLength =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n"
            let first = try await self.rawExchange(
                port: baseURL.port!, request: transferAndLength, expectedResponses: 1
            )
            XCTAssertTrue(first.contains("HTTP/1.1 400 Bad Request"))

            let duplicateLength =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n"
            let second = try await self.rawExchange(
                port: baseURL.port!, request: duplicateLength, expectedResponses: 1
            )
            XCTAssertTrue(second.contains("HTTP/1.1 400 Bad Request"))
        }
    }

    func testNormalChunkedTrailersAndFollowingPipelinedRequestStillWork() async throws {
        try await withServer { baseURL in
            let body = #"{"state":"chunked-trailer","questions":{}}"#
            let first =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nTrailer: X-Trace\r\n\r\n"
                + String(body.utf8.count, radix: 16) + "\r\n" + body
                + "\r\n0\r\nX-Trace: ok\r\n\r\n"
            let second = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            let response = try await self.rawExchange(
                port: baseURL.port!, request: first + second, expectedResponses: 2
            )
            XCTAssertEqual(response.components(separatedBy: "HTTP/1.1 200 OK").count - 1, 2)
            XCTAssertTrue(response.contains(#""model":"von-latest""#))
            XCTAssertTrue(response.contains(#"{"status":"ok""#))
        }
    }

    func testShortHeaderDeadlineDoesNotKillActiveHandler() async throws {
        let configuration = VonHTTPServerConfiguration(
            headerDeadline: .milliseconds(30),
            idleKeepAliveTimeout: .seconds(1),
            requestDeadline: .milliseconds(500)
        )
        try await withServer(
            configuration: configuration,
            handler: { request in
                try await Task.sleep(nanoseconds: 90_000_000)
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let response = try await self.request(
                baseURL.appendingPathComponent("v1/systemone"),
                method: "POST",
                body: Data(self.validRequest.utf8)
            )
            XCTAssertEqual(response.status, 200)
        }
    }

    func testDeclaredOversizePrecedesCapacityRejection() async throws {
        let entered = TestSignal()
        let release = TestSignal()
        let configuration = VonHTTPServerConfiguration(
            maximumBodyBytes: 128,
            maximumActiveRequests: 1,
            maximumQueuedRequests: 0
        )
        try await withServer(
            configuration: configuration,
            handler: { request in
                await entered.signal()
                await release.wait()
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            let first = Task {
                try await self.request(endpoint, method: "POST", body: Data(self.validRequest.utf8))
                    .status
            }
            await entered.wait()
            let oversized =
                "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 129\r\n\r\n"
            let response = try await self.rawExchange(
                port: baseURL.port!, request: oversized, expectedResponses: 1)
            XCTAssertTrue(response.contains("HTTP/1.1 413 Payload Too Large"))
            await release.signal()
            let firstStatus = try await first.value
            XCTAssertEqual(firstStatus, 200)
        }
    }

    func testIdleKeepAliveConnectionIsClosed() async throws {
        let configuration = VonHTTPServerConfiguration(
            headerDeadline: .seconds(1),
            idleKeepAliveTimeout: .milliseconds(50)
        )
        try await withServer(configuration: configuration) { baseURL in
            let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
            let channel = try await ClientBootstrap(group: group)
                .connect(host: "127.0.0.1", port: baseURL.port!).get()
            let closed = try await self.waitForClose(channel, timeout: .seconds(1))
            XCTAssertTrue(closed)
            try await group.shutdownGracefully()
        }
    }

    func testGlobalConnectionCapRecoversAfterClose() async throws {
        let configuration = VonHTTPServerConfiguration(maximumConnections: 1)
        try await withServer(configuration: configuration) { baseURL in
            let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
            let first = try await ClientBootstrap(group: group)
                .connect(host: "127.0.0.1", port: baseURL.port!).get()
            let rejected = try await ClientBootstrap(group: group)
                .connect(host: "127.0.0.1", port: baseURL.port!).get()
            let rejectedClosed = try await self.waitForClose(rejected, timeout: .seconds(1))
            XCTAssertTrue(rejectedClosed)
            try await first.close().get()

            var recovered = false
            for _ in 0..<20 where !recovered {
                do {
                    let response = try await self.rawExchange(
                        port: baseURL.port!,
                        request:
                            "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                        expectedResponses: 1
                    )
                    recovered = response.contains("HTTP/1.1 200 OK")
                } catch {
                    try await Task.sleep(nanoseconds: 10_000_000)
                }
            }
            XCTAssertTrue(recovered)
            try await group.shutdownGracefully()
        }
    }

    func testRequestDeadlineCancelsButRetainsActivePermitUntilHandlerExits() async throws {
        let entered = TestCounter()
        let release = TestSignal()
        let configuration = VonHTTPServerConfiguration(
            maximumActiveRequests: 1,
            maximumQueuedRequests: 1,
            requestDeadline: .milliseconds(80),
            shutdownGracePeriod: .milliseconds(50)
        )
        try await withServer(
            configuration: configuration,
            handler: { request in
                await entered.increment()
                await release.wait()
                return SystemOneResponse(model: request.model, answers: [], usage: .init())
            }
        ) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            let first = Task {
                try await self.request(endpoint, method: "POST", body: Data(self.validRequest.utf8))
                    .status
            }
            while await entered.value() == 0 { await Task.yield() }
            let second = Task {
                try await self.request(endpoint, method: "POST", body: Data(self.validRequest.utf8))
                    .status
            }
            let firstStatus = try await first.value
            let secondStatus = try await second.value
            let enteredCount = await entered.value()
            XCTAssertEqual(firstStatus, 504)
            XCTAssertEqual(secondStatus, 504)
            XCTAssertEqual(enteredCount, 1)
            await release.signal()
        }
    }

    func testShutdownIsBoundedWhenHandlerIgnoresCancellation() async throws {
        let entered = TestSignal()
        let release = TestSignal()
        let configuration = VonHTTPServerConfiguration(
            requestDeadline: .seconds(30),
            shutdownGracePeriod: .milliseconds(60)
        )
        let server = VonHTTPServer(configuration: configuration) { request in
            await entered.signal()
            await release.wait()
            return SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        try await server.start()
        let endpoint = URL(string: "http://127.0.0.1:\(try XCTUnwrap(server.port))/v1/systemone")!
        let requestTask = Task {
            try? await self.request(endpoint, method: "POST", body: Data(self.validRequest.utf8))
                .status
        }
        await entered.wait()
        let clock = ContinuousClock()
        let start = clock.now
        try await server.shutdown()
        XCTAssertLessThan(start.duration(to: clock.now), .milliseconds(500))
        await release.signal()
        _ = await requestTask.value
    }

    func testNonLoopbackWithoutAPIKeyFailsBeforeBindUnlessExplicitlyAllowed() async throws {
        let denied = VonHTTPServer(configuration: .init(host: "0.0.0.0")) { request in
            SystemOneResponse(model: request.model, answers: [], usage: .init())
        }
        do {
            try await denied.start()
            XCTFail("Expected fail-closed binding policy")
        } catch {
            XCTAssertEqual(error as? VonHTTPServerError, .insecureNonLoopbackBinding)
        }
        try await denied.shutdown()

        let explicitOptIn = VonHTTPServerConfiguration(
            host: "0.0.0.0",
            allowUnsafeNonLoopbackWithoutAPIKey: true
        )
        XCTAssertTrue(explicitOptIn.allowUnsafeNonLoopbackWithoutAPIKey)
    }

    func testConcurrencyIsBoundedWithoutBlockingEventLoops() async throws {
        actor Meter {
            var active = 0
            var peak = 0
            func enter() {
                active += 1
                peak = max(peak, active)
            }
            func leave() { active -= 1 }
            func peakValue() -> Int { peak }
        }
        let meter = Meter()
        try await withServer(handler: { request in
            await meter.enter()
            try await Task.sleep(nanoseconds: 60_000_000)
            await meter.leave()
            return SystemOneResponse(model: request.model, answers: [], usage: .init())
        }) { baseURL in
            let endpoint = baseURL.appendingPathComponent("v1/systemone")
            try await withThrowingTaskGroup(of: Int.self) { group in
                for _ in 0..<6 {
                    group.addTask {
                        try await self.request(
                            endpoint, method: "POST", body: Data(self.validRequest.utf8)
                        )
                        .status
                    }
                }
                for try await status in group { XCTAssertEqual(status, 200) }
            }
            let peak = await meter.peakValue()
            XCTAssertEqual(peak, 2)
        }
    }

    private func withServer<T>(
        configuration: VonHTTPServerConfiguration = .init(),
        handler: @escaping VonHTTPSystemOneHandler = { request in
            SystemOneResponse(model: request.model, answers: [], usage: .init())
        },
        operation: (URL) async throws -> T
    ) async throws -> T {
        let server = VonHTTPServer(configuration: configuration, handler: handler)
        try await server.start()
        let port = try XCTUnwrap(server.port)
        do {
            let result = try await operation(URL(string: "http://127.0.0.1:\(port)/")!)
            try await server.shutdown()
            try await server.shutdown()
            return result
        } catch {
            try? await server.shutdown()
            throw error
        }
    }

    private func header(_ name: String, in headers: [AnyHashable: Any]) -> String? {
        headers.first { key, _ in
            String(describing: key).caseInsensitiveCompare(name) == .orderedSame
        }?.value as? String
    }

    private func request(
        _ url: URL,
        method: String = "GET",
        body: Data? = nil,
        headers: [String: String] = [:]
    ) async throws -> (status: Int, body: Data, headers: [AnyHashable: Any]) {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = 5
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        return (http.statusCode, data, http.allHeaderFields)
    }

    private func rawTrickledExchange(
        port: Int,
        writes: [String],
        intervalNanoseconds: UInt64,
        expectedResponses: Int = 1
    ) async throws -> String {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let promise = group.next().makePromise(of: String.self)
        let collector = RawResponseCollector(expectedResponses: expectedResponses, promise: promise)
        var channel: Channel?
        do {
            channel = try await ClientBootstrap(group: group)
                .channelInitializer { $0.pipeline.addHandler(collector) }
                .connect(host: "127.0.0.1", port: port).get()
            for (index, text) in writes.enumerated() {
                if index > 0 { try await Task.sleep(nanoseconds: intervalNanoseconds) }
                var buffer = channel!.allocator.buffer(capacity: text.utf8.count)
                buffer.writeString(text)
                do {
                    try await channel!.writeAndFlush(buffer).get()
                } catch {
                    // On a slow host the server can answer and close before the
                    // last trickled write; the collector still decides the outcome.
                    break
                }
            }
            let response = try await promise.futureResult.get()
            try? await channel!.close().get()
            try await group.shutdownGracefully()
            return response
        } catch {
            if let channel { try? await channel.close().get() }
            try? await group.shutdownGracefully()
            throw error
        }
    }

    private func rawDelayedExchange(
        port: Int,
        initial: String,
        expectedResponses: Int
    ) async throws -> String {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let promise = group.next().makePromise(of: String.self)
        let collector = RawResponseCollector(expectedResponses: expectedResponses, promise: promise)
        var channel: Channel?
        do {
            channel = try await ClientBootstrap(group: group)
                .channelInitializer { $0.pipeline.addHandler(collector) }
                .connect(host: "127.0.0.1", port: port).get()
            var buffer = channel!.allocator.buffer(capacity: initial.utf8.count)
            buffer.writeString(initial)
            try await channel!.writeAndFlush(buffer).get()
            let response = try await promise.futureResult.get()
            try? await channel!.close().get()
            try await group.shutdownGracefully()
            return response
        } catch {
            if let channel { try? await channel.close().get() }
            try? await group.shutdownGracefully()
            throw error
        }
    }

    private func waitForClose(_ channel: Channel, timeout: TimeAmount) async throws -> Bool {
        let promise = channel.eventLoop.makePromise(of: Bool.self)
        let completion = TestOneShotBool(promise: promise)
        channel.closeFuture.whenComplete { _ in completion.resolve(true) }
        let timeoutTask = channel.eventLoop.scheduleTask(in: timeout) { completion.resolve(false) }
        let result = try await promise.futureResult.get()
        timeoutTask.cancel()
        return result
    }

    private func rawExpectContinueExchange(port: Int, head: String, body: String) async throws
        -> String
    {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let eventLoop = group.next()
        let responsePromise = eventLoop.makePromise(of: String.self)
        let continuePromise = eventLoop.makePromise(of: Void.self)
        let collector = RawResponseCollector(
            expectedResponses: 2,
            promise: responsePromise,
            informationalPromise: continuePromise
        )
        var channel: Channel?
        do {
            channel = try await ClientBootstrap(group: group)
                .channelInitializer { $0.pipeline.addHandler(collector) }
                .connect(host: "127.0.0.1", port: port)
                .get()
            let timeout = eventLoop.scheduleTask(in: .seconds(5)) {
                collector.failIfPending(RawClientError.timeout)
            }
            var headBuffer = channel!.allocator.buffer(capacity: head.utf8.count)
            headBuffer.writeString(head)
            try await channel!.writeAndFlush(headBuffer).get()
            try await continuePromise.futureResult.get()
            var bodyBuffer = channel!.allocator.buffer(capacity: body.utf8.count)
            bodyBuffer.writeString(body)
            try await channel!.writeAndFlush(bodyBuffer).get()
            let response = try await responsePromise.futureResult.get()
            timeout.cancel()
            try? await channel!.close().get()
            try await group.shutdownGracefully()
            return response
        } catch {
            collector.failIfPending(error)
            if let channel { try? await channel.close().get() }
            try? await group.shutdownGracefully()
            throw error
        }
    }

    private func rawExchange(port: Int, request: String, expectedResponses: Int) async throws
        -> String
    {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let promise = group.next().makePromise(of: String.self)
        let collector = RawResponseCollector(expectedResponses: expectedResponses, promise: promise)
        var channel: Channel?
        do {
            channel = try await ClientBootstrap(group: group)
                .channelInitializer { $0.pipeline.addHandler(collector) }
                .connect(host: "127.0.0.1", port: port)
                .get()
            var buffer = channel!.allocator.buffer(capacity: request.utf8.count)
            buffer.writeString(request)
            try await channel!.writeAndFlush(buffer).get()
            let timeout = channel!.eventLoop.scheduleTask(in: .seconds(5)) {
                collector.failIfPending(RawClientError.timeout)
            }
            let response = try await promise.futureResult.get()
            timeout.cancel()
            try? await channel!.close().get()
            try await group.shutdownGracefully()
            return response
        } catch {
            if let channel { try? await channel.close().get() }
            try? await group.shutdownGracefully()
            throw error
        }
    }
}

private final class TestOneShotBool: @unchecked Sendable {
    private let lock = NSLock()
    private let promise: EventLoopPromise<Bool>
    private var completed = false

    init(promise: EventLoopPromise<Bool>) { self.promise = promise }

    func resolve(_ value: Bool) {
        let shouldResolve = lock.withLock { () -> Bool in
            guard !completed else { return false }
            completed = true
            return true
        }
        if shouldResolve { promise.succeed(value) }
    }
}

private actor TestCounter {
    private var count = 0

    func increment() { count += 1 }
    func value() -> Int { count }
}

private actor TestSignal {
    private var signaled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        signaled = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        if signaled { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

private struct TestResponse: Sendable {
    let status: Int
    let body: String
}

private actor TestResponseMailbox {
    private var values: [TestResponse] = []
    private var waiters: [CheckedContinuation<TestResponse, Never>] = []

    func submit(_ value: TestResponse) {
        if waiters.isEmpty {
            values.append(value)
        } else {
            waiters.removeFirst().resume(returning: value)
        }
    }

    func next() async -> TestResponse {
        if !values.isEmpty { return values.removeFirst() }
        return await withCheckedContinuation { waiters.append($0) }
    }
}

private actor TestFlag {
    private var isSet = false

    func set() { isSet = true }
    func value() -> Bool { isSet }
}

private enum RawClientError: Error {
    case timeout
    case closedBeforeExpectedResponses
    case malformedResponse
}

private final class RawResponseCollector: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer

    private let expectedResponses: Int
    private let promise: EventLoopPromise<String>
    private let informationalPromise: EventLoopPromise<Void>?
    private var bytes: [UInt8] = []
    private var parseIndex = 0
    private var parsedResponses = 0
    private var completed = false
    private var informationalCompleted = false

    init(
        expectedResponses: Int,
        promise: EventLoopPromise<String>,
        informationalPromise: EventLoopPromise<Void>? = nil
    ) {
        self.expectedResponses = expectedResponses
        self.promise = promise
        self.informationalPromise = informationalPromise
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        var buffer = unwrapInboundIn(data)
        if let incoming = buffer.readBytes(length: buffer.readableBytes) {
            bytes.append(contentsOf: incoming)
        }
        parseAvailableResponses()
    }

    func channelInactive(context: ChannelHandlerContext) {
        failIfPending(RawClientError.closedBeforeExpectedResponses)
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        failIfPending(error)
        context.close(promise: nil)
    }

    func failIfPending(_ error: Error) {
        guard !completed else { return }
        completed = true
        promise.fail(error)
        if !informationalCompleted {
            informationalCompleted = true
            informationalPromise?.fail(error)
        }
    }

    private func parseAvailableResponses() {
        while !completed, parsedResponses < expectedResponses {
            guard let headerEnd = findHeaderEnd(from: parseIndex) else { return }
            let headerData = Data(bytes[parseIndex..<headerEnd])
            guard let header = String(data: headerData, encoding: .utf8) else {
                failIfPending(RawClientError.malformedResponse)
                return
            }
            let informational = isInformational(header)
            guard let contentLength = informational ? 0 : parseContentLength(header) else {
                failIfPending(RawClientError.malformedResponse)
                return
            }
            let bodyStart = headerEnd + 4
            let (bodyEnd, overflow) = bodyStart.addingReportingOverflow(contentLength)
            guard !overflow else {
                failIfPending(RawClientError.malformedResponse)
                return
            }
            guard bytes.count >= bodyEnd else { return }
            parseIndex = bodyEnd
            parsedResponses += 1
            if informational, !informationalCompleted {
                informationalCompleted = true
                informationalPromise?.succeed(())
            }
        }
        if parsedResponses == expectedResponses {
            completed = true
            promise.succeed(String(decoding: bytes[0..<parseIndex], as: UTF8.self))
        }
    }

    private func findHeaderEnd(from start: Int) -> Int? {
        guard bytes.count >= start + 4 else { return nil }
        for index in start...(bytes.count - 4)
        where bytes[index] == 13 && bytes[index + 1] == 10
            && bytes[index + 2] == 13 && bytes[index + 3] == 10
        {
            return index
        }
        return nil
    }

    private func isInformational(_ header: String) -> Bool {
        guard let statusLine = header.components(separatedBy: "\r\n").first else { return false }
        let parts = statusLine.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 2, let status = Int(parts[1]) else { return false }
        return (100..<200).contains(status) && status != 101
    }

    private func parseContentLength(_ header: String) -> Int? {
        let values = header.components(separatedBy: "\r\n").dropFirst().compactMap {
            line -> String? in
            let parts = line.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2,
                parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length"
            else {
                return nil
            }
            return parts[1].trimmingCharacters(in: .whitespaces)
        }
        guard values.count == 1, let value = values.first, !value.isEmpty else { return nil }
        var length = 0
        for byte in value.utf8 {
            guard byte >= 48, byte <= 57 else { return nil }
            let (multiplied, overflow1) = length.multipliedReportingOverflow(by: 10)
            let (next, overflow2) = multiplied.addingReportingOverflow(Int(byte - 48))
            guard !overflow1, !overflow2 else { return nil }
            length = next
        }
        return length
    }
}
