import CryptoKit
import Foundation
import JSON
import NIOCore
import NIOHTTP1
import NIOPosix
import SystemOne

public typealias SystemOneHTTPHandler =
    @Sendable (SystemOneRequest) async throws -> SystemOneResponse

public enum SystemOneHTTPServerError: Error, Equatable, Sendable {
    case alreadyStarted
    case alreadyShutdown
    case missingBoundPort
    case insecureNonLoopbackBinding
}

public struct SystemOneHTTPServerConfiguration: Equatable, Sendable {
    public let host: String
    public let port: Int
    public let apiKey: String?
    public let cors: CORSConfiguration
    public let maximumBodyBytes: Int
    public let maximumHeaderBytes: Int
    public let maximumHeaderCount: Int
    public let maximumTrailerBytes: Int
    public let maximumConnections: Int
    public let maximumActiveRequests: Int
    public let maximumQueuedRequests: Int
    public let headerDeadline: TimeAmount
    public let bodyDeadline: TimeAmount
    public let idleKeepAliveTimeout: TimeAmount
    public let requestDeadline: TimeAmount
    public let shutdownGracePeriod: TimeAmount
    public let allowUnsafeNonLoopbackWithoutAPIKey: Bool

    public init(
        host: String = "127.0.0.1",
        port: Int = 0,
        apiKey: String? = ProcessInfo.processInfo.environment["SYSTEMONE_API_KEY"],
        cors: CORSConfiguration = .disabled,
        maximumBodyBytes: Int = 1_048_576,
        maximumHeaderBytes: Int = 32_768,
        maximumHeaderCount: Int = 64,
        maximumTrailerBytes: Int = 8_192,
        maximumConnections: Int = 64,
        maximumActiveRequests: Int = 2,
        maximumQueuedRequests: Int = 8,
        headerDeadline: TimeAmount = .seconds(10),
        bodyDeadline: TimeAmount = .seconds(15),
        idleKeepAliveTimeout: TimeAmount = .seconds(15),
        requestDeadline: TimeAmount = .seconds(5),
        shutdownGracePeriod: TimeAmount = .seconds(2),
        allowUnsafeNonLoopbackWithoutAPIKey: Bool = false
    ) {
        self.host = host
        self.port = port
        self.apiKey = apiKey.flatMap { $0.isEmpty ? nil : $0 }
        self.cors = cors
        self.maximumBodyBytes = min(max(0, maximumBodyBytes), 1_048_576)
        self.maximumHeaderBytes = min(65_536, max(1_024, maximumHeaderBytes))
        self.maximumHeaderCount = min(256, max(1, maximumHeaderCount))
        self.maximumTrailerBytes = min(32_768, max(256, maximumTrailerBytes))
        self.maximumConnections = min(1_024, max(1, maximumConnections))
        self.maximumActiveRequests = min(2, max(1, maximumActiveRequests))
        self.maximumQueuedRequests = min(8, max(0, maximumQueuedRequests))
        self.headerDeadline = Self.positive(headerDeadline)
        self.bodyDeadline = Self.positive(bodyDeadline)
        self.idleKeepAliveTimeout = Self.positive(idleKeepAliveTimeout)
        self.requestDeadline = Self.positive(requestDeadline)
        self.shutdownGracePeriod = Self.positive(shutdownGracePeriod)
        self.allowUnsafeNonLoopbackWithoutAPIKey = allowUnsafeNonLoopbackWithoutAPIKey
    }

    private static func positive(_ value: TimeAmount) -> TimeAmount {
        .nanoseconds(max(1, value.nanoseconds))
    }
}

/// A small SwiftNIO HTTP/1.1 server for the Python-compatible Von API.
/// Call `start()` once, inspect `port`, and finish with the idempotent `shutdown()`.
public final class SystemOneHTTPServer: @unchecked Sendable {
    private enum Lifecycle {
        case initialized
        case starting
        case listening(Channel)
        case shutdown
    }

    private let configuration: SystemOneHTTPServerConfiguration
    private let handler: SystemOneHTTPHandler
    private let codec: SystemOneWireCodec
    private let group: MultiThreadedEventLoopGroup
    private let gate: ModelConcurrencyGate
    private let requestCapacity: RequestCapacity
    private let connections: ConnectionRegistry
    private let work = AsyncWorkRegistry()
    private let lock = NSLock()
    private var lifecycle: Lifecycle = .initialized
    private var bindingFuture: EventLoopFuture<Channel>?
    private var startupResolved = false
    private var shutdownTask: Task<Void, Error>?
    private var boundPort: Int?

    public init(
        configuration: SystemOneHTTPServerConfiguration = .init(),
        codec: SystemOneWireCodec,
        handler: @escaping SystemOneHTTPHandler
    ) {
        self.configuration = configuration
        self.codec = codec
        self.handler = handler
        self.group = MultiThreadedEventLoopGroup(numberOfThreads: max(1, min(2, System.coreCount)))
        self.gate = ModelConcurrencyGate(
            maximumActive: configuration.maximumActiveRequests,
            maximumQueued: configuration.maximumQueuedRequests
        )
        let (capacity, overflow) = configuration.maximumActiveRequests.addingReportingOverflow(
            configuration.maximumQueuedRequests
        )
        self.requestCapacity = RequestCapacity(limit: overflow ? Int.max : capacity)
        self.connections = ConnectionRegistry(limit: configuration.maximumConnections)
    }

    /// The actual bound TCP port, available after `start()` returns.
    public var port: Int? {
        lock.withLock { boundPort }
    }

    public func start() async throws {
        let mayStart = lock.withLock { () -> Bool in
            guard case .initialized = lifecycle else { return false }
            lifecycle = .starting
            return true
        }
        guard mayStart else { throw SystemOneHTTPServerError.alreadyStarted }

        do {
            try validateBindingPolicy()
            let configuration = self.configuration
            let codec = self.codec
            let handler = self.handler
            let gate = self.gate
            let requestCapacity = self.requestCapacity
            let connections = self.connections
            let work = self.work
            let future = ServerBootstrap(group: group)
                .serverChannelOption(
                    ChannelOptions.backlog, value: Int32(configuration.maximumConnections)
                )
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
                .childChannelInitializer { channel in
                    guard connections.admit(channel) else {
                        channel.close(promise: nil)
                        return channel.eventLoop.makeSucceededFuture(())
                    }
                    let application = SystemOneHTTPChannelHandler(
                        configuration: configuration,
                        codec: codec,
                        modelHandler: handler,
                        gate: gate,
                        requestCapacity: requestCapacity,
                        work: work
                    )
                    do {
                        try channel.pipeline.syncOperations.addHandler(HTTPResponseEncoder())
                        try channel.pipeline.syncOperations.addHandler(
                            IdleStateHandler(allTimeout: configuration.idleKeepAliveTimeout)
                        )
                        try channel.pipeline.syncOperations.addHandler(
                            HTTP1FramingBudgetHandler(configuration: configuration)
                        )
                        try channel.pipeline.syncOperations.addHandler(
                            ByteToMessageHandler(
                                HTTPRequestDecoder(leftOverBytesStrategy: .dropBytes))
                        )
                        try channel.pipeline.syncOperations.addHandler(
                            NIOHTTPResponseHeadersValidator())
                        try channel.pipeline.syncOperations.addHandler(
                            HTTPServerProtocolErrorHandler())
                        try channel.pipeline.syncOperations.addHandler(application)
                        return channel.eventLoop.makeSucceededFuture(())
                    } catch {
                        return channel.eventLoop.makeFailedFuture(error)
                    }
                }
                .bind(host: configuration.host, port: configuration.port)
            lock.withLock {
                bindingFuture = future
                startupResolved = true
            }
            let channel = try await future.get()
            guard let actualPort = channel.localAddress?.port else {
                try await channel.close().get()
                throw SystemOneHTTPServerError.missingBoundPort
            }
            let published = lock.withLock { () -> Bool in
                guard case .starting = lifecycle else { return false }
                boundPort = actualPort
                lifecycle = .listening(channel)
                return true
            }
            guard published else {
                // A concurrent shutdown owns listener closure after awaiting bindingFuture.
                throw SystemOneHTTPServerError.alreadyShutdown
            }
        } catch {
            lock.withLock { startupResolved = true }
            let cleanup = cleanupAfterFailedStart()
            try? await cleanup.value
            throw error
        }
    }

    /// Stops admission, bounds application-handler draining to the configured grace period,
    /// then force-closes channels before awaiting normal SwiftNIO cleanup.
    public func shutdown() async throws {
        let task: Task<Void, Error> = lock.withLock {
            if let shutdownTask { return shutdownTask }
            let listeningChannel: Channel?
            let waitForBinding: Bool
            switch lifecycle {
            case .initialized:
                listeningChannel = nil
                waitForBinding = false
            case .starting:
                listeningChannel = nil
                waitForBinding = true
            case .listening(let channel):
                listeningChannel = channel
                waitForBinding = false
            case .shutdown:
                listeningChannel = nil
                waitForBinding = false
            }
            lifecycle = .shutdown
            let task = Task { [self, listeningChannel, waitForBinding] in
                let channel = waitForBinding ? await boundChannelAfterStarting() : listeningChannel
                connections.stopAdmissionAndBeginGracefulShutdown()
                await gate.shutdown()
                work.cancelAll()
                var firstError: Error?
                if let channel {
                    do { try await channel.close().get() } catch { firstError = error }
                }
                _ = await work.waitUntilIdle(for: configuration.shutdownGracePeriod)
                work.detachAndCancelAll()
                connections.closeAll()
                await connections.waitUntilIdle()
                do { try await group.shutdownGracefully() } catch {
                    if firstError == nil { firstError = error }
                }
                if let firstError { throw firstError }
            }
            shutdownTask = task
            return task
        }
        try await task.value
    }

    private func validateBindingPolicy() throws {
        let loopbackHosts: Set<String> = ["127.0.0.1", "::1", "localhost"]
        guard
            loopbackHosts.contains(configuration.host) || configuration.apiKey != nil
                || configuration.allowUnsafeNonLoopbackWithoutAPIKey
        else {
            throw SystemOneHTTPServerError.insecureNonLoopbackBinding
        }
    }

    private func cleanupAfterFailedStart() -> Task<Void, Error> {
        lock.withLock {
            if let shutdownTask { return shutdownTask }
            lifecycle = .shutdown
            let task = Task { [self] in
                connections.stopAdmissionAndBeginGracefulShutdown()
                await gate.shutdown()
                work.detachAndCancelAll()
                connections.closeAll()
                try await group.shutdownGracefully()
            }
            shutdownTask = task
            return task
        }
    }

    private func boundChannelAfterStarting() async -> Channel? {
        while true {
            let snapshot = lock.withLock { (future: bindingFuture, resolved: startupResolved) }
            if let future = snapshot.future { return try? await future.get() }
            if snapshot.resolved { return nil }
            await Task.yield()
        }
    }
}

private final class ConnectionRegistry: @unchecked Sendable {
    private let limit: Int
    private let lock = NSLock()
    private var channels: [ObjectIdentifier: Channel] = [:]
    private var accepting = true
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []

    init(limit: Int) {
        self.limit = limit
    }

    func admit(_ channel: Channel) -> Bool {
        let identifier = ObjectIdentifier(channel)
        let admitted = lock.withLock { () -> Bool in
            guard accepting, channels.count < limit else { return false }
            channels[identifier] = channel
            return true
        }
        guard admitted else { return false }
        channel.closeFuture.whenComplete { [weak self] _ in self?.remove(identifier) }
        return true
    }

    func stopAdmissionAndBeginGracefulShutdown() {
        let snapshot = lock.withLock { () -> [Channel] in
            accepting = false
            return Array(channels.values)
        }
        for channel in snapshot where channel.isActive {
            channel.pipeline.fireUserInboundEventTriggered(GracefulShutdownEvent())
        }
    }

    func closeAll() {
        let snapshot = lock.withLock { Array(channels.values) }
        for channel in snapshot { channel.close(promise: nil) }
    }

    func waitUntilIdle() async {
        await withCheckedContinuation { continuation in
            let resumeNow = lock.withLock { () -> Bool in
                guard !channels.isEmpty else { return true }
                idleWaiters.append(continuation)
                return false
            }
            if resumeNow { continuation.resume() }
        }
    }

    private func remove(_ identifier: ObjectIdentifier) {
        let waiters = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
            channels.removeValue(forKey: identifier)
            guard channels.isEmpty else { return [] }
            let waiters = idleWaiters
            idleWaiters.removeAll()
            return waiters
        }
        for waiter in waiters { waiter.resume() }
    }
}

private final class RequestCapacity: @unchecked Sendable {
    private let limit: Int
    private let lock = NSLock()
    private var count = 0

    init(limit: Int) { self.limit = limit }

    func acquire() -> Bool {
        lock.withLock {
            guard count < limit else { return false }
            count += 1
            return true
        }
    }

    func release() {
        lock.withLock {
            precondition(count > 0)
            count -= 1
        }
    }
}

private final class AsyncWorkRegistry: @unchecked Sendable {
    private struct Entry {
        var cancel: (@Sendable () -> Void)?
    }

    private let lock = NSLock()
    private var accepting = true
    private var detached = false
    private var entries: [UUID: Entry] = [:]
    private var idleWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]

    func reserve() -> UUID? {
        lock.withLock {
            guard accepting, !detached else { return nil }
            let identifier = UUID()
            entries[identifier] = Entry()
            return identifier
        }
    }

    func attach(_ identifier: UUID, cancel: @escaping @Sendable () -> Void) {
        let cancelNow = lock.withLock { () -> Bool in
            guard entries[identifier] != nil else { return true }
            entries[identifier]?.cancel = cancel
            return !accepting
        }
        if cancelNow { cancel() }
    }

    /// Atomically keeps the entry live while enqueueing its completion. The callback
    /// rechecks attachment before touching event-loop-confined state and removes the
    /// entry only after the callback has run.
    func enqueueCompletion(
        _ identifier: UUID,
        on eventLoop: EventLoop,
        _ body: @escaping @Sendable () -> Void
    ) {
        lock.lock()
        guard entries[identifier] != nil, !detached else {
            lock.unlock()
            return
        }
        eventLoop.execute { [weak self] in
            guard let self, self.beginCompletion(identifier) else { return }
            body()
            self.finishCompletion(identifier)
        }
        lock.unlock()
    }

    private func beginCompletion(_ identifier: UUID) -> Bool {
        lock.withLock { entries[identifier] != nil && !detached }
    }

    private func finishCompletion(_ identifier: UUID) {
        let waiters = lock.withLock { () -> [CheckedContinuation<Bool, Never>] in
            guard entries.removeValue(forKey: identifier) != nil, entries.isEmpty else { return [] }
            let waiters = Array(idleWaiters.values)
            idleWaiters.removeAll()
            return waiters
        }
        for waiter in waiters { waiter.resume(returning: true) }
    }

    func cancel(_ identifier: UUID) {
        let cancellation = lock.withLock { entries[identifier]?.cancel }
        cancellation?()
    }

    func cancelAll() {
        let cancellations = lock.withLock { () -> [@Sendable () -> Void] in
            accepting = false
            return entries.values.compactMap(\.cancel)
        }
        for cancel in cancellations { cancel() }
    }

    func detachAndCancelAll() {
        let result = lock.withLock {
            () -> ([@Sendable () -> Void], [CheckedContinuation<Bool, Never>]) in
            accepting = false
            detached = true
            let cancellations = entries.values.compactMap(\.cancel)
            entries.removeAll()
            let waiters = Array(idleWaiters.values)
            idleWaiters.removeAll()
            return (cancellations, waiters)
        }
        for cancel in result.0 { cancel() }
        for waiter in result.1 { waiter.resume(returning: true) }
    }

    func waitUntilIdle(for amount: TimeAmount) async -> Bool {
        await withCheckedContinuation { continuation in
            let identifier = UUID()
            let resumeNow = lock.withLock { () -> Bool in
                guard !entries.isEmpty else { return true }
                idleWaiters[identifier] = continuation
                return false
            }
            if resumeNow {
                continuation.resume(returning: true)
                return
            }
            let delay = DispatchTimeInterval.nanoseconds(Self.dispatchNanoseconds(amount))
            DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
                self?.timeoutWaiter(identifier)
            }
        }
    }

    private func timeoutWaiter(_ identifier: UUID) {
        let waiter = lock.withLock { idleWaiters.removeValue(forKey: identifier) }
        waiter?.resume(returning: false)
    }

    private static func dispatchNanoseconds(_ amount: TimeAmount) -> Int {
        Int(min(Int64(Int.max), max(1, amount.nanoseconds)))
    }
}

private enum RawFramingFailure: Sendable {
    case requestHeadersTooLarge
    case badRequest
}

private struct RawFramingFailureEvent: Sendable {
    let failure: RawFramingFailure
}

private struct RawHeaderTimeoutEvent: Sendable {}

/// Bounds raw HTTP metadata before NIO's decoder can allocate one String per field.
/// It also tracks framing so pipelined headers and chunked trailers are budgeted without
/// mistaking CRLF bytes inside a request body for metadata boundaries.
private final class HTTP1FramingBudgetHandler: ChannelInboundHandler {
    typealias InboundIn = ByteBuffer
    typealias InboundOut = ByteBuffer

    private enum Phase {
        case headers
        case fixedBody(Int)
        case chunkSize
        case chunkData(Int)
        case chunkDataTerminator(Int)
        case trailers
    }

    private let maximumHeaderBytes: Int
    private let maximumHeaderCount: Int
    private let maximumTrailerBytes: Int
    private let headerDeadline: TimeAmount
    private var phase: Phase = .headers
    private var heldHeaders: [UInt8] = []
    private var chunkSizeLine: [UInt8] = []
    private var heldTrailers: [UInt8] = []
    private var headerTimer: Scheduled<Void>?
    private var failed = false
    private var pendingFailure: RawFramingFailure?

    init(configuration: SystemOneHTTPServerConfiguration) {
        maximumHeaderBytes = configuration.maximumHeaderBytes
        maximumHeaderCount = configuration.maximumHeaderCount
        maximumTrailerBytes = configuration.maximumTrailerBytes
        headerDeadline = configuration.headerDeadline
        heldHeaders.reserveCapacity(min(configuration.maximumHeaderBytes, 4_096))
        heldTrailers.reserveCapacity(min(configuration.maximumTrailerBytes, 1_024))
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard !failed else { return }
        var input = unwrapInboundIn(data)
        var output = context.channel.allocator.buffer(capacity: input.readableBytes)

        while let byte: UInt8 = input.readInteger() {
            guard consume(byte, output: &output, context: context) else {
                forwardValidated(output, context: context)
                if let failure = pendingFailure {
                    pendingFailure = nil
                    context.fireUserInboundEventTriggered(RawFramingFailureEvent(failure: failure))
                }
                return
            }
        }
        forwardValidated(output, context: context)
    }

    func channelInactive(context: ChannelHandlerContext) {
        cancelHeaderTimer()
        context.fireChannelInactive()
    }

    func handlerRemoved(context: ChannelHandlerContext) {
        cancelHeaderTimer()
    }

    private func consume(
        _ byte: UInt8,
        output: inout ByteBuffer,
        context: ChannelHandlerContext
    ) -> Bool {
        switch phase {
        case .headers:
            if heldHeaders.isEmpty { startHeaderTimer(context) }
            heldHeaders.append(byte)
            guard heldHeaders.count <= maximumHeaderBytes else {
                fail(.requestHeadersTooLarge, context: context)
                return false
            }
            guard Self.endsWith(heldHeaders, Self.doubleCRLF) else { return true }
            cancelHeaderTimer()
            if Self.lines(in: heldHeaders).dropFirst().dropLast().count > maximumHeaderCount {
                fail(.requestHeadersTooLarge, context: context)
                return false
            }
            guard let nextPhase = parseHeaderBlock(heldHeaders) else {
                fail(.badRequest, context: context)
                return false
            }
            output.writeBytes(heldHeaders)
            heldHeaders.removeAll(keepingCapacity: true)
            phase = nextPhase
            return true

        case .fixedBody(let remaining):
            output.writeInteger(byte)
            phase = remaining == 1 ? .headers : .fixedBody(remaining - 1)
            return true

        case .chunkSize:
            output.writeInteger(byte)
            chunkSizeLine.append(byte)
            guard chunkSizeLine.count <= 1_024 else {
                fail(.badRequest, context: context)
                return false
            }
            guard Self.endsWith(chunkSizeLine, Self.crlf) else { return true }
            guard let size = parseChunkSize(chunkSizeLine) else {
                fail(.badRequest, context: context)
                return false
            }
            chunkSizeLine.removeAll(keepingCapacity: true)
            phase = size == 0 ? .trailers : .chunkData(size)
            return true

        case .chunkData(let remaining):
            output.writeInteger(byte)
            phase = remaining == 1 ? .chunkDataTerminator(0) : .chunkData(remaining - 1)
            return true

        case .chunkDataTerminator(let offset):
            guard byte == Self.crlf[offset] else {
                fail(.badRequest, context: context)
                return false
            }
            output.writeInteger(byte)
            phase = offset == 1 ? .chunkSize : .chunkDataTerminator(1)
            return true

        case .trailers:
            heldTrailers.append(byte)
            guard heldTrailers.count <= maximumTrailerBytes else {
                fail(.requestHeadersTooLarge, context: context)
                return false
            }
            let empty = heldTrailers == Self.crlf
            let complete = Self.endsWith(heldTrailers, Self.doubleCRLF)
            guard empty || complete else { return true }
            guard validateTrailerBlock(heldTrailers) else {
                fail(.badRequest, context: context)
                return false
            }
            output.writeBytes(heldTrailers)
            heldTrailers.removeAll(keepingCapacity: true)
            phase = .headers
            return true
        }
    }

    private func parseHeaderBlock(_ bytes: [UInt8]) -> Phase? {
        let lines = Self.lines(in: bytes)
        guard lines.count >= 2, !lines[0].isEmpty else { return nil }
        let requestLine = String(decoding: lines[0], as: UTF8.self)
        guard requestLine.split(separator: " ", omittingEmptySubsequences: false).count == 3 else {
            return nil
        }
        let fieldLines = lines.dropFirst().dropLast()
        guard fieldLines.count <= maximumHeaderCount else { return nil }

        var contentLengths: [String] = []
        var transferEncodings: [String] = []
        for line in fieldLines {
            guard let colon = line.firstIndex(of: 58), colon > line.startIndex else { return nil }
            let nameBytes = line[..<colon]
            guard nameBytes.allSatisfy(Self.isHeaderNameByte) else { return nil }
            let valueStart = line.index(after: colon)
            let name = String(decoding: nameBytes, as: UTF8.self).lowercased()
            let value = String(decoding: line[valueStart...], as: UTF8.self)
                .trimmingCharacters(in: .whitespaces)
            if name == "content-length" { contentLengths.append(value) }
            if name == "transfer-encoding" { transferEncodings.append(value) }
        }

        guard contentLengths.isEmpty || transferEncodings.isEmpty else { return nil }
        if !transferEncodings.isEmpty {
            let codings = transferEncodings.joined(separator: ",").split(
                separator: ",", omittingEmptySubsequences: false
            )
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            guard !codings.isEmpty, !codings.contains(where: \.isEmpty), codings.last == "chunked"
            else {
                return nil
            }
            return .chunkSize
        }
        if !contentLengths.isEmpty {
            guard contentLengths.count == 1,
                !contentLengths[0].contains(","),
                let length = Self.parseDecimal(contentLengths[0])
            else { return nil }
            return length == 0 ? .headers : .fixedBody(length)
        }
        return .headers
    }

    private func validateTrailerBlock(_ bytes: [UInt8]) -> Bool {
        if bytes == Self.crlf { return true }
        let lines = Self.lines(in: bytes)
        let fields = lines.dropLast()
        guard fields.count <= maximumHeaderCount else { return false }
        return fields.allSatisfy { line in
            guard let colon = line.firstIndex(of: 58), colon > line.startIndex else { return false }
            return line[..<colon].allSatisfy(Self.isHeaderNameByte)
        }
    }

    private func parseChunkSize(_ bytes: [UInt8]) -> Int? {
        guard bytes.count >= 3 else { return nil }
        let raw = bytes.dropLast(2)
        let sizeBytes = raw.prefix { $0 != 59 }
        guard !sizeBytes.isEmpty else { return nil }
        var value = 0
        for byte in sizeBytes {
            let digit: Int
            switch byte {
            case 48...57: digit = Int(byte - 48)
            case 65...70: digit = Int(byte - 65 + 10)
            case 97...102: digit = Int(byte - 97 + 10)
            default: return nil
            }
            let (multiplied, overflow1) = value.multipliedReportingOverflow(by: 16)
            let (next, overflow2) = multiplied.addingReportingOverflow(digit)
            guard !overflow1, !overflow2 else { return nil }
            value = next
        }
        return value
    }

    private func startHeaderTimer(_ context: ChannelHandlerContext) {
        headerTimer = context.eventLoop.scheduleTask(in: headerDeadline) { [weak self] in
            guard let self, !self.failed, !self.heldHeaders.isEmpty else { return }
            self.failed = true
            self.headerTimer = nil
            context.fireUserInboundEventTriggered(RawHeaderTimeoutEvent())
        }
    }

    private func cancelHeaderTimer() {
        headerTimer?.cancel()
        headerTimer = nil
    }

    private func fail(_ failure: RawFramingFailure, context: ChannelHandlerContext) {
        guard !failed else { return }
        failed = true
        pendingFailure = failure
        cancelHeaderTimer()
        heldHeaders.removeAll(keepingCapacity: false)
        heldTrailers.removeAll(keepingCapacity: false)
        chunkSizeLine.removeAll(keepingCapacity: false)
    }

    private func forwardValidated(_ output: ByteBuffer, context: ChannelHandlerContext) {
        guard output.readableBytes > 0 else { return }
        context.fireChannelRead(wrapInboundOut(output))
    }

    private static func lines(in bytes: [UInt8]) -> [[UInt8]] {
        var result: [[UInt8]] = []
        var start = 0
        var index = 0
        while index + 1 < bytes.count {
            if bytes[index] == 13, bytes[index + 1] == 10 {
                result.append(Array(bytes[start..<index]))
                index += 2
                start = index
            } else {
                index += 1
            }
        }
        return result
    }

    private static func endsWith(_ bytes: [UInt8], _ suffix: [UInt8]) -> Bool {
        bytes.count >= suffix.count && bytes.suffix(suffix.count).elementsEqual(suffix)
    }

    private static func parseDecimal(_ value: String) -> Int? {
        guard !value.isEmpty else { return nil }
        var result = 0
        for byte in value.utf8 {
            guard byte >= 48, byte <= 57 else { return nil }
            let (multiplied, overflow1) = result.multipliedReportingOverflow(by: 10)
            let (next, overflow2) = multiplied.addingReportingOverflow(Int(byte - 48))
            guard !overflow1, !overflow2 else { return nil }
            result = next
        }
        return result
    }

    private static func isHeaderNameByte(_ byte: UInt8) -> Bool {
        switch byte {
        case 33, 35...39, 42, 43, 45, 46, 48...57, 65...90, 94...122, 124, 126:
            return true
        default:
            return false
        }
    }

    private static let crlf: [UInt8] = [13, 10]
    private static let doubleCRLF: [UInt8] = [13, 10, 13, 10]
}

private struct GracefulShutdownEvent: Sendable {}

private enum GateResult: Sendable {
    case acquired
    case rejected
    case timedOut
    case shutdown
}

private actor ModelConcurrencyGate {
    private struct Waiter {
        let continuation: CheckedContinuation<GateResult, Never>
        let timeout: Task<Void, Never>
    }

    private let maximumActive: Int
    private let maximumQueued: Int
    private var active = 0
    private var accepting = true
    private var order: [UUID] = []
    private var waiters: [UUID: Waiter] = [:]

    init(maximumActive: Int, maximumQueued: Int) {
        self.maximumActive = maximumActive
        self.maximumQueued = maximumQueued
    }

    func acquire(timeout: TimeAmount) async -> GateResult {
        guard accepting, !Task.isCancelled else { return .shutdown }
        if active < maximumActive {
            active += 1
            return .acquired
        }
        guard waiters.count < maximumQueued else { return .rejected }
        let identifier = UUID()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let timeoutTask = Task { [weak self] in
                    do {
                        try await Task.sleep(nanoseconds: UInt64(max(1, timeout.nanoseconds)))
                    } catch {
                        return
                    }
                    await self?.timeout(identifier)
                }
                order.append(identifier)
                waiters[identifier] = Waiter(continuation: continuation, timeout: timeoutTask)
            }
        } onCancel: {
            Task { await self.cancel(identifier) }
        }
    }

    func release() {
        while !order.isEmpty {
            let identifier = order.removeFirst()
            guard let waiter = waiters.removeValue(forKey: identifier) else { continue }
            waiter.timeout.cancel()
            waiter.continuation.resume(returning: .acquired)
            return
        }
        precondition(active > 0)
        active -= 1
    }

    func shutdown() {
        accepting = false
        let pending = waiters.values
        waiters.removeAll()
        order.removeAll()
        for waiter in pending {
            waiter.timeout.cancel()
            waiter.continuation.resume(returning: .shutdown)
        }
    }

    private func timeout(_ identifier: UUID) {
        guard let waiter = waiters.removeValue(forKey: identifier) else { return }
        order.removeAll { $0 == identifier }
        waiter.continuation.resume(returning: .timedOut)
    }

    private func cancel(_ identifier: UUID) {
        guard let waiter = waiters.removeValue(forKey: identifier) else { return }
        order.removeAll { $0 == identifier }
        waiter.timeout.cancel()
        waiter.continuation.resume(returning: .shutdown)
    }
}

private struct InboundRequest {
    let sequence: Int
    let head: HTTPRequestHead
    var body = Data()
    let admitted: Bool
    let keepAlive: Bool
}

private struct OutboundResponse {
    let status: HTTPResponseStatus
    let body: Data
    let origin: String?
    let keepAlive: Bool
    var extraHeaders = HTTPHeaders()
}

private struct ProcessingState {
    let workIdentifier: UUID
    let timeout: Scheduled<Void>
    let origin: String?
}

private enum RequestExpectation: Equatable {
    case none
    case continueRequest
    case unsupported
}

private final class SystemOneHTTPChannelHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let configuration: SystemOneHTTPServerConfiguration
    private let codec: SystemOneWireCodec
    private let modelHandler: SystemOneHTTPHandler
    private let gate: ModelConcurrencyGate
    private let requestCapacity: RequestCapacity
    private let work: AsyncWorkRegistry
    private var current: InboundRequest?
    private var nextSequence = 0
    private var nextResponseSequence = 0
    private var completed: [Int: OutboundResponse] = [:]
    private var processing: [Int: ProcessingState] = [:]
    private var bodyTimer: Scheduled<Void>?
    private var gracefulShutdownRequested = false
    private var terminal = false

    init(
        configuration: SystemOneHTTPServerConfiguration,
        codec: SystemOneWireCodec,
        modelHandler: @escaping SystemOneHTTPHandler,
        gate: ModelConcurrencyGate,
        requestCapacity: RequestCapacity,
        work: AsyncWorkRegistry
    ) {
        self.configuration = configuration
        self.codec = codec
        self.modelHandler = modelHandler
        self.gate = gate
        self.requestCapacity = requestCapacity
        self.work = work
    }

    func channelActive(context: ChannelHandlerContext) {
        context.fireChannelActive()
    }

    func channelInactive(context: ChannelHandlerContext) {
        cancelBodyTimer()
        for state in processing.values {
            state.timeout.cancel()
            work.cancel(state.workIdentifier)
        }
        processing.removeAll()
        if let request = current, request.admitted { requestCapacity.release() }
        current = nil
        context.fireChannelInactive()
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard !terminal else { return }
        switch unwrapInboundIn(data) {
        case .head(let head):
            receiveHead(head, context: context)
        case .body(var buffer):
            receiveBody(&buffer, context: context)
        case .end(let trailers):
            receiveEnd(trailers, context: context)
        }
    }

    private func receiveHead(_ head: HTTPRequestHead, context: ChannelHandlerContext) {
        guard current == nil, !gracefulShutdownRequested else {
            closeWithoutResponse(context)
            return
        }
        let sequence = nextSequence
        nextSequence += 1
        let keepAlive = head.isKeepAlive

        guard decodedMetadataWithinLimits(head) else {
            complete(
                sequence,
                response: jsonError(
                    .requestHeaderFieldsTooLarge,
                    "Request Header Fields Too Large",
                    nil,
                    false
                ), context: context)
            enterTerminal()
            return
        }

        if requiresAuthentication(head), !isAuthorized(head.headers) {
            complete(
                sequence,
                response: jsonError(.unauthorized, "Unauthorized", corsOrigin(head), false),
                context: context)
            enterTerminal()
            return
        }

        if declaredBodyIsInvalidOrOversized(head.headers) {
            complete(
                sequence,
                response: jsonError(.payloadTooLarge, "Payload Too Large", corsOrigin(head), false),
                context: context)
            enterTerminal()
            return
        }

        let expectation = requestExpectation(head)
        guard expectation != .unsupported else {
            complete(
                sequence,
                response: jsonError(
                    .expectationFailed,
                    "Expectation Failed",
                    corsOrigin(head),
                    false
                ), context: context)
            enterTerminal()
            return
        }

        guard requestCapacity.acquire() else {
            complete(
                sequence,
                response: jsonError(.serviceUnavailable, "Service Unavailable", nil, false),
                context: context)
            enterTerminal()
            return
        }

        let request = InboundRequest(
            sequence: sequence, head: head, admitted: true, keepAlive: keepAlive)
        current = request
        scheduleBodyDeadline(context)
        if expectation == .continueRequest {
            writeContinue(context: context)
        }
    }

    private func receiveBody(_ buffer: inout ByteBuffer, context: ChannelHandlerContext) {
        guard var request = current else { return }
        let byteCount = buffer.readableBytes
        let (total, overflow) = request.body.count.addingReportingOverflow(byteCount)
        guard !overflow, total <= configuration.maximumBodyBytes else {
            current = nil
            requestCapacity.release()
            complete(
                request.sequence,
                response: jsonError(
                    .payloadTooLarge,
                    "Payload Too Large",
                    corsOrigin(request.head),
                    false
                ), context: context)
            enterTerminal()
            return
        }
        if let bytes = buffer.readBytes(length: byteCount) {
            request.body.append(contentsOf: bytes)
        }
        current = request
    }

    private func receiveEnd(_ trailers: HTTPHeaders?, context: ChannelHandlerContext) {
        guard let request = current else { return }
        current = nil
        cancelBodyTimer()
        guard decodedTrailersWithinLimits(trailers) else {
            requestCapacity.release()
            complete(
                request.sequence,
                response: jsonError(.badRequest, "Bad Request", corsOrigin(request.head), false),
                context: context)
            enterTerminal()
            return
        }
        dispatch(request, context: context)
    }

    private func dispatch(_ request: InboundRequest, context: ChannelHandlerContext) {
        let origin = corsOrigin(request.head)
        let keepAlive = request.keepAlive && !gracefulShutdownRequested
        let path =
            request.head.uri.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
            .first.map(String.init) ?? request.head.uri
        let known =
            path == "/" || path == "/health" || path == "/v1/models" || path == "/v1/systemone"

        if request.head.method == .OPTIONS, known, origin != nil {
            var headers = HTTPHeaders()
            headers.add(name: "access-control-allow-methods", value: "GET, POST, OPTIONS")
            headers.add(name: "access-control-allow-headers", value: "Authorization, Content-Type")
            finishImmediate(
                request,
                response: OutboundResponse(
                    status: .ok, body: Data(), origin: origin, keepAlive: keepAlive,
                    extraHeaders: headers
                ), context: context)
            return
        }
        guard known else {
            finishImmediate(
                request, response: jsonError(.notFound, "Not Found", origin, keepAlive),
                context: context)
            return
        }

        switch (request.head.method, path) {
        case (.GET, "/"), (.GET, "/health"):
            finishImmediate(
                request, response: json(.ok, Data(codec.profile.healthJSON.utf8), origin, keepAlive), context: context)
        case (.GET, "/v1/models"):
            finishImmediate(
                request, response: json(.ok, Data(codec.profile.modelsJSON.utf8), origin, keepAlive), context: context)
        case (.POST, "/v1/systemone"):
            evaluate(request, origin: origin, keepAlive: keepAlive, context: context)
        default:
            var headers = HTTPHeaders()
            headers.add(name: "allow", value: path == "/v1/systemone" ? "POST" : "GET")
            finishImmediate(
                request,
                response: OutboundResponse(
                    status: .methodNotAllowed,
                    body: errorBody("Method Not Allowed"),
                    origin: origin,
                    keepAlive: keepAlive,
                    extraHeaders: headers
                ), context: context)
        }
    }

    private func finishImmediate(
        _ request: InboundRequest, response: OutboundResponse, context: ChannelHandlerContext
    ) {
        requestCapacity.release()
        complete(request.sequence, response: response, context: context)
    }

    private func evaluate(
        _ inbound: InboundRequest,
        origin: String?,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        guard let workIdentifier = work.reserve() else {
            requestCapacity.release()
            complete(
                inbound.sequence,
                response: jsonError(.serviceUnavailable, "Service Unavailable", origin, false),
                context: context)
            return
        }

        let sequence = inbound.sequence
        let eventLoop = context.eventLoop
        let contextBox = SendableContext(context)
        let timeout = eventLoop.scheduleTask(in: configuration.requestDeadline) { [weak self] in
            guard let self else { return }
            self.processingTimedOut(sequence: sequence, context: contextBox.value)
        }
        processing[sequence] = ProcessingState(
            workIdentifier: workIdentifier,
            timeout: timeout,
            origin: origin
        )

        let body = inbound.body
        let codec = self.codec
        let handler = self.modelHandler
        let gate = self.gate
        let work = self.work
        let requestCapacity = self.requestCapacity
        let deadline = configuration.requestDeadline
        let task = Task { [weak self] in
            let response: OutboundResponse
            do {
                let request = try codec.decodeRequest(body)
                let admission = await gate.acquire(timeout: deadline)
                switch admission {
                case .acquired:
                    do {
                        let modelResponse = try await handler(request)
                        response =
                            self?.json(
                                .ok, try codec.encodeResponse(modelResponse), origin, keepAlive)
                            ?? OutboundResponse(
                                status: .internalServerError, body: Data(), origin: origin,
                                keepAlive: false)
                    } catch is CancellationError {
                        response =
                            self?.jsonError(.gatewayTimeout, "Gateway Timeout", origin, false)
                            ?? OutboundResponse(
                                status: .gatewayTimeout, body: Data(), origin: origin,
                                keepAlive: false)
                    } catch {
                        response =
                            self?.jsonError(
                                .internalServerError, "Internal Server Error", origin, keepAlive)
                            ?? OutboundResponse(
                                status: .internalServerError, body: Data(), origin: origin,
                                keepAlive: false)
                    }
                    await gate.release()
                case .rejected:
                    response =
                        self?.jsonError(.serviceUnavailable, "Service Unavailable", origin, false)
                        ?? OutboundResponse(
                            status: .serviceUnavailable, body: Data(), origin: origin,
                            keepAlive: false)
                case .timedOut:
                    response =
                        self?.jsonError(.gatewayTimeout, "Gateway Timeout", origin, false)
                        ?? OutboundResponse(
                            status: .gatewayTimeout, body: Data(), origin: origin, keepAlive: false)
                case .shutdown:
                    response =
                        self?.jsonError(.serviceUnavailable, "Service Unavailable", origin, false)
                        ?? OutboundResponse(
                            status: .serviceUnavailable, body: Data(), origin: origin,
                            keepAlive: false)
                }
            } catch {
                response =
                    self?.jsonError(.unprocessableEntity, "Invalid request", origin, keepAlive)
                    ?? OutboundResponse(
                        status: .unprocessableEntity, body: Data(), origin: origin, keepAlive: false
                    )
            }
            requestCapacity.release()
            work.enqueueCompletion(workIdentifier, on: eventLoop) { [weak self] in
                guard contextBox.value.channel.isActive else { return }
                self?.processingFinished(
                    sequence: sequence, response: response, context: contextBox.value)
            }
        }
        work.attach(workIdentifier, cancel: { task.cancel() })
    }

    private func processingTimedOut(sequence: Int, context: ChannelHandlerContext) {
        guard let state = processing.removeValue(forKey: sequence) else { return }
        work.cancel(state.workIdentifier)
        complete(
            sequence,
            response: jsonError(.gatewayTimeout, "Gateway Timeout", state.origin, false),
            context: context
        )
        enterTerminal()
    }

    private func processingFinished(
        sequence: Int, response: OutboundResponse, context: ChannelHandlerContext
    ) {
        guard let state = processing.removeValue(forKey: sequence) else { return }
        state.timeout.cancel()
        complete(sequence, response: response, context: context)
    }

    private func complete(
        _ sequence: Int, response: OutboundResponse, context: ChannelHandlerContext
    ) {
        completed[sequence] = response
        while let response = completed.removeValue(forKey: nextResponseSequence) {
            write(response, context: context)
            nextResponseSequence += 1
            if !response.keepAlive {
                enterTerminal()
                break
            }
        }
        if gracefulShutdownRequested, current == nil, processing.isEmpty,
            nextResponseSequence == nextSequence
        {
            context.close(promise: nil)
        }
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        if event is RawHeaderTimeoutEvent {
            handleRawTerminal(
                status: .requestTimeout,
                detail: "Request Timeout",
                context: context
            )
            return
        }
        if let framing = event as? RawFramingFailureEvent {
            switch framing.failure {
            case .requestHeadersTooLarge:
                handleRawTerminal(
                    status: .requestHeaderFieldsTooLarge,
                    detail: "Request Header Fields Too Large",
                    context: context
                )
            case .badRequest:
                handleRawTerminal(status: .badRequest, detail: "Bad Request", context: context)
            }
            return
        }
        if event is GracefulShutdownEvent {
            gracefulShutdownRequested = true
            cancelBodyTimer()
            if let request = current {
                current = nil
                if request.admitted { requestCapacity.release() }
                closeWithoutResponse(context)
            } else if processing.isEmpty, nextResponseSequence == nextSequence {
                context.close(promise: nil)
            }
            return
        }
        if event is IdleStateHandler.IdleStateEvent {
            if current == nil, processing.isEmpty { context.close(promise: nil) }
            return
        }
        context.fireUserInboundEventTriggered(event)
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        closeWithoutResponse(context)
    }

    private func handleRawTerminal(
        status: HTTPResponseStatus,
        detail: String,
        context: ChannelHandlerContext
    ) {
        if let request = current {
            current = nil
            if request.admitted { requestCapacity.release() }
            complete(
                request.sequence,
                response: jsonError(
                    status,
                    detail,
                    corsOrigin(request.head),
                    false
                ), context: context)
            enterTerminal()
            return
        }
        let sequence = nextSequence
        nextSequence += 1
        complete(
            sequence,
            response: jsonError(status, detail, nil, false),
            context: context
        )
        enterTerminal()
    }

    private func scheduleBodyDeadline(_ context: ChannelHandlerContext) {
        cancelBodyTimer()
        guard !terminal, !gracefulShutdownRequested else { return }
        bodyTimer = context.eventLoop.scheduleTask(in: configuration.bodyDeadline) { [weak self] in
            guard let self else { return }
            self.bodyDeadlineExpired(context: context)
        }
    }

    private func bodyDeadlineExpired(context: ChannelHandlerContext) {
        bodyTimer = nil
        guard !terminal, let request = current else { return }
        current = nil
        if request.admitted { requestCapacity.release() }
        complete(
            request.sequence,
            response: jsonError(
                .requestTimeout,
                "Request Timeout",
                corsOrigin(request.head),
                false
            ), context: context)
        enterTerminal()
    }

    private func cancelBodyTimer() {
        bodyTimer?.cancel()
        bodyTimer = nil
    }

    private func enterTerminal() {
        terminal = true
        cancelBodyTimer()
    }

    private func closeWithoutResponse(_ context: ChannelHandlerContext) {
        enterTerminal()
        context.close(promise: nil)
    }

    private func decodedMetadataWithinLimits(_ head: HTTPRequestHead) -> Bool {
        guard head.headers.count <= configuration.maximumHeaderCount else { return false }
        let requestLineBytes = head.method.rawValue.utf8.count + head.uri.utf8.count + 16
        let (total, overflow) = serializedBytes(in: head.headers).addingReportingOverflow(
            requestLineBytes)
        return !overflow && total <= configuration.maximumHeaderBytes
    }

    private func decodedTrailersWithinLimits(_ trailers: HTTPHeaders?) -> Bool {
        guard let trailers else { return true }
        return trailers.count <= configuration.maximumHeaderCount
            && serializedBytes(in: trailers) <= configuration.maximumTrailerBytes
    }

    private func serializedBytes(in headers: HTTPHeaders) -> Int {
        var total = 2
        for (name, value) in headers {
            let (withName, overflow1) = total.addingReportingOverflow(name.utf8.count)
            let (withValue, overflow2) = withName.addingReportingOverflow(value.utf8.count)
            let (withFraming, overflow3) = withValue.addingReportingOverflow(4)
            guard !overflow1, !overflow2, !overflow3 else { return Int.max }
            total = withFraming
        }
        return total
    }

    private func declaredBodyIsInvalidOrOversized(_ headers: HTTPHeaders) -> Bool {
        let raw = headers["content-length"]
        guard !raw.isEmpty else { return false }
        guard raw.count == 1, headers[canonicalForm: "content-length"].count == 1,
            let length = Self.parseContentLength(raw[0])
        else { return true }
        return length > configuration.maximumBodyBytes
    }

    private func requestExpectation(_ head: HTTPRequestHead) -> RequestExpectation {
        let values = head.headers["expect"]
        guard !values.isEmpty, head.version == .http1_1 else { return .none }
        let tokens = values.flatMap {
            $0.split(separator: ",", omittingEmptySubsequences: false).map {
                $0.trimmingCharacters(in: .whitespaces).lowercased()
            }
        }
        guard !tokens.isEmpty,
            tokens.allSatisfy({ $0 == "100-continue" })
        else {
            return .unsupported
        }
        return requestHasBody(head.headers) ? .continueRequest : .none
    }

    private func requestHasBody(_ headers: HTTPHeaders) -> Bool {
        if let contentLength = headers.first(name: "content-length"),
            let length = Self.parseContentLength(contentLength), length > 0
        {
            return true
        }
        return !headers["transfer-encoding"].isEmpty
    }

    private func writeContinue(context: ChannelHandlerContext) {
        let head = HTTPResponseHead(version: .http1_1, status: .continue)
        context.writeAndFlush(wrapOutboundOut(.head(head)), promise: nil)
    }

    private func requiresAuthentication(_ head: HTTPRequestHead) -> Bool {
        guard configuration.apiKey != nil, head.method == .POST else { return false }
        let path = head.uri.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
            .first
        return path == "/v1/systemone"
    }

    private func isAuthorized(_ headers: HTTPHeaders) -> Bool {
        guard let expected = configuration.apiKey else { return true }
        let raw = headers["authorization"]
        let canonical = headers[canonicalForm: "authorization"]
        guard raw.count == 1, canonical.count == 1 else { return false }
        let value = raw[0]
        guard value.hasPrefix("Bearer ") else { return false }
        let token = PythonSemantics.pythonStrip(String(value.dropFirst("Bearer ".count)))
        return Self.constantTimeEqual(token, expected)
    }

    private func corsOrigin(_ head: HTTPRequestHead) -> String? {
        configuration.cors.allowedOrigin(in: head.headers)
    }

    private func write(_ response: OutboundResponse, context: ChannelHandlerContext) {
        var headers = response.extraHeaders
        headers.replaceOrAdd(name: "content-length", value: String(response.body.count))
        headers.replaceOrAdd(name: "content-type", value: "application/json")
        if let origin = response.origin {
            headers.replaceOrAdd(name: "access-control-allow-origin", value: origin)
            headers.replaceOrAdd(name: "access-control-allow-credentials", value: "true")
            headers.replaceOrAdd(name: "vary", value: "Origin")
        }
        if !response.keepAlive { headers.replaceOrAdd(name: "connection", value: "close") }
        let head = HTTPResponseHead(version: .http1_1, status: response.status, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        if !response.body.isEmpty {
            var buffer = context.channel.allocator.buffer(capacity: response.body.count)
            buffer.writeBytes(response.body)
            context.write(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        }
        let promise = response.keepAlive ? nil : context.eventLoop.makePromise(of: Void.self)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: promise)
        promise?.futureResult.whenComplete { _ in context.close(promise: nil) }
    }

    private func json(
        _ status: HTTPResponseStatus, _ body: Data, _ origin: String?, _ keepAlive: Bool
    ) -> OutboundResponse {
        OutboundResponse(status: status, body: body, origin: origin, keepAlive: keepAlive)
    }

    private func jsonError(
        _ status: HTTPResponseStatus, _ detail: String, _ origin: String?, _ keepAlive: Bool
    ) -> OutboundResponse {
        json(status, errorBody(detail), origin, keepAlive)
    }

    private func errorBody(_ detail: String) -> Data {
        (try? OrderedJSONSerializer().serialize(
            .object([
                .init(key: "detail", value: .string(detail))
            ]))) ?? Data(#"{"detail":"Internal Server Error"}"#.utf8)
    }

    private static func parseContentLength(_ value: String) -> Int? {
        guard !value.isEmpty else { return nil }
        var result = 0
        for byte in value.utf8 {
            guard byte >= 48, byte <= 57 else { return nil }
            let (multiplied, overflow1) = result.multipliedReportingOverflow(by: 10)
            let (next, overflow2) = multiplied.addingReportingOverflow(Int(byte - 48))
            guard !overflow1, !overflow2 else { return nil }
            result = next
        }
        return result
    }

    private static func constantTimeEqual(_ lhs: String, _ rhs: String) -> Bool {
        let key = SymmetricKey(data: Data(rhs.utf8))
        let expectedCode = HMAC<SHA256>.authenticationCode(for: Data(rhs.utf8), using: key)
        return HMAC<SHA256>.isValidAuthenticationCode(
            expectedCode, authenticating: Data(lhs.utf8), using: key)
    }
}

private final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext
    init(_ value: ChannelHandlerContext) { self.value = value }
}

extension NSLock {
    fileprivate func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
