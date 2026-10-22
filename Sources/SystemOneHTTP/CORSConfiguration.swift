import Foundation
import NIOHTTP1

/// Explicit cross-origin policy. An empty allowlist disables CORS.
public struct CORSConfiguration: Equatable, Sendable {
    public let allowedOrigins: [String]

    public init(allowedOrigins: [String] = []) {
        self.allowedOrigins = allowedOrigins.filter { !$0.isEmpty && $0 != "*" }
    }

    public static let disabled = CORSConfiguration()

    func allowedOrigin(in headers: HTTPHeaders) -> String? {
        guard let origin = headers.first(name: "origin"), allowedOrigins.contains(origin) else {
            return nil
        }
        return origin
    }
}
