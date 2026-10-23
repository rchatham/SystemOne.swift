import SystemOneHTTP
import SystemOneVon

extension SystemOneWireCodec {
    /// The HTTP tests check parity with Python von's server.
    static let von = SystemOneWireCodec(profile: VonModelProfile())
}
