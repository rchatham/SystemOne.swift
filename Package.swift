// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VonSwift",
    targets: [
        .target(
            name: "VonSwift",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "von",
            dependencies: ["VonSwift"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "VonSwiftTests",
            dependencies: ["VonSwift"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)