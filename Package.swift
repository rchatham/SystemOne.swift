// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VonSwift",
    platforms: [
        .macOS(.v13),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            exact: "1.1.6"
        ),
    ],
    targets: [
        .target(
            name: "VonSwift",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
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
            resources: [
                .copy("Resources/Tokenizer"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)