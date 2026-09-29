// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VonSwift",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            exact: "1.1.6"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            exact: "0.31.4"
        ),
    ],
    targets: [
        .target(
            name: "VonSwift",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
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
            dependencies: [
                "VonSwift",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources/Tokenizer"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)