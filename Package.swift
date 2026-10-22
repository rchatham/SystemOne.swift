// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SystemOne",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "SystemOne", targets: ["SystemOne"]),
        .library(name: "SystemOneVon", targets: ["SystemOneVon"]),
        .library(name: "SystemOneHTTP", targets: ["SystemOneHTTP"]),
        .executable(name: "systemone", targets: ["SystemOneCLI"]),
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
        .package(
            url: "https://github.com/apple/swift-nio.git",
            exact: "2.84.0"
        ),
        .package(
            url: "https://github.com/rchatham/JSON.swift.git",
            exact: "1.1.0"
        ),
    ],
    targets: [
        .target(
            name: "SystemOne",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "SystemOneVon",
            dependencies: [
                "SystemOne",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "SystemOneHTTP",
            dependencies: [
                "SystemOne",
                "SystemOneVon",
                .product(name: "JSON", package: "JSON.swift"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "SystemOneCLI",
            dependencies: ["SystemOne", "SystemOneVon", "SystemOneHTTP"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "SystemOneBenchmark",
            dependencies: [
                "SystemOne",
                "SystemOneVon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
                        swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SystemOneTests",
            dependencies: ["SystemOne"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SystemOneVonTests",
            dependencies: [
                "SystemOne",
                "SystemOneVon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources/Tokenizer"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SystemOneHTTPTests",
            dependencies: [
                "SystemOneHTTP",
                "SystemOne",
                "SystemOneVon",
                .product(name: "JSON", package: "JSON.swift"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
