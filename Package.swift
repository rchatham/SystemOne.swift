// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VonSwift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "VonSwift", targets: ["VonSwift"]),
        .library(name: "VonHTTP", targets: ["VonHTTP"]),
        .executable(name: "von", targets: ["von"]),
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
            name: "VonSwift",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "VonHTTP",
            dependencies: [
                "VonSwift",
                .product(name: "JSON", package: "JSON.swift"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "von",
            dependencies: ["VonSwift", "VonHTTP"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "VonBenchmark",
            dependencies: [
                "VonSwift",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/VonBenchmark",
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
        .testTarget(
            name: "VonHTTPTests",
            dependencies: [
                "VonHTTP",
                "VonSwift",
                .product(name: "JSON", package: "JSON.swift"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
