// swift-tools-version:5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MPJPKeyboardSuggestion",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "MPJPKeyboardSuggestion",
            targets: ["MPJPKeyboardSuggestion"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MPJPKeyboardSuggestion",
            dependencies: [],
            path: "Sources/MPJPKeyboardSuggestion",
            resources: [
                // Use .copy() for .mlmodelc directory (pre-compiled model for fast loading)
                .copy("Resources/zenz-v3_1-xsmall_coreml.mlmodelc"),
                .process("Resources/vocab.json"),
                .process("Resources/merges.txt"),
            ]
        ),
        .testTarget(
            name: "MPJPKeyboardSuggestionTests",
            dependencies: ["MPJPKeyboardSuggestion"],
            path: "Tests/MPJPKeyboardSuggestionTests"
        ),
    ]
)
