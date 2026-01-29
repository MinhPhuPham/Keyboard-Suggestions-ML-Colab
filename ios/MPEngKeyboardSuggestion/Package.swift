// swift-tools-version:5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MPEngKeyboardSuggestion",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "MPEngKeyboardSuggestion",
            targets: ["MPEngKeyboardSuggestion"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MPEngKeyboardSuggestion",
            dependencies: [],
            path: "Sources/MPEngKeyboardSuggestion",
            resources: [
                // Models - CoreML
                .process("Resources/gru_keyboard_ios.mlpackage"),
                // JSON Resources
                .process("Resources/word_to_index.json"),
                .process("Resources/vocab_high.json"),
                .process("Resources/vocab_medium.json"),
                .process("Resources/vocab_low.json"),
                .process("Resources/compact_trie.json"),
                .process("Resources/symspell_index.json"),
                .process("Resources/keyboard_adjacent.json"),
            ]
        ),
        .testTarget(
            name: "MPEngKeyboardSuggestionTests",
            dependencies: ["MPEngKeyboardSuggestion"],
            path: "Tests/MPEngKeyboardSuggestionTests"
        ),
    ]
)
