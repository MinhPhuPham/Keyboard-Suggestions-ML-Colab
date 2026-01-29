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
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "MPEngKeyboardSuggestionTests",
            dependencies: ["MPEngKeyboardSuggestion"],
            path: "Tests/MPEngKeyboardSuggestionTests"
        ),
    ]
)
