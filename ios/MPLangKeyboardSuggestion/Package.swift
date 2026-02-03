// swift-tools-version:5.8
// MPLangKeyboardSuggestion - Wrapper package for multi-language keyboard testing

import PackageDescription

let package = Package(
    name: "MPLangKeyboardSuggestion",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "MPLangKeyboardSuggestion",
            targets: ["MPLangKeyboardSuggestion"]
        ),
    ],
    dependencies: [
        .package(path: "../MPEngKeyboardSuggestion"),
        .package(path: "../MPJPKeyboardSuggestion"),
    ],
    targets: [
        .target(
            name: "MPLangKeyboardSuggestion",
            dependencies: [
                "MPEngKeyboardSuggestion",
                "MPJPKeyboardSuggestion"
            ],
            path: "Sources/MPLangKeyboardSuggestion"
        ),
    ]
)
