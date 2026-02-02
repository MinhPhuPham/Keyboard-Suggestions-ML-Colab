# MPJPKeyboardSuggestion

Japanese keyboard suggestion library using zenz CoreML model.

## Features

- **Kana-Kanji Conversion**: ひらがな → 漢字 (e.g., `ありがとう` → `有難う`)
- **Next Phrase Prediction**: Context → Next phrase (e.g., `ありがとう` → `ございます`)

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(path: "../ios/MPJPKeyboardSuggestion")
]
```

## Usage

```swift
import MPJPKeyboardSuggestion

let keyboard = MPJPKeyboardSuggestion()

// Combined suggestions (recommended)
let suggestions = keyboard.getSuggestions(for: "ありがとう")
for s in suggestions {
    print("\(s.text) (\(s.type)): \(s.probability)")
}

// Kana-Kanji conversion only
let conversions = keyboard.getConversions(for: "ありがとう")

// Next phrase prediction only
let predictions = keyboard.getPredictions(for: "ありがとう")
```

## API

| Method | Description |
|--------|-------------|
| `getSuggestions(for:context:limit:)` | Combined conversions + predictions |
| `getConversions(for:context:limit:)` | Kana-Kanji conversion only |
| `getPredictions(for:limit:)` | Next phrase prediction only |
| `isReady` | Check if model is loaded |

## Model

Uses `zenz-v3.1-xsmall` (~50MB) - a GPT-2 based model for Japanese text.
