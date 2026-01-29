# MPEngKeyboardSuggestion

A high-performance English keyboard suggestion library for iOS using hybrid ML + Trie approach.

## Features

- 🧠 **GRU-based ML Prediction** - Context-aware next-word suggestion using CoreML
- ⚡ **O(1) Prefix Search** - Fast word completion with chunked vocabulary
- 📚 **User Learning** - Personalized suggestions based on typing history
- 🔤 **Smart Typo Correction** - Soundex phonetic + keyboard proximity
- ⌨️ **Custom Shortcuts** - User-defined text expansion (e.g., "brb" → "be right back")

## Requirements

- iOS 13.0+
- Swift 5.8+
- CoreML framework

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(path: "../MPEngKeyboardSuggestion")
]
```

Or in Xcode:
1. File → Add Package Dependencies
2. Enter the package path

## Quick Start

```swift
import MPEngKeyboardSuggestion

// Get shared instance
let keyboard = MPKeyboardSuggestion.shared

// Get suggestions
let suggestions = keyboard.getSuggestions(for: "how are y")
// → ["you", "your", "yet", "young", "years"]

// Next-word prediction (ends with space)
let nextWord = keyboard.getSuggestions(for: "how are ")
// → ["you", "they", "we", "things", "people"]

// Record selection for learning
keyboard.recordSelection("you", context: "how are")

// Add custom shortcut
keyboard.addShortcut("brb", "be right back")
keyboard.addShortcut("omw", "on my way")
```

## API Reference

### Main Class: `MPKeyboardSuggestion`

```swift
public final class MPKeyboardSuggestion {
    
    /// Shared singleton instance
    public static let shared: MPKeyboardSuggestion
    
    /// Get suggestions for input text
    /// - Parameters:
    ///   - input: Current input text
    ///   - limit: Max suggestions (default: 5)
    /// - Returns: Array of suggestions with scores
    public func getSuggestions(for input: String, limit: Int = 5) -> [MPSuggestion]
    
    /// Record user selection for learning
    public func recordSelection(_ word: String, context: String? = nil)
    
    /// Add custom shortcut
    public func addShortcut(_ input: String, _ suggestion: String)
    
    /// Remove shortcut
    public func removeShortcut(_ input: String)
    
    /// Preload all data for faster first response
    public func preload(completion: (() -> Void)? = nil)
    
    /// Release memory (keeps high-freq words)
    public func releaseMemory()
    
    /// Get system statistics
    public func getStats() -> MPStats
    
    /// Clear all learning data
    public func clearLearningData()
    
    // Configuration
    public var useGRU: Bool        // Enable/disable ML model (default: true)
    public var useLearning: Bool   // Enable/disable learning (default: true)
    public var gruWeight: Double   // ML weight in reranking (default: 0.7)
}
```

### Models

```swift
public struct MPSuggestion: Equatable {
    public let word: String
    public let score: Double
    public let source: MPSuggestionSource
}

public enum MPSuggestionSource: String {
    case gru        // ML model prediction
    case trie       // Prefix completion
    case learning   // User behavior
    case typo       // Typo correction
    case shortcut   // Custom shortcut
    case hybrid     // Combined sources
}

public struct MPStats {
    public let vocabSize: Int
    public let learnedWords: Int
    public let learnedBigrams: Int
    public let shortcuts: Int
    public let gruReady: Bool
}
```

## Architecture

```
MPKeyboardSuggestion (Main Facade)
├── MPResourceLoader      → Vocab chunks, indices
├── MPModelInference      → CoreML GRU model
├── MPTrieHelper          → Prefix search
├── MPLearningManager     → User behavior
├── MPTypoCorrector       → Soundex + keyboard
└── MPShortcutManager     → Custom shortcuts
```

## Required Bundle Resources

Add these files to your app bundle:

| File | Size | Purpose |
|------|------|---------|
| `gru_keyboard_ios.mlmodelc` | ~10MB | CoreML model |
| `word_to_index.json` | ~300KB | Tokenization |
| `vocab_high.json` | ~50KB | High-freq words |
| `vocab_medium.json` | ~150KB | Medium-freq words |
| `vocab_low.json` | ~100KB | Low-freq words |
| `prefix_index.json` | ~200KB | Prefix lookup |
| `soundex_index.json` | ~50KB | Phonetic codes |
| `keyboard_adjacent.json` | ~1KB | Key proximity |

## Performance

| Metric | Value |
|--------|-------|
| First suggestion | <50ms |
| Subsequent | <5ms |
| Memory (loaded) | ~5MB |
| Bundle size | ~10MB |

## License

MIT License
