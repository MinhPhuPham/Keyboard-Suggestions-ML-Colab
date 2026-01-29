# MPEngKeyboardSuggestion (Android)

High-performance English keyboard suggestion library for Android using hybrid ML + Trie approach.

## Features

- 🧠 **GRU Neural Network** - Context-aware next-word prediction via TensorFlow Lite
- ⚡ **Trie Completion** - Instant prefix matching from 25k+ word vocabulary
- 📚 **Personalized Learning** - Adapts to user typing patterns (unigram/bigram/recency)
- ✏️ **Typo Correction** - Soundex phonetic matching + keyboard proximity
- 💬 **Custom Shortcuts** - User-defined text expansions

## Installation

### Gradle (AAR)

1. Copy the `MPEngKeyboardSuggestion` module to your project
2. Add to `settings.gradle`:
```groovy
include ':MPEngKeyboardSuggestion'
```

3. Add dependency in your app's `build.gradle`:
```groovy
dependencies {
    implementation project(':MPEngKeyboardSuggestion')
}
```

### Required Assets

Copy these files to `app/src/main/assets/`:

| File | Purpose |
|------|---------|
| `gru_model_fp16.tflite` | TFLite model |
| `word_to_index.json` | Tokenization |
| `vocab_high.json` | High-freq words |
| `vocab_medium.json` | Medium-freq words |
| `vocab_low.json` | Low-freq words |
| `prefix_index.json` | Prefix lookup |
| `soundex_index.json` | Phonetic codes |
| `keyboard_adjacent.json` | Key proximity |

## Quick Start

```kotlin
import com.mp.keyboard.suggestion.MPKeyboardSuggestion

// Create instance (user manages singleton if needed)
val keyboard = MPKeyboardSuggestion(context)

// Get suggestions
val suggestions = keyboard.getSuggestions("how are y")
// Returns: [MPSuggestion(word="you", score=95.0, source=HYBRID), ...]

// Record user selection for learning
keyboard.recordSelection("you", "how are")

// Add shortcuts
keyboard.addShortcut("brb", "be right back")

// Get stats
val stats = keyboard.getStats()
println("Vocab: ${stats.vocabSize}, Learned: ${stats.learnedWords}")

// Clean up when done
keyboard.destroy()
```

## API Reference

### Main Class: `MPKeyboardSuggestion`

```kotlin
class MPKeyboardSuggestion(context: Context) {
    // Configuration
    var gruWeight: Double      // GRU influence (0-1), default: 0.7
    var useGRU: Boolean        // Enable GRU, default: true
    var useLearning: Boolean   // Enable learning, default: true
    
    // Core
    fun getSuggestions(input: String, limit: Int = 5): List<MPSuggestion>
    fun getSuggestionWords(input: String, limit: Int = 5): List<String>
    fun recordSelection(word: String, context: String? = null)
    
    // Shortcuts
    fun addShortcut(input: String, suggestion: String)
    fun removeShortcut(input: String)
    fun getAllShortcuts(): Map<String, List<String>>
    
    // Memory
    fun preload(onComplete: (() -> Unit)? = null)
    fun releaseMemory()
    fun clearLearningData()
    fun getStats(): MPStats
    fun destroy()
}
```

### Data Classes

```kotlin
data class MPSuggestion(
    val word: String,
    val score: Double,
    val source: MPSuggestionSource
)

enum class MPSuggestionSource {
    GRU, TRIE, LEARNING, TYPO, SHORTCUT, HYBRID
}

data class MPStats(
    val vocabSize: Int,
    val learnedWords: Int,
    val learnedBigrams: Int,
    val shortcuts: Int,
    val gruReady: Boolean
)
```

## Architecture

```
MPKeyboardSuggestion (Main API)
├── MPResourceLoader    - Vocab chunks, prefix index
├── MPModelInference    - TFLite GRU model
├── MPTrieHelper        - Prefix search
├── MPLearningManager   - User patterns (SharedPreferences)
├── MPTypoCorrector     - Soundex + edit distance
└── MPShortcutManager   - Custom shortcuts
```

## Singleton Pattern (Optional)

The library does NOT use internal singletons. If you need a singleton:

```kotlin
object KeyboardManager {
    private var instance: MPKeyboardSuggestion? = null
    
    fun getInstance(context: Context): MPKeyboardSuggestion {
        return instance ?: MPKeyboardSuggestion(context.applicationContext).also {
            instance = it
        }
    }
    
    fun destroy() {
        instance?.destroy()
        instance = null
    }
}
```

## Requirements

- Android SDK 21+ (Lollipop)
- TensorFlow Lite 2.14+
- Kotlin 1.8+

## License

MIT License - Minh Phu Pham
