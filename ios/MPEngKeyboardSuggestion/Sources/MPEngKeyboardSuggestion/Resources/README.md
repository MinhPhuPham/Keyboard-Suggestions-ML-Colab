# Resources Directory

Place your model and JSON resource files here. The package uses `Bundle.module` to automatically access these resources.

## Required Files

| File | Purpose | Export From |
|------|---------|-------------|
| `gru_keyboard_ios.mlmodelc` | CoreML model (compiled) | Xcode or coremltools |
| `gru_keyboard_ios.mlpackage` | CoreML model (package) | scripts/convert_to_coreml.py |
| `word_to_index.json` | Tokenization mapping | Notebook Cell 10 |
| `vocab_high.json` | High-freq words | Notebook Cell 10 |
| `vocab_medium.json` | Medium-freq words | Notebook Cell 10 |
| `vocab_low.json` | Low-freq words | Notebook Cell 10 |
| `prefix_index.json` | Prefix lookup | Notebook Cell 10 |
| `soundex_index.json` | Phonetic codes | Notebook Cell 10 |
| `keyboard_adjacent.json` | Key proximity | Notebook Cell 10 |

## How to Add Resources

1. Run the training notebook (`train_gru_model_english_raw.ipynb`) to export all files
2. Download files from Google Drive `/models/gru_keyboard/`
3. Copy files to this `Resources` directory
4. Build your project - resources are bundled via `Bundle.module`

## Usage in Code

```swift
// Default: Uses Bundle.module (resources in this directory)
let keyboard = MPKeyboardSuggestion()

// Custom bundle (if resources are in your app bundle)
let keyboard = MPKeyboardSuggestion(bundle: Bundle.main)
```

## Debugging

If resources aren't loading, check console for:
```
🐛 [MP] DEBUG: [MPResourceLoader] init - vocabSize=0
❌ [MP] ERROR: [MPResourceLoader] loadWordMappings - ❌ word_to_index.json not found
```

This means files are missing from this directory.
