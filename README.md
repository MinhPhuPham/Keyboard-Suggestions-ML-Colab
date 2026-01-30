# Keyboard Suggestions ML

Machine learning models for keyboard next-word suggestions.

## Languages Supported

| Language | Model | Architecture | Size |
|----------|-------|--------------|------|
| **English** | Custom GRU | 6-layer transformer | ~14MB |
| **Japanese** | zenz-v2.5 | GPT-2 (kana-kanji) | ~50MB |

## Quick Start

### English Model

```bash
cd scripts/english
./run_test.sh
```

### Japanese Model

```bash
cd scripts/japanese
./setup.sh                  # First time only
./run_download_and_test.sh  # Download and test
```

## Project Structure

```
KeyboardSuggestionsML/
├── scripts/
│   ├── english/           # English model training & testing
│   └── japanese/          # Japanese model (zenz) download & testing
├── models/
│   ├── english/           # Trained English models
│   └── japanese/          # Downloaded Japanese models
├── ios/                   # iOS SDK (MPEngKeyboardSuggestion)
├── android/               # Android SDK
└── docs/                  # Documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [IOS_INTEGRATION.md](docs/IOS_INTEGRATION.md) | iOS CoreML integration guide |
| [ANDROID_INTEGRATION.md](docs/ANDROID_INTEGRATION.md) | Android TFLite integration |
| [LITERT_MIGRATION_GUIDE.md](docs/LITERT_MIGRATION_GUIDE.md) | TFLite to LiteRT migration |
| [scripts/japanese/README.md](scripts/japanese/README.md) | Japanese model usage |

## Model Details

### English (Custom GRU)
- Next-word prediction from context
- 10,000 word vocabulary
- ~50ms latency on mobile

### Japanese (zenz-v2.5)
- **Kana-Kanji conversion**: ひらがな → 漢字
- **Next character prediction**: 文脈 → 次の文字
- Context-aware suggestions

## Mobile Integration

| Platform | Format | SDK |
|----------|--------|-----|
| iOS | CoreML (.mlpackage) | MPEngKeyboardSuggestion |
| Android | TFLite (.tflite) | MPEngKeyboardSuggestion |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- For Japanese: separate venv (auto-created by setup.sh)

## License

MIT License
