# Japanese Kana-Kanji Conversion

Scripts to download and test zenz Japanese model for iOS keyboard.

## Quick Start

```bash
# 1. Setup (first time only)
./setup.sh

# 2. Download model and test
./run_download_and_test.sh

# 3. Download + CoreML conversion (for iOS)
./run_download.sh
```

## Test Modes

| Mode | Command | Input → Output |
|------|---------|----------------|
| **Convert** | `:convert` | ひらがな → 漢字 (e.g. `ありがとう` → `有難う`) |
| **Predict** | `:predict` | 文脈 → 次の文字 (e.g. `私の名前は` → `田`, `山`) |
| **Generate** | `:generate` | 文脈 → 続きのテキスト |

## Context Usage

Set context with `:ctx` for better predictions:

```
:ctx サッカーの試合を見た。
:predict
今日は
→ 勝った、負けた (sports-related predictions)
```

**Why context matters:**
- Topic awareness (sports → sports words)
- Style matching (formal → formal)
- Grammar correctness
- Natural continuations

## Model Format

| Version | Format |
|---------|--------|
| v1 | `\uEE00<katakana>\uEE01<output>` |
| v2 | `\uEE00<katakana>\uEE02<context>\uEE01<output>` |
| Predict | `\uEE00。\uEE02<leftContext>` |

## Output Files

```
models/japanese/
├── zenz-v2.5-small/                  # HuggingFace model
└── zenz-v2_5-small_coreml.mlpackage  # CoreML for iOS
```