# Japanese Kana-Kanji Conversion

Scripts to download, convert, and test zenz Japanese model for iOS keyboard.

## Quick Start

```bash
# 1. Setup (first time only)
./setup.sh

# 2. Download model and test
./run_download_and_test.sh

# 3. Download + CoreML conversion (for iOS)
./run_download.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Create Python venv, install dependencies |
| `run_download_and_test.sh` | Download model + interactive test |
| `run_download.sh` | Download model + convert to CoreML |
| `run_test.sh` | Test existing model only |

## Model Info

- **zenz-v2.5-small**: Kana-to-kanji conversion (かな漢字変換)
- **Input**: ひらがな → **Output**: 漢字
- **Example**: `ありがとう` → `有難う`

## Output Files

```
models/japanese/
├── zenz-v2.5-small/          # HuggingFace model
└── zenz-v2_5-small_coreml.mlpackage  # CoreML for iOS
```